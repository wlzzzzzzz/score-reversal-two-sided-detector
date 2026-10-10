#!/usr/bin/env python3
"""Score clean80 cells with the official TempTest or NTS statistic.

This adapter writes one atomic JSON artifact per dataset-generator cell,
allowing model scoring to resume at cell boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TEMPERATURES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
TOKEN_COUNTS = (25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300)
NTS_T1 = 0.7
NTS_T2 = 1.4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def temperature_key(temperature: float, token_count: int) -> str:
    return f"score_t{temperature:.1f}_n{token_count}"


def _masked_prefix_means(values: torch.Tensor, mask: torch.Tensor) -> dict[int, torch.Tensor]:
    cumulative = (values * mask).cumsum(dim=1)
    counts = mask.cumsum(dim=1).clamp_min(1)
    result = {}
    width = values.shape[1]
    for token_count in TOKEN_COUNTS:
        index = min(token_count, width) - 1
        result[token_count] = cumulative[:, index] / counts[:, index]
    return result


def temptest_scores_from_logits(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Published TempTest statistic, algebraically equivalent to src/metrics.py.

    The repository first computes p=softmax(logits), then
    log(sum(p**(1/tau))).  The log-sum-exp expression below is identical but
    avoids materializing six additional full-vocabulary probability tensors.
    Raw TempTest is Human-high, exactly as in the official AUROC script.
    """
    z = logits.float()
    log_z = torch.logsumexp(z, dim=-1)
    label_z = z.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    label_log_probability = label_z - log_z
    output: dict[str, torch.Tensor] = {}
    for temperature in TEMPERATURES:
        log_temperature_norm = (
            torch.logsumexp(z / temperature, dim=-1) - log_z / temperature
        )
        per_token = log_temperature_norm - (
            (1.0 / temperature) - 1.0
        ) * label_log_probability
        for token_count, score in _masked_prefix_means(per_token, mask).items():
            output[temperature_key(temperature, token_count)] = score
    return output


def nts_score_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    t1: float = NTS_T1,
    t2: float = NTS_T2,
) -> torch.Tensor:
    """Official normalized temperature sensitivity (TS_norm).

    For d(v)=log p_t1(v)-log p_t2(v), d(v)=a*z(v)-c.  Therefore its
    expectation and variance under p_1 can be computed from the first two
    logit moments.  This is mathematically identical to the official dense
    vocabulary implementation while using substantially less GPU memory.
    """
    z = logits.float()
    log_z1 = torch.logsumexp(z / t1, dim=-1)
    log_z2 = torch.logsumexp(z / t2, dim=-1)
    difference_in_inverse_temperature = (1.0 / t1) - (1.0 / t2)
    normalization_difference = log_z1 - log_z2
    label_z = z.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    label_sensitivity = (
        difference_in_inverse_temperature * label_z - normalization_difference
    )

    probabilities = torch.softmax(z, dim=-1)
    expected_z = (probabilities * z).sum(dim=-1)
    expected_z_squared = (probabilities * z.square()).sum(dim=-1)
    variance_z = (expected_z_squared - expected_z.square()).clamp_min(0.0)
    expected_sensitivity = (
        difference_in_inverse_temperature * expected_z - normalization_difference
    )
    sensitivity_std = abs(difference_in_inverse_temperature) * variance_z.sqrt()

    count = mask.sum(dim=1).clamp_min(1.0)
    observed = ((label_sensitivity * mask).sum(dim=1) / count).abs()
    expected = (expected_sensitivity * mask).sum(dim=1) / count
    std = (sensitivity_std * mask).sum(dim=1) / count
    return (observed - expected) / std.clamp_min(1e-12)


def iter_rows(cell: dict) -> list[dict]:
    path = Path(cell["data_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    lengths = {len(value) for value in payload.values()}
    if lengths != {cell["source_count"]}:
        raise RuntimeError(f"unaligned streams in {path}: {sorted(lengths)}")
    rows = []
    for method, texts in payload.items():
        normalized_method = "human" if method == "original" else method
        for source_index, text in enumerate(texts):
            sample_suffix = "human" if normalized_method == "human" else normalized_method
            rows.append({
                "sample_id": f"{cell['dataset']}:{cell['generator']}:{source_index}:{sample_suffix}",
                "source_id": f"{cell['dataset']}:{source_index}",
                "dataset": cell["dataset"],
                "generator": cell["generator"],
                "watermark_method": normalized_method,
                "label": 0 if normalized_method == "human" else 1,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text": text,
            })
    return rows


def load_model(model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda()
    model.eval()
    return tokenizer, model


def score_batch(method: str, texts: list[str], tokenizer, model) -> tuple[list[dict], list[int]]:
    if method == "temptest":
        encoding = tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max(TOKEN_COUNTS) + 1,
            return_tensors="pt",
            return_token_type_ids=False,
        )
    else:
        encoding = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_token_type_ids=False,
        )
    encoding = {key: value.cuda(non_blocking=True) for key, value in encoding.items()}
    token_lengths = encoding["attention_mask"].sum(dim=1).cpu().tolist()
    with torch.inference_mode():
        logits = model(**encoding, use_cache=False).logits[:, :-1, :]
        labels = encoding["input_ids"][:, 1:]
        mask = encoding["attention_mask"][:, 1:].float()
        if method == "temptest":
            score_tensors = temptest_scores_from_logits(logits, labels, mask)
            score_rows = []
            for row_index in range(len(texts)):
                score_rows.append({
                    key: float(value[row_index].detach().cpu())
                    for key, value in score_tensors.items()
                })
        else:
            scores = nts_score_from_logits(logits, labels, mask)
            score_rows = [{"score_ts_norm": float(value)} for value in scores.detach().cpu()]
    return score_rows, [int(value) for value in token_lengths]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("temptest", "nts"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--max-cells", type=int)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cells = sorted(manifest["cells"], key=lambda item: item["cell"])
    cells = [cell for index, cell in enumerate(cells) if index % args.num_shards == args.shard_index]
    if args.max_cells is not None:
        cells = cells[: args.max_cells]
    tokenizer, model = load_model(args.model)

    for cell in cells:
        output_path = args.output_dir / f"{cell['dataset']}_{cell['generator']}.json"
        if output_path.is_file():
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if existing.get("status") == "PASS" and existing.get("method") == args.method:
                print(f"SKIP {cell['cell']}", flush=True)
                continue
        started = time.time()
        rows = iter_rows(cell)
        if args.sample_limit is not None:
            rows = rows[: args.sample_limit]
        output_rows = []
        for start in range(0, len(rows), args.batch_size):
            part = rows[start : start + args.batch_size]
            scores, token_lengths = score_batch(
                args.method, [row["text"] for row in part], tokenizer, model
            )
            for row, score, token_length in zip(part, scores, token_lengths):
                output_row = {key: value for key, value in row.items() if key != "text"}
                output_row["input_token_length"] = token_length
                output_row.update(score)
                output_rows.append(output_row)
            if start and start % (50 * args.batch_size) == 0:
                print(
                    f"{cell['cell']} {start}/{len(rows)} elapsed={time.time()-started:.1f}s",
                    flush=True,
                )
        payload = {
            "schema_version": "official-temperature-baseline-cell-v1",
            "status": "PASS",
            "method": args.method,
            "cell": cell["cell"],
            "split": cell["split"],
            "input_path": str(Path(cell["data_path"]).resolve()),
            "input_sha256": sha256(Path(cell["data_path"])),
            "model_path": str(args.model.resolve()),
            "model_config_sha256": sha256(args.model / "config.json"),
            "official_parameters": (
                {"temperatures": TEMPERATURES, "token_counts": TOKEN_COUNTS, "raw_direction": "human_high"}
                if args.method == "temptest"
                else {"criterion": "TS_norm", "t1": NTS_T1, "t2": NTS_T2, "max_length": 512}
            ),
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "batch_size": args.batch_size,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            },
            "row_count": len(output_rows),
            "elapsed_seconds": time.time() - started,
            "rows": output_rows,
        }
        temporary = output_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(output_path)
        print(
            f"DONE {cell['cell']} rows={len(output_rows)} elapsed={payload['elapsed_seconds']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
