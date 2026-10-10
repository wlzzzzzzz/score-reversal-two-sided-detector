#!/usr/bin/env python3
"""Batched classical baseline scoring for the 80 benchmark cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from model import load_model, load_tokenizer


METHOD_FIELD = "watermark_method"




def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)




def scalar_statistics(logits: torch.Tensor, labels: torch.Tensor,
                      chunk_tokens: int = 256) -> dict[str, float]:
    """Classic statistics with token-chunked vocabulary tensors."""
    token_count = int(labels.numel()); vocab = int(logits.shape[-1])
    target_sum = mean_sum = variance_sum = entropy_sum = rank_sum = logrank_sum = 0.0
    for start in range(0, token_count, chunk_tokens):
        end = min(start + chunk_tokens, token_count)
        current_logits = logits[:, start:end].reshape(-1, vocab)
        current_labels = labels[:, start:end].reshape(-1)
        log_probs = F.log_softmax(current_logits.float(), dim=-1)
        probs = torch.exp(log_probs)
        target = log_probs.gather(-1, current_labels.unsqueeze(-1)).squeeze(-1)
        mean_ref = (probs * log_probs).sum(dim=-1)
        variance_ref = (probs * log_probs.square()).sum(dim=-1) - mean_ref.square()
        entropy = -(probs * log_probs).sum(dim=-1) / math.log(vocab)
        ordered = current_logits.argsort(dim=-1, descending=True)
        matches = (ordered == current_labels.unsqueeze(-1)).nonzero(as_tuple=False)
        if matches.shape[0] != current_labels.numel():
            raise RuntimeError("rank lookup lost tokens")
        ranks = matches[:, -1].float() + 1.0
        target_sum += float(target.sum().item()); mean_sum += float(mean_ref.sum().item())
        variance_sum += float(variance_ref.clamp_min(1e-9).sum().item())
        entropy_sum += float(entropy.sum().item()); rank_sum += float(ranks.sum().item())
        logrank_sum += float(torch.log(ranks).sum().item())
        del log_probs, probs, target, mean_ref, variance_ref, entropy, ordered, matches, ranks
    discrepancy = (target_sum - mean_sum) / math.sqrt(max(variance_sum, 1e-9))
    likelihood = target_sum / token_count
    return {
        "likelihood": likelihood, "fastdetectgpt": float(discrepancy),
        "rank": -rank_sum / token_count, "logrank": -logrank_sum / token_count,
        "entropy": entropy_sum / token_count,
    }




def batches(lengths: list[int], max_batch_size: int, token_budget: int):
    order = sorted(range(len(lengths)), key=lengths.__getitem__)
    batch: list[int] = []
    for index in order:
        proposed = batch + [index]
        if batch and (len(proposed) > max_batch_size or max(lengths[i] for i in proposed) * len(proposed) > token_budget):
            yield batch
            batch = [index]
        else:
            batch = proposed
    if batch:
        yield batch


@torch.inference_mode()
def score_stream(model, tokenizer, texts: list[str], device: str, max_tokens: int,
                 max_batch_size: int, token_budget: int) -> list[dict]:
    encoded = tokenizer(
        texts, add_special_tokens=True, return_attention_mask=False,
        return_token_type_ids=False, truncation=False,
    )["input_ids"]
    full_lengths = [len(ids) for ids in encoded]
    clipped = [ids[:max_tokens] for ids in encoded]
    lengths = [len(ids) for ids in clipped]
    if min(lengths, default=0) < 2:
        raise RuntimeError("text has fewer than two tokens")
    result: list[dict | None] = [None] * len(texts)
    pad_id = int(tokenizer.pad_token_id)
    for indices in batches(lengths, max_batch_size, token_budget):
        width = max(lengths[index] for index in indices)
        input_ids = torch.full((len(indices), width), pad_id, dtype=torch.long, device=device)
        attention = torch.zeros((len(indices), width), dtype=torch.long, device=device)
        for offset, index in enumerate(indices):
            ids = torch.tensor(clipped[index], dtype=torch.long, device=device)
            input_ids[offset, :len(ids)] = ids
            attention[offset, :len(ids)] = 1
        batch_logits = model(input_ids=input_ids, attention_mask=attention).logits
        for offset, index in enumerate(indices):
            length = lengths[index]
            logits = batch_logits[offset:offset + 1, :length - 1]
            labels = input_ids[offset:offset + 1, 1:length]
            scalar = scalar_statistics(logits, labels)
            values = {
                **scalar, "input_token_length": full_lengths[index],
                "token_length": length - 1, "context_truncated": full_lengths[index] > max_tokens,
            }
            if not all(math.isfinite(float(v)) for v in values.values() if not isinstance(v, bool)):
                raise RuntimeError("non-finite proxy statistic")
            result[index] = values
        del batch_logits, input_ids, attention
    if any(item is None for item in result):
        raise RuntimeError("scoring lost rows")
    return [dict(item) for item in result if item is not None]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--proxy-key", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", default="models")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--batch-token-budget", type=int, default=1200)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if args.max_tokens != int(source["max_tokens"]):
        raise RuntimeError("source/worker max token mismatch")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    audit = {
        "source_manifest_sha256": sha256(args.source_manifest), "model_name": args.model_name,
        "proxy_key": args.proxy_key, "max_tokens": args.max_tokens,
        "config": {"statistics": ["likelihood", "entropy", "rank", "logrank", "fastdetectgpt"]},
        "worker_sha256": sha256(Path(__file__)),
    }
    config_hash = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.model_name, args.cache_dir)
    tokenizer.padding_side = "right"
    model = load_model(args.model_name, args.device, args.cache_dir).eval()
    selected = [cell for index, cell in enumerate(source["cells"]) if index % args.shard_count == args.shard_index]
    completed = total = truncated = 0
    started_all = time.perf_counter()
    for position, cell in enumerate(selected, 1):
        output = args.output_dir / f"{cell['dataset']}_{cell['generator']}.json"
        if output.is_file():
            try:
                cached = json.loads(output.read_text(encoding="utf-8"))
                cached_config = cached.get("config", {})
                compatible = (
                    cached.get("row_count") == cell["expected_rows"]
                    and cached_config.get("source_manifest_sha256") == audit["source_manifest_sha256"]
                    and cached_config.get("model_name") == audit["model_name"]
                    and int(cached_config.get("max_tokens", -1)) == audit["max_tokens"]
                    and cached_config.get("config") == audit["config"]
                )
                if compatible:
                    completed += 1; total += cached["row_count"]; truncated += cached["truncated_rows"]
                    print(f"[skip {position}/{len(selected)}] {cell['cell']}", flush=True)
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        path = Path(cell["dataset_file"])
        if sha256(path) != cell["sha256"]:
            raise RuntimeError(f"input hash changed: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        streams = [("human", 0, payload["original"])] + [
            (method, 1, payload[method]) for method in source["methods"]
        ]
        rows = []; started = time.perf_counter()
        for method, label, values in streams:
            scores = score_stream(model, tokenizer, [str(v) for v in values], args.device,
                                  args.max_tokens, args.max_batch_size,
                                  args.batch_token_budget)
            for index, (text, score) in enumerate(zip(values, scores)):
                rows.append({
                    "sample_id": f"{cell['dataset']}:{cell['generator']}:{index}:{method}",
                    "source_id": f"{cell['dataset']}:{index}", "dataset": cell["dataset"],
                    "generator": cell["generator"], METHOD_FIELD: method, "label": label,
                    "text_sha256": hashlib.sha256(str(text).encode()).hexdigest(), **score,
                })
        if len(rows) != cell["expected_rows"]:
            raise RuntimeError(f"row mismatch: {cell['cell']}")
        cell_truncated = sum(bool(row["context_truncated"]) for row in rows)
        elapsed = time.perf_counter() - started
        atomic_json(output, {
            "schema_version": "clean80-fulltext-proxy-cell-v1", "config": audit,
            "config_hash": config_hash, "cell": cell["cell"], "input_sha256": cell["sha256"],
            "row_count": len(rows), "truncated_rows": cell_truncated,
            "elapsed_seconds": elapsed, "seconds_per_row": elapsed / len(rows), "rows": rows,
        })
        completed += 1; total += len(rows); truncated += cell_truncated
        print(f"[done {position}/{len(selected)}] {cell['cell']} rows={len(rows)} "
              f"truncated={cell_truncated} elapsed={elapsed:.1f}s", flush=True)
    print(json.dumps({"status": "PASS", "proxy": args.proxy_key, "cells": completed,
                      "rows": total, "truncated_rows": truncated,
                      "elapsed_seconds": time.perf_counter() - started_all}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
