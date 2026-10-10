#!/usr/bin/env python3
"""Recompute clean41 Binoculars/AdaDetectGPT scores at 128 and 160 tokens.

The two token budgets are derived from one 160-token pair of Falcon forwards.
The shared witness and scoring functions also support the full-text baseline runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADA_ROOT = Path("external_code/AdaDetectGPT/scripts")
ADA_BETA = [0.0, -0.011333, -0.037667, -0.056667, -0.281667, -0.592, 0.157833, 0.727333]
ADA_CONFIG = {"start": -32, "end": 0, "n_bases": 7, "spline_order": 2, "intercept": 1}
CAPS = (128, 160)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def normalize_generator(value: str) -> str:
    return "falcon" if value in {"falcon", "falcon-7b", "falcon_7B"} else value


class FastFixedBSplineWitness(torch.nn.Module):
    """Exact fixed Ada witness without materializing all seven basis columns."""

    def __init__(self, ada_root: Path, device: torch.device):
        super().__init__()
        if str(ada_root) not in sys.path:
            sys.path.insert(0, str(ada_root))
        from BSpline import BSpline  # pylint: disable=import-outside-toplevel

        spline = BSpline(**ADA_CONFIG)
        self.start = float(spline.start)
        self.order = int(spline.spline_order)
        self.register_buffer("knots", spline.knots.to(device=device, dtype=torch.float32))
        self.register_buffer(
            "beta", torch.tensor(ADA_BETA, dtype=torch.float32, device=device),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        from helper import _torch_splev  # pylint: disable=import-outside-toplevel

        shape = values.shape
        flat = values.clamp_min(self.start).reshape(-1).float()
        result = _torch_splev(flat, self.knots, self.beta[1:], self.order)
        return (result + self.beta[0]).reshape(shape)


def load_witness(ada_root: Path, device: torch.device):
    if str(ada_root) not in sys.path:
        sys.path.insert(0, str(ada_root))
    return FastFixedBSplineWitness(ada_root, device).eval()


def prefix_scores(
    input_ids: torch.Tensor,
    observer_logits: torch.Tensor,
    performer_logits: torch.Tensor,
    witness,
    caps: tuple[int, ...] = CAPS,
    vocab_chunk_size: int = 4096,
) -> dict[int, tuple[float, float]]:
    """Return machine-high Binoculars and Ada scores for each token cap.

    The official formulas require expectations over the complete vocabulary.
    Evaluating the B-spline witness on ``tokens x vocabulary x bases`` at once
    has a multi-GB transient allocation for Falcon. The vocabulary reduction
    below is mathematically identical, but bounds that transient allocation.
    """
    length = int(input_ids.numel())
    if length < 2:
        return {cap: (0.0, 0.0) for cap in caps}
    observer_logits = observer_logits[:length]
    performer_logits = performer_logits[:length]

    # Binoculars performer perplexity term (official implementation uses mean NLL,
    # despite the historical function name ``perplexity``).
    labels = input_ids[1:length]
    performer_log_z = torch.logsumexp(performer_logits.float(), dim=-1)
    observer_log_z = torch.logsumexp(observer_logits.float(), dim=-1)
    selected_logits = performer_logits[:-1].gather(-1, labels[:, None]).squeeze(-1).float()
    selected_log_probs = selected_logits - performer_log_z[:-1]
    token_nll = -selected_log_probs

    cross_entropy = torch.zeros(length, dtype=torch.float32, device=performer_logits.device)
    expected = torch.zeros(length - 1, dtype=torch.float32, device=performer_logits.device)
    second_moment = torch.zeros_like(expected)
    vocabulary = int(performer_logits.shape[-1])
    for begin in range(0, vocabulary, int(vocab_chunk_size)):
        end = min(vocabulary, begin + int(vocab_chunk_size))
        performer_log_probs = (
            performer_logits[:, begin:end].float() - performer_log_z[:, None]
        )
        observer_probs = torch.exp(
            observer_logits[:, begin:end].float() - observer_log_z[:, None]
        )
        cross_entropy.add_(-(observer_probs * performer_log_probs).sum(dim=-1))

        ada_log_probs = witness(performer_log_probs[:-1])
        observer_next_probs = observer_probs[:-1]
        expected.add_((observer_next_probs * ada_log_probs).sum(dim=-1))
        second_moment.add_((observer_next_probs * ada_log_probs.square()).sum(dim=-1))
        del performer_log_probs, observer_probs, ada_log_probs, observer_next_probs

    observed = witness(selected_log_probs)
    variance = (second_moment - expected.square()).clamp_min(0.0)

    output: dict[int, tuple[float, float]] = {}
    for cap in caps:
        capped_length = min(length, int(cap))
        next_count = max(1, capped_length - 1)
        bino_ppl = token_nll[:next_count].mean()
        bino_cross = cross_entropy[:capped_length].mean().clamp_min(1e-8)
        # Official Binoculars is machine-low; negate to the common machine-high direction.
        binoculars_machine_high = -(bino_ppl / bino_cross)

        numerator = (observed[:next_count] - expected[:next_count]).sum()
        denominator = variance[:next_count].sum().clamp_min(1e-8).sqrt()
        ada_machine_high = numerator / denominator
        output[cap] = (
            float(binoculars_machine_high.detach().cpu()),
            float(ada_machine_high.detach().cpu()),
        )
    return output


class DualCapScorer:
    def __init__(self, manifest: dict, worker: dict):
        self.max_tokens = max(CAPS)
        self.batch_size = int(manifest["batch_size"])
        self.vocab_chunk_size = int(manifest.get("vocab_chunk_size", 4096))
        self.observer_device = torch.device(worker["device_observer"])
        self.performer_device = torch.device(worker["device_performer"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            manifest["observer"], trust_remote_code=True, local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        performer_tokenizer = AutoTokenizer.from_pretrained(
            manifest["performer"], trust_remote_code=True, local_files_only=True,
        )
        if self.tokenizer.get_vocab() != performer_tokenizer.get_vocab():
            raise RuntimeError("Falcon observer and performer tokenizers differ")
        self.observer = AutoModelForCausalLM.from_pretrained(
            manifest["observer"], trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).to(self.observer_device).eval()
        self.performer = AutoModelForCausalLM.from_pretrained(
            manifest["performer"], trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).to(self.performer_device).eval()
        self.witness = load_witness(Path(manifest["ada_scripts"]), self.performer_device)

    @torch.inference_mode()
    def score(self, texts: list[str]) -> list[dict]:
        rows: list[dict] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoding = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_tokens, return_token_type_ids=False,
            )
            observer_inputs = {key: value.to(self.observer_device) for key, value in encoding.items()}
            performer_inputs = {key: value.to(self.performer_device) for key, value in encoding.items()}
            observer_logits = self.observer(**observer_inputs).logits
            performer_logits = self.performer(**performer_inputs).logits
            lengths = encoding["attention_mask"].sum(dim=1).tolist()
            for index, length in enumerate(lengths):
                length = int(length)
                scores = prefix_scores(
                    performer_inputs["input_ids"][index, :length],
                    observer_logits[index, :length].to(self.performer_device),
                    performer_logits[index, :length], self.witness,
                    vocab_chunk_size=self.vocab_chunk_size,
                )
                item = {"token_length": length}
                for cap, (binoculars, ada) in scores.items():
                    item[f"binoculars_{cap}"] = binoculars
                    item[f"adadetectgpt_pretrained_{cap}"] = ada
                rows.append(item)
            del observer_logits, performer_logits
        return rows


def completed(path: Path, manifest: dict, job: dict) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("schema_version") == "clean41-dual-cap-strong-baseline-v1"
        and payload.get("config_hash") == manifest["config_hash"]
        and int(payload.get("row_count", -1)) == int(job["expected_rows"])
    )


def run_worker(manifest_path: Path, worker_index: int, smoke_test: bool = False) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    worker = manifest["workers"][worker_index]
    output_dir = Path(manifest["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    scorer = DualCapScorer(manifest, worker)
    if smoke_test:
        result = scorer.score([
            "A short human-written sentence used only for a memory smoke test.",
            "Language models generate text by predicting the next token in context.",
        ])
        print(json.dumps({"smoke_test": "PASS", "scores": result}, indent=2), flush=True)
        return
    for job in worker["jobs"]:
        output_path = output_dir / f"{job['dataset']}_{job['generator']}.json"
        if completed(output_path, manifest, job):
            print(f"[skip] {job['cell']}", flush=True)
            continue
        payload = json.loads(Path(job["dataset_file"]).read_text(encoding="utf-8"))
        methods = [name for name in manifest["methods"] if name in payload]
        count = min(
            int(manifest["target_count"]), len(payload["original"]),
            *(len(payload[name]) for name in methods),
        )
        started = time.time()
        score_streams: dict[str, list[dict]] = {}
        for name in ["original", *methods]:
            print(f"[{job['cell']}] {name} n={count}", flush=True)
            score_streams[name] = scorer.score([str(value) for value in payload[name][:count]])
        rows = []
        generator = normalize_generator(job["generator"])
        for index, item in enumerate(score_streams["original"]):
            rows.append({
                "sample_id": f"{job['dataset']}:{generator}:{index}:human",
                "source_id": f"{job['dataset']}:{index}",
                "dataset": job["dataset"], "generator": generator,
                "method": "human", "label": 0, **item,
            })
        for method in methods:
            for index, item in enumerate(score_streams[method]):
                rows.append({
                    "sample_id": f"{job['dataset']}:{generator}:{index}:{method}",
                    "source_id": f"{job['dataset']}:{index}",
                    "dataset": job["dataset"], "generator": generator,
                    "method": method, "label": 1, **item,
                })
        elapsed = time.time() - started
        result = {
            "schema_version": "clean41-dual-cap-strong-baseline-v1",
            "config_hash": manifest["config_hash"], "cell": job["cell"],
            "dataset_file": job["dataset_file"], "dataset_file_sha256": job["sha256"],
            "count": count, "methods": methods, "row_count": len(rows),
            "elapsed_seconds": elapsed, "seconds_per_row": elapsed / max(1, len(rows)),
            "rows": rows,
        }
        if len(rows) != int(job["expected_rows"]):
            raise RuntimeError(f"{job['cell']}: {len(rows)} != {job['expected_rows']}")
        atomic_write_json(output_path, result)
        print(f"[done] {job['cell']} rows={len(rows)} elapsed={elapsed:.1f}s", flush=True)


def prepare(
    source_manifest: Path, output_root: Path, batch_size: int, vocab_chunk_size: int,
) -> Path:
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    jobs = []
    for cell in source["cells"]:
        jobs.append({
            "cell": cell["cell"], "dataset": cell["dataset"],
            "generator": normalize_generator(cell["generator"]),
            "dataset_file": cell["dataset_file"], "sha256": cell["sha256"],
            "expected_rows": int(cell["expected_rows"]),
        })
    workers = [{
        "index": index, "visible_gpus": f"{2 * index},{2 * index + 1}",
        "device_observer": "cuda:0", "device_performer": "cuda:1", "jobs": [],
    } for index in range(3)]
    loads = [0, 0, 0]
    for job in sorted(jobs, key=lambda value: int(value["expected_rows"]), reverse=True):
        index = min(range(3), key=loads.__getitem__)
        workers[index]["jobs"].append(job)
        loads[index] += int(job["expected_rows"])
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / "baseline_scores"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": "clean41-dual-cap-strong-baseline-launch-v1",
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256(source_manifest),
        "observer": "models/falcon-7b",
        "performer": "models/falcon-7b-instruct",
        "ada_scripts": str(DEFAULT_ADA_ROOT), "caps": list(CAPS),
        "batch_size": int(batch_size), "target_count": int(source["target_count"]),
        "vocab_chunk_size": int(vocab_chunk_size),
        "methods": list(source["methods"]), "output_dir": str(output_dir.resolve()),
        "workers": workers, "worker_expected_rows": loads,
        "code_sha256": sha256(Path(__file__)),
        "ada_nuisance_sha256": sha256(DEFAULT_ADA_ROOT / "nuisance_func.py"),
        "ada_bspline_sha256": sha256(DEFAULT_ADA_ROOT / "BSpline.py"),
        "ada_helper_sha256": sha256(DEFAULT_ADA_ROOT / "helper.py"),
        "environment": {
            "OPENBLAS_NUM_THREADS": 1, "OMP_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1, "PYTHONHASHSEED": 20260809,
        },
        "python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
    }
    hash_payload = {key: value for key, value in config.items() if key != "workers"}
    config["config_hash"] = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    path = output_root / "baseline_launcher_manifest.json"
    atomic_write_json(path, config)
    print(json.dumps({"manifest": str(path), "worker_expected_rows": loads}, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-chunk-size", type=int, default=4096)
    parser.add_argument("--worker-manifest", type=Path)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.worker_manifest is not None:
        if args.worker_index is None:
            raise ValueError("--worker-index is required")
        run_worker(args.worker_manifest, args.worker_index, smoke_test=args.smoke_test)
        return
    if args.source_manifest is None or args.output_root is None:
        raise ValueError("--source-manifest and --output-root are required")
    prepare(
        args.source_manifest, args.output_root, args.batch_size, args.vocab_chunk_size,
    )


if __name__ == "__main__":
    main()
