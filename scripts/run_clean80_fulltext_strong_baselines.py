#!/usr/bin/env python3
"""Resumable 2048-token AdaDetectGPT/Binoculars scoring for clean80."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_clean41_strong_baseline_scores import DEFAULT_ADA_ROOT, load_witness, prefix_scores


OBSERVER = "models/falcon-7b"
PERFORMER = "models/falcon-7b-instruct"


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


class Scorer:
    def __init__(self, max_tokens: int, batch_size: int, vocab_chunk_size: int):
        self.max_tokens = max_tokens; self.batch_size = batch_size
        self.vocab_chunk_size = vocab_chunk_size
        self.observer_device = torch.device("cuda:0"); self.performer_device = torch.device("cuda:1")
        self.tokenizer = AutoTokenizer.from_pretrained(
            OBSERVER, trust_remote_code=True, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        performer_tokenizer = AutoTokenizer.from_pretrained(
            PERFORMER, trust_remote_code=True, local_files_only=True)
        if self.tokenizer.get_vocab() != performer_tokenizer.get_vocab():
            raise RuntimeError("observer/performer tokenizers differ")
        self.observer = AutoModelForCausalLM.from_pretrained(
            OBSERVER, trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16).to(self.observer_device).eval()
        self.performer = AutoModelForCausalLM.from_pretrained(
            PERFORMER, trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16).to(self.performer_device).eval()
        self.witness = load_witness(DEFAULT_ADA_ROOT, self.performer_device)

    @torch.inference_mode()
    def score(self, texts: list[str]) -> list[dict]:
        raw_ids = [self.tokenizer(text, add_special_tokens=True, return_attention_mask=False,
                                  return_token_type_ids=False, truncation=False)["input_ids"]
                   for text in texts]
        raw_lengths = [len(ids) for ids in raw_ids]
        order = sorted(range(len(texts)), key=raw_lengths.__getitem__)
        result: list[dict | None] = [None] * len(texts)
        for start in range(0, len(order), self.batch_size):
            indices = order[start:start + self.batch_size]
            batch = [texts[index] for index in indices]
            encoding = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_tokens, return_token_type_ids=False)
            observer_inputs = {key: value.to(self.observer_device) for key, value in encoding.items()}
            performer_inputs = {key: value.to(self.performer_device) for key, value in encoding.items()}
            observer_logits = self.observer(**observer_inputs).logits
            performer_logits = self.performer(**performer_inputs).logits
            lengths = encoding["attention_mask"].sum(dim=1).tolist()
            for offset, (index, length) in enumerate(zip(indices, lengths)):
                length = int(length)
                score = prefix_scores(
                    performer_inputs["input_ids"][offset, :length],
                    observer_logits[offset, :length].to(self.performer_device),
                    performer_logits[offset, :length], self.witness,
                    caps=(self.max_tokens,), vocab_chunk_size=self.vocab_chunk_size,
                )[self.max_tokens]
                result[index] = {
                    "input_token_length": raw_lengths[index], "token_length": length - 1,
                    "context_truncated": raw_lengths[index] > self.max_tokens,
                    "binoculars_full": float(score[0]),
                    "adadetectgpt_pretrained_full": float(score[1]),
                }
            del observer_logits, performer_logits, observer_inputs, performer_inputs
        if any(item is None for item in result):
            raise RuntimeError("strong scoring lost rows")
        return [dict(item) for item in result if item is not None]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--vocab-chunk-size", type=int, default=4096)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if args.max_tokens != int(source["max_tokens"]):
        raise RuntimeError("source/worker max token mismatch")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    config = {"source_manifest_sha256": sha256(args.source_manifest), "max_tokens": args.max_tokens,
              "batch_size": args.batch_size, "vocab_chunk_size": args.vocab_chunk_size,
              "observer": OBSERVER, "performer": PERFORMER, "ada_root": str(DEFAULT_ADA_ROOT),
              "worker_sha256": sha256(Path(__file__))}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorer = Scorer(args.max_tokens, args.batch_size, args.vocab_chunk_size)
    total = truncated = 0
    selected = [cell for index, cell in enumerate(source["cells"])
                if index % args.shard_count == args.shard_index]
    for position, cell in enumerate(selected, 1):
        output = args.output_dir / f"{cell['dataset']}_{cell['generator']}.json"
        if output.is_file():
            try:
                cached = json.loads(output.read_text(encoding="utf-8"))
                cached_config = cached.get("config", {})
                compatible = (
                    cached.get("row_count") == cell["expected_rows"]
                    and cached_config.get("source_manifest_sha256") == config["source_manifest_sha256"]
                    and int(cached_config.get("max_tokens", -1)) == config["max_tokens"]
                    and int(cached_config.get("batch_size", -1)) == config["batch_size"]
                    and int(cached_config.get("vocab_chunk_size", -1)) == config["vocab_chunk_size"]
                    and cached_config.get("observer") == config["observer"]
                    and cached_config.get("performer") == config["performer"]
                )
                if compatible:
                    total += cached["row_count"]; truncated += cached["truncated_rows"]
                    print(f"[skip {position}/{len(selected)}] {cell['cell']}", flush=True); continue
            except (OSError, json.JSONDecodeError):
                pass
        path = Path(cell["dataset_file"])
        if sha256(path) != cell["sha256"]:
            raise RuntimeError(f"input hash changed: {path}")
        payload = json.loads(path.read_text(encoding="utf-8")); started = time.perf_counter()
        streams = [("human", 0, payload["original"])] + [
            (method, 1, payload[method]) for method in source["methods"]]
        rows = []
        for method, label, values in streams:
            scores = scorer.score([str(value) for value in values])
            for index, (text, score) in enumerate(zip(values, scores)):
                rows.append({
                    "sample_id": f"{cell['dataset']}:{cell['generator']}:{index}:{method}",
                    "source_id": f"{cell['dataset']}:{index}", "dataset": cell["dataset"],
                    "generator": cell["generator"], "watermark_method": method, "label": label,
                    "text_sha256": hashlib.sha256(str(text).encode()).hexdigest(), **score,
                })
        if len(rows) != cell["expected_rows"]:
            raise RuntimeError(f"row mismatch: {cell['cell']}")
        cell_truncated = sum(bool(row["context_truncated"]) for row in rows)
        elapsed = time.perf_counter() - started
        atomic_json(output, {"schema_version": "clean80-fulltext-strong-cell-v1",
                    "config": config, "config_hash": config_hash, "cell": cell["cell"],
                    "input_sha256": cell["sha256"], "row_count": len(rows),
                    "truncated_rows": cell_truncated, "elapsed_seconds": elapsed,
                    "seconds_per_row": elapsed / len(rows), "rows": rows})
        total += len(rows); truncated += cell_truncated
        print(f"[done {position}/{len(selected)}] {cell['cell']} rows={len(rows)} "
              f"truncated={cell_truncated} elapsed={elapsed:.1f}s", flush=True)
    print(json.dumps({"status": "PASS", "rows": total, "truncated_rows": truncated}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
