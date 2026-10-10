#!/usr/bin/env python3
"""Independent structural and Falcon-likelihood audit for broad watermark records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


from text_utils import text_metrics as metrics

MODEL = Path("models/falcon-7b")


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(rows[0]) if rows else ["dataset", "generator", "method", "source_id"]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=7)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=-8.0)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    requests = json.loads(args.request_manifest.read_text(encoding="utf-8"))
    tasks = []
    for row in requests:
        material = f"{row['dataset']}\x1f{row['generator']}\x1f{row['source_id']}\x1f{row['method']}"
        shard = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") % args.shard_count
        if shard == args.shard_index: tasks.append(row)
    if not tasks:
        atomic_csv(output / "quality_scores" / f"shard_{args.shard_index:02d}_of_{args.shard_count:02d}.csv", [])
        print(f"[{args.shard_index}] 0/0", flush=True)
        return 0
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=False, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map="auto").eval()
    result = []
    for index, request in enumerate(tasks, 1):
        record_path = output / "records" / request["dataset"] / request["generator"] / request["method"] / f"{int(request['slot_index']):04d}.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        text = str(record.get("continuation", ""))
        lexical, unique, repeated, cjk, controls = metrics(text)
        structural = lexical >= 32 and unique >= 0.15 and repeated <= 0.35 and cjk <= 0.05 and controls == 0
        mean_ll, scored_tokens = float("nan"), 0
        if structural:
            prompt_ids = tokenizer(request["prompt"], add_special_tokens=False)["input_ids"]
            encoded = tokenizer(request["prompt"] + text, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.inference_mode(): logits = model(**encoded).logits[:, :-1].float()
            targets = encoded["input_ids"][:, 1:]
            start = max(0, len(prompt_ids) - 1)
            values = torch.log_softmax(logits[:, start:], dim=-1).gather(-1, targets[:, start:].unsqueeze(-1)).squeeze(-1)
            mean_ll, scored_tokens = float(values.mean()), int(values.numel())
        diagnostics = record.get("attempts", [{}])[-1].get("diagnostics", {})
        result.append({
            "dataset": request["dataset"], "generator": request["generator"], "method": request["method"],
            "source_id": request["source_id"], "slot_index": request["slot_index"], "record_status": record.get("status"),
            "lexical_units": lexical, "unique_lexical_ratio": unique, "repeated_4gram_ratio": repeated,
            "cjk_ratio": cjk, "control_characters": int(controls), "scored_tokens": scored_tokens,
            "falcon_observer_mean_log_likelihood": mean_ll, "quality_threshold": args.threshold,
            "quality_pass": int(record.get("status") == "SUCCESS" and structural and mean_ll >= args.threshold),
            "self_detection_neg_log10_p": diagnostics.get("self_detection_neg_log10_p", ""),
            "self_detection_z": diagnostics.get("self_detection_z", ""),
        })
        if index % 20 == 0 or index == len(tasks):
            atomic_csv(output / "quality_scores" / f"shard_{args.shard_index:02d}_of_{args.shard_count:02d}.csv", result)
            print(f"[{args.shard_index}] {index}/{len(tasks)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
