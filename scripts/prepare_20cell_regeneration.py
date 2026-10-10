#!/usr/bin/env python3
"""Prepare a non-destructive, full-factorial 20-cell regeneration workspace.

Human text is never synthesized.  SQuAD/Writing/XSum human continuations are
recovered from the historical source files.  Corrupt C4 human sources are
replaced deterministically with clean, previously unused rows from the same
processed C4 corpus.  Machine streams that are missing, corrupt, too short, or
violate the exact prompt contract are marked for sparse GPU regeneration.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from io_utils import atomic_json, sha256 as sha256_file, sha256_text
from text_utils import lexical_count, normalize_text

DATASETS = ("c4", "squad", "writing", "xsum")
GENERATORS = (
    ("GLM-4-9B-0414", "GLM-4-9B-0414", "GLM-4-9B-0414"),
    ("falcon", "falcon-7b", "falcon_7B"),
    ("llama-2-7b", "llama-2-7b", "llama-2-7b"),
    ("opt-2.7B", "opt-2.7B", "opt-2.7B"),
    ("opt-6.7B", "opt-6.7B", "opt-6.7B"),
)
METHODS = (
    "unwatermarked", "KGW", "SWEET", "Unigram",
    "EXP", "SynthID", "DIP", "ITSEdit",
)
PENDING = "<GENERATION_PENDING>"
FAILED = "<GENERATION_FAILED>"
SENTINELS = {"", PENDING, FAILED}


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def forbidden_reasons(value: object) -> List[str]:
    text = str(value)
    reasons = []
    if text.strip() in SENTINELS:
        reasons.append("EMPTY_OR_SENTINEL")
    if "\ufffd" in text:
        reasons.append("UNICODE_REPLACEMENT_CHARACTER")
    if "\x00" in text:
        reasons.append("NUL_CHARACTER")
    if any(unicodedata.category(char) == "Cs" for char in text):
        reasons.append("UNPAIRED_SURROGATE")
    if any(
        unicodedata.category(char) == "Cc" and char not in "\t\n\r"
        for char in text
    ):
        reasons.append("UNSUPPORTED_CONTROL_CHARACTER")
    return reasons


def infer_prompt(human: str, sampled: str) -> str:
    common = os.path.commonprefix([human, sampled])
    if " " in common:
        common = common.rsplit(" ", 1)[0]
    prompt = common.strip()
    if not prompt or not human.startswith(prompt):
        raise ValueError("unable to recover an exact prompt boundary")
    return prompt


def continuation_from_machine(raw: object, prompt: str, method: str) -> Tuple[str, str]:
    text = str(raw)
    if text.strip() in SENTINELS:
        return "", "EMPTY_OR_SENTINEL"
    if method == "ITSEdit":
        # Historical MarkLLM ITSEdit serializes only the edited continuation.
        continuation = normalize_text(text[len(prompt):] if text.startswith(prompt) else text)
        return continuation, "ITSEDIT_CONTINUATION"
    if not text.startswith(prompt):
        return "", "EXACT_PROMPT_PREFIX_MISSING"
    return normalize_text(text[len(prompt):]), "EXACT_PROMPT_REMOVED"


def raw_cell_path(dataset: str, disk_name: str, c4_name: str, watermark_root: Path, c4_root: Path) -> Path:
    if dataset == "c4":
        return c4_root / c4_name / "data.json"
    return watermark_root / dataset / disk_name / "data.json"


def build_source_registry(
    legacy_root: Path,
    c4_processed: Path,
    target_count: int,
    min_lexical_units: int,
) -> Tuple[Dict[Tuple[str, int], dict], List[dict], List[dict]]:
    registry: Dict[Tuple[str, int], dict] = {}
    source_rows: List[dict] = []
    replacements: List[dict] = []

    c4_rows = load_jsonl(c4_processed)
    if len(c4_rows) <= target_count:
        raise RuntimeError("processed C4 corpus has no reserve rows for human recovery")
    bad_slots = []
    for slot in range(target_count):
        row = c4_rows[slot]
        prompt, human = str(row.get("prompt", "")), str(row.get("natural_text", ""))
        if forbidden_reasons(prompt) or forbidden_reasons(human) or lexical_count(human) < min_lexical_units:
            bad_slots.append(slot)
    clean_reserve = []
    for corpus_index in range(target_count, len(c4_rows)):
        row = c4_rows[corpus_index]
        prompt, human = str(row.get("prompt", "")), str(row.get("natural_text", ""))
        if (
            not forbidden_reasons(prompt)
            and not forbidden_reasons(human)
            and lexical_count(prompt) >= 8
            and lexical_count(human) >= min_lexical_units
        ):
            clean_reserve.append(corpus_index)
            if len(clean_reserve) == len(bad_slots):
                break
    if len(clean_reserve) != len(bad_slots):
        raise RuntimeError("not enough verified Human C4 reserve rows")
    replacement_map = dict(zip(bad_slots, clean_reserve))

    for slot in range(target_count):
        corpus_index = replacement_map.get(slot, slot)
        row = c4_rows[corpus_index]
        item = {
            "dataset": "c4",
            "slot_index": slot,
            "source_id": f"c4:processed_c4:{corpus_index}",
            "corpus_index": corpus_index,
            "prompt": str(row["prompt"]),
            "human_raw": str(row["natural_text"]),
            "human_continuation": normalize_text(row["natural_text"]),
            "human_provenance": "processed_c4.natural_text",
            "replaced": slot in replacement_map,
        }
        registry[("c4", slot)] = item
        if slot in replacement_map:
            replacements.append({
                "dataset": "c4",
                "slot_index": slot,
                "old_source_id": f"c4:processed_c4:{slot}",
                "new_source_id": item["source_id"],
                "reason": "CORRUPT_HUMAN_OR_PROMPT",
                "policy": "FIRST_CLEAN_UNUSED_ROW_AFTER_TARGET_WINDOW",
            })

    for dataset in ("squad", "writing", "xsum"):
        path = legacy_root / f"{dataset}_gpt-j-6B.raw_data.json"
        payload = load_json(path)
        if min(len(payload["original"]), len(payload["sampled"])) < target_count:
            raise RuntimeError(f"{dataset}: insufficient legacy aligned sources")
        for slot in range(target_count):
            human_raw = str(payload["original"][slot])
            prompt = infer_prompt(human_raw, str(payload["sampled"][slot]))
            human_continuation = normalize_text(human_raw[len(prompt):])
            if forbidden_reasons(human_continuation) or lexical_count(human_continuation) < min_lexical_units:
                raise RuntimeError(f"{dataset}:{slot}: real Human continuation is unusable")
            registry[(dataset, slot)] = {
                "dataset": dataset,
                "slot_index": slot,
                "source_id": f"{dataset}:{slot}",
                "corpus_index": slot,
                "prompt": prompt,
                "human_raw": human_raw,
                "human_continuation": human_continuation,
                "human_provenance": str(path.resolve()),
                "replaced": False,
            }

    for dataset in DATASETS:
        for slot in range(target_count):
            item = registry[(dataset, slot)]
            source_rows.append({
                "dataset": dataset,
                "slot_index": slot,
                "source_id": item["source_id"],
                "corpus_index": item["corpus_index"],
                "human_provenance": item["human_provenance"],
                "human_replaced": item["replaced"],
                "prompt_chars": len(item["prompt"]),
                "prompt_lexical_units": lexical_count(item["prompt"]),
                "human_chars": len(item["human_continuation"]),
                "human_lexical_units": lexical_count(item["human_continuation"]),
                "prompt_sha256": sha256_text(item["prompt"]),
                "human_sha256": sha256_text(item["human_continuation"]),
            })
    return registry, source_rows, replacements


def prepare(args: argparse.Namespace) -> dict:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry, source_rows, replacements = build_source_registry(
        args.legacy_root,
        args.c4_processed,
        args.target_count,
        args.min_lexical_units,
    )
    pending_rows: List[dict] = []
    kept_rows: List[dict] = []
    input_rows: List[dict] = []

    for dataset in DATASETS:
        for generator, disk_name, c4_name in GENERATORS:
            raw_path = raw_cell_path(dataset, disk_name, c4_name, args.watermark_root, args.c4_root)
            raw = load_json(raw_path)
            input_rows.append({
                "dataset": dataset,
                "generator": generator,
                "path": str(raw_path.resolve()),
                "bytes": raw_path.stat().st_size,
                "sha256": sha256_file(raw_path),
                "methods": ",".join(key for key in METHODS if isinstance(raw.get(key), list)),
            })
            prepared = {
                "original": [registry[(dataset, slot)]["human_raw"] for slot in range(args.target_count)],
                **{method: [PENDING] * args.target_count for method in METHODS},
            }
            cell_requests = []
            for slot in range(args.target_count):
                source = registry[(dataset, slot)]
                prompt = str(source["prompt"])
                for method in METHODS:
                    reasons = []
                    raw_values = raw.get(method)
                    if not isinstance(raw_values, list) or slot >= len(raw_values):
                        reasons.append("METHOD_OR_RECORD_MISSING")
                        raw_value = ""
                    else:
                        raw_value = str(raw_values[slot])
                    if source["replaced"]:
                        reasons.append("HUMAN_SOURCE_REPLACED")
                    continuation, split_status = continuation_from_machine(raw_value, prompt, method)
                    if split_status in {"EMPTY_OR_SENTINEL", "EXACT_PROMPT_PREFIX_MISSING"}:
                        reasons.append(split_status)
                    reasons.extend(forbidden_reasons(continuation))
                    if continuation and lexical_count(continuation) < args.min_lexical_units:
                        reasons.append("CONTINUATION_TOO_SHORT")
                    reasons = sorted(set(reasons))
                    if not reasons:
                        prepared[method][slot] = raw_value
                        kept_rows.append({
                            "dataset": dataset,
                            "generator": generator,
                            "source_id": source["source_id"],
                            "slot_index": slot,
                            "method": method,
                            "split_status": split_status,
                            "continuation_lexical_units": lexical_count(continuation),
                            "raw_sha256": sha256_text(raw_value),
                        })
                        continue
                    request = {
                        "dataset": dataset,
                        "generator": generator,
                        "source_id": source["source_id"],
                        "slot_index": slot,
                        "method": method,
                        "prompt": prompt,
                        "prompt_sha256": sha256_text(prompt),
                        "reasons": ";".join(reasons),
                        "status": "PENDING",
                    }
                    pending_rows.append(request)
                    cell_requests.append(request)
            prepared_path = output / "prepared_raw" / dataset / generator / "data.json"
            atomic_json(prepared_path, prepared)
            atomic_json(prepared_path.with_name("generation_requests.json"), cell_requests)

    write_csv(
        output / "source_registry.csv",
        source_rows,
        (
            "dataset", "slot_index", "source_id", "corpus_index", "human_provenance",
            "human_replaced", "prompt_chars", "prompt_lexical_units", "human_chars",
            "human_lexical_units", "prompt_sha256", "human_sha256",
        ),
    )
    write_csv(
        output / "human_replacements.csv",
        replacements,
        ("dataset", "slot_index", "old_source_id", "new_source_id", "reason", "policy"),
    )
    write_csv(
        output / "pending_records.csv",
        pending_rows,
        (
            "dataset", "generator", "source_id", "slot_index", "method", "prompt_sha256",
            "reasons", "status",
        ),
    )
    write_csv(
        output / "retained_machine_records.csv",
        kept_rows,
        (
            "dataset", "generator", "source_id", "slot_index", "method", "split_status",
            "continuation_lexical_units", "raw_sha256",
        ),
    )
    atomic_json(output / "input_manifest.json", input_rows)
    atomic_json(output / "source_registry.json", {
        f"{dataset}:{slot}": registry[(dataset, slot)]
        for dataset in DATASETS for slot in range(args.target_count)
    })

    reason_counts = Counter(
        reason for row in pending_rows for reason in str(row["reasons"]).split(";") if reason
    )
    per_generator = Counter(str(row["generator"]) for row in pending_rows)
    per_dataset = Counter(str(row["dataset"]) for row in pending_rows)
    summary = {
        "schema_version": "clean20-regeneration-preparation-v1",
        "status": "READY_FOR_SPARSE_GPU_GENERATION" if pending_rows else "COMPLETE",
        "target_sources_per_dataset": args.target_count,
        "datasets": list(DATASETS),
        "generators": [row[0] for row in GENERATORS],
        "methods": list(METHODS),
        "expected_cells": len(DATASETS) * len(GENERATORS),
        "expected_atomic_tasks": len(DATASETS) * len(GENERATORS) * len(METHODS),
        "expected_final_text_rows": len(DATASETS) * len(GENERATORS) * args.target_count * (1 + len(METHODS)),
        "human_policy": "RECOVER_OR_SAME_CORPUS_REPLACE_ONLY_NEVER_GENERATE",
        "c4_human_replacements": len(replacements),
        "retained_machine_records": len(kept_rows),
        "pending_machine_records": len(pending_rows),
        "pending_by_generator": dict(sorted(per_generator.items())),
        "pending_by_dataset": dict(sorted(per_dataset.items())),
        "pending_reason_counts": dict(sorted(reason_counts.items())),
        "min_lexical_units": args.min_lexical_units,
        "seed": args.seed,
        "python": sys.version,
        "platform": platform.platform(),
    }
    atomic_json(output / "preparation_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("scratch/clean20_complete_regeneration"))
    parser.add_argument("--watermark-root", type=Path, default=Path("external_data/watermark_data"))
    parser.add_argument("--c4-root", type=Path, default=Path("external_data/fast_detect_dataset"))
    parser.add_argument("--legacy-root", type=Path, default=Path("exp_main/data"))
    parser.add_argument("--c4-processed", type=Path, default=Path("third_party/MarkLLM/dataset/c4/processed_c4.json"))
    parser.add_argument("--target-count", type=int, default=200)
    parser.add_argument("--min-lexical-units", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
