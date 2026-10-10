#!/usr/bin/env python3
"""Prepare source-aligned CGW/HeavyWater/SemStamp completion for English 20/60."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from io_utils import atomic_json, sha256, sha256_text as sha_text

ROOT = Path(__file__).resolve().parents[1]
BASE_METHODS = ("unwatermarked", "KGW", "SWEET", "Unigram", "EXP", "SynthID", "DIP", "ITSEdit")
ADDED_METHODS = ("CGW", "HeavyWater", "SemStamp")
ALL_METHODS = ("unwatermarked", "KGW", "SWEET", "Unigram", "EXP", "SynthID",
               "DIP", "ITSEdit", "CGW", "HeavyWater", "SemStamp")
PENDING = "<GENERATION_PENDING>"


def load_registries(clean80: Path, clean20: Path) -> tuple[dict, dict]:
    reg80 = {row["source_id"]: row for row in json.loads(
        (clean80 / "source_registry.json").read_text(encoding="utf-8"))["sources"]}
    reg20 = json.loads((clean20 / "source_registry.json").read_text(encoding="utf-8"))
    return reg80, reg20


def compatible_candidate(dataset: str, generator: str, original: list[str], roots: list[Path]):
    for root in roots:
        path = root / dataset / generator / "data.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("original") == original:
            return path, payload
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path(
        "scratch/clean80_english20_test60_benchmark/source_manifest.json"))
    parser.add_argument("--output-root", type=Path, default=Path(
        "scratch/clean80_english20_test60_plus3wm"))
    parser.add_argument("--clean80-root", type=Path, default=Path(
        "scratch/clean80_fullfactorial_completion"))
    parser.add_argument("--clean20-registry-root", type=Path, default=Path(
        "scratch/clean20_complete_regeneration"))
    args = parser.parse_args()
    protocol_path = args.protocol.resolve(); output = args.output_root.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "PASS" or len(protocol.get("cells", [])) != 80:
        raise RuntimeError("English 20/60 parent protocol is not PASS/80-cell")
    clean80 = args.clean80_root.resolve(); clean20 = args.clean20_registry_root.resolve()
    reg80, reg20 = load_registries(clean80, clean20)
    reuse_roots = [
        ROOT / "scratch/clean20_all_anomalies_repaired/complete_prompt_free",
        ROOT / "scratch/broad_watermark_dataset/complete_prompt_free",
    ]
    extension_requests = []
    reused = Counter(); generated = Counter(); cells = []
    for cell in protocol["cells"]:
        dataset, generator = cell["dataset"], cell["generator"]
        base_path = Path(cell["dataset_file"])
        base = json.loads(base_path.read_text(encoding="utf-8"))
        count = int(cell["source_count"])
        if any(len(base[name]) != count for name in ("original", *BASE_METHODS)):
            raise RuntimeError(f"base stream mismatch: {cell['cell']}")
        registry = reg20 if dataset == "c4" else reg80
        sources = []
        for slot, human in enumerate(base["original"]):
            key = f"{dataset}:{slot}"
            if key not in registry or registry[key]["human_continuation"] != human:
                raise RuntimeError(f"source registry mismatch: {cell['cell']} slot={slot}")
            sources.append(registry[key])
        candidate_path, candidate = compatible_candidate(
            dataset, generator, base["original"], reuse_roots)
        prepared = {name: list(base[name]) for name in ("original", *BASE_METHODS)}
        reuse_methods = []
        for method in ADDED_METHODS:
            values = candidate.get(method) if candidate is not None else None
            if isinstance(values, list) and len(values) == count and all(str(value).strip() for value in values):
                prepared[method] = [str(value) for value in values]
                reused[method] += count; reuse_methods.append(method)
            else:
                prepared[method] = [PENDING] * count
                generated[method] += count
        cell_dir = output / "dataset/prepared_raw" / dataset / generator
        for slot, source in enumerate(sources):
            source_id = str(source["source_id"]); prompt = str(source["prompt"])
            for method in ("CGW", "HeavyWater", "SemStamp"):
                if prepared[method][slot] == PENDING:
                    extension_requests.append({
                        "dataset": dataset, "generator": generator, "method": method,
                        "slot_index": slot, "source_id": source_id, "prompt": prompt,
                        "prompt_sha256": sha_text(prompt),
                        "human_continuation_sha256": sha_text(str(source["human_continuation"])),
                    })
        atomic_json(cell_dir / "data.json", prepared, sort_keys=False)
        cells.append({
            "cell": cell["cell"], "dataset": dataset, "generator": generator,
            "split": cell["split"], "source_count": count,
            "parent_data": str(base_path.resolve()), "parent_sha256": sha256(base_path),
            "prepared_data": str((cell_dir / "data.json").resolve()),
            "reuse_source": str(candidate_path.resolve()) if candidate_path else None,
            "reuse_methods": reuse_methods,
        })
    extension_requests.sort(key=lambda row: (
        row["method"], row["generator"], row["dataset"], row["slot_index"]))
    atomic_json(output / "extension/generation_requests.json", extension_requests, sort_keys=False)
    manifest = {
        "schema_version": "clean80-english-plus3wm-preparation-v1", "status": "READY",
        "parent_protocol": str(protocol_path), "parent_protocol_sha256": sha256(protocol_path),
        "methods": list(ALL_METHODS), "added_methods": list(ADDED_METHODS),
        "cells": cells, "cell_count": 80,
        "reused_rows_by_method": dict(reused), "generation_rows_by_method": dict(generated),
        "total_generation_rows": sum(generated.values()),
        "extension_request_count": len(extension_requests),
        "test_labels_used": False, "original_data_overwritten": False,
    }
    atomic_json(output / "dataset/preparation_manifest.json", manifest, sort_keys=False)
    print(json.dumps({"status": "READY", "cells": 80,
                      "reused": dict(reused), "to_generate": dict(generated),
                      "total_to_generate": sum(generated.values())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
