#!/usr/bin/env python3
"""Prepare a 42-cell/16-domain English watermark extension without touching inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from io_utils import atomic_json, sha256

ROOT = Path(".")
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from clean_clean41_text_separation import load_json, load_mage_humans, recover_sources


NEW_METHODS = ("CGW", "HeavyWater", "SemStamp")
REUSED_DATASETS = {"squad", "writing", "xsum"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean41-dir", type=Path, default=Path("scratch/clean41_strict_text_separation"))
    parser.add_argument("--source-manifest", type=Path, default=Path("scratch/clean41_llama_proxy_necessity/clean41_source_manifest.json"))
    parser.add_argument("--current-clean20", type=Path, default=Path("scratch/clean20_all_anomalies_repaired"))
    parser.add_argument("--output-dir", type=Path, default=Path("scratch/broad_watermark_extension"))
    args = parser.parse_args()
    clean41, output = args.clean41_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = load_json(args.source_manifest)
    mage, mage_inputs = load_mage_humans(Path("external_data/MAGE"))
    recovered, _, legacy_inputs = recover_sources(source_manifest, mage, ROOT / "exp_main/data")
    tasks = []
    cells = []
    input_hashes = []
    for cell in source_manifest["cells"]:
        dataset, generator = str(cell["dataset"]), str(cell["generator"])
        if dataset == "chinese":
            continue
        full_dir = clean41 / "cleaned_full" / dataset / generator
        manifest_path, data_path = full_dir / "manifest.json", full_dir / "data.json"
        full_manifest = load_json(manifest_path)
        source_ids = list(full_manifest["source_ids"])
        cells.append({"dataset": dataset, "generator": generator, "source_count": len(source_ids),
                      "source_ids": source_ids, "mode": "REUSE" if dataset in REUSED_DATASETS else "GENERATE"})
        input_hashes.extend([{"path": str(manifest_path), "sha256": sha256(manifest_path)},
                             {"path": str(data_path), "sha256": sha256(data_path)}])
        if dataset in REUSED_DATASETS:
            continue
        for source_id in source_ids:
            source_index = int(source_id.rsplit(":", 1)[1])
            source = recovered[(dataset, source_index)]
            if not source["human_continuation"]:
                raise RuntimeError(f"missing Human continuation for deployable task {source_id}")
            for method in NEW_METHODS:
                tasks.append({
                    "dataset": dataset, "generator": generator, "method": method,
                    "source_id": source_id, "slot_index": source_index, "prompt": source["prompt"],
                    "prompt_sha256": hashlib.sha256(source["prompt"].encode()).hexdigest(),
                    "human_continuation_sha256": hashlib.sha256(source["human_continuation"].encode()).hexdigest(),
                })
    tasks.sort(key=lambda row: (row["method"], row["generator"], row["dataset"], row["slot_index"]))
    if len(cells) != 37 or len(tasks) != 11049:
        raise RuntimeError(f"broad frozen contract changed: cells={len(cells)} tasks={len(tasks)}")
    atomic_json(output / "generation_requests.json", tasks)
    atomic_json(output / "cell_registry.json", cells)
    per_method = Counter(row["method"] for row in tasks)
    per_generator = Counter(row["generator"] for row in tasks)
    per_dataset = Counter(row["dataset"] for row in tasks)
    atomic_json(output / "preparation_manifest.json", {
        "schema_version": "broad-watermark-extension-preparation-v1", "status": "READY",
        "scope": "15 English clean41 domains plus c4 at final merge; Chinese remains blocked for independent Human continuation",
        "clean41_english_cells": len(cells), "reused_cells": sum(row["mode"] == "REUSE" for row in cells),
        "generation_cells": sum(row["mode"] == "GENERATE" for row in cells), "new_generation_rows": len(tasks),
        "new_methods": list(NEW_METHODS), "counts_by_method": dict(per_method),
        "counts_by_generator": dict(per_generator), "counts_by_dataset": dict(per_dataset),
        "request_manifest_sha256": sha256(output / "generation_requests.json"),
        "source_manifest_sha256": sha256(args.source_manifest), "input_hashes": input_hashes,
        "mage_inputs": mage_inputs, "legacy_inputs": legacy_inputs,
        "test_labels_used": False, "original_data_overwritten": False,
    })
    with (output / "generation_requests.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in tasks[0] if key != "prompt"])
        writer.writeheader()
        writer.writerows(({key: value for key, value in row.items() if key != "prompt"} for row in tasks))
    print(json.dumps({"status": "READY", "cells": len(cells), "tasks": len(tasks), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
