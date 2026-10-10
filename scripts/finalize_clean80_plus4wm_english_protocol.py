#!/usr/bin/env python3
"""Merge completed source-aligned watermarks and freeze the expanded 20/60 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from io_utils import atomic_json, sha256, sha256_text as sha_text

ALL_METHODS = ("unwatermarked", "KGW", "SWEET", "Unigram", "EXP", "SynthID",
               "DIP", "ITSEdit", "CGW", "HeavyWater", "SemStamp")
PENDING = {"", "<GENERATION_PENDING>", "<GENERATION_FAILED>"}


def normalize(value: str) -> str:
    return " ".join(str(value).split()).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(
        "scratch/clean80_english20_test60_plus3wm"))
    args = parser.parse_args(); root = args.root.resolve()
    preparation = json.loads((root / "dataset/preparation_manifest.json").read_text(encoding="utf-8"))
    parent = json.loads(Path(preparation["parent_protocol"]).read_text(encoding="utf-8"))
    final_cells = []; unresolved = []
    for cell in preparation["cells"]:
        dataset, generator = cell["dataset"], cell["generator"]
        payload = json.loads(Path(cell["prepared_data"]).read_text(encoding="utf-8"))
        count = int(cell["source_count"])
        parent_data = json.loads(Path(cell["parent_data"]).read_text(encoding="utf-8"))
        registry_path = (
            Path("scratch/clean20_complete_regeneration/source_registry.json")
            if dataset == "c4" else Path("scratch/clean80_fullfactorial_completion/source_registry.json"))
        registry_raw = json.loads(registry_path.read_text(encoding="utf-8"))
        registry = registry_raw if dataset == "c4" else {row["source_id"]: row for row in registry_raw["sources"]}
        for method in ("CGW", "HeavyWater", "SemStamp"):
            for slot, value in enumerate(payload[method]):
                if str(value) not in PENDING:
                    payload[method][slot] = normalize(str(value)); continue
                source = registry[f"{dataset}:{slot}"]; prompt = str(source["prompt"])
                record_path = root / "extension/records" / dataset / generator / method / f"{slot:04d}.json"
                try:
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    unresolved.append({"cell": cell["cell"], "method": method, "slot": slot})
                    continue
                valid = (
                    record.get("status") == "SUCCESS"
                    and record.get("source_id") == source["source_id"]
                    and record.get("prompt_sha256") == sha_text(prompt)
                    and str(record.get("continuation", "")).strip()
                )
                if not valid:
                    unresolved.append({"cell": cell["cell"], "method": method, "slot": slot})
                    continue
                payload[method][slot] = normalize(str(record["continuation"]))
        final = {"original": list(parent_data["original"])}
        for method in ALL_METHODS:
            final[method] = list(payload[method])
            if len(final[method]) != count or any(str(value).strip() in PENDING for value in final[method]):
                unresolved.append({"cell": cell["cell"], "method": method, "slot": "stream"})
        out = root / "dataset/complete_prompt_free" / dataset / generator / "data.json"
        if not unresolved or not any(row["cell"] == cell["cell"] for row in unresolved):
            atomic_json(out, final, sort_keys=False)
            final_cells.append({
                "cell": cell["cell"], "dataset": dataset, "generator": generator,
                "split": cell["split"], "source_count": count,
                # `data_path` is retained for the dataset audit while
                # `dataset_file` is the established clean80 scoring contract.
                # They intentionally resolve to the same immutable JSON file.
                "data_path": str(out), "dataset_file": str(out),
                "sha256": sha256(out), "expected_rows": count * (len(ALL_METHODS) + 1),
            })
    atomic_json(root / "dataset/unresolved_records.json", unresolved, sort_keys=False)
    if unresolved:
        atomic_json(root / "dataset/FINAL_DATASET_MANIFEST.json", {
            "status": "BLOCKED", "unresolved_count": len(unresolved), "cells_complete": len(final_cells)}, sort_keys=False)
        print(json.dumps({"status": "BLOCKED", "unresolved": len(unresolved)}, indent=2))
        return 2
    source_manifest = {
        "schema_version": "clean80-english20-test60-plus3wm-source-v1", "status": "PASS",
        "evidence_scope": "EXPANDED_12_METHOD_60_CELL_TEST_EVIDENCE",
        "parent_protocol": preparation["parent_protocol"],
        "parent_protocol_sha256": preparation["parent_protocol_sha256"],
        "view": "complete_prompt_free", "max_tokens": 2048,
        "context_policy": "first_2048_tokens_matching_current_fulltext_implementation",
        "methods": list(ALL_METHODS), "calibration_datasets": parent["calibration_datasets"],
        "test_datasets": parent["test_datasets"], "generators": parent["generators"],
        "cell_count": 80, "calibration_cell_count": 20, "test_cell_count": 60,
        "calibration_source_count": parent["calibration_source_count"],
        "test_source_count": parent["test_source_count"],
        "expanded_row_count": sum(cell["expected_rows"] for cell in final_cells),
        "cells": final_cells,
        "labels_allowed": "calibration rows only for detector calibration; test labels evaluation only",
    }
    manifest_path = root / "benchmark/source_manifest.json"
    atomic_json(manifest_path, source_manifest, sort_keys=False)
    atomic_json(root / "dataset/FINAL_DATASET_MANIFEST.json", {
        "schema_version": "clean80-english-plus3wm-final-v1", "status": "PASS",
        "cell_count": 80, "methods": list(ALL_METHODS), "cells": final_cells,
        "source_manifest": str(manifest_path), "source_manifest_sha256": sha256(manifest_path),
    }, sort_keys=False)
    print(json.dumps({"status": "PASS", "cells": 80, "methods": len(ALL_METHODS),
                      "rows": source_manifest["expanded_row_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
