#!/usr/bin/env python3
"""Merge, validate, and report the broad 42-cell English watermark benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path

from io_utils import atomic_json, sha256

OLD_METHODS = ("unwatermarked", "KGW", "SWEET", "Unigram", "EXP", "SynthID", "DIP", "ITSEdit")
NEW_METHODS = ("CGW", "HeavyWater", "SemStamp")
COMMON_METHODS = (*OLD_METHODS, *NEW_METHODS)
WORDS = re.compile(r"[A-Za-z0-9]+(?:['`-][A-Za-z0-9]+)*|[\u4e00-\u9fff]")
DOMAIN_TYPE = {
    "c4": "web text", "cmv": "opinion debate", "cnn": "news", "dialogsum": "dialogue summary",
    "eli5": "long-form Q&A", "hswag": "commonsense continuation", "imdb": "movie review", "pubmed": "biomedical text",
    "roct": "story continuation", "sci": "scientific text", "squad": "factual Q&A", "tldr": "social media summary",
    "wp": "creative writing", "writing": "creative writing", "xsum": "news summary", "yelp": "consumer review",
}


def lexical_count(text: str) -> int:
    return len(WORDS.findall(str(text)))


def structurally_strict(text: str) -> bool:
    lexical = [token.lower() for token in WORDS.findall(text)]
    controls = sum(unicodedata.category(char) in {"Cc", "Cs"} and char not in "\n\r\t" for char in text)
    return len(lexical) >= 32 and controls == 0


def load_quality(extension: Path) -> list[dict]:
    rows = []
    for path in sorted((extension / "quality_scores").glob("shard_*.csv")):
        with path.open(encoding="utf-8") as handle: rows.extend(csv.DictReader(handle))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--clean41-dir", type=Path, required=True)
    parser.add_argument("--clean20-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("scratch/broad_watermark_dataset"))
    parser.add_argument("--max-characters", type=int, default=640)
    args = parser.parse_args()
    extension, clean41, clean20, output = map(Path.resolve, (args.extension_dir, args.clean41_dir, args.clean20_dir, args.output_dir))
    preparation = json.loads((extension / "preparation_manifest.json").read_text())
    clean20_manifest = json.loads((clean20 / "data_manifest.json").read_text())
    scheduler = json.loads((extension / "scheduler_audit/summary.json").read_text())
    recovery_path = extension / "sparse_recovery_audit/summary.json"
    recovery = json.loads(recovery_path.read_text()) if recovery_path.exists() else None
    recovery_initial_rows = recovery["rounds"][0]["input_failures"] if recovery and recovery.get("rounds") else 0
    recovery_attempted_rows = sum(row["input_failures"] for row in recovery.get("rounds", [])) if recovery else 0
    recovery_initial_by_dataset = defaultdict(int)
    recovery_initial_by_status = defaultdict(int)
    if recovery:
        for row in recovery.get("archived_pre_retry_records", []):
            if row.get("round") == 1:
                recovery_initial_by_dataset[row["dataset"]] += 1
                recovery_initial_by_status[row["status"]] += 1
    quality_scheduler = json.loads((extension / "quality_audit/summary.json").read_text())
    clean41_likelihood_audit = json.loads((extension / "clean41_likelihood_anomaly_audit.json").read_text())
    requests = json.loads((extension / "generation_requests.json").read_text())
    failures = []
    if scheduler["status"] != "PASS" and not (recovery and recovery["status"] == "PASS"):
        failures.append("GENERATION_NOT_PASS_AND_RECOVERY_NOT_PASS")
    if quality_scheduler["status"] != "PASS": failures.append("QUALITY_SCHEDULER_NOT_PASS")
    if clean41_likelihood_audit["status"] != "PASS": failures.append("CLEAN41_OLD_STREAM_ANOMALIES_FOUND")
    if sha256(extension / "generation_requests.json") != preparation["request_manifest_sha256"]:
        failures.append("REQUEST_MANIFEST_CHANGED")
    for row in preparation.get("input_hashes", []):
        path = Path(row["path"])
        if not path.exists() or sha256(path) != row["sha256"]:
            failures.append(f"FROZEN_CLEAN41_INPUT_CHANGED={path}")
    quality = load_quality(extension)
    pilot_quality = []
    for path in sorted((extension / "pilot_quality_scores").glob("shard_*.csv")):
        with path.open(encoding="utf-8") as handle: pilot_quality.extend(csv.DictReader(handle))
    if len(pilot_quality) != 6 or any(int(row["quality_pass"]) != 1 for row in pilot_quality):
        failures.append(f"PILOT_QUALITY_CONTRACT={len(pilot_quality)}")
    if len(quality) != len(requests): failures.append(f"QUALITY_ROWS={len(quality)}")
    quality_failures = [row for row in quality if int(row["quality_pass"]) != 1]
    if quality_failures: failures.append(f"QUALITY_FAILURES={len(quality_failures)}")
    record_keys = set()
    generation_attempts = 0
    semstamp_candidate_trials = 0
    for request in requests:
        path = extension / "records" / request["dataset"] / request["generator"] / request["method"] / f"{int(request['slot_index']):04d}.json"
        if not path.exists():
            failures.append(f"MISSING_RECORD={path}")
            continue
        record = json.loads(path.read_text())
        generation_attempts += len(record.get("attempts", []))
        for attempt in record.get("attempts", []):
            semstamp_candidate_trials += sum(attempt.get("diagnostics", {}).get("trials_per_segment", []))
        record_identity = (record.get("dataset"), record.get("generator"), record.get("method"), record.get("source_id"))
        request_identity = (request["dataset"], request["generator"], request["method"], request["source_id"])
        expected_prompt_hash = hashlib.sha256(request["prompt"].encode()).hexdigest()
        if (record.get("status") != "SUCCESS" or record_identity != request_identity
                or record.get("prompt_sha256") != expected_prompt_hash):
            failures.append(f"INVALID_RECORD={path}")
        record_keys.add((request["dataset"], request["generator"], request["method"], request["source_id"]))
    if len(record_keys) != len(requests): failures.append(f"RECORD_KEYS={len(record_keys)}")

    cells = []
    cell_registry = json.loads((extension / "cell_registry.json").read_text())
    for cell in cell_registry:
        dataset, generator = cell["dataset"], cell["generator"]
        base_dir = clean41 / "cleaned_full" / dataset / generator
        payload = json.loads((base_dir / "data.json").read_text())
        source_ids = json.loads((base_dir / "manifest.json").read_text())["source_ids"]
        full = {"original": list(payload["original"]), **{method: list(payload[method]) for method in OLD_METHODS}}
        for method in NEW_METHODS: full[method] = []
        if cell["mode"] == "REUSE":
            current_dir = clean20 / "complete_prompt_free" / dataset / generator
            current = json.loads((current_dir / "data.json").read_text())
            current_ids = json.loads((current_dir / "manifest.json").read_text())["source_ids"]
            index = {source_id: slot for slot, source_id in enumerate(current_ids)}
            for source_id in source_ids:
                if source_id not in index: failures.append(f"REUSE_SOURCE_MISSING={dataset}/{generator}/{source_id}")
                for method in NEW_METHODS: full[method].append(current[method][index[source_id]])
        else:
            for source_id in source_ids:
                source_index = int(source_id.rsplit(":", 1)[1])
                for method in NEW_METHODS:
                    path = extension / "records" / dataset / generator / method / f"{source_index:04d}.json"
                    full[method].append(json.loads(path.read_text())["continuation"])
        cells.append({"dataset": dataset, "generator": generator, "source_ids": source_ids, "full": full, "origin": "clean41"})
    for generator in ("GLM-4-9B-0414", "falcon", "llama-2-7b", "opt-2.7B", "opt-6.7B"):
        base_dir = clean20 / "complete_prompt_free" / "c4" / generator
        payload = json.loads((base_dir / "data.json").read_text())
        source_ids = json.loads((base_dir / "manifest.json").read_text())["source_ids"]
        full = {"original": list(payload["original"]), **{method: list(payload[method]) for method in COMMON_METHODS}}
        cells.append({"dataset": "c4", "generator": generator, "source_ids": source_ids, "full": full, "origin": "clean20"})
    if len(cells) != 42: failures.append(f"CELL_COUNT={len(cells)}")

    collision_map = defaultdict(list)
    for cell in cells:
        for slot, source_id in enumerate(cell["source_ids"]):
            for method, texts in cell["full"].items():
                label = "Human" if method == "original" else "Machine"
                collision_map[hashlib.sha256(texts[slot].encode()).hexdigest()].append(
                    (label, cell["dataset"], cell["generator"], source_id, method)
                )
    collision_rows = [entries for entries in collision_map.values() if {row[0] for row in entries} == {"Human", "Machine"}]
    if collision_rows: failures.append(f"CROSS_LABEL_COLLISIONS={len(collision_rows)}")

    full_exposures = strict_exposures = total_rows = strict_rows = 0
    dataset_counts = defaultdict(lambda: {"cells": 0, "full_sources": 0, "strict_sources": 0})
    if not failures:
        for cell in cells:
            dataset, generator, source_ids, full = cell["dataset"], cell["generator"], cell["source_ids"], cell["full"]
            sizes = {len(values) for values in full.values()}
            if sizes != {len(source_ids)} or set(full) != {"original", *COMMON_METHODS}:
                raise RuntimeError(f"stream contract failed: {dataset}/{generator}")
            selected, budgets = [], []
            for slot in range(len(source_ids)):
                texts = [full[key][slot] for key in ("original", *COMMON_METHODS)]
                budget = min(args.max_characters, *(len(text) for text in texts))
                # Validate the actual equal-character strings, not only the full
                # pre-truncation streams.  Otherwise a verbose stream could have
                # >=32 words overall but fewer than 32 inside the shared budget.
                if budget > 0 and all(structurally_strict(text[:budget]) for text in texts):
                    selected.append(slot); budgets.append(budget)
            strict = {key: [values[slot][:budgets[pos]] for pos, slot in enumerate(selected)] for key, values in full.items()}
            strict_ids = [source_ids[slot] for slot in selected]
            for view, values, ids in (("complete_prompt_free", full, source_ids), ("complete_strict_char_matched", strict, strict_ids)):
                target = output / view / dataset / generator
                atomic_json(target / "data.json", values)
                atomic_json(target / "manifest.json", {
                    "schema_version": "broad-watermark-dataset-cell-v1", "dataset": dataset, "generator": generator,
                    "source_ids": ids, "source_count": len(ids), "methods": list(COMMON_METHODS),
                    "source_grouping_contract": "same source_id and all mechanisms remain in one fold",
                    "strict_character_budgets": budgets if view.endswith("matched") else None,
                })
            full_exposures += len(source_ids); strict_exposures += len(strict_ids)
            total_rows += len(source_ids) * 12; strict_rows += len(strict_ids) * 12
            dataset_counts[dataset]["cells"] += 1; dataset_counts[dataset]["full_sources"] += len(source_ids)
            dataset_counts[dataset]["strict_sources"] += len(strict_ids)

    detection = {}
    for method in NEW_METHODS:
        rows = [row for row in quality if row["method"] == method]
        field, threshold = ("self_detection_neg_log10_p", 1.3010299956639813) if method == "CGW" else ("self_detection_z", 1.6448536269514722)
        values = [float(row[field]) for row in rows if row[field] not in {"", None}]
        detection[method] = {"rows": len(rows), "available": len(values), "metric": field,
                             "mean": statistics.fmean(values) if values else None,
                             "median": statistics.median(values) if values else None,
                             "detection_fraction_at_one_sided_5pct": sum(value >= threshold for value in values) / len(values) if values else None}
    status = "PASS" if not failures else "FAILED_VALIDATION"
    atomic_json(extension / "finalization_audit.json", {"status": status, "failures": failures,
        "generation_rows": len(requests), "quality_rows": len(quality), "quality_failures": len(quality_failures),
        "record_keys": len(record_keys), "cells": len(cells), "full_cell_source_exposures": full_exposures,
        "strict_cell_source_exposures": strict_exposures, "cross_label_collisions": len(collision_rows),
        "watermark_self_detection": detection, "output_dir": str(output) if status == "PASS" else None})
    if status == "PASS":
        manifest = {"schema_version": "broad-watermark-dataset-manifest-v1", "status": "PASS",
            "datasets": len(dataset_counts), "dataset_names": sorted(dataset_counts), "cells": 42,
            "generator_families": 5, "generators": ["GLM-4-9B-0414", "falcon", "llama-2-7b", "opt-2.7B", "opt-6.7B"],
            "machine_channels": len(COMMON_METHODS), "methods": list(COMMON_METHODS),
            "full_cell_source_exposures": full_exposures, "strict_cell_source_exposures": strict_exposures,
            "full_text_rows": total_rows, "strict_text_rows": strict_rows, "quality_threshold": -8.0,
            "new_generation_rows": len(requests), "reused_new_watermark_rows": (full_exposures - len(requests) // 3) * 3,
            "generation_attempts_in_final_records": generation_attempts,
            "semstamp_candidate_trials_in_final_records": semstamp_candidate_trials,
            "first_pass_generation_status": scheduler["status"],
            "sparse_recovery_status": recovery["status"] if recovery else "NOT_REQUIRED",
            "sparse_recovery_initial_rows": recovery_initial_rows,
            "sparse_recovery_initial_content_failures": recovery_initial_by_status.get("ARCHIVED", 0),
            "sparse_recovery_initial_missing_records": recovery_initial_by_status.get("MISSING_BEFORE_RETRY", 0),
            "sparse_recovery_attempted_rows_across_rounds": recovery_attempted_rows,
            "known_quality_failures": 0, "cross_label_collisions": 0, "test_labels_used_for_generation_or_filtering": False,
            "clean41_existing_machine_rows_likelihood_audited": clean41_likelihood_audit["machine_rows"],
            "clean41_existing_rows_below_minus8": clean41_likelihood_audit["failed_rows"],
            "pilot_rows": len(pilot_quality), "pilot_quality_failures": sum(int(row["quality_pass"]) != 1 for row in pilot_quality),
            "original_inputs_overwritten": False, "chinese_status": "NOT_INCLUDED_MISSING_INDEPENDENT_HUMAN_CONTINUATION",
            "dataset_summary": {key: {**value, "domain_type": DOMAIN_TYPE[key]} for key, value in dataset_counts.items()},
            "watermark_self_detection": detection,
            "extension_manifest_sha256": sha256(extension / "preparation_manifest.json"),
            "clean20_manifest_sha256": sha256(clean20 / "data_manifest.json"),
            "clean41_manifest_sha256": sha256(clean41 / "data_manifest.json")}
        atomic_json(output / "data_manifest.json", manifest)
        report_lines = [
            '# Broad watermark benchmark', '',
            '## Dataset summary', '',
            f'The benchmark includes {len(dataset_counts)} datasets, 42 '
            f'dataset-generator cells, five generator models, and 11 machine '
            f'conditions. The full view contains {full_exposures} cell-source '
            f'exposures and {total_rows} text rows. The character-matched '
            f'strict view contains {strict_exposures} cell-source exposures '
            f'and {strict_rows} text rows.', '',
            'The machine conditions are unwatermarked, KGW, SWEET, Unigram, '
            'EXP, SynthID, DIP, ITSEdit, CGW, HeavyWater, and SemStamp.', '',
            '## Input and generation audits', '',
            f"Clean20 repaired {clean20_manifest['initial_near_complete_repair_rows']} "
            f"near-complete rows and {clean20_manifest['residual_sparse_repair_rows']} "
            f"sparse rows. Its manifest records {clean20_manifest['total_rows']} "
            'text rows.', '',
            f"Clean41 checked {clean41_likelihood_audit['machine_rows']} "
            f"machine texts for likelihood anomalies; "
            f"{clean41_likelihood_audit['failed_rows']} fell below the -8 "
            'threshold.', '',
            f"The pilot quality audit passed {len(pilot_quality)} of six rows. "
            f"Its Falcon observer mean log likelihood ranged from "
            f"{min(float(row['falcon_observer_mean_log_likelihood']) for row in pilot_quality):.3f} "
            f"to {max(float(row['falcon_observer_mean_log_likelihood']) for row in pilot_quality):.3f}.", '',
            f"Generation scheduler status: {scheduler['status']}. "
            f"Sparse recovery status: "
            f"{recovery['status'] if recovery else 'NOT_REQUIRED'}. "
            f"Initially archived rows: "
            f"{recovery_initial_by_status.get('ARCHIVED', 0)}; initially "
            f"missing records: "
            f"{recovery_initial_by_status.get('MISSING_BEFORE_RETRY', 0)}.", '',
            '## File layout', '',
            '`complete_prompt_free/<dataset>/<generator>/data.json` contains '
            'the original human continuation and eleven machine streams. '
            'The neighboring `manifest.json` records source IDs. '
            '`complete_strict_char_matched/` stores the selected '
            'character-matched sources with the same stream structure.', '',
            '## Dataset coverage', '',
            '| Dataset | Domain | Cells | Full sources | Strict sources |',
            '|---|---|---:|---:|---:|',
        ]
        for dataset, row in sorted(dataset_counts.items()):
            report_lines.append(
                f"| {dataset} | {DOMAIN_TYPE[dataset]} | {row['cells']} | "
                f"{row['full_sources']} | {row['strict_sources']} |"
            )
        report_lines += [
            '', '## Watermark self-detection', '',
            '| Method | Available scores | Detection fraction at 5% |',
            '|---|---:|---:|',
        ]
        for method, row in detection.items():
            value = row['detection_fraction_at_one_sided_5pct']
            report_lines.append(f"| {method} | {row['available']} | {value:.1%} |")
        report_lines += [
            '', '## Scope', '',
            'The Chinese-language benchmark was excluded because independent '
            'human continuations were unavailable. English benchmark rows and '
            'all recorded machine streams are retained.', '',
            '**Finalization status: PASS**',
        ]
        (output / "BROAD_DATASET_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        code_files = [
            Path("scripts/clean_clean41_text_separation.py"), Path("scripts/run_clean20_watermark_extension_worker.py"),
            Path("scripts/prepare_broad_watermark_extension.py"), Path("scripts/run_broad_watermark_extension.py"),
            Path("scripts/retry_broad_watermark_failed_records.py"),
            Path("scripts/score_broad_watermark_quality.py"), Path("scripts/run_broad_watermark_quality.py"),
            Path("scripts/finalize_broad_watermark_dataset.py"), Path("scripts/audit_clean41_likelihood_anomalies.py"),
        ]
        artifact_paths = [output / "data_manifest.json", output / "BROAD_DATASET_REPORT.md", output / "REPRODUCE.sh",
                          extension / "preparation_manifest.json", extension / "scheduler_audit/summary.json",
                          extension / "quality_audit/summary.json", extension / "finalization_audit.json",
                          extension / "clean41_likelihood_anomaly_audit.json",
                          extension / "pilot_requests.json", extension / "pilot_scheduler_audit/summary.json",
                          extension / "pilot_quality_audit/summary.json",
                          clean20 / "data_manifest.json", clean41 / "data_manifest.json"]
        if recovery_path.exists():
            artifact_paths.append(recovery_path)
        artifact_paths.extend(sorted((output / "complete_prompt_free").glob("*/*/data.json")))
        artifact_paths.extend(sorted((output / "complete_prompt_free").glob("*/*/manifest.json")))
        artifact_paths.extend(sorted((output / "complete_strict_char_matched").glob("*/*/data.json")))
        artifact_paths.extend(sorted((output / "complete_strict_char_matched").glob("*/*/manifest.json")))
        atomic_json(output / "artifact_manifest.json", {
            "schema_version": "broad-watermark-artifact-manifest-v1", "status": "PASS", "seed": 20260816,
            "new_generation_target_rows": len(requests),
            "generation_attempts_in_final_records": generation_attempts,
            "semstamp_candidate_trials_in_final_records": semstamp_candidate_trials,
            "generation_elapsed_seconds": scheduler.get("elapsed_seconds"),
            "quality_elapsed_seconds": quality_scheduler.get("elapsed_seconds"),
            "code_sha256": {str(path): sha256(path) for path in code_files},
            "artifacts": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in artifact_paths],
        })
    print(json.dumps({"status": status, "failures": failures, "output": str(output)}, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
