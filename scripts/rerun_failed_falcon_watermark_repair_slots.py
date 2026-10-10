#!/usr/bin/env python3
"""Sparsely regenerate only records rejected by the model-based quality audit."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

from io_utils import atomic_json
from gpu_jobs import parse_gpus, run_jobs

ROOT = Path(".")
PYTHON = Path("python3")
NEW_WORKER = ROOT / "scripts/run_clean20_watermark_extension_worker.py"
OLD_WORKER = ROOT / "scripts/run_20cell_regeneration_worker.py"
NEW_METHODS = {"CGW", "HeavyWater", "SemStamp"}


def load_failures(repair: Path) -> list[dict]:
    rows = []
    for path in sorted((repair / "quality_scores").glob("shard_*.csv")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(row for row in csv.DictReader(handle) if int(row["quality_pass"]) == 0)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    repair = args.repair_dir.resolve()
    gpus = parse_gpus(args.gpus)
    failures = load_failures(repair)
    audit = repair / "quality_retry_audit" / f"round_{args.round:02d}"
    audit.mkdir(parents=True, exist_ok=True)
    with (audit / "failed_quality.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(failures[0]) if failures else ["status"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)
    if not failures:
        atomic_json(audit / "summary.json", {"status": "NOTHING_TO_DO", "failed_rows": 0}, sort_keys=False)
        return 0
    previous_records = []
    for row in failures:
        dataset, method, slot = row["dataset"], row["method"], int(row["slot_index"])
        if method in NEW_METHODS:
            record_path = repair / "records" / dataset / "falcon" / method / f"{slot:04d}.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            previous_records.append({
                "dataset": dataset, "method": method, "slot_index": slot,
                "record_path": str(record_path), "record": record,
                "failed_quality": row,
            })
        else:
            data_path = repair / "prepared_raw" / dataset / "falcon" / "data.json"
            payload = json.loads(data_path.read_text(encoding="utf-8"))
            previous_records.append({
                "dataset": dataset, "method": method, "slot_index": slot,
                "data_path": str(data_path), "raw_text": payload[method][slot],
                "failed_quality": row,
            })
    atomic_json(audit / "previous_failed_records.json", previous_records, sort_keys=False)
    grouped = defaultdict(list)
    for row in failures:
        grouped[(row["method"], row["dataset"])].append(int(row["slot_index"]))
    retry_seed = args.seed + args.round
    jobs = []
    for (method, dataset), all_slots in sorted(grouped.items()):
        # A full damaged CGW cell can contain ~200 rows.  Split only large
        # sparse groups into disjoint slot chunks so released GPUs are not
        # idle behind one cell-level process/lock.
        chunk_size = 100
        unique_slots = sorted(set(all_slots))
        slot_chunks = [unique_slots[start:start + chunk_size]
                       for start in range(0, len(unique_slots), chunk_size)]
        for chunk_index, slots in enumerate(slot_chunks):
            slot_text = ",".join(map(str, slots))
            suffix = f"_part{chunk_index:02d}" if len(slot_chunks) > 1 else ""
            if method in NEW_METHODS:
                command = [
                    str(PYTHON), str(NEW_WORKER), "--method", method,
                    "--generator", "falcon", "--datasets", dataset,
                    "--output-dir", str(repair), "--slot-indices", slot_text,
                    "--max-new-tokens", "200", "--max-attempts", "10",
                    "--base-seed", str(retry_seed), "--native-falcon", "--force",
                    "--strict-text-quality",
                ]
                if method in {"CGW", "HeavyWater"}:
                    command.extend([
                        "--repetition-penalty", "1.1", "--no-repeat-ngram-size", "4",
                    ])
            else:
                command = [
                    str(PYTHON), str(OLD_WORKER), "--output-dir", str(repair),
                    "--generator", "falcon", "--methods", method,
                    "--datasets", dataset, "--slot-indices", slot_text,
                    "--audit-tag", f"quality_retry_{args.round}_{method}_{dataset}{suffix}",
                    "--max-new-tokens", "200", "--max-attempts", "10",
                    "--seed", str(retry_seed), "--native-falcon", "--force",
                    "--strict-text-quality",
                ]
            jobs.append({
                "name": f"{method}_{dataset}{suffix}", "command": command, "slots": slots,
            })
    started = time.time()
    completed = [{key: row[key] for key in ("name", "command", "slots", "gpu", "pid", "returncode")}
                 for row in run_jobs(jobs, gpus, audit, args.seed)]
    archive = audit / "previous_quality_scores"
    archive.mkdir(parents=True, exist_ok=True)
    for path in sorted((repair / "quality_scores").glob("shard_*.csv")):
        shutil.move(str(path), archive / path.name)
    failed_jobs = [row for row in completed if row["returncode"] != 0]
    status = "PASS" if not failed_jobs else "FAILED_OR_INCOMPLETE"
    atomic_json(audit / "summary.json", {
        "status": status, "failed_rows": len(failures), "retry_seed": retry_seed,
        "jobs": completed, "failed_jobs": failed_jobs,
        "elapsed_seconds": time.time() - started,
        "quality_scores_archived_to": str(archive),
    }, sort_keys=False)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
