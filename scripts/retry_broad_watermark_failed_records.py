#!/usr/bin/env python3
"""Audited sparse recovery for broad-generation records that failed frozen first pass."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from io_utils import sha256
from gpu_jobs import parse_gpus, run_jobs

ROOT = Path(".")
PYTHON = Path("python3")
WORKER = ROOT / "scripts/run_clean20_watermark_extension_worker.py"


def record_path(output: Path, row: dict) -> Path:
    return output / "records" / row["dataset"] / row["generator"] / row["method"] / f"{int(row['slot_index']):04d}.json"


def is_success(output: Path, row: dict) -> bool:
    path = record_path(output, row)
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(value, dict):
        return False
    expected = (row["dataset"], row["generator"], row["method"], row["source_id"])
    actual = (value.get("dataset"), value.get("generator"), value.get("method"), value.get("source_id"))
    return (value.get("status") == "SUCCESS" and actual == expected
            and value.get("prompt_sha256") == hashlib.sha256(row["prompt"].encode()).hexdigest())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--first-pass-seed", type=int, default=20260816)
    parser.add_argument("--max-rounds", type=int, default=3)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    requests = json.loads(args.request_manifest.read_text())
    gpus = parse_gpus(args.gpus)
    audit = output / "sparse_recovery_audit"
    audit.mkdir(parents=True, exist_ok=True)
    round_rows = []
    archived_records = []
    failed_retry_counts = defaultdict(int)
    for round_index in range(1, args.max_rounds + 1):
        failed = [row for row in requests if not is_success(output, row)]
        if not failed:
            break
        # Preserve every negative record before --force replaces it.  This keeps
        # first-pass and later-round failures auditable rather than only retaining
        # the list of failed request keys.
        for row in failed:
            source = record_path(output, row)
            if not source.exists():
                archived_records.append({"round": round_index, "source_id": row["source_id"],
                    "dataset": row["dataset"], "generator": row["generator"], "method": row["method"],
                    "status": "MISSING_BEFORE_RETRY"})
                continue
            target = audit / "pre_retry_records" / f"round_{round_index:02d}" / row["dataset"] / row["generator"] / row["method"] / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            archived_records.append({"round": round_index, "source_id": row["source_id"],
                "dataset": row["dataset"], "generator": row["generator"], "method": row["method"],
                "status": "ARCHIVED", "path": str(target), "sha256": sha256(target)})
        retry_path = audit / f"retry_requests_round_{round_index:02d}.json"
        retry_path.write_text(json.dumps(failed, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        groups = defaultdict(list)
        for row in failed:
            mode = "FAILED_RECORD" if record_path(output, row).exists() else "MISSING_RECORD"
            groups[(row["method"], row["generator"], mode)].append(row)
        jobs = []
        for (method, generator, mode), rows in sorted(groups.items()):
            group_path = audit / f"round_{round_index:02d}_{method}_{generator}_{mode}.json"
            group_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            base_seed = args.first_pass_seed if mode == "MISSING_RECORD" else args.seed + round_index
            command = [
                str(PYTHON), str(WORKER), "--method", method, "--generator", generator,
                "--request-manifest", str(group_path), "--output-dir", str(output), "--force",
                "--strict-text-quality", "--repetition-penalty", "1.1", "--no-repeat-ngram-size", "4",
                "--max-new-tokens", "200", "--max-attempts", "10",
                "--base-seed", str(base_seed),
                "--audit-tag", f"recovery_round_{round_index:02d}",
            ]
            if method == "SemStamp":
                if mode == "MISSING_RECORD":
                    # A scheduler/code failure did not constitute a generation
                    # attempt: rerun the frozen first-pass semantics exactly.
                    min_new_tokens = 0
                else:
                    min_new_tokens = 64 + 32 * failed_retry_counts[(method, generator)]
                    failed_retry_counts[(method, generator)] += 1
                command += ["--semstamp-min-new-tokens", str(min_new_tokens)]
            if generator == "falcon":
                command.append("--native-falcon")
            jobs.append({"name": f"round_{round_index:02d}_{method}_{generator}_{mode}",
                         "method": method, "generator": generator, "recovery_mode": mode,
                         "command": command, "rows": len(rows)})
        completed = [{key: row[key] for key in
                      ("method", "generator", "recovery_mode", "gpu", "rows", "returncode", "command")}
                     for row in sorted(run_jobs(jobs, gpus, audit, args.seed), key=lambda row: row["name"])]
        remaining = [row for row in requests if not is_success(output, row)]
        round_rows.append({"round": round_index, "input_failures": len(failed),
                           "remaining_failures": len(remaining), "jobs": completed})
    remaining = [row for row in requests if not is_success(output, row)]
    summary = {"schema_version": "broad-watermark-sparse-recovery-v1",
               "status": "PASS" if not remaining else "FAILED", "requested_rows": len(requests),
               "remaining_failures": len(remaining), "rounds": round_rows, "remaining": remaining,
               "first_pass_failures_preserved": True,
               "archived_pre_retry_records": archived_records,
               "archived_pre_retry_record_count": sum(row["status"] == "ARCHIVED" for row in archived_records),
               "quality_thresholds_unchanged": True}
    (audit / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "remaining": len(remaining), "rounds": len(round_rows)}, indent=2))
    return 0 if not remaining else 2


if __name__ == "__main__":
    raise SystemExit(main())
