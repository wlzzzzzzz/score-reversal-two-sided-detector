#!/usr/bin/env python3
"""Schedule broad CGW/HeavyWater/SemStamp generation across the requested GPUs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from io_utils import atomic_json
from gpu_jobs import parse_gpus, run_jobs

ROOT = Path(".")
PYTHON = Path("python3")
WORKER = ROOT / "scripts/run_clean20_watermark_extension_worker.py"
GENERATORS = ("GLM-4-9B-0414", "falcon", "llama-2-7b", "opt-2.7B", "opt-6.7B")
METHODS = ("CGW", "HeavyWater", "SemStamp")


def build_jobs(output: Path, request: Path, seed: int) -> list[dict]:
    jobs = []
    for method in METHODS:
        for generator in GENERATORS:
            shards = 4 if generator == "llama-2-7b" and method == "SemStamp" else (2 if generator == "llama-2-7b" else 1)
            for shard in range(shards):
                command = [
                    str(PYTHON), str(WORKER), "--method", method, "--generator", generator,
                    "--request-manifest", str(request), "--output-dir", str(output),
                    "--shard-count", str(shards), "--shard-index", str(shard),
                    "--max-new-tokens", "200", "--max-attempts", "10", "--base-seed", str(seed),
                    "--strict-text-quality", "--repetition-penalty", "1.1", "--no-repeat-ngram-size", "4",
                ]
                if generator == "falcon": command.append("--native-falcon")
                jobs.append({"name": f"{method}_{generator}_{shard:02d}_of_{shards:02d}", "command": command})
    # Token-level methods finish first; use all GPUs for the slower sentence-level stage afterwards.
    jobs.sort(key=lambda row: (row["name"].startswith("SemStamp"), row["name"]))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--audit-name", default="scheduler_audit")
    args = parser.parse_args()
    output, request = args.output_dir.resolve(), args.request_manifest.resolve()
    gpus = parse_gpus(args.gpus)
    queue = build_jobs(output, request, args.seed)
    audit = output / args.audit_name
    started = time.time()
    completed = run_jobs(queue, gpus, audit, args.seed, progress=True)
    failures = [row for row in completed if row["returncode"] != 0]
    summary = {"status": "PASS" if not failures else "FAILED", "job_count": len(completed),
               "failure_count": len(failures), "failures": failures, "jobs": completed,
               "elapsed_seconds": time.time() - started}
    atomic_json(audit / "summary.json", summary, sort_keys=False)
    print(json.dumps({"status": summary["status"], "jobs": len(completed), "failures": len(failures)}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
