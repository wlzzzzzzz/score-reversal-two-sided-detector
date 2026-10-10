#!/usr/bin/env python3
"""Schedule the independent broad-record quality audit."""

import argparse
import json
import time
from pathlib import Path

from gpu_jobs import parse_gpus, run_jobs


ROOT = Path(".")
PYTHON = Path("python3")
SCORER = ROOT / "scripts/score_broad_watermark_quality.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6")
    parser.add_argument("--threshold", type=float, default=-8.0)
    args = parser.parse_args()
    gpus = parse_gpus(args.gpus)
    audit = args.output_dir.resolve() / "quality_audit"
    jobs, started = [], time.time()
    for shard in range(len(gpus)):
        command = [str(PYTHON), str(SCORER), "--output-dir", str(args.output_dir.resolve()),
                   "--request-manifest", str(args.request_manifest.resolve()), "--shard-count", str(len(gpus)),
                   "--shard-index", str(shard), "--threshold", str(args.threshold)]
        jobs.append({"name": f"shard_{shard:02d}", "shard": shard, "command": command})
    completed = [{key: row[key] for key in ("shard", "gpu", "returncode")}
                 for row in sorted(run_jobs(jobs, gpus, audit, 20260816), key=lambda row: row["shard"])]
    failures = [row for row in completed if row["returncode"]]
    summary = {"status": "PASS" if not failures else "FAILED", "jobs": completed,
               "failure_count": len(failures), "elapsed_seconds": time.time() - started}
    (audit / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
