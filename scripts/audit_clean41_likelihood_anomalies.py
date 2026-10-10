#!/usr/bin/env python3
"""Audit frozen clean41 rows for the same gross low-likelihood corruption mode."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from io_utils import atomic_json, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=Path("scratch/clean41_llama_proxy_necessity/clean41_five_proxy_rows.json"))
    parser.add_argument("--output", type=Path, default=Path("scratch/broad_watermark_extension/clean41_likelihood_anomaly_audit.json"))
    parser.add_argument("--threshold", type=float, default=-8.0)
    args = parser.parse_args()
    payload = json.loads(args.rows.read_text(encoding="utf-8"))
    proxies = payload["default_proxies"]
    failures = []
    machine_rows = 0
    for row in payload["rows"]:
        if int(row["label"]) != 1: continue
        machine_rows += 1
        value = float(np.median([row[f"{proxy}__likelihood"] for proxy in proxies]))
        if value < args.threshold:
            failures.append({"sample_id": row["sample_id"], "source_id": row["source_id"],
                             "dataset": row["dataset"], "generator": row["generator"],
                             "method": row["watermark_method"], "median_four_proxy_ll": value})
    result = {"schema_version": "clean41-likelihood-anomaly-audit-v1",
        "status": "PASS" if not failures else "ANOMALIES_FOUND", "threshold": args.threshold,
        "machine_rows": machine_rows, "failed_rows": len(failures),
        "failed_cells": len({(r["dataset"], r["generator"], r["method"]) for r in failures}),
        "failures_by_generator": dict(Counter(r["generator"] for r in failures)),
        "failures_by_dataset": dict(Counter(r["dataset"] for r in failures)),
        "failures_by_method": dict(Counter(r["method"] for r in failures)),
        "proxies": proxies, "input": str(args.rows.resolve()),
        "input_sha256": sha256(args.rows), "failures": failures}
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "machine_rows", "failed_rows", "failed_cells")}, indent=2))


if __name__ == "__main__": main()
