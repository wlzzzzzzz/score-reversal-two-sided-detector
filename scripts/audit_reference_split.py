#!/usr/bin/env python3
"""Audit the development split, then an explicitly unlocked test split.

No detector scores or metrics are read. Outputs are immutable: an existing
destination is an error. Relative paths in the config resolve from the repo.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_OUTPUT = "development_source_manifest.json"
TEST_OUTPUT = "test_source_audit.json"
SHINGLE_SIZE = 5
MIN_SHINGLES = 20
MIN_CONTAINMENT = 0.8


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def text_sha256(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def split_name(value: str) -> str:
    return "development" if value == "calibration" else value


def shingles(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + SHINGLE_SIZE])
            for i in range(max(0, len(words) - SHINGLE_SIZE + 1))}


def immutable_json(path: Path, value: dict) -> None:
    """Publish a complete file without replacing an existing destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Fails atomically if path already exists.
    finally:
        os.unlink(temporary)


def read_config(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict) or not isinstance(config.get("data_cells"), list):
        raise ValueError("config.data_cells must be a list")
    if not re.fullmatch(r"[0-9a-f]{64}", config.get("frozen_protocol_sha256", "")):
        raise ValueError("config requires frozen_protocol_sha256")
    if "protocol_path" in config:
        protocol_hash = sha256_bytes(resolve_path(config["protocol_path"]).read_bytes())
        if protocol_hash != config["frozen_protocol_sha256"]:
            raise ValueError("protocol_path does not match frozen_protocol_sha256")
    methods = config.get("methods", config.get("machine_methods"))
    if (not isinstance(methods, list) or not methods
            or any(not isinstance(m, str) or not m for m in methods)
            or len(set(methods)) != len(methods)
            or {"original", "human"}.intersection(methods)):
        raise ValueError("methods/machine_methods must list distinct machine channels")
    return config, sha256_bytes(raw)


def selected_cells(config: dict, split: str) -> list[dict]:
    """Inspect manifest metadata only; do not open any source here."""
    cells = [c for c in config["data_cells"] if split_name(c.get("split")) == split]
    expected_cells, expected_domains = (20, 4) if split == "development" else (60, 12)
    if len(cells) != expected_cells:
        raise ValueError(f"{split} requires exactly {expected_cells} cells")
    groups = defaultdict(set)
    identities = set()
    counts = {}
    for cell in cells:
        for key in ("dataset", "generator", "data_path", "sha256", "source_count"):
            if key not in cell:
                raise ValueError(f"cell lacks {key}")
        dataset, generator = cell["dataset"], cell["generator"]
        if not isinstance(dataset, str) or not dataset or ":" in dataset:
            raise ValueError("dataset must be a nonempty name without colons")
        if not isinstance(generator, str) or not generator:
            raise ValueError("generator must be a nonempty name")
        identity = (dataset, generator)
        if identity in identities:
            raise ValueError(f"duplicate cell {identity}")
        identities.add(identity)
        groups[dataset].add(generator)
        count = cell["source_count"]
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"invalid source_count for {identity}")
        if dataset in counts and counts[dataset] != count:
            raise ValueError(f"generator source counts disagree for {dataset}")
        counts[dataset] = count
        if not re.fullmatch(r"[0-9a-f]{64}", cell["sha256"]):
            raise ValueError(f"invalid source SHA-256 for {identity}")
    if len(groups) != expected_domains or any(len(g) != 5 for g in groups.values()):
        raise ValueError(f"{split} must be {expected_domains} domains x 5 generators")
    if len({tuple(sorted(g)) for g in groups.values()}) != 1:
        raise ValueError("generator sets must agree across domains")
    configured_domains = config.get(split + "_datasets")
    if configured_domains is not None and set(configured_domains) != set(groups):
        raise ValueError(f"{split} datasets disagree with config")
    return sorted(cells, key=lambda c: (c["dataset"], c["generator"]))


def load_sources(config: dict, cells: list[dict]) -> dict:
    """Open only the supplied whitelist and verify every input before use."""
    methods = config.get("methods", config.get("machine_methods"))
    human = {}
    text_groups = {}
    metadata = []
    row_count = 0
    for cell in cells:
        path = resolve_path(cell["data_path"])
        raw = path.read_bytes()
        digest = sha256_bytes(raw)
        if digest != cell["sha256"]:
            raise ValueError(f"input SHA-256 mismatch: {path}")
        data = json.loads(raw)
        if not isinstance(data, dict) or set(data) != {"original", *methods}:
            raise ValueError(f"unexpected method keys: {path}")
        count = cell["source_count"]
        for method, values in data.items():
            if (not isinstance(values, list) or len(values) != count
                    or any(not isinstance(text, str) for text in values)):
                raise ValueError(f"invalid {method} text list/count: {path}")
        expected_rows = count * (len(methods) + 1)
        if cell.get("expected_rows", expected_rows) != expected_rows:
            raise ValueError(f"expected_rows disagrees with source count: {path}")
        for index, text in enumerate(data["original"]):
            source_id = f"{cell['dataset']}:{index}"
            if source_id in human:
                if human[source_id]["text"] != text:
                    raise ValueError(f"Human text differs across generators: {source_id}")
                human[source_id]["generators"].append(cell["generator"])
            else:
                human[source_id] = {
                    "source_id": source_id, "dataset": cell["dataset"],
                    "source_index": index, "text": text,
                    "text_sha256": text_sha256(text), "generators": [cell["generator"]],
                }
            for method in ["human", *methods]:
                value = text if method == "human" else data[method][index]
                digest_text = text_sha256(value)
                group = text_groups.setdefault(digest_text, {
                    "rawtext_dedup_id": "sha256:" + digest_text,
                    "text_sha256": digest_text, "occurrences": [],
                })
                group["occurrences"].append({
                    "source_id": source_id, "dataset": cell["dataset"],
                    "source_index": index, "generator": cell["generator"],
                    "watermark_method": method,
                    "row_id": f"{cell['dataset']}:{cell['generator']}:{index}:{method}",
                })
        metadata.append({
            "cell": cell.get("cell", f"{cell['dataset']}::{cell['generator']}"),
            "dataset": cell["dataset"], "generator": cell["generator"],
            "split": split_name(cell["split"]), "data_path": str(path),
            "sha256": digest, "source_count": count, "rows": expected_rows,
            "methods": ["human", *methods],
        })
        row_count += expected_rows
    return {"human": human, "text_groups": text_groups,
            "cells": metadata, "rows": row_count}


def public_human(human: dict) -> list[dict]:
    return [{**{k: v for k, v in value.items() if k != "text"},
             "rawtext_dedup_id": "sha256:" + value["text_sha256"]}
            for _, value in sorted(human.items())]


def development_report(config: dict, config_hash: str, sources: dict) -> dict:
    return {
        "schema_version": "reference-shaping-development-source-manifest-v1",
        "status": "PASS", "split": "development", "test_sources_opened": 0,
        "config_sha256": config_hash,
        "frozen_protocol_sha256": config["frozen_protocol_sha256"],
        "cells": sources["cells"], "cell_count": len(sources["cells"]),
        "row_count": sources["rows"], "human_source_count": len(sources["human"]),
        "unique_raw_text_count": len(sources["text_groups"]),
        "raw_text_identity": "SHA-256 of exact UTF-8 text; no normalization",
        "human_sources": public_human(sources["human"]),
        "raw_text_groups": [v for _, v in sorted(sources["text_groups"].items())],
        "human_consistent_across_all_five_generators": True,
    }


def authorize_test(config: dict, config_hash: str, output: Path) -> dict:
    """Every authorization check precedes all test-source file access."""
    frozen_path = output / "FROZEN_SELECTION.json"
    authorization_path = output / "TEST_ACCESS_AUTHORIZATION.json"
    frozen_bytes = frozen_path.read_bytes()
    authorization_bytes = authorization_path.read_bytes()
    frozen, authorization = json.loads(frozen_bytes), json.loads(authorization_bytes)
    if authorization.get("authorized") is not True:
        raise PermissionError("TEST_ACCESS_AUTHORIZATION.json must set authorized=true")
    expected = {"config_sha256": config_hash,
                "frozen_protocol_sha256": config["frozen_protocol_sha256"]}
    for key, value in expected.items():
        if frozen.get(key) != value or authorization.get(key) != value:
            raise PermissionError(f"frozen selection/authorization {key} mismatch")
    frozen_hash = sha256_bytes(frozen_bytes)
    if authorization.get("frozen_selection_sha256") != frozen_hash:
        raise PermissionError("authorization frozen_selection_sha256 mismatch")
    code_key = "scripts/audit_reference_split.py"
    code_hashes = frozen.get("code_sha256")
    current_code_hash = sha256_bytes(Path(__file__).resolve().read_bytes())
    if not isinstance(code_hashes, dict) or code_hashes.get(code_key) != current_code_hash:
        raise PermissionError("frozen selection audit code_sha256 mismatch")
    return {**expected, "frozen_selection_sha256": frozen_hash,
            "test_access_authorization_sha256": sha256_bytes(authorization_bytes),
            "audit_code_sha256": current_code_hash}


def compare_splits(development: dict, test: dict) -> dict:
    """Exclude whole test sources on exact text or eligible Human near matches."""
    exact_matches = []
    excluded = set()
    for digest, group in sorted(test["text_groups"].items()):
        development_group = development["text_groups"].get(digest)
        if development_group is None:
            continue
        dev_ids = sorted({r["source_id"] for r in development_group["occurrences"]})
        test_ids = sorted({r["source_id"] for r in group["occurrences"]})
        excluded.update(test_ids)
        exact_matches.append({
            "reason": "exact_raw_text_cross_split", "text_sha256": digest, "ratio": 1.0,
            "test_source_ids": test_ids, "matched_development_source_ids": dev_ids,
            "test_occurrences": group["occurrences"],
            "matched_development_occurrences": development_group["occurrences"],
        })

    dev_shingles = {}
    postings = defaultdict(list)
    for source_id, source in development["human"].items():
        tokens = shingles(source["text"])
        if len(tokens) < MIN_SHINGLES:
            continue
        dev_shingles[source_id] = tokens
        for token in tokens:
            postings[token].append(source_id)
    near_matches = []
    for source_id, source in sorted(test["human"].items()):
        tokens = shingles(source["text"])
        if len(tokens) < MIN_SHINGLES:
            continue
        candidates = Counter(dev_id for token in tokens for dev_id in postings.get(token, ()))
        for dev_id, intersection in sorted(candidates.items()):
            denominator = min(len(tokens), len(dev_shingles[dev_id]))
            if intersection * 5 < denominator * 4:
                continue
            # Exact matches already have complete channel/source provenance above.
            if source["text_sha256"] == development["human"][dev_id]["text_sha256"]:
                continue
            excluded.add(source_id)
            near_matches.append({
                "reason": "human_near_overlap", "test_source_id": source_id,
                "matched_development_source_id": dev_id,
                "ratio": intersection / denominator,
                "shared_shingles": intersection, "test_shingles": len(tokens),
                "development_shingles": len(dev_shingles[dev_id]),
                "test_text_sha256": source["text_sha256"],
                "development_text_sha256": development["human"][dev_id]["text_sha256"],
            })
    excluded_rows = sum(r["source_id"] in excluded
                        for group in test["text_groups"].values()
                        for r in group["occurrences"])
    return {
        "overlap_policy": {
            "exact": "all Human and Machine raw texts across splits, including short text",
            "near": "Human only; lowercase Unicode word tokens; unique 5-word shingles",
            "minimum_unique_shingles_in_each_text": MIN_SHINGLES,
            "containment_denominator": "min(number of development shingles, number of test shingles)",
            "minimum_containment": MIN_CONTAINMENT,
            "exclusion_unit": "entire dataset:index bundle across every generator and channel",
        },
        "excluded_source_ids": sorted(excluded), "excluded_source_count": len(excluded),
        "excluded_rows": excluded_rows, "included_rows": test["rows"] - excluded_rows,
        "included_source_count": len(test["human"]) - len(excluded),
        "exact_matches": exact_matches, "human_near_matches": near_matches,
    }


def run(config_path: Path, split: str) -> Path:
    config, config_hash = read_config(config_path)
    output = resolve_path(config["output_root"])
    destination = output / (DEVELOPMENT_OUTPUT if split == "development" else TEST_OUTPUT)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite audit artifact: {destination}")
    if split == "development":
        development = load_sources(config, selected_cells(config, "development"))
        report = development_report(config, config_hash, development)
    elif split == "test":
        authorization = authorize_test(config, config_hash, output)
        manifest_path = output / DEVELOPMENT_OUTPUT
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        for key in ("config_sha256", "frozen_protocol_sha256"):
            if manifest.get(key) != authorization[key]:
                raise PermissionError(f"development manifest {key} mismatch")
        development = load_sources(config, selected_cells(config, "development"))
        if manifest != development_report(config, config_hash, development):
            raise ValueError("development manifest does not reproduce from frozen source inputs")
        test_cells = selected_cells(config, "test")
        dev_domains = {c["dataset"] for c in development["cells"]}
        if dev_domains.intersection(c["dataset"] for c in test_cells):
            raise ValueError("development and test domain names must be disjoint")
        test = load_sources(config, test_cells)
        report = {
            "schema_version": "reference-shaping-test-source-audit-v1",
            "status": "PASS", "split": "test", **authorization,
            "development_source_manifest_sha256": sha256_bytes(manifest_bytes),
            "cells": test["cells"], "cell_count": len(test["cells"]),
            "row_count": test["rows"], "human_source_count": len(test["human"]),
            "human_sources": public_human(test["human"]),
            "human_consistent_across_all_five_generators": True,
            **compare_splits(development, test),
            "original_data_modified": False,
        }
    else:
        raise ValueError(f"unknown split: {split}")
    immutable_json(destination, report)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "test"), required=True)
    args = parser.parse_args()
    path = run(args.config, args.split)
    print(json.dumps({"status": "PASS", "split": args.split, "output": str(path)}))


if __name__ == "__main__":
    main()
