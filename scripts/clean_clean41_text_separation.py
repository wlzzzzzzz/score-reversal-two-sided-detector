#!/usr/bin/env python3
"""Build a non-destructive, prompt-free clean41 benchmark.

The historical watermark files mix two incompatible schemas:

* thirteen datasets store ``original`` as the generation prompt and store most
  machine streams as ``prompt + continuation``;
* SQuAD, Writing and XSum store a complete human document, while the machine
  streams were generated from an earlier prompt recovered from exp_main;
* ITSEdit always returns continuation-only text.

This script recovers the complete human source where it exists, removes the
exact generation prompt from both classes, and writes a strict character-
matched view.  Raw inputs are read-only and are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from io_utils import sha256 as sha256_file, sha256_text
from text_utils import lexical_count

METHODS = (
    "unwatermarked", "KGW", "SWEET", "Unigram",
    "EXP", "SynthID", "DIP", "ITSEdit",
)
LEGACY_FULL_HUMAN_DATASETS = {"squad", "writing", "xsum"}
MAGE_PROMPT_DATASETS = {
    "cmv", "cnn", "dialogsum", "eli5", "hswag", "imdb",
    "pubmed", "roct", "sci", "tldr", "wp", "yelp",
}
UNRECOVERED_DATASETS = {"chinese"}
SENTINELS = {"", "<GENERATION_PENDING>", "<GENERATION_FAILED>"}


def trim_exact_chars(text: str, count: int) -> str:
    """Return exactly ``count`` Unicode code points after boundary stripping."""
    value = str(text).strip()
    if count < 0 or count > len(value):
        raise ValueError("character budget is outside the text")
    return value[:count]


def infer_legacy_prompt(human: str, sampled: str) -> str:
    """Reproduce the word-boundary common-prefix rule used by MarkLLM setup."""
    common = os.path.commonprefix([str(human), str(sampled)])
    if " " in common:
        common = common.rsplit(" ", 1)[0]
    return common.strip()


def infer_repaired_falcon_prompt(human: str, llama_machine: str) -> str:
    """Reproduce the Falcon repair's whitespace-token prefix reconstruction."""
    left_tokens = re.findall(r"\S+", str(human))
    right_tokens = re.findall(r"\S+", str(llama_machine))
    common = []
    for left, right in zip(left_tokens, right_tokens):
        if left != right:
            break
        common.append(left)
    if len(common) < 8:
        raise ValueError("repaired Falcon prompt has fewer than eight exact tokens")
    return " ".join(common)


def split_machine_continuation(raw: str, prompt: str, method: str) -> Tuple[str, str]:
    """Split a machine stream according to its generating implementation.

    ITSEdit decodes only tokens after the prompt.  Every other current stream
    decodes the complete model sequence and therefore must begin with the exact
    prompt.  Approximate prefix deletion is deliberately forbidden.
    """
    value = str(raw)
    if value.strip() in SENTINELS:
        return "", "generation_sentinel"
    if method == "ITSEdit":
        return value.strip(), "already_continuation"
    if not value.startswith(prompt):
        return "", "exact_prompt_prefix_missing"
    return value[len(prompt):].strip(), "exact_prompt_removed"


def stable_fold(source_id: str) -> int:
    digest = hashlib.sha256(source_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 5


def source_fold_macro_auc(source_ids: Sequence[str], human: Sequence[float], machine: Sequence[float]) -> float:
    aucs = []
    folds = [stable_fold(value) for value in source_ids]
    for fold in range(5):
        selected = [index for index, value in enumerate(folds) if value == fold]
        if not selected:
            continue
        labels = [0] * len(selected) + [1] * len(selected)
        scores = [human[index] for index in selected] + [machine[index] for index in selected]
        aucs.append(float(roc_auc_score(labels, scores)))
    return float(np.mean(aucs)) if aucs else float("nan")


def load_mage_humans(mage_root: Path) -> Tuple[Dict[str, List[str]], List[dict]]:
    corpus: Dict[str, List[str]] = defaultdict(list)
    files = []
    for name in ("train.csv", "valid.csv", "test.csv", "test_ood_set_gpt.csv"):
        path = mage_root / name
        frame = pd.read_csv(path, usecols=["src", "text"])
        files.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        for row in frame.itertuples(index=False):
            source = str(row.src).lower()
            if "human" not in source:
                continue
            dataset = source.split("_", 1)[0]
            corpus[dataset].append(str(row.text).strip())
    return {key: list(dict.fromkeys(values)) for key, values in corpus.items()}, files


def unique_prefix_match(prompt: str, candidates: Sequence[str]) -> Tuple[Optional[str], int]:
    matches = [value for value in candidates if value.startswith(prompt)]
    if len(matches) != 1:
        return None, len(matches)
    return matches[0], 1


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def git_state(repo: Path) -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(args, cwd=str(repo), stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "NOT_AVAILABLE"
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "status_porcelain_sha256": sha256_text(run("git", "status", "--porcelain")),
    }


def recover_sources(
    manifest: Mapping[str, object],
    mage: Mapping[str, Sequence[str]],
    legacy_root: Path,
) -> Tuple[Dict[Tuple[str, int], dict], List[dict], List[dict]]:
    """Recover one canonical prompt and human continuation per source ID."""
    cells = list(manifest["cells"])
    datasets = sorted({str(cell["dataset"]) for cell in cells})
    raw_originals: Dict[str, Dict[int, str]] = defaultdict(dict)
    recovery_issues: List[dict] = []
    legacy_inputs: List[dict] = []

    for cell in cells:
        payload = load_json(Path(str(cell["dataset_file"])))
        count = int(cell["n_sources"])
        for index, original in enumerate(payload["original"][:count]):
            previous = raw_originals[str(cell["dataset"])].get(index)
            if previous is not None and previous != str(original):
                raise RuntimeError(
                    "source identity disagreement across generators: "
                    f"{cell['dataset']}:{index}"
                )
            raw_originals[str(cell["dataset"])][index] = str(original)

    legacy_payloads = {}
    for dataset in sorted(LEGACY_FULL_HUMAN_DATASETS):
        path = legacy_root / f"{dataset}_gpt-j-6B.raw_data.json"
        legacy_payloads[dataset] = load_json(path)
        legacy_inputs.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)})

    recovered: Dict[Tuple[str, int], dict] = {}
    recovery_rows = []
    for dataset in datasets:
        for index, raw_original in sorted(raw_originals[dataset].items()):
            source_id = f"{dataset}:{index}"
            human_full: Optional[str] = None
            prompt = ""
            recovery = ""
            multiplicity = 0

            if dataset in MAGE_PROMPT_DATASETS:
                prompt = raw_original
                human_full, multiplicity = unique_prefix_match(prompt, mage.get(dataset, []))
                recovery = "MAGE_EXACT_UNIQUE_PREFIX" if human_full is not None else "MAGE_MATCH_FAILED"
            elif dataset in LEGACY_FULL_HUMAN_DATASETS:
                legacy = legacy_payloads[dataset]
                if index >= len(legacy["original"]) or str(legacy["original"][index]) != raw_original:
                    recovery = "LEGACY_HUMAN_INDEX_MISMATCH"
                else:
                    human_full = raw_original
                    prompt = infer_legacy_prompt(legacy["original"][index], legacy["sampled"][index])
                    multiplicity = 1
                    recovery = "EXP_MAIN_EXACT_INDEX_AND_COMMON_PREFIX"
            elif dataset in UNRECOVERED_DATASETS:
                prompt = raw_original
                recovery = "MISSING_INDEPENDENT_HUMAN_CONTINUATION"
            else:
                recovery = "UNKNOWN_DATASET_SCHEMA"

            human_continuation = ""
            if human_full is not None:
                if not prompt or not human_full.startswith(prompt):
                    recovery = "HUMAN_EXACT_PROMPT_PREFIX_MISSING"
                    human_full = None
                else:
                    human_continuation = human_full[len(prompt):].strip()

            record = {
                "dataset": dataset,
                "source_index": index,
                "source_id": source_id,
                "prompt": prompt,
                "human_full": human_full,
                "human_continuation": human_continuation,
                "recovery": recovery,
                "match_multiplicity": multiplicity,
            }
            recovered[(dataset, index)] = record
            recovery_rows.append({
                "dataset": dataset,
                "source_index": index,
                "source_id": source_id,
                "recovery": recovery,
                "match_multiplicity": multiplicity,
                "prompt_chars": len(prompt),
                "human_full_chars": len(human_full or ""),
                "human_continuation_chars": len(human_continuation),
                "human_continuation_lexical_units": lexical_count(human_continuation),
                "prompt_sha256": sha256_text(prompt),
                "human_full_sha256": sha256_text(human_full or ""),
            })
            if human_full is None:
                recovery_issues.append({
                    "dataset": dataset, "source_id": source_id, "reason": recovery,
                })
    return recovered, recovery_rows, legacy_inputs


def build_cell_records(manifest: Mapping[str, object], recovered: Mapping[Tuple[str, int], dict]) -> Tuple[List[dict], List[dict]]:
    # The repaired Falcon SQuAD/XSum cells explicitly reconstructed their
    # prompts from aligned Llama human/unwatermarked pairs.  This differs from
    # the older exp_main common-prefix prompt for a small number of sources.
    llama_reference = {}
    for reference_cell in manifest["cells"]:
        if (
            str(reference_cell["dataset"]) in {"squad", "xsum"}
            and str(reference_cell["generator"]) == "llama-2-7b"
        ):
            llama_reference[str(reference_cell["dataset"])] = load_json(
                Path(str(reference_cell["dataset_file"]))
            )

    cells = []
    raw_length_rows = []
    for cell in sorted(manifest["cells"], key=lambda item: (str(item["dataset"]), str(item["generator"]))):
        dataset = str(cell["dataset"])
        generator = str(cell["generator"])
        payload = load_json(Path(str(cell["dataset_file"])))
        count = int(cell["n_sources"])
        source_records = []
        for index in range(count):
            base = recovered[(dataset, index)]
            prompt = str(base["prompt"])
            recovery_name = str(base["recovery"])
            if dataset in {"squad", "xsum"} and generator == "falcon":
                reference = llama_reference[dataset]
                prompt = infer_repaired_falcon_prompt(
                    str(reference["original"][index]), str(reference["unwatermarked"][index])
                )
                recovery_name += "+FALCON_REPAIRED_LLAMA_TOKEN_PREFIX"
            human_full = base["human_full"]
            if human_full is not None and str(human_full).startswith(prompt):
                human_continuation = str(human_full)[len(prompt):].strip()
            else:
                human_continuation = ""
            machine = {}
            split_modes = {}
            raw_starts_with_original = {}
            reasons = []
            if human_full is None:
                reasons.append(str(base["recovery"]))
            elif not str(human_full).startswith(prompt):
                reasons.append("HUMAN_EXACT_CELL_PROMPT_PREFIX_MISSING")
            elif not human_continuation:
                reasons.append("EMPTY_HUMAN_CONTINUATION")

            for method in METHODS:
                raw_starts_with_original[method] = str(payload[method][index]).startswith(
                    str(payload["original"][index])
                )
                continuation, split_mode = split_machine_continuation(payload[method][index], prompt, method)
                machine[method] = continuation
                split_modes[method] = split_mode
                if split_mode in {"generation_sentinel", "exact_prompt_prefix_missing"}:
                    reasons.append(f"{method}:{split_mode}")
                elif not continuation:
                    reasons.append(f"{method}:EMPTY_MACHINE_CONTINUATION")
                elif human_continuation and continuation == human_continuation:
                    reasons.append(f"{method}:EXACT_CROSS_LABEL_DUPLICATE_FULL")

            all_streams = [human_continuation] + [machine[method] for method in METHODS]
            lexical_min = min((lexical_count(value) for value in all_streams), default=0)
            structural_valid = not reasons
            strict32_candidate = structural_valid and lexical_min >= 32
            char_budget = min(640, *(len(value) for value in all_streams)) if strict32_candidate else 0
            char_human = trim_exact_chars(human_continuation, char_budget) if char_budget else ""
            char_machine = {
                method: trim_exact_chars(machine[method], char_budget) if char_budget else ""
                for method in METHODS
            }
            if strict32_candidate and any(char_machine[method] == char_human for method in METHODS):
                reasons.append("EXACT_CROSS_LABEL_DUPLICATE_AFTER_CHAR_MATCH")
                strict32_candidate = False

            source_records.append({
                "dataset": dataset,
                "generator": generator,
                "source_index": index,
                "source_id": f"{dataset}:{index}",
                "prompt": prompt,
                "human_full": human_full,
                "human_continuation": human_continuation,
                "machine_continuations": machine,
                "split_modes": split_modes,
                "raw_starts_with_original": raw_starts_with_original,
                "recovery": recovery_name,
                "reasons": sorted(set(reasons)),
                "structural_valid": structural_valid,
                "strict32_candidate": strict32_candidate,
                "lexical_min": lexical_min,
                "char_budget": char_budget,
                "char_human": char_human,
                "char_machine": char_machine,
            })

        source_ids = [f"{dataset}:{index}" for index in range(count)]
        human_raw_lengths = [len(str(value)) for value in payload["original"][:count]]
        for method in METHODS:
            machine_raw_lengths = [len(str(value)) for value in payload[method][:count]]
            raw_length_rows.append({
                "dataset": dataset,
                "generator": generator,
                "method": method,
                "sources": count,
                "source_fold_macro_character_length_auc": source_fold_macro_auc(
                    source_ids, human_raw_lengths, machine_raw_lengths,
                ),
            })
        cells.append({"cell": dict(cell), "payload": payload, "sources": source_records})
    return cells, raw_length_rows


def cross_label_collision_sources(cells: Sequence[dict]) -> Tuple[set, List[dict]]:
    by_hash: Dict[str, List[Tuple[int, str, str, str]]] = defaultdict(list)
    for cell_index, cell in enumerate(cells):
        for source in cell["sources"]:
            if not source["strict32_candidate"]:
                continue
            key = (cell_index, str(source["source_id"]))
            by_hash[sha256_text(source["char_human"])].append((*key, "human", "human"))
            for method in METHODS:
                by_hash[sha256_text(source["char_machine"][method])].append((*key, "machine", method))

    rejected = set()
    audit = []
    for digest, entries in by_hash.items():
        labels = {entry[2] for entry in entries}
        if labels != {"human", "machine"}:
            continue
        sources = sorted({(entry[0], entry[1]) for entry in entries})
        rejected.update(sources)
        audit.append({
            "text_sha256": digest,
            "entry_count": len(entries),
            "source_count": len(sources),
            "entries": json.dumps(entries, ensure_ascii=False),
        })
    return rejected, audit


def write_cell_outputs(output: Path, cells: Sequence[dict], rejected: set) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    cell_rows = []
    pair_rows = []
    excluded_rows = []
    strict_pair_rows = []

    for cell_index, cell_state in enumerate(cells):
        cell = cell_state["cell"]
        dataset, generator = str(cell["dataset"]), str(cell["generator"])
        full_sources = [source for source in cell_state["sources"] if source["structural_valid"]]
        strict_sources = [
            source for source in cell_state["sources"]
            if source["strict32_candidate"] and (cell_index, source["source_id"]) not in rejected
        ]

        def compatible_payload(sources: Sequence[dict], strict: bool) -> dict:
            if strict:
                human = [source["char_human"] for source in sources]
                machines = {
                    method: [source["char_machine"][method] for source in sources]
                    for method in METHODS
                }
            else:
                human = [source["human_continuation"] for source in sources]
                machines = {
                    method: [source["machine_continuations"][method] for source in sources]
                    for method in METHODS
                }
            return {"original": human, **machines}

        full_path = output / "cleaned_full" / dataset / generator / "data.json"
        strict_path = output / "cleaned_strict_char_matched" / dataset / generator / "data.json"
        dump_json(full_path, compatible_payload(full_sources, strict=False))
        dump_json(strict_path, compatible_payload(strict_sources, strict=True))
        dump_json(full_path.with_name("manifest.json"), {
            "schema_version": "clean41-prompt-free-full-v1",
            "dataset": dataset, "generator": generator,
            "source_ids": [source["source_id"] for source in full_sources],
            "source_count": len(full_sources), "methods": list(METHODS),
            "input_path": str(Path(str(cell["dataset_file"])).resolve()),
            "input_sha256": str(cell["sha256"]),
        })
        dump_json(strict_path.with_name("manifest.json"), {
            "schema_version": "clean41-prompt-free-char-matched-v1",
            "dataset": dataset, "generator": generator,
            "source_ids": [source["source_id"] for source in strict_sources],
            "source_count": len(strict_sources), "methods": list(METHODS),
            "min_lexical_units_before_matching": 32,
            "max_characters": 640,
            "character_budgets": [source["char_budget"] for source in strict_sources],
            "input_path": str(Path(str(cell["dataset_file"])).resolve()),
            "input_sha256": str(cell["sha256"]),
        })

        if dataset in UNRECOVERED_DATASETS:
            machine_only = {
                "prompt": [source["prompt"] for source in cell_state["sources"]],
                **{
                    method: [source["machine_continuations"][method] for source in cell_state["sources"]]
                    for method in METHODS
                },
            }
            path = output / "machine_only_unpaired" / dataset / generator / "data.json"
            dump_json(path, machine_only)
            dump_json(path.with_name("manifest.json"), {
                "schema_version": "clean41-machine-continuation-only-unpaired-v1",
                "dataset": dataset, "generator": generator,
                "source_ids": [source["source_id"] for source in cell_state["sources"]],
                "deployable_binary_benchmark": False,
                "reason": "independent human continuation is unavailable",
            })

        for source in cell_state["sources"]:
            strict_valid = source in strict_sources
            strict_reason = list(source["reasons"])
            if source["strict32_candidate"] and (cell_index, source["source_id"]) in rejected:
                strict_reason.append("GLOBAL_CROSS_LABEL_TEXT_HASH_COLLISION")
            if source["structural_valid"] and source["lexical_min"] < 32:
                strict_reason.append("MIN_LEXICAL_UNITS_BELOW_32")
            for method in METHODS:
                machine = source["machine_continuations"][method]
                row = {
                    "dataset": dataset, "generator": generator,
                    "source_id": source["source_id"], "source_index": source["source_index"],
                    "method": method, "recovery": source["recovery"],
                    "split_mode": source["split_modes"][method],
                    "raw_starts_with_complete_original": source["raw_starts_with_original"][method],
                    "structural_valid_source": source["structural_valid"],
                    "strict_char_matched_source": strict_valid,
                    "human_chars": len(source["human_continuation"]),
                    "machine_chars": len(machine),
                    "human_lexical_units": lexical_count(source["human_continuation"]),
                    "machine_lexical_units": lexical_count(machine),
                    "common_min_lexical_units": source["lexical_min"],
                    "strict_character_budget": source["char_budget"] if strict_valid else 0,
                    "human_sha256": sha256_text(source["human_continuation"]),
                    "machine_sha256": sha256_text(machine),
                    "exclusion_reasons": ";".join(sorted(set(strict_reason))),
                }
                pair_rows.append(row)
                if not strict_valid:
                    excluded_rows.append(row)
                else:
                    strict_pair_rows.append({
                        "dataset": dataset, "generator": generator,
                        "source_id": source["source_id"], "source_index": source["source_index"],
                        "method": method, "label_human": 0, "label_machine": 1,
                        "prompt": source["prompt"],
                        "human_continuation_full": source["human_continuation"],
                        "machine_continuation_full": machine,
                        "human_text": source["char_human"],
                        "machine_text": source["char_machine"][method],
                        "character_budget": source["char_budget"],
                    })

        strict_length_ok = all(
            len(source["char_human"]) == source["char_budget"]
            and all(len(source["char_machine"][method]) == source["char_budget"] for method in METHODS)
            for source in strict_sources
        )
        cell_rows.append({
            "dataset": dataset, "generator": generator,
            "raw_sources": len(cell_state["sources"]),
            "prompt_free_structural_sources": len(full_sources),
            "strict_char_matched_sources": len(strict_sources),
            "structural_retention": len(full_sources) / max(1, len(cell_state["sources"])),
            "strict_retention": len(strict_sources) / max(1, len(cell_state["sources"])),
            "strict_character_length_invariant": strict_length_ok,
            "strict_evaluable": len(strict_sources) > 0,
            "input_sha256": str(cell["sha256"]),
            "cleaned_full_path": str(full_path.resolve()),
            "cleaned_strict_path": str(strict_path.resolve()),
        })
    return cell_rows, pair_rows, excluded_rows, strict_pair_rows


def write_dataset_summary(
    output: Path,
    recovery: pd.DataFrame,
    cells: pd.DataFrame,
    raw_length: pd.DataFrame,
) -> None:
    """Record source recovery and cleaning statistics by dataset."""
    dataset_rows = []
    for dataset, group in cells.groupby("dataset", sort=True):
        rec = recovery[recovery.dataset == dataset]
        dataset_rows.append({
            "dataset": dataset,
            "cells": len(group),
            "unique_sources": int(rec.source_id.nunique()),
            "human_recovery_rate": float((rec.human_continuation_chars > 0).mean()),
            "structural_retention": float(group.prompt_free_structural_sources.sum() / group.raw_sources.sum()),
            "strict32_retention": float(group.strict_char_matched_sources.sum() / group.raw_sources.sum()),
            "raw_character_length_auc": float(raw_length[raw_length.dataset == dataset].source_fold_macro_character_length_auc.mean()),
            "strict_character_length_auc": 0.5 if int(group.strict_char_matched_sources.sum()) > 0 else float("nan"),
        })
    dataset = pd.DataFrame(dataset_rows)
    dataset.to_csv(output / "dataset_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest", type=Path,
        default=Path("scratch/clean41_llama_proxy_necessity/clean41_source_manifest.json"),
    )
    parser.add_argument(
        "--mage-root", type=Path,
        default=Path("external_data/MAGE"),
    )
    parser.add_argument("--legacy-root", type=Path, default=Path("exp_main/data"))
    parser.add_argument(
        "--prefix-audit-dir", type=Path,
        default=Path("scratch/compact_sarm_generalization/prefix_audit"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("scratch/clean41_strict_text_separation"),
    )
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    manifest = load_json(args.source_manifest)
    if int(manifest.get("cell_count", -1)) != 41:
        raise RuntimeError("source manifest is not the frozen clean-41 definition")
    if tuple(manifest.get("methods", [])) != METHODS:
        raise RuntimeError("method inventory differs from the frozen cleaning contract")
    source_hash_mismatches = []
    for cell in manifest["cells"]:
        path = Path(str(cell["dataset_file"]))
        actual = sha256_file(path)
        if actual != str(cell["sha256"]):
            source_hash_mismatches.append({
                "cell": str(cell["cell"]), "expected": str(cell["sha256"]), "actual": actual,
            })
    if source_hash_mismatches:
        raise RuntimeError(f"raw source hash mismatch: {source_hash_mismatches[:3]}")

    mage, mage_inputs = load_mage_humans(args.mage_root)
    recovered, recovery_rows, legacy_inputs = recover_sources(manifest, mage, args.legacy_root)
    cells, raw_length_rows = build_cell_records(manifest, recovered)
    rejected, collision_rows = cross_label_collision_sources(cells)
    cell_rows, pair_rows, excluded_rows, strict_pair_rows = write_cell_outputs(output, cells, rejected)

    recovery_frame = pd.DataFrame(recovery_rows).sort_values(["dataset", "source_index"])
    cell_frame = pd.DataFrame(cell_rows).sort_values(["dataset", "generator"])
    pair_frame = pd.DataFrame(pair_rows).sort_values(["dataset", "generator", "source_index", "method"])
    excluded_frame = pd.DataFrame(excluded_rows).sort_values(["dataset", "generator", "source_index", "method"])
    strict_frame = pd.DataFrame(strict_pair_rows).sort_values(["dataset", "generator", "source_index", "method"])
    raw_length_frame = pd.DataFrame(raw_length_rows).sort_values(["dataset", "generator", "method"])
    collision_frame = pd.DataFrame(collision_rows)

    recovery_frame.to_parquet(output / "source_recovery.parquet", index=False, compression="zstd")
    cell_frame.to_csv(output / "cell_manifest.csv", index=False)
    pair_frame.to_parquet(output / "pair_audit.parquet", index=False, compression="zstd")
    excluded_frame.to_csv(output / "excluded_pairs.csv", index=False)
    strict_frame.to_parquet(output / "strict_pairs.parquet", index=False, compression="zstd")
    raw_length_frame.to_csv(output / "raw_character_length_auc.csv", index=False)
    collision_frame.to_csv(output / "cross_label_hash_collisions.csv", index=False)

    written_json_files_validated = 0
    written_json_schema_failures = []
    for row in cell_frame.itertuples():
        for path_value, count_value in (
            (row.cleaned_full_path, row.prompt_free_structural_sources),
            (row.cleaned_strict_path, row.strict_char_matched_sources),
        ):
            path = Path(path_value)
            payload = load_json(path)
            sizes = {key: len(payload.get(key, [])) for key in ("original", *METHODS)}
            if set(sizes.values()) != {int(count_value)}:
                written_json_schema_failures.append({"path": str(path), "sizes": sizes})
            written_json_files_validated += 1
    if written_json_schema_failures:
        raise RuntimeError(f"written JSON schema validation failed: {written_json_schema_failures[:3]}")

    validation = {
        "status": "PARTIAL_PASS",
        "raw_cell_count": len(cells),
        "raw_dataset_count": len({row["dataset"] for row in manifest["cells"]}),
        "raw_cell_sources": int(sum(len(cell["sources"]) for cell in cells)),
        "raw_pair_count": len(pair_frame),
        "strict_pair_count": len(strict_frame),
        "strict_cell_count": int(cell_frame.strict_evaluable.sum()),
        "exact_source_identity_across_generators": True,
        "raw_source_hash_mismatches": len(source_hash_mismatches),
        "written_json_files_validated": written_json_files_validated,
        "written_json_schema_failures": len(written_json_schema_failures),
        "non_itse_prefix_split_failures": int((pair_frame.split_mode == "exact_prompt_prefix_missing").sum()),
        "strict_character_length_mismatches": int(sum(
            len(row.human_text) != row.character_budget or len(row.machine_text) != row.character_budget
            for row in strict_frame.itertuples()
        )),
        "strict_exact_cross_label_duplicates": int(sum(
            row.human_text == row.machine_text for row in strict_frame.itertuples()
        )),
        "strict_source_ids_are_dataset_scoped": bool(all(
            str(row.source_id).startswith(f"{row.dataset}:") for row in strict_frame.itertuples()
        )),
        "raw_character_length_auc_mean": float(raw_length_frame.source_fold_macro_character_length_auc.mean()),
        "strict_character_length_auc": 0.5,
        "new_llm_forward_count": 0,
    }
    if validation["non_itse_prefix_split_failures"] != 0:
        raise RuntimeError("exact prefix splitting failed for a non-ITSEdit row")
    if validation["strict_character_length_mismatches"] != 0:
        raise RuntimeError("strict character matching invariant failed")
    if validation["strict_exact_cross_label_duplicates"] != 0:
        raise RuntimeError("strict output retained an exact cross-label duplicate")
    dump_json(output / "validation.json", validation)

    write_dataset_summary(output, recovery_frame, cell_frame, raw_length_frame)

    script_path = Path(__file__).resolve()
    input_files = [
        {"path": str(args.source_manifest.resolve()), "bytes": args.source_manifest.stat().st_size, "sha256": sha256_file(args.source_manifest)},
        *mage_inputs, *legacy_inputs,
    ]
    input_files.extend({
        "path": str(Path(str(cell["dataset_file"])).resolve()),
        "bytes": int(cell["bytes"]), "sha256": str(cell["sha256"]),
    } for cell in manifest["cells"])
    for name in ("raw_prefix_audit.csv", "raw_length_baseline.csv", "score_duplicate_audit.csv"):
        path = args.prefix_audit_dir / name
        if path.is_file():
            input_files.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)})

    reproduce = """#!/usr/bin/env bash
set -euo pipefail
cd .
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=20260809
python3 scripts/clean_clean41_text_separation.py \\
  --source-manifest scratch/clean41_llama_proxy_necessity/clean41_source_manifest.json \\
  --output-dir scratch/clean41_strict_text_separation
"""
    reproduce_path = output / "REPRODUCE.sh"
    reproduce_path.write_text(reproduce, encoding="utf-8")
    reproduce_path.chmod(0o755)

    output_files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "data_manifest.json":
            output_files.append({
                "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path),
            })
    data_manifest = {
        "schema_version": "clean41-strict-text-separation-manifest-v1",
        "cleaning_contract": {
            "exact_prompt_removal_only": True,
            "itse_is_continuation_only_by_implementation": True,
            "minimum_lexical_units_all_nine_streams": 32,
            "maximum_common_character_budget": 640,
            "raw_inputs_modified": False,
            "labels_used_for_cleaning_or_selection": False,
        },
        "git": git_state(Path.cwd()),
        "runtime": {
            "python": sys.version, "platform": platform.platform(),
            "pandas": pd.__version__, "numpy": np.__version__,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", "NOT_SET"),
        },
        "code": {"path": str(script_path), "sha256": sha256_file(script_path)},
        "inputs": input_files,
        "outputs": output_files,
        "validation": validation,
        "exact_reproduction_command": str(reproduce_path.resolve()),
    }
    dump_json(output / "data_manifest.json", data_manifest)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
