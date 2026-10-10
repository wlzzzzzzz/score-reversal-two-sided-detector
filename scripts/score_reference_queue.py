#!/usr/bin/env python3
"""Queue cells across GPU workers without changing the frozen numerical module.

The original precompute main is a monolithic CLI. This explicit driver reuses
its features, split authorization, model validation, and model loader; its
tokenization order, batch construction, precision, and parity check are kept
identical. Legacy workers must stop before queue workers take over: only queue
workers participate in the advisory per-cell locks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time
import uuid

import torch

import score_reference_moments as numerical


RUNNER_CODE_KEY = "scripts/score_reference_queue.py"
PROVENANCE_NOTE = (
    "script_sha256 identifies the imported numerical precompute module; "
    "runner_script_sha256 identifies the queue execution driver. Both are "
    "bound by the frozen selection. physical_gpus is runtime scheduling only."
)


def queue_access(config, split, config_hash):
    """Keep the original access record unchanged, adding a separate runner gate."""
    access = numerical.validate_access(config, split, config_hash)
    if split == "test":
        frozen = json.loads((Path(config["output_root"]) / "FROZEN_SELECTION.json").read_text())
        hashes = frozen.get("code_sha256", {})
        if not isinstance(hashes, dict) or hashes.get(RUNNER_CODE_KEY) != numerical.sha(__file__):
            raise PermissionError("queue runner differs from frozen code")
    return access


def immutable_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    try:
        with temporary.open("x") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def cell_lock(groups, stem):
    """The descriptor owns the lock for the entire claim/compute/commit cycle."""
    directory = Path(groups) / ".queue_locks"
    directory.mkdir(exist_ok=True)
    with (directory / (stem + ".lock")).open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def proxy_worker(config, proxy, gpu_argument):
    matches = [worker for worker in config["workers"] if worker["proxy"] == proxy]
    if not matches or len({(w["q"], w["p"]) for w in matches}) != 1:
        raise ValueError("the requested proxy must identify one unambiguous model pair")
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", gpu_argument):
        raise ValueError("--gpus must be comma-separated nonnegative physical GPU IDs")
    gpus = [int(value) for value in gpu_argument.split(",")]
    required = 1 if proxy == "OPT" else 2
    if len(gpus) != required or len(set(gpus)) != required:
        raise ValueError("OPT requires one GPU; Falcon requires two distinct GPUs")
    # Deliberately do not change the config, pair, proxy, or any numerical setting.
    return {**matches[0], "gpus": gpus, "name": proxy + "_queue"}


@dataclass
class ModelRuntime:
    tokenizers: list
    qm: object
    pm: object
    qdevice: str
    pdevice: str
    same: bool


def load_runtime(worker):
    visibility = ",".join(str(gpu) for gpu in worker["gpus"])
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before setting the queue GPU assignment")
    os.environ["CUDA_VISIBLE_DEVICES"] = visibility
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from model import load_model, load_tokenizer

    same = worker["q"] == worker["p"]
    tokenizers = [load_tokenizer(path, str(Path(path).parent))
                  for path in (worker["q"], worker["p"])]
    if tokenizers[0].get_vocab() != tokenizers[1].get_vocab():
        raise ValueError("reference/scoring tokenizer vocabularies differ")
    qdevice, pdevice = "cuda:0", "cuda:0" if same else "cuda:1"
    qm = load_model(worker["q"], qdevice, str(Path(worker["q"]).parent)).eval()
    pm = qm if same else load_model(worker["p"], pdevice, str(Path(worker["p"]).parent)).eval()
    for model in (qm, pm):
        model.config.use_cache = False
    return ModelRuntime(tokenizers, qm, pm, qdevice, pdevice, same)


def records_for_cell(config, split, proxy, cell, runtime):
    source = Path(cell["data_path"])
    if numerical.sha(source) != cell["sha256"]:
        raise ValueError(f"input hash mismatch: {source}")
    data = json.loads(source.read_text())
    count, records = int(cell["source_count"]), []
    for method in ["human"] + config["methods"]:
        values = data["original" if method == "human" else method]
        if len(values) != count:
            raise ValueError(f"source count mismatch: {cell['cell']} / {method}")
        for index, text_value in enumerate(values):
            ids = runtime.tokenizers[0](text_value, add_special_tokens=True,
                                        return_token_type_ids=False)["input_ids"]
            other = runtime.tokenizers[1](text_value, add_special_tokens=True,
                                          return_token_type_ids=False)["input_ids"]
            if ids != other:
                raise ValueError("reference/scoring tokenization differs")
            length = len(ids)
            ids = ids[:config["max_tokens"]]
            if len(ids) < 2:
                raise ValueError("text must contain at least two input tokens")
            row = {
                "row_id": f"{cell['dataset']}:{cell['generator']}:{index}:{method}",
                "source_id": f"{cell['dataset']}:{index}", "source_index": index,
                "cell": cell["cell"], "dataset": cell["dataset"], "generator": cell["generator"],
                "proxy": proxy, "watermark_method": method, "label": int(method != "human"),
                "split": split, "token_length": len(ids)-1,
                "original_input_token_length": length, "context_truncated": length > config["max_tokens"],
                "text_sha256": hashlib.sha256(text_value.encode()).hexdigest(),
            }
            records.append((ids, row))
    records.sort(key=lambda item: len(item[0]))
    return records


def synchronize(device):
    # CPU is used only by synthetic driver tests; CLI runtime always uses CUDA.
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def write_cell_features(config, records, runtime, partial, report, announce):
    """The same padding, stable length ordering, and token budget as original main."""
    parity, written, offset = [], 0, 0
    with partial.open("x") as handle:
        while offset < len(records):
            stop = min(offset + config["batch_size"], len(records))
            while (stop > offset + 1
                   and max(len(item[0]) for item in records[offset:stop]) * (stop-offset)
                   > config["max_batch_tokens"]):
                stop -= 1
            batch = records[offset:stop]
            encoded = runtime.tokenizers[0].pad(
                [{"input_ids": ids, "attention_mask": [1]*len(ids)} for ids, _ in batch],
                return_tensors="pt")
            tick = time.perf_counter()
            with torch.inference_mode():
                lr = runtime.qm(**{key: value.to(runtime.qdevice) for key, value in encoded.items()},
                                use_cache=False).logits
                ls = lr if runtime.same else runtime.pm(
                    **{key: value.to(runtime.pdevice) for key, value in encoded.items()},
                    use_cache=False).logits
            synchronize(runtime.qdevice)
            if not runtime.same:
                synchronize(runtime.pdevice)
            report["forward_seconds"] += time.perf_counter() - tick
            tick = time.perf_counter()
            for position, (ids, row) in enumerate(batch):
                n = len(ids)-1
                labels = encoded["input_ids"][position, 1:len(ids)].to(runtime.qdevice)
                r, s = lr[position, :n], ls[position, :n]
                computed = numerical.features(r, s, labels)
                if written == 0:
                    from run_review_supplement_scores import features as old_features, tag
                    old = old_features(r, s, labels, [1.0, 1.2], [1.0, 0.8])
                    errors = {
                        f"{numerical.endpoint_name(*key)}_{weight}": abs(
                            old[tag(*key, weight)] - computed["stats"][weight]["endpoints"][
                                numerical.endpoint_name(*key)]["R"])
                        for key in numerical.ENDPOINTS for weight in ("uniform", "entropy")
                    }
                    maximum = max(errors.values())
                    if maximum >= 1e-4:
                        raise ValueError(f"endpoint parity failure: {errors}")
                    report["max_endpoint_parity_error"] = max(report["max_endpoint_parity_error"], maximum)
                    parity.append({"row_id": row["row_id"], "errors": errors})
                handle.write(json.dumps({**row, **computed}, separators=(",", ":"), allow_nan=False) + "\n")
                written += 1
            report["module_seconds"] += time.perf_counter() - tick
            del lr, ls
            offset = stop
            if written % 200 < config["batch_size"]:
                handle.flush()
                announce(written)
        handle.flush()
        os.fsync(handle.fileno())
    return written, parity


def completed_cell(output, manifest, expected, runner_hash, expected_rows):
    if not output.exists() and not manifest.exists():
        return None
    if not output.is_file() or not manifest.is_file():
        raise RuntimeError(f"incomplete published cell; preserve and inspect: {output}")
    old = json.loads(manifest.read_text())
    if any(old.get(key) != value for key, value in expected.items()):
        raise ValueError(f"immutable existing cell failed resume validation: {output}")
    if (old.get("sha256") != numerical.sha(output) or old.get("rows") != expected_rows
            or ("runner_script_sha256" in old and old["runner_script_sha256"] != runner_hash)):
        raise ValueError(f"immutable existing cell hash/rows/runner mismatch: {output}")
    return old




def run(config_path, split, proxy, gpu_argument):
    config = json.loads(Path(config_path).read_text())
    config_hash = numerical.sha(config_path)
    module_hash, runner_hash = numerical.sha(numerical.__file__), numerical.sha(__file__)
    access = queue_access(config, split, config_hash)
    worker = proxy_worker(config, proxy, gpu_argument)
    numerical.check_model_paths(worker)
    if (config["max_tokens"], config["batch_size"], config["max_batch_tokens"]) != (2048, 8, 2048):
        raise ValueError("parity protocol requires max_tokens=2048, batch_size=8, max_batch_tokens=2048")
    if len(set(config["methods"])) != len(config["methods"]) or "human" in config["methods"]:
        raise ValueError("methods must be unique and exclude human")
    cells = [cell for cell in config["data_cells"] if cell["split"] == split]
    root = Path(config["output_root"])
    groups = root / split / proxy
    groups.mkdir(parents=True, exist_ok=True)
    claims = groups / ".queue_claims"
    claims.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex
    identity = f"queue_{proxy}_gpus{'_'.join(map(str, worker['gpus']))}_{run_id}"
    progress = groups / (identity + "_progress.json")
    report = {
        "schema_version": numerical.SCHEMA_VERSION, "name": identity, "run_id": run_id,
        "proxy": proxy, "split": split, "smoke_rows": 0, "status": "RUNNING",
        "config_sha256": config_hash, "script_sha256": module_hash,
        "runner_script_sha256": runner_hash, "physical_gpus": worker["gpus"],
        "provenance_note": PROVENANCE_NOTE, "access": access,
        "rows": 0, "cells": 0, "resumed_cells": 0, "locked_cells_skipped": 0,
        "forward_seconds": 0.0, "module_seconds": 0.0, "max_endpoint_parity_error": 0.0,
    }
    started = time.perf_counter()
    numerical.atomic_json(progress, report)
    runtime = None
    for cell in cells:
        # Gate check also precedes reads of previously completed test scores.
        if queue_access(config, split, config_hash) != access:
            raise PermissionError("access artifacts changed during the run")
        stem = cell["dataset"] + "_" + cell["generator"]
        with cell_lock(groups, stem) as acquired:
            if not acquired:
                report["locked_cells_skipped"] += 1
                continue
            output = groups / (stem + ".jsonl")
            manifest = output.with_suffix(".manifest.json")
            expected = {
                "config_sha256": config_hash, "script_sha256": module_hash,
                "input_sha256": cell["sha256"], "access": access,
                "smoke_rows": 0, "proxy": proxy,
            }
            expected_rows = int(cell["source_count"]) * (len(config["methods"]) + 1)
            old = completed_cell(output, manifest, expected, runner_hash, expected_rows)
            if old is not None:
                report["resumed_cells"] += 1
                continue
            claim = {**expected, "runner_script_sha256": runner_hash,
                     "physical_gpus": worker["gpus"], "run_id": run_id,
                     "cell": cell["cell"], "pid": os.getpid(), "host": socket.gethostname(),
                     "claimed_at_unix": time.time(), "provenance_note": PROVENANCE_NOTE}
            immutable_json(claims / (stem + "." + run_id + ".json"), claim)
            if runtime is None:
                runtime = load_runtime(worker)
            # Recheck after loading models, immediately before opening raw text.
            if queue_access(config, split, config_hash) != access:
                raise PermissionError("access artifacts changed during model loading")
            records = records_for_cell(config, split, proxy, cell, runtime)
            partial = output.with_name(output.name + ".partial." + run_id)

            def announce(written):
                print(json.dumps({"name": identity, "split": split, "cell": cell["cell"],
                                  "cell_rows": written, "total_rows": report["rows"] + written,
                                  "elapsed": round(time.perf_counter() - started, 1)}), flush=True)

            written, parity = write_cell_features(config, records, runtime, partial, report, announce)
            if written != expected_rows:
                raise ValueError("computed cell row count disagrees with the frozen source count")
            if output.exists() or manifest.exists():
                raise RuntimeError(f"concurrent output collision; preserved attempt: {partial}")
            final_manifest = {
                "schema_version": numerical.SCHEMA_VERSION, **expected,
                "sha256": numerical.sha(partial), "rows": written,
                "runner_script_sha256": runner_hash, "physical_gpus": worker["gpus"],
                "provenance_note": PROVENANCE_NOTE, "parity_checks": parity, "run_id": run_id,
            }
            os.link(partial, output)
            immutable_json(manifest, final_manifest)
            partial.unlink()
            report["rows"] += written
            report["cells"] += 1
            report["elapsed_seconds"] = time.perf_counter() - started
            numerical.atomic_json(progress, report)
    report["status"] = "COMPLETE"
    report["completion_scope"] = "this queue scan; locked cells remain owned by other queue processes"
    report["elapsed_seconds"] = time.perf_counter() - started
    report["gpu_peak_allocated_bytes"] = (
        {str(i): torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())}
        if runtime is not None else {})
    numerical.atomic_json(progress, report)
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "test"), required=True)
    parser.add_argument("--proxy", choices=("OPT", "Falcon"), required=True)
    parser.add_argument("--gpus", required=True)
    args = parser.parse_args()
    run(args.config, args.split, args.proxy, args.gpus)


if __name__ == "__main__":
    main()
