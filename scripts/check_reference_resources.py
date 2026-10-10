#!/usr/bin/env python3
"""Pre-freeze GPU resource check using one synthetic 2048-token input.

No development/test corpus or detector output is opened. This is a resource
and numerical-parity check, not an additional scientific candidate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import torch

import score_reference_moments as numerical
from score_reference_queue import immutable_json, load_runtime, proxy_worker


def gpu_resources(gpus):
    result = []
    for logical, physical in enumerate(gpus):
        free, total = torch.cuda.mem_get_info(logical)
        result.append({
            "logical_gpu": logical, "physical_gpu": physical,
            "name": torch.cuda.get_device_name(logical),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(logical),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(logical),
            "allocated_bytes": torch.cuda.memory_allocated(logical),
            "reserved_bytes": torch.cuda.memory_reserved(logical),
            "free_bytes": free, "total_bytes": total,
        })
    return result


def synthetic_inputs(runtime):
    """Choose a vocabulary-valid non-special token from a fixed synthetic literal."""
    tokenizers = runtime.tokenizers
    candidate_ids = [tokenizer(" the", add_special_tokens=False,
                               return_token_type_ids=False)["input_ids"]
                     for tokenizer in tokenizers]
    if candidate_ids[0] != candidate_ids[1]:
        raise ValueError("reference/scoring tokenization differs for the synthetic literal")
    special = set(tokenizers[0].all_special_ids) | set(tokenizers[1].all_special_ids)
    common = min(runtime.qm.config.vocab_size, runtime.pm.config.vocab_size)
    candidates = [token for token in candidate_ids[0]
                  if token not in special and 0 <= token < common]
    if not candidates:
        raise ValueError("synthetic literal did not yield a shared valid ordinary token")
    token_id = candidates[0]
    return {
        "input_ids": torch.full((1, 2048), token_id, dtype=torch.long),
        "attention_mask": torch.ones((1, 2048), dtype=torch.long),
    }, token_id


def run(config_path, proxy, gpu_argument):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    config_hash = numerical.sha(config_path)
    # Only roster and protocol metadata are checked; validate_access opens no corpus.
    numerical.validate_access(config, "development", config_hash)
    root = Path(config["output_root"])
    if (root / "FROZEN_SELECTION.json").exists() or (root / "TEST_ACCESS_AUTHORIZATION.json").exists():
        raise PermissionError("resource smoke must run before freeze and test authorization")
    if (config["max_tokens"], config["batch_size"], config["max_batch_tokens"]) != (2048, 8, 2048):
        raise ValueError("resource smoke must use the unchanged 2048/8/2048 protocol")
    worker = proxy_worker(config, proxy, gpu_argument)
    numerical.check_model_paths(worker)
    directory = root / "resource_smoke"
    directory.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = directory / f"{timestamp}_{proxy}_2048_{run_id}.json"
    report = {
        "schema_version": "reference-shaping-resource-smoke-v1", "status": "RUNNING",
        "proxy": proxy, "physical_gpus": worker["gpus"], "run_id": run_id,
        "config_sha256": config_hash, "frozen_protocol_sha256": config["frozen_protocol_sha256"],
        "script_sha256": numerical.sha(__file__),
        "numerical_module_sha256": numerical.sha(numerical.__file__),
        "queue_driver_sha256": numerical.sha(Path(__file__).with_name("score_reference_queue.py")),
        "model_loader_sha256": numerical.sha(Path(__file__).with_name("model.py")),
        "reference_model": worker["q"], "scoring_model": worker["p"],
        "input_kind": "one ordinary token from fixed literal ' the', repeated 2048 times",
        "input_shape": [1, 2048], "scored_tokens": 2047,
        "source_corpus_files_opened": 0, "test_scores_read": 0,
        "scientific_candidates_evaluated": 0,
        "output_path": str(destination), "finite": False, "parity_pass": False,
    }
    started = time.perf_counter()
    runtime = None
    try:
        runtime = load_runtime(worker)
        report["model_loading_seconds"] = time.perf_counter() - started
        encoded, token_id = synthetic_inputs(runtime)
        report["synthetic_token_id"] = token_id
        for logical in range(len(worker["gpus"])):
            torch.cuda.synchronize(logical)
            torch.cuda.reset_peak_memory_stats(logical)
        report["resources_before_forward"] = gpu_resources(worker["gpus"])
        tick = time.perf_counter()
        with torch.inference_mode():
            lr = runtime.qm(**{key: value.to(runtime.qdevice) for key, value in encoded.items()},
                            use_cache=False).logits
            ls = lr if runtime.same else runtime.pm(
                **{key: value.to(runtime.pdevice) for key, value in encoded.items()},
                use_cache=False).logits
        torch.cuda.synchronize(runtime.qdevice)
        if not runtime.same:
            torch.cuda.synchronize(runtime.pdevice)
        report["forward_seconds"] = time.perf_counter() - tick
        labels = encoded["input_ids"][0, 1:].to(runtime.qdevice)
        ref, score = lr[0, :2047], ls[0, :2047]
        tick = time.perf_counter()
        computed = numerical.features(ref, score, labels)
        report["features_seconds"] = time.perf_counter() - tick
        json.dumps(computed, allow_nan=False)
        report["finite"] = True
        report["synthetic_features"] = computed
        from run_review_supplement_scores import features as old_features, tag
        tick = time.perf_counter()
        old = old_features(ref, score, labels, [1.0, 1.2], [1.0, 0.8])
        report["parity_seconds"] = time.perf_counter() - tick
        errors = {
            f"{numerical.endpoint_name(*key)}_{weight}": abs(
                old[tag(*key, weight)] - computed["stats"][weight]["endpoints"][
                    numerical.endpoint_name(*key)]["R"])
            for key in numerical.ENDPOINTS for weight in ("uniform", "entropy")
        }
        report["parity_errors"] = errors
        report["max_endpoint_parity_error"] = max(errors.values())
        report["parity_pass"] = max(errors.values()) < 1e-4
        if not report["parity_pass"]:
            raise ValueError(f"endpoint parity failure: {errors}")
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["exception_type"] = type(error).__name__
        report["exception_message"] = str(error)
        report["failure_kind"] = (
            "out_of_memory" if "out of memory" in str(error).lower() else "runtime_or_parity_error")
    finally:
        if torch.cuda.is_initialized():
            try:
                report["resources_after_attempt"] = gpu_resources(worker["gpus"])
            except Exception as error:
                report["resource_collection_error"] = f"{type(error).__name__}: {error}"
        report["elapsed_seconds"] = time.perf_counter() - started
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        immutable_json(destination, report)
    print(json.dumps({key: report.get(key) for key in (
        "status", "proxy", "physical_gpus", "finite", "parity_pass",
        "max_endpoint_parity_error", "failure_kind", "exception_message", "output_path")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--proxy", choices=("OPT", "Falcon"), required=True)
    parser.add_argument("--gpus", required=True)
    args = parser.parse_args()
    return 0 if run(args.config, args.proxy, args.gpus)["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
