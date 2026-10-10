#!/usr/bin/env python3
"""Sparse, resumable GPU regeneration worker for one 20-cell generator.

The preparation stage decides which machine records require regeneration.
This worker never writes ``original`` (Human) values.  Each generator writes
only its own cell directories, allowing five independent GPU workers.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from generation_utils import load_model as load_generation_model, seed_everything, stable_seed
from text_utils import sanitize_text as sanitize_machine_serialization, strict_text_quality_reasons

MARKLLM_ROOT = Path("third_party/MarkLLM")
if str(MARKLLM_ROOT) not in sys.path:
    sys.path.append(str(MARKLLM_ROOT))

from watermark.auto_watermark import AutoWatermark  # noqa: E402
from utils.transformers_config import TransformersConfig  # noqa: E402

from prepare_20cell_regeneration import (  # noqa: E402
    DATASETS,
    FAILED,
    METHODS,
    PENDING,
    atomic_json,
    write_csv as atomic_csv,
    continuation_from_machine,
    forbidden_reasons,
    lexical_count,
)


MODEL_DIR = {
    "GLM-4-9B-0414": "GLM-4-9B-0414",
    "falcon": "falcon-7b",
    "llama-2-7b": "llama-2-7b",
    "opt-2.7B": "opt-2.7B",
    "opt-6.7B": "opt-6.7B",
}


def exp_hash_key_for_attempt(default_key: int, attempt_seed: int, attempt: int) -> int:
    """Keep the historical EXP key first, then use deterministic fallback keys.

    EXP is deterministic for a fixed prompt/key and can repeatedly emit EOS or
    an unusably short serialization.  Global RNG retries therefore do nothing.
    A per-source watermark key is a valid EXP deployment condition; only failed
    records use this auditable fallback, while attempt 1 keeps the historical
    key exactly.
    """
    if attempt == 1:
        return int(default_key)
    return int(attempt_seed) or 1


def parse_exp_temperature_schedule(raw: str) -> Tuple[float, ...]:
    """Parse an explicit, auditable EXP temperature retry schedule."""
    values = tuple(float(value) for value in str(raw).split(",") if value.strip())
    if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("EXP temperature schedule must contain positive finite values")
    return values


def exp_temperature_for_attempt(schedule: Sequence[float], attempt: int) -> float:
    """Return the registered temperature, repeating the last value if needed."""
    if attempt < 1 or not schedule:
        raise ValueError("invalid EXP attempt or empty temperature schedule")
    return float(schedule[min(attempt - 1, len(schedule) - 1)])


def request_shard(request: Mapping[str, object], shard_count: int) -> int:
    material = "\x1f".join(
        str(request[key]) for key in ("dataset", "generator", "source_id", "method")
    )
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big") % shard_count


def store_machine_result(
    data_path: Path,
    method: str,
    slot: int,
    value: str,
    expected_human_hash: str,
) -> None:
    """Lock and merge one Machine result without losing concurrent shards."""
    lock_path = data_path.with_suffix(data_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            latest = json.loads(data_path.read_text(encoding="utf-8"))
            latest_human_hash = hashlib.sha256(
                json.dumps(latest["original"], ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            if latest_human_hash != expected_human_hash:
                raise AssertionError("worker attempted to mutate Human text")
            latest[method][slot] = value
            atomic_json(data_path, latest)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def load_model(model_path: Path, max_new_tokens: int, native_falcon: bool = False):
    model, tokenizer = load_generation_model(model_path, max_new_tokens, native_falcon)
    model.generation_config.max_new_tokens = max_new_tokens
    model.generation_config.do_sample = True
    model.generation_config.temperature = 1.0
    model.generation_config.use_cache = model.config.use_cache
    if tokenizer.pad_token_id is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def generate_unwatermarked(
    model, tokenizer, prompt: str, max_new_tokens: int, min_new_tokens: int = 0,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max(0, min(int(min_new_tokens), int(max_new_tokens))),
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    continuation = tokenizer.decode(output[0, prompt_tokens:], skip_special_tokens=True)
    return prompt + continuation


def serialized_continuation(raw: str, prompt: str, method: str) -> str:
    """Match the prompt-removal contract used by the clean80 finalizer.

    Unlike ``continuation_from_machine``, this deliberately does not collapse
    internal whitespace.  The finalizer evaluates the serialized continuation,
    so resume and generation quality checks must use the same denominator for
    language-ratio validation.
    """
    text = str(raw)
    if method == "ITSEdit":
        return text.strip()
    if text.startswith(prompt):
        return text[len(prompt):].strip()
    return ""


def validate_generated(raw: str, prompt: str, method: str, min_lexical_units: int) -> Tuple[bool, str, List[str]]:
    continuation, split_status = continuation_from_machine(raw, prompt, method)
    reasons = []
    if split_status in {"EMPTY_OR_SENTINEL", "EXACT_PROMPT_PREFIX_MISSING"}:
        reasons.append(split_status)
    reasons.extend(forbidden_reasons(continuation))
    if continuation and lexical_count(continuation) < min_lexical_units:
        reasons.append("CONTINUATION_TOO_SHORT")
    return not reasons, continuation, sorted(set(reasons))


def build_watermarkers(
    methods: Sequence[str], model, tokenizer, max_new_tokens: int,
    min_new_tokens: int = 0,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        min_new_tokens=max(0, min(int(min_new_tokens), int(max_new_tokens))),
        do_sample=True,
        temperature=1.0,
    )
    watermarkers: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    for method in methods:
        if method == "unwatermarked":
            continue
        try:
            watermarkers[method] = AutoWatermark.load(
                algorithm_name=method,
                algorithm_config=str(MARKLLM_ROOT / "config" / f"{method}.json"),
                transformers_config=config,
            )
        except Exception as exc:  # keep the negative result in the worker audit
            errors[method] = f"{type(exc).__name__}: {exc}"
    return watermarkers, errors


def read_requests(
    root: Path,
    generator: str,
    selected_methods: Sequence[str],
    selected_datasets: Sequence[str],
) -> List[dict]:
    allowed = set(selected_methods)
    requests = []
    for dataset in selected_datasets:
        path = root / "prepared_raw" / dataset / generator / "generation_requests.json"
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row["method"] in allowed:
                requests.append(row)
    return sorted(requests, key=lambda row: (row["dataset"], int(row["slot_index"]), row["method"]))


def request_is_complete(
    root: Path,
    generator: str,
    request: Mapping[str, object],
    min_lexical_units: int,
    strict_text_quality: bool = False,
) -> bool:
    data_path = root / "prepared_raw" / str(request["dataset"]) / generator / "data.json"
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    value = str(payload[str(request["method"])][int(request["slot_index"])])
    if value in {PENDING, FAILED, ""}:
        return False
    valid, _, _ = validate_generated(
        value, str(request["prompt"]), str(request["method"]), min_lexical_units
    )
    if valid and strict_text_quality:
        language = "zh" if str(request["dataset"]) == "chinese" else "en"
        continuation = serialized_continuation(
            value, str(request["prompt"]), str(request["method"])
        )
        valid = not strict_text_quality_reasons(continuation, language=language)
    return valid


def run(args: argparse.Namespace) -> int:
    if args.generator not in MODEL_DIR:
        raise ValueError(f"unknown generator: {args.generator}")
    selected_methods = [method for method in METHODS if method in set(args.methods.split(","))]
    selected_datasets = list(dict.fromkeys(
        dataset.strip() for dataset in args.datasets.split(",") if dataset.strip()
    ))
    if not selected_methods:
        raise ValueError("no valid methods selected")
    if not selected_datasets:
        raise ValueError("no valid datasets selected")
    exp_temperature_schedule = parse_exp_temperature_schedule(
        args.exp_temperature_schedule
    )
    all_requests = read_requests(
        args.output_dir, args.generator, selected_methods, selected_datasets
    )
    selected_slots = {
        int(value) for value in args.slot_indices.split(",") if value.strip()
    }
    if selected_slots:
        all_requests = [
            request for request in all_requests
            if int(request["slot_index"]) in selected_slots
        ]
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    all_requests = [
        request for request in all_requests
        if request_shard(request, args.shard_count) == args.shard_index
    ]
    resume_skipped = 0
    if args.pilot or args.force:
        requests = all_requests
    else:
        requests = []
        for request in all_requests:
            if request_is_complete(
                args.output_dir,
                args.generator,
                request,
                args.min_lexical_units,
                strict_text_quality=args.strict_text_quality,
            ):
                resume_skipped += 1
            else:
                requests.append(request)
    if args.max_records > 0:
        requests = requests[: args.max_records]
    model_path = args.model_root / MODEL_DIR[args.generator]
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    audit_dir = args.output_dir / ("pilot" if args.pilot else "generation_audit") / args.generator
    if args.audit_tag:
        audit_dir = audit_dir / args.audit_tag
    audit_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "schema_version": "clean20-sparse-worker-v1",
        "generator": args.generator,
        "model_path": str(model_path.resolve()),
        "methods": selected_methods,
        "datasets": selected_datasets,
        "request_count": len(requests),
        "resume_skipped_count": resume_skipped,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "pilot": args.pilot,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "min_lexical_units": args.min_lexical_units,
        "max_attempts": args.max_attempts,
        "base_seed": args.seed,
        "native_falcon": args.native_falcon,
        "slot_indices": sorted(selected_slots),
        "strict_text_quality": args.strict_text_quality,
        "exp_temperature_schedule": list(exp_temperature_schedule),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_json(audit_dir / "run_config.json", run_config)
    if not requests:
        atomic_json(audit_dir / "summary.json", {**run_config, "status": "NOTHING_TO_DO"})
        return 0

    model, tokenizer = load_model(
        model_path, args.max_new_tokens, native_falcon=args.native_falcon,
    )
    needed_methods = sorted({str(row["method"]) for row in requests})
    watermarkers, initialization_errors = build_watermarkers(
        needed_methods, model, tokenizer, args.max_new_tokens, args.min_new_tokens
    )
    for method, watermarker in watermarkers.items():
        if method == "EXP":
            watermarker._clean20_default_hash_key = int(watermarker.config.hash_key)
    atomic_json(audit_dir / "watermarker_initialization.json", {
        "loaded": sorted(watermarkers),
        "errors": initialization_errors,
    })

    results: List[dict] = []
    success = failed = 0
    start = time.time()
    for position, request in enumerate(requests, start=1):
        dataset = str(request["dataset"])
        method = str(request["method"])
        slot = int(request["slot_index"])
        prompt = str(request["prompt"])
        data_path = args.output_dir / "prepared_raw" / dataset / args.generator / "data.json"
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        before_human_hash = hashlib.sha256(
            json.dumps(payload["original"], ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        final_raw = FAILED
        final_continuation = ""
        attempt_rows = []
        error = ""
        for attempt in range(1, args.max_attempts + 1):
            attempt_seed = stable_seed(
                args.seed, dataset, args.generator, request["source_id"], method, attempt
            )
            seed_everything(attempt_seed)
            exp_hash_key = ""
            exp_temperature = ""
            try:
                if method == "unwatermarked":
                    candidate = generate_unwatermarked(
                        model, tokenizer, prompt, args.max_new_tokens, args.min_new_tokens
                    )
                elif method not in watermarkers:
                    raise RuntimeError(initialization_errors.get(method, "watermarker unavailable"))
                else:
                    if method == "EXP":
                        default_key = int(watermarkers[method]._clean20_default_hash_key)
                        exp_hash_key = exp_hash_key_for_attempt(
                            default_key, attempt_seed, attempt
                        )
                        watermarkers[method].config.hash_key = exp_hash_key
                        exp_temperature = exp_temperature_for_attempt(
                            exp_temperature_schedule, attempt
                        )
                        watermarkers[method].config.temperature = exp_temperature
                    candidate = str(watermarkers[method].generate_watermarked_text(prompt))
                serialized_candidate, sanitized_characters = sanitize_machine_serialization(candidate)
                valid, continuation, reasons = validate_generated(
                    serialized_candidate, prompt, method, args.min_lexical_units
                )
                if args.strict_text_quality:
                    strict_continuation = serialized_continuation(
                        serialized_candidate, prompt, method
                    )
                    reasons.extend(strict_text_quality_reasons(
                        strict_continuation,
                        language="zh" if dataset == "chinese" else "en",
                    ))
                    valid = not reasons
                attempt_rows.append({
                    "attempt": attempt,
                    "seed": attempt_seed,
                    "valid": valid,
                    "reasons": ";".join(reasons),
                    "raw_chars": len(candidate),
                    "sanitized_characters": sanitized_characters,
                    "exp_hash_key": exp_hash_key,
                    "exp_temperature": exp_temperature,
                    "continuation_lexical_units": lexical_count(continuation),
                })
                if valid:
                    final_raw = serialized_candidate
                    final_continuation = continuation
                    break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                attempt_rows.append({
                    "attempt": attempt,
                    "seed": attempt_seed,
                    "valid": False,
                    "reasons": error,
                    "raw_chars": 0,
                    "sanitized_characters": 0,
                    "exp_hash_key": exp_hash_key,
                    "exp_temperature": exp_temperature,
                    "continuation_lexical_units": 0,
                })
                torch.cuda.empty_cache()

        record_status = "SUCCESS" if final_raw != FAILED else "FAILED"
        if record_status == "SUCCESS":
            success += 1
            if not args.pilot:
                store_machine_result(
                    data_path, method, slot, final_raw, before_human_hash
                )
        else:
            failed += 1
        row = {
            "dataset": dataset,
            "generator": args.generator,
            "source_id": request["source_id"],
            "slot_index": slot,
            "method": method,
            "status": record_status,
            "attempt_count": len(attempt_rows),
            "final_continuation_lexical_units": lexical_count(final_continuation),
            "final_raw_sha256": hashlib.sha256(final_raw.encode("utf-8")).hexdigest(),
            "error": error,
            "attempts_json": json.dumps(attempt_rows, ensure_ascii=False, sort_keys=True),
        }
        results.append(row)
        atomic_csv(
            audit_dir / "records.csv",
            results,
            (
                "dataset", "generator", "source_id", "slot_index", "method", "status",
                "attempt_count", "final_continuation_lexical_units", "final_raw_sha256",
                "error", "attempts_json",
            ),
        )
        atomic_json(audit_dir / "progress.json", {
            "generator": args.generator,
            "processed": position,
            "total": len(requests),
            "success": success,
            "failed": failed,
            "current": {key: request[key] for key in ("dataset", "source_id", "slot_index", "method")},
            "elapsed_seconds": time.time() - start,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        print(
            f"[{position}/{len(requests)}] {dataset}/{args.generator}/{slot}/{method}: "
            f"{record_status}",
            flush=True,
        )

    summary = {
        **run_config,
        "status": "PASS" if failed == 0 else "PARTIAL_FAILURE",
        "success": success,
        "failed": failed,
        "elapsed_seconds": time.time() - start,
        "new_llm_forward_records": len(results),
        "resume_skipped_count": resume_skipped,
        "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_json(audit_dir / "summary.json", summary)
    return 0 if failed == 0 else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("scratch/clean20_complete_regeneration"))
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--generator", required=True, choices=sorted(MODEL_DIR))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--audit-tag", default="")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--min-new-tokens", type=int, default=0,
        help="Optional generation token floor; default zero preserves prior runs.",
    )
    parser.add_argument("--min-lexical-units", type=int, default=32)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--exp-temperature-schedule",
        default="0.7",
        help=(
            "Comma-separated EXP generation temperatures used by retry attempt; "
            "the last value is repeated. Default 0.7 preserves the historical setup."
        ),
    )
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--slot-indices", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict-text-quality", action="store_true")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--native-falcon", action="store_true",
        help="Use maintained Transformers Falcon code and checkpoint bfloat16 dtype.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
