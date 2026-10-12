#!/usr/bin/env python3
"""Resumable GPU worker for CGW, HeavyWater, and official SemStamp channels."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from transformers import (
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteria,
)

from generation_utils import load_model, seed_everything, stable_seed
from io_utils import atomic_json, sha256_text
from text_utils import lexical_count, normalize_text, sanitize_text, strict_text_quality_reasons

ROOT = Path(".")
BASE = ROOT / "scratch/clean20_complete_regeneration"
MODEL_DIR = {
    "GLM-4-9B-0414": "models/GLM-4-9B-0414",
    "falcon": "models/falcon-7b",
    "llama-2-7b": "models/llama-2-7b",
    "opt-2.7B": "models/opt-2.7B",
    "opt-6.7B": "models/opt-6.7B",
}
DATASETS = ("c4", "squad", "writing", "xsum")
METHODS = ("CGW", "HeavyWater", "SemStamp")


class CGWAdapter:
    """Use the upstream MarkMyWords Binary generator with its CGW semantics."""

    def __init__(self, model, tokenizer, secret_key: int):
        compat = ROOT / "scripts/markmywords_compat"
        benchmark = ROOT / "third_party/external_watermarks/MarkMyWords/src"
        apps = ROOT / "third_party/external_watermarks/MarkMyWords_apps/src"
        for path in (compat, benchmark, apps):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from watermark_benchmark.utils.bit_tokenizer import Binarization
        from watermark_benchmark.watermark.schemes.binary import BinaryGenerator
        from watermark_benchmark.watermark.templates.random import EmbeddedRandomness

        device = next(model.parameters()).device
        vocab_size = len(tokenizer)
        if vocab_size < 2:
            raise RuntimeError("tokenizer vocabulary is invalid")
        self.binarizer = Binarization(
            tokenizer, devices=[device], use_huffman_coding=False,
        )
        self.rng = EmbeddedRandomness(
            secret_key, [device], vocab_size, hash_len=1, min_hash=False,
        )
        self.engine = BinaryGenerator(
            self.rng, [], tokenizer, 1.0, self.binarizer, skip_prob=0.0,
        )

    def __call__(self, input_ids, scores):
        return self.engine.process(scores, input_ids, [0])

    def reset(self) -> None:
        self.engine.reset()


class NullAppendList:
    def append(self, value: object) -> None:
        return None

    def __len__(self) -> int:
        return 0


def build_heavywater(model, tokenizer):
    repo = ROOT / "third_party/external_watermarks/HeavyWater_SimplexWater"
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from watermark.heavy_tail_randscore import HeavyTailLogitsProcessor

    # Some checkpoints (notably OPT) pad the LM head beyond the tokenizer
    # vocabulary.  HeavyWater indexes its cost matrix with *logit* indices, so
    # it must use the LM-head width rather than len(tokenizer); otherwise a
    # padded token selected by top-p causes an out-of-bounds CUDA assertion.
    vocab_size = int(model.get_output_embeddings().weight.shape[0])
    vocab = list(range(vocab_size))
    processor = HeavyTailLogitsProcessor(
        bad_words_ids=None,
        eos_token_id=tokenizer.eos_token_id,
        vocab=vocab,
        vocab_size=vocab_size,
        bl_proportion=0.5,
        bl_logit_bias=5.0,
        bl_type="soft",
        initial_seed=1234,
        dynamic_seed="markov_1",
        store_bl_ids=False,
        store_spike_ents=False,
        noop_blacklist=False,
        tilt=False,
        tilting_delta=0.0,
        top_p=0.999,
        context=1,
        hashing=None,
        hash_key=15485863,
        k=1024,
        dist="lognormal",
        temperature=1.0,
        sinkhorn_reg=0.05,
        sinkhorn_thresh=1e-5,
    )
    processor.saved_distributions = NullAppendList()
    return processor


def heavywater_z(processor, prompt_ids: torch.Tensor, generated_ids: torch.Tensor) -> float:
    if generated_ids.numel() == 0:
        return float("nan")
    previous = int(prompt_ids[0, -1])
    total = 0.0
    generator = torch.Generator(device=generated_ids.device)
    for token in generated_ids:
        generator.manual_seed(int(processor.large_prime) * previous)
        side = int(torch.randint(
            1, processor.k + 1, (1,), generator=generator, device=generated_ids.device,
        ).item())
        total += float(processor.cost_matrix[int(token), side - 1])
        previous = int(token)
    return total / math.sqrt(int(generated_ids.numel()))


def cgw_self_detection(
    adapter: CGWAdapter, prompt_ids: torch.Tensor, generated_ids: torch.Tensor,
) -> dict:
    """Evaluate the upstream BinaryVerifier statistic with prompt context."""
    from scipy.stats import gamma
    import hash_cpp

    if generated_ids.numel() == 0:
        return {"self_detection_p": float("nan"), "self_detection_neg_log10_p": float("nan"), "detected_bits": 0}
    bit_rows = adapter.binarizer.to_bit(generated_ids).detach().cpu()
    previous_token = int(prompt_ids[0, -1])
    secret_key = adapter.rng.secret_key
    total = 0.0
    count = 0
    for token, bits in zip(generated_ids.detach().cpu(), bit_rows):
        seed = f"{previous_token}SEED{secret_key}"
        for bit_index, bit_tensor in enumerate(bits):
            bit = int(bit_tensor)
            if bit < 0:
                continue
            random_value = float(hash_cpp.index_hash([seed], bit_index)[0])
            likelihood = random_value if bit == 1 else 1.0 - random_value
            total += -math.log(max(likelihood, 1e-38))
            count += 1
        previous_token = int(token)
    p_value = float(gamma.sf(total, count)) if count else float("nan")
    return {
        "self_detection_p": p_value,
        "self_detection_neg_log10_p": -math.log10(max(p_value, 1e-300)),
        "detected_bits": count,
    }


def init_semstamp():
    repo = ROOT / "external_code/SemStamp"
    if not (repo / "sampling_lsh_utils.py").is_file():
        raise FileNotFoundError("SemStamp requires a local upstream checkout at external_code/SemStamp; see docs/REPRODUCE.md#optional-generation-and-baseline-dependencies")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import nltk
    nltk.data.path.insert(0, "models/stage3_watermarks/nltk_data")
    from sampling_lsh_utils import get_mask_from_seed, reject_close_generation
    from sampling_utils import MAX_TRIALS, SentenceEndCriteria, gen_sent
    from sbert_lsh_model import SBERTLSHModel

    lsh_model = SBERTLSHModel(
        lsh_model_path="models/stage3_watermarks/SemStamp-c4-sbert",
        device="cuda",
        batch_size=1,
        lsh_dim=3,
        sbert_type="base",
    )
    return {
        "get_mask_from_seed": get_mask_from_seed,
        "reject_close_generation": reject_close_generation,
        "SentenceEndCriteria": SentenceEndCriteria,
        "gen_sent": gen_sent,
        "MAX_TRIALS": int(MAX_TRIALS),
        "lsh_model": lsh_model,
    }


def generate_semstamp(
    components: Mapping[str, Any], model, tokenizer, prompt: str, max_new_tokens: int,
    min_new_tokens: int = 0,
) -> tuple[str, dict]:
    get_mask = components["get_mask_from_seed"]
    reject_close = components["reject_close_generation"]
    SentenceEndCriteria = components["SentenceEndCriteria"]
    gen_sent = components["gen_sent"]
    max_trials = components["MAX_TRIALS"]
    lsh_model = components["lsh_model"]
    lsh_dim, accept_rate, margin = 3, 0.25, 0.0
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        min_new_tokens=max(0, min(int(min_new_tokens), int(max_new_tokens))),
        do_sample=True,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    criteria = SentenceEndCriteria(tokenizer)

    class DelayedSentenceEndCriteria(StoppingCriteria):
        """Preserve SemStamp sentence stopping after an audited token floor."""

        def __init__(self, wrapped, token_floor: int):
            self.wrapped = wrapped
            self.token_floor = int(token_floor)
            self.start_length = 0

        def reset_start(self, length: int) -> None:
            self.start_length = int(length)

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            if int(input_ids.shape[1]) - self.start_length < self.token_floor:
                return False
            return bool(self.wrapped(input_ids, scores, **kwargs))

    delayed_criteria = (
        DelayedSentenceEndCriteria(criteria, min_new_tokens)
        if min_new_tokens > 0 else criteria
    )
    lsh_seed = int(lsh_model.get_hash([prompt])[0])
    accept_mask = get_mask(lsh_dim, accept_rate, lsh_seed)
    text_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    prompt_length = int(text_ids.shape[1])
    criteria.update(prompt)
    accepted_segments: list[str] = []
    trials_per_segment: list[int] = []
    signature_hits: list[bool] = []
    fallback_count = 0
    empty_stop = False
    while int(text_ids.shape[1]) - prompt_length < max_new_tokens - 1:
        current_trials = 0
        while True:
            current_trials += 1
            from transformers import StoppingCriteriaList
            if isinstance(delayed_criteria, DelayedSentenceEndCriteria):
                delayed_criteria.reset_start(int(text_ids.shape[1]))
            candidate, candidate_ids = gen_sent(
                model=model,
                tokenizer=tokenizer,
                text_ids=text_ids,
                gen_config=generation_config,
                stopping_criteria=StoppingCriteriaList([delayed_criteria]),
            )
            if not candidate:
                empty_stop = True
                break
            sufficiently_far, _ = reject_close(
                lsh_model, [candidate], margin=margin, cutoff=None,
            )
            candidate_hash = int(lsh_model.get_hash([candidate])[0])
            in_signature = bool(candidate_hash in accept_mask)
            if sufficiently_far and in_signature:
                break
            if current_trials >= max_trials:
                fallback_count += 1
                break
        if empty_stop:
            break
        accepted_segments.append(candidate)
        trials_per_segment.append(current_trials)
        signature_hits.append(in_signature)
        text_ids = candidate_ids
        criteria.update(prompt + "".join(accepted_segments))
        lsh_seed = candidate_hash
        accept_mask = get_mask(lsh_dim, accept_rate, lsh_seed)
    continuation = "".join(accepted_segments)
    trials = len(signature_hits)
    if trials:
        hits = sum(signature_hits)
        z_score = (hits - accept_rate * trials) / math.sqrt(
            trials * accept_rate * (1.0 - accept_rate)
        )
    else:
        z_score = float("nan")
    return continuation, {
        "segments": trials,
        "signature_hits": sum(signature_hits),
        "self_detection_z": z_score,
        "fallback_segments": fallback_count,
        "trials_per_segment": trials_per_segment,
        "empty_stop": empty_stop,
        "recovery_min_new_tokens": int(min_new_tokens),
    }


def generate_token_method(
    method: str, model, tokenizer, prompt: str, max_new_tokens: int, adapter,
    repetition_penalty: float = 1.0, no_repeat_ngram_size: int = 0,
    min_new_tokens: int = 64,
) -> tuple[str, dict]:
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = int(encoded["input_ids"].shape[1])
    if method == "CGW":
        adapter.reset()
    elif method == "HeavyWater":
        adapter.seed_increment = 0
        adapter.saved_distributions = NullAppendList()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max(0, min(int(min_new_tokens), int(max_new_tokens))),
            do_sample=True,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            logits_processor=LogitsProcessorList([adapter]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, prompt_tokens:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    diagnostics = {"generated_tokens": int(generated.numel())}
    if method == "CGW":
        diagnostics.update(cgw_self_detection(adapter, encoded["input_ids"], generated))
    elif method == "HeavyWater":
        diagnostics["self_detection_z"] = heavywater_z(
            adapter, encoded["input_ids"], generated,
        )
    return text, diagnostics


def valid_record(
    path: Path, min_lexical_units: int, *, dataset: Optional[str] = None,
    generator: Optional[str] = None, method: Optional[str] = None,
    source_id: Optional[str] = None, prompt: Optional[str] = None,
) -> bool:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(row, dict):
        return False
    valid = (
        row.get("status") == "SUCCESS"
        and lexical_count(str(row.get("continuation", ""))) >= min_lexical_units
    )
    expected = {"dataset": dataset, "generator": generator, "method": method, "source_id": source_id}
    valid = valid and all(value is None or row.get(key) == value for key, value in expected.items())
    if prompt is not None:
        valid = valid and row.get("prompt_sha256") == sha256_text(prompt)
    return valid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--generator", choices=tuple(MODEL_DIR), required=True)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("scratch/clean20_plus_cgw_heavywater_semstamp"),
    )
    parser.add_argument("--base-dir", type=Path, default=BASE)
    parser.add_argument(
        "--request-manifest", type=Path, default=None,
        help="Optional generalized JSON task list with dataset/generator/source_id/slot_index/prompt.",
    )
    parser.add_argument("--audit-tag", default="")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--token-min-new-tokens", type=int, default=64,
        help="Generation floor for CGW/HeavyWater; default 64 preserves prior runs.",
    )
    parser.add_argument("--min-lexical-units", type=int, default=32)
    parser.add_argument(
        "--semstamp-min-new-tokens", type=int, default=0,
        help="Audited early-EOS recovery constraint; zero preserves the official first pass.",
    )
    # Sentence-level semantic rejection sampling can hit EOS unusually early
    # for a small subset of prompts.  Keep deterministic failure traces, but
    # allow enough independent seeds to fill every registered source.
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260811)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--slot-indices", default="",
        help="Optional comma-separated slot indices for audited sparse reruns.",
    )
    parser.add_argument("--strict-text-quality", action="store_true")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument(
        "--native-falcon", action="store_true",
        help="Use maintained Transformers Falcon code and checkpoint bfloat16 dtype.",
    )
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    args.output_dir = args.output_dir.resolve()
    args.base_dir = args.base_dir.resolve()
    selected_slots = {
        int(value) for value in args.slot_indices.split(",") if value.strip()
    }
    tasks = []
    registry = None
    request_manifest_sha256 = None
    if args.request_manifest is not None:
        request_path = args.request_manifest.resolve()
        request_manifest_sha256 = hashlib.sha256(request_path.read_bytes()).hexdigest()
        request_rows = json.loads(request_path.read_text(encoding="utf-8"))
        for row in request_rows:
            if row["generator"] != args.generator or row["method"] != args.method:
                continue
            dataset, slot, source_id = str(row["dataset"]), int(row["slot_index"]), str(row["source_id"])
            if selected_slots and slot not in selected_slots:
                continue
            material = f"{dataset}\x1f{args.generator}\x1f{source_id}\x1f{args.method}"
            shard = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") % args.shard_count
            if shard == args.shard_index:
                tasks.append((dataset, slot, source_id, str(row["prompt"])))
        selected_datasets = sorted({row[0] for row in tasks})
    else:
        selected_datasets = [d for d in DATASETS if d in set(args.datasets.split(","))]
        if not selected_datasets:
            raise ValueError("no valid datasets selected")
        registry = json.loads((args.base_dir / "source_registry.json").read_text(encoding="utf-8"))
        for dataset in selected_datasets:
            manifest = json.loads((
                args.base_dir / "complete_prompt_free" / dataset / args.generator / "manifest.json"
            ).read_text(encoding="utf-8"))
            for slot, source_id in enumerate(manifest["source_ids"]):
                if selected_slots and slot not in selected_slots:
                    continue
                material = f"{dataset}\x1f{args.generator}\x1f{source_id}\x1f{args.method}"
                shard = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") % args.shard_count
                if shard == args.shard_index:
                    tasks.append((dataset, slot, source_id, None))
    if args.limit > 0:
        tasks = tasks[:args.limit]

    model_path = Path(MODEL_DIR[args.generator])
    run_dir = args.output_dir / "generation_audit" / args.method / args.generator / f"shard_{args.shard_index:02d}_of_{args.shard_count:02d}"
    sparse_scope = ""
    if selected_slots:
        scope_material = ",".join(selected_datasets) + "\x1f" + ",".join(map(str, sorted(selected_slots)))
        sparse_scope = "sparse_" + "_".join(selected_datasets) + "_" + sha256_text(scope_material)[:12]
        run_dir = args.output_dir / "generation_audit" / args.method / args.generator / sparse_scope
    if args.audit_tag:
        run_dir = args.output_dir / "generation_audit" / args.method / args.generator / args.audit_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "schema_version": "clean20-watermark-extension-worker-v1",
        "method": args.method,
        "generator": args.generator,
        "model_path": str(model_path),
        "datasets": selected_datasets,
        "task_count": len(tasks),
        "max_new_tokens": args.max_new_tokens,
        "token_min_new_tokens": args.token_min_new_tokens,
        "semstamp_min_new_tokens": args.semstamp_min_new_tokens,
        "min_lexical_units": args.min_lexical_units,
        "max_attempts": args.max_attempts,
        "base_seed": args.base_seed,
        "native_falcon": args.native_falcon,
        "request_manifest": str(args.request_manifest.resolve()) if args.request_manifest else None,
        "request_manifest_sha256": request_manifest_sha256,
        "audit_tag": args.audit_tag,
        "slot_indices": sorted(selected_slots),
        "strict_text_quality": args.strict_text_quality,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    atomic_json(run_dir / "run_config.json", run_config)
    shard_suffix = (
        f"__shard_{args.shard_index:02d}_of_{args.shard_count:02d}"
        if args.shard_count > 1 else ""
    )
    lock_path = args.output_dir / "locks" / (
        f"{args.method}__{args.generator}{('__' + sparse_scope) if sparse_scope else shard_suffix}.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(json.dumps({
        "pid": os.getpid(), "method": args.method, "generator": args.generator,
        "base_seed": args.base_seed, "acquired_unix": time.time(),
    }) + "\n")
    lock_handle.flush()

    # A scheduler may conservatively relaunch a combination while another
    # resumable worker is finishing it.  Avoid loading a multi-GB model when all
    # registered records already pass the frozen per-record contract.
    if not args.force:
        valid_before_load = sum(
            valid_record(
                args.output_dir / "records" / dataset / args.generator /
                args.method / f"{slot:04d}.json",
                args.min_lexical_units,
                dataset=dataset, generator=args.generator, method=args.method,
                source_id=source_id, prompt=task_prompt,
            )
            for dataset, slot, source_id, task_prompt in tasks
        )
        if valid_before_load == len(tasks):
            summary = {
                **run_config, "completed": 0, "failed": 0,
                "skipped": len(tasks), "elapsed_seconds": 0.0,
                "status": "PASS", "preflight_complete": True,
            }
            atomic_json(run_dir / "summary.json", summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0

    model, tokenizer = load_model(
        model_path, args.max_new_tokens, native_falcon=args.native_falcon,
    )
    fixed_adapter = None
    if args.method == "CGW":
        fixed_adapter = CGWAdapter(model, tokenizer, 15485863)
    elif args.method == "HeavyWater":
        fixed_adapter = build_heavywater(model, tokenizer)
    elif args.method == "SemStamp":
        fixed_adapter = init_semstamp()

    completed = skipped = failed = 0
    started_run = time.perf_counter()
    for task_index, (dataset, slot, source_id, task_prompt) in enumerate(tasks):
        record_path = args.output_dir / "records" / dataset / args.generator / args.method / f"{slot:04d}.json"
        if not args.force and valid_record(
            record_path, args.min_lexical_units, dataset=dataset,
            generator=args.generator, method=args.method, source_id=source_id,
            prompt=task_prompt,
        ):
            skipped += 1
            continue
        if task_prompt is None:
            source = registry[f"{dataset}:{slot}"]
            if source["source_id"] != source_id:
                raise RuntimeError("source registry mismatch")
            prompt = str(source["prompt"])
        else:
            prompt = task_prompt
        attempt_rows = []
        success_row = None
        for attempt in range(1, args.max_attempts + 1):
            seed = stable_seed(
                args.base_seed, dataset, args.generator, source_id, args.method, attempt,
            )
            seed_everything(seed)
            started = time.perf_counter()
            try:
                if args.method == "CGW":
                    key = 15485863 if attempt == 1 else stable_seed(seed, "CGW_KEY")
                    # Reuse the expensive token-to-bit table for the registered
                    # key.  Only a rare short-output retry builds an alternate
                    # auditable per-source key.
                    adapter = fixed_adapter if attempt == 1 else CGWAdapter(model, tokenizer, key)
                    raw, diagnostics = generate_token_method(
                        args.method, model, tokenizer, prompt, args.max_new_tokens, adapter,
                        repetition_penalty=args.repetition_penalty,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        min_new_tokens=args.token_min_new_tokens,
                    )
                    diagnostics["secret_key"] = key
                elif args.method == "HeavyWater":
                    raw, diagnostics = generate_token_method(
                        args.method, model, tokenizer, prompt, args.max_new_tokens, fixed_adapter,
                        repetition_penalty=args.repetition_penalty,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        min_new_tokens=args.token_min_new_tokens,
                    )
                else:
                    raw, diagnostics = generate_semstamp(
                        fixed_adapter, model, tokenizer, prompt, args.max_new_tokens,
                        min_new_tokens=args.semstamp_min_new_tokens,
                    )
                raw, removed = sanitize_text(raw)
                continuation = normalize_text(raw)
                reasons = []
                if not continuation:
                    reasons.append("EMPTY")
                if lexical_count(continuation) < args.min_lexical_units:
                    reasons.append("TOO_SHORT")
                if prompt and continuation.startswith(prompt):
                    reasons.append("PROMPT_NOT_REMOVED")
                if args.strict_text_quality:
                    reasons.extend(strict_text_quality_reasons(continuation))
                attempt_row = {
                    "attempt": attempt,
                    "seed": seed,
                    "seconds": time.perf_counter() - started,
                    "characters": len(continuation),
                    "lexical_units": lexical_count(continuation),
                    "sanitized_characters": removed,
                    "reasons": reasons,
                    "diagnostics": diagnostics,
                }
                attempt_rows.append(attempt_row)
                if not reasons:
                    success_row = {
                        "schema_version": "clean20-watermark-extension-record-v1",
                        "status": "SUCCESS",
                        "dataset": dataset,
                        "generator": args.generator,
                        "method": args.method,
                        "slot_index": slot,
                        "source_id": source_id,
                        "prompt_sha256": sha256_text(prompt),
                        "continuation": continuation,
                        "continuation_sha256": sha256_text(continuation),
                        "attempts": attempt_rows,
                        "selected_attempt": attempt,
                    }
                    break
            except Exception as exc:
                attempt_rows.append({
                    "attempt": attempt,
                    "seed": seed,
                    "seconds": time.perf_counter() - started,
                    "exception": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if success_row is None:
            failed += 1
            success_row = {
                "schema_version": "clean20-watermark-extension-record-v1",
                "status": "FAILED",
                "dataset": dataset,
                "generator": args.generator,
                "method": args.method,
                "slot_index": slot,
                "source_id": source_id,
                "prompt_sha256": sha256_text(prompt),
                "attempts": attempt_rows,
            }
        else:
            completed += 1
        atomic_json(record_path, success_row)
        progress = {
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "processed": task_index + 1,
            "total": len(tasks),
            "elapsed_seconds": time.perf_counter() - started_run,
        }
        atomic_json(run_dir / "progress.json", progress)
        print(
            f"[{args.method}/{args.generator}] {task_index + 1}/{len(tasks)} "
            f"{dataset}:{slot} {success_row['status']}", flush=True,
        )
    summary = {
        **run_config,
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "elapsed_seconds": time.perf_counter() - started_run,
        "status": "PASS" if failed == 0 else "FAILED_RECORDS_PRESENT",
    }
    atomic_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
