#!/usr/bin/env python3
"""Precompute reference-shaping sufficient statistics with a 20/60 split gate.

No evaluation labels enter feature computation.  Development invocations only
open development input files.  Test invocations require an externally created,
hash-linked frozen selection and authorization before models or data are read.
Existing completed cells are immutable and partial attempts are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import torch

from fastdetect_official_module_core import _document_score, _weights


ENDPOINTS = ((1.0, 1.0), (1.0, 0.8), (1.2, 1.0), (1.2, 0.8))
EPSILON = 1e-9
SCHEMA_VERSION = 1
DEVELOPMENT_DOMAINS = frozenset(("c4", "squad", "writing", "xsum"))
TEST_DOMAINS = frozenset(("cmv", "cnn", "dialogsum", "eli5", "hswag", "imdb",
                          "pubmed", "roct", "sci", "tldr", "wp", "yelp"))
GENERATORS = frozenset(("GLM-4-9B-0414", "falcon", "llama-2-7b", "opt-2.7B", "opt-6.7B"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    with temporary.open("x") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def endpoint_name(temperature, mass):
    return "T%s_P%s" % (str(temperature).replace(".", "p"), str(mass).replace(".", "p"))


def mixture_score(stats, strength):
    """Exact token-distribution mixture, including the between-mean variance."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("mixture strength must be in [0, 1]")
    numerator = (1.0 - strength) * stats["N0"] + strength * stats["N1"]
    variance = ((1.0 - strength) * stats["V0"] + strength * stats["V1"]
                + strength * (1.0 - strength) * stats["C01"])
    return numerator / max(math.sqrt(max(variance, 0.0)), EPSILON)


def cross_score(stats, center, scale):
    """Center/scale crossing is diagnostic, not a distribution identity."""
    if center not in (0, 1) or scale not in (0, 1):
        raise ValueError("center and scale must each be 0 or 1")
    return stats[f"N{center}"] / max(math.sqrt(max(stats[f"V{scale}"], 0.0)), EPSILON)


def features(ref, score, labels, chunk=128):
    """Float32 token moments, matching the existing supplement nucleus rule.

    ref/score are [tokens, vocabulary] logits; labels are observed next-token
    IDs. These labels are unrelated to the human/machine evaluation label.
    """
    if ref.ndim != 2 or score.ndim != 2 or labels.ndim != 1:
        raise ValueError("expected two [tokens, vocab] logits and [tokens] token IDs")
    n = int(labels.numel())
    if n < 1 or ref.shape[0] != n or score.shape[0] != n or chunk < 1:
        raise ValueError("nonempty aligned token sequences and a positive chunk are required")
    common = min(ref.shape[-1], score.shape[-1])
    if common < 2 or int(labels.max()) >= common or int(labels.min()) < 0:
        raise ValueError("token IDs must belong to the common vocabulary")
    target_parts, entropy_parts, divergence_parts = [], [], []
    means, variances = {key: [] for key in ENDPOINTS}, {key: [] for key in ENDPOINTS}
    outside, retained = {key: [] for key in ENDPOINTS}, {key: [] for key in ENDPOINTS}
    minima = {key: common for key in ENDPOINTS}
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        r = ref[start:stop, :common].float()
        s = score[start:stop, :common].to(ref.device).float()
        observed = labels[start:stop].to(ref.device)
        lp = torch.log_softmax(s, dim=-1)
        lq = torch.log_softmax(r, dim=-1)
        q = lq.exp()
        target_parts.append(lp.gather(-1, observed[:, None]).squeeze(-1).cpu())
        entropy_parts.append((-(q * lq).sum(-1) / math.log(common)).cpu())
        divergence_parts.append((q * (lq - lp)).sum(-1).cpu())
        for temperature in (1.0, 1.2):
            qt = torch.softmax(r / temperature, dim=-1)
            sorted_q, indices = torch.sort(qt, dim=-1, descending=True)
            sorted_lp = lp.gather(-1, indices)
            cumulative = sorted_q.cumsum(-1)
            removal = cumulative > 0.8
            mask = torch.ones_like(removal)
            mask[:, 1:] = ~removal[:, :-1]
            for mass in (1.0, 0.8):
                key = (temperature, mass)
                if mass == 1.0:
                    dist, logprob = qt, lp
                    kept_mass = torch.ones(stop - start, device=ref.device)
                    outside_chunk = torch.zeros(stop - start, dtype=torch.bool, device=ref.device)
                else:
                    values = sorted_q * mask
                    kept_mass = values.sum(-1)
                    dist, logprob = values / kept_mass[:, None], sorted_lp
                    # Every common-vocabulary ID occurs exactly once per row.
                    positions = (indices == observed[:, None]).nonzero(as_tuple=False)[:, 1]
                    outside_chunk = ~mask[torch.arange(stop-start, device=ref.device), positions]
                    minima[key] = min(minima[key], int(mask.sum(-1).min()))
                mean = (dist * logprob).sum(-1)
                variance = ((dist * logprob.square()).sum(-1) - mean.square()).clamp_min(0.0)
                means[key].append(mean.cpu())
                variances[key].append(variance.cpu())
                outside[key].append(outside_chunk.cpu())
                retained[key].append(kept_mass.cpu())
    target, entropy = torch.cat(target_parts), torch.cat(entropy_parts)
    means = {key: torch.cat(value) for key, value in means.items()}
    variances = {key: torch.cat(value) for key, value in variances.items()}
    outside = {key: torch.cat(value).float() for key, value in outside.items()}
    retained = {key: torch.cat(value) for key, value in retained.items()}
    weight_map = {"uniform": _weights(entropy, "uniform", 1.5),
                  "entropy": _weights(entropy, "linear_entropy", 1.5)}
    full, shaped = (1.0, 1.0), (1.2, 0.8)
    stats = {}
    for name, weight in weight_map.items():
        endpoints = {}
        for key in ENDPOINTS:
            numerator = float((weight * target).sum() - (weight * means[key]).sum())
            variance = float((weight.square() * variances[key]).sum())
            endpoints[endpoint_name(*key)] = {
                "N": numerator, "V": variance,
                "R": _document_score(target, means[key], variances[key], weight, EPSILON),
            }
        a, b = endpoints[endpoint_name(*full)], endpoints[endpoint_name(*shaped)]
        stats[name] = {
            "N0": a["N"], "N1": b["N"], "V0": a["V"], "V1": b["V"],
            "C01": float((weight.square() * (means[full] - means[shaped]).square()).sum()),
            "R0": a["R"], "R1": b["R"],
            "R_T1_P0p8": endpoints[endpoint_name(1.0, 0.8)]["R"],
            "R_T1p2_P1": endpoints[endpoint_name(1.2, 1.0)]["R"],
            "neff": float(1.0 / weight.square().sum()),
            "weighted_entropy": float((weight * entropy).sum()),
            "weighted_outside_fraction": float((weight * outside[shaped]).sum()),
            "endpoints": endpoints,
        }
    diagnostics = {
        "entropy_mean": float(entropy.mean()),
        "entropy_sd": float(entropy.std(unbiased=False)),
        "reference_score_kl_mean": float(torch.cat(divergence_parts).mean()),
        "observed_outside_nucleus_fraction": float(outside[shaped].mean()),
        "retained_mass_mean": float(retained[shaped].mean()),
        "min_support_tokens": minima[shaped],
        "nucleus_degenerate_fraction": float((variances[shaped] <= EPSILON).float().mean()),
        "untempered_outside_nucleus_fraction": float(outside[(1.0, 0.8)].mean()),
        "untempered_retained_mass_mean": float(retained[(1.0, 0.8)].mean()),
        "token_count": n,
        "common_vocabulary_size": common,
    }
    result = {"stats": stats, "diagnostics": diagnostics}
    # Reject nonfinite nested features before a cell can become complete.
    json.dumps(result, allow_nan=False)
    return result


def validate_access(config, split, config_sha256=None):
    """Validate split metadata and the test gate without opening any raw input."""
    if split not in ("development", "test"):
        raise ValueError("split must be development or test")
    cells = config["data_cells"]
    counts = {name: sum(cell["split"] == name for cell in cells) for name in ("development", "test")}
    if counts != {"development": 20, "test": 60} or len(cells) != 80:
        raise ValueError(f"expected exactly 20 development and 60 test cells, got {counts}")
    for name, domains in (("development", DEVELOPMENT_DOMAINS), ("test", TEST_DOMAINS)):
        actual_pairs = {(cell["dataset"], cell["generator"]) for cell in cells if cell["split"] == name}
        expected_pairs = {(domain, generator) for domain in domains for generator in GENERATORS}
        if actual_pairs != expected_pairs:
            raise ValueError(f"{name} domain/generator whitelist differs from the fixed 20/60 protocol")
    names = [(cell["dataset"], cell["generator"]) for cell in cells]
    if len(set(names)) != len(names):
        raise ValueError("development and test cell identities must be disjoint")
    paths = [str(Path(cell["data_path"]).absolute()) for cell in cells]
    if len(set(paths)) != len(paths):
        raise ValueError("raw data paths must be disjoint across all 80 cells")
    protocol_hash = config["frozen_protocol_sha256"]
    if len(protocol_hash) != 64 or any(c not in "0123456789abcdef" for c in protocol_hash):
        raise ValueError("frozen_protocol_sha256 must be a lowercase SHA-256 hex digest")
    if config.get("protocol_path") and sha(config["protocol_path"]) != protocol_hash:
        raise ValueError("protocol file no longer matches frozen_protocol_sha256")
    if split == "development":
        return {"split": split, "frozen_protocol_sha256": protocol_hash}
    root = Path(config["output_root"])
    selection_path = root / "FROZEN_SELECTION.json"
    authorization_path = root / "TEST_ACCESS_AUTHORIZATION.json"
    if not selection_path.is_file() or not authorization_path.is_file():
        raise PermissionError("test scoring requires frozen development results and explicit test authorization")
    selection = json.loads(selection_path.read_text())
    authorization = json.loads(authorization_path.read_text())
    if not config_sha256 or selection.get("config_sha256") != config_sha256:
        raise PermissionError("frozen selection config hash mismatch")
    if selection.get("frozen_protocol_sha256") != protocol_hash:
        raise PermissionError("frozen selection protocol hash mismatch")
    code_hashes = selection.get("code_sha256", {})
    if not isinstance(code_hashes, dict) or code_hashes.get("scripts/score_reference_moments.py") != sha(__file__):
        raise PermissionError("precompute script differs from frozen code")
    if authorization.get("authorized") is not True or authorization.get("frozen_protocol_sha256") != protocol_hash:
        raise PermissionError("test authorization missing or protocol hash mismatch")
    if authorization.get("config_sha256") != config_sha256:
        raise PermissionError("test authorization config hash mismatch")
    selection_hash = sha(selection_path)
    if authorization.get("frozen_selection_sha256") != selection_hash:
        raise PermissionError("test authorization does not bind the current frozen selection")
    return {"split": split, "frozen_protocol_sha256": protocol_hash,
            "config_sha256": config_sha256,
            "frozen_selection_sha256": selection_hash,
            "test_authorization_sha256": sha(authorization_path)}


def assigned_cells(config, split, worker):
    cells = [cell for cell in config["data_cells"] if cell["split"] == split]
    modulus, remainder = int(worker.get("cell_modulus", 1)), int(worker.get("cell_remainder", 0))
    if modulus < 1 or not 0 <= remainder < modulus:
        raise ValueError("invalid worker cell modulo assignment")
    return [cell for index, cell in enumerate(cells) if index % modulus == remainder]


def check_model_paths(worker):
    for key in ("q", "p"):
        if not Path(worker[key]).is_dir():
            raise ValueError(f"only an existing local model directory is allowed: {worker[key]}")
    proxy = worker["proxy"]
    qname, pname = str(worker["q"]).lower(), str(worker["p"]).lower()
    if proxy == "OPT":
        if worker["q"] != worker["p"] or "opt" not in qname or "2.7" not in qname:
            raise ValueError("OPT requires the same OPT-2.7B reference and scoring model")
    elif proxy == "Falcon":
        if "falcon" not in qname or "7b" not in qname or "instruct" in qname:
            raise ValueError("Falcon reference must be Falcon-7B base")
        if "falcon" not in pname or "7b" not in pname or "instruct" not in pname:
            raise ValueError("Falcon scoring must be Falcon-7B-Instruct")
    else:
        raise ValueError("proxy must be OPT or Falcon")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "test"), required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--smoke-rows", type=int, default=0)
    args = parser.parse_args()
    if args.smoke_rows < 0:
        raise ValueError("smoke rows cannot be negative")
    config = json.loads(args.config.read_text())
    config_hash, code_hash = sha(args.config), sha(__file__)
    access = validate_access(config, args.split, config_hash)
    worker = config["workers"][args.worker]
    check_model_paths(worker)
    cells = assigned_cells(config, args.split, worker)
    if not cells:
        raise ValueError("worker has no cells")
    if (config["max_tokens"], config["batch_size"], config["max_batch_tokens"]) != (2048, 8, 2048):
        raise ValueError("parity protocol requires max_tokens=2048, batch_size=8, max_batch_tokens=2048")
    methods = config["methods"]
    if len(set(methods)) != len(methods) or "human" in methods:
        raise ValueError("methods must be unique and exclude human")
    same = worker["q"] == worker["p"]
    gpus = [str(value) for value in worker["gpus"]]
    if len(gpus) != (1 if same else 2) or len(set(gpus)) != len(gpus):
        raise ValueError("OPT needs one GPU and Falcon needs two distinct GPUs")
    requested_visibility = ",".join(gpus)
    if os.environ.get("CUDA_VISIBLE_DEVICES", requested_visibility) != requested_visibility:
        raise ValueError("CUDA_VISIBLE_DEVICES conflicts with configured worker GPUs")
    os.environ["CUDA_VISIBLE_DEVICES"] = requested_visibility
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from model import load_model, load_tokenizer

    root = Path(config["output_root"])
    groups = root / (args.split + ("_smoke" if args.smoke_rows else "")) / worker["proxy"]
    groups.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    report = {"schema_version": SCHEMA_VERSION, "worker": args.worker,
              "name": worker["name"], "proxy": worker["proxy"], "status": "RUNNING",
              "split": args.split, "smoke_rows": args.smoke_rows,
              "rows": 0, "cells": 0, "config_sha256": config_hash,
              "script_sha256": code_hash, "access": access,
              "forward_seconds": 0.0, "module_seconds": 0.0,
              "max_endpoint_parity_error": 0.0, "run_id": run_id}
    started = time.perf_counter()
    progress_path = groups / f"worker_{args.worker}_progress.json"
    atomic_json(progress_path, report)
    tokenizers = [load_tokenizer(path, str(Path(path).parent)) for path in (worker["q"], worker["p"])]
    if tokenizers[0].get_vocab() != tokenizers[1].get_vocab():
        raise ValueError("reference/scoring tokenizer vocabularies differ")
    qdevice, pdevice = "cuda:0", "cuda:0" if same else "cuda:1"
    qm = load_model(worker["q"], qdevice, str(Path(worker["q"]).parent)).eval()
    pm = qm if same else load_model(worker["p"], pdevice, str(Path(worker["p"]).parent)).eval()
    for model in (qm, pm):
        model.config.use_cache = False
    for cell in cells:
        # Revalidate immediately before each raw-file access, including resumes.
        if validate_access(config, args.split, config_hash) != access:
            raise PermissionError("access artifacts changed during the run")
        source = Path(cell["data_path"])
        if sha(source) != cell["sha256"]:
            raise ValueError(f"input hash mismatch: {source}")
        output = groups / (cell["dataset"] + "_" + cell["generator"] + ".jsonl")
        manifest = output.with_suffix(".manifest.json")
        if output.exists():
            if not manifest.exists():
                raise RuntimeError(f"completed path has no manifest; preserve and inspect: {output}")
            old = json.loads(manifest.read_text())
            expected = {"config_sha256": config_hash, "script_sha256": code_hash,
                        "input_sha256": cell["sha256"], "access": access,
                        "smoke_rows": args.smoke_rows}
            if any(old.get(key) != value for key, value in expected.items()) or old["sha256"] != sha(output):
                raise ValueError(f"immutable existing cell failed resume validation: {output}")
            report["rows"] += old["rows"]
            report["cells"] += 1
            continue
        data = json.loads(source.read_text())
        count, records = int(cell["source_count"]), []
        for method in ["human"] + methods:
            values = data["original" if method == "human" else method]
            if len(values) != count:
                raise ValueError(f"source count mismatch: {cell['cell']} / {method}")
            for index, text_value in enumerate(values):
                ids = tokenizers[0](text_value, add_special_tokens=True, return_token_type_ids=False)["input_ids"]
                other = tokenizers[1](text_value, add_special_tokens=True, return_token_type_ids=False)["input_ids"]
                if ids != other:
                    raise ValueError("reference/scoring tokenization differs")
                length = len(ids)
                ids = ids[:config["max_tokens"]]
                if len(ids) < 2:
                    raise ValueError("text must contain at least two input tokens")
                row = {"row_id": f"{cell['dataset']}:{cell['generator']}:{index}:{method}",
                       "source_id": f"{cell['dataset']}:{index}", "source_index": index,
                       "cell": cell["cell"], "dataset": cell["dataset"], "generator": cell["generator"],
                       "proxy": worker["proxy"], "watermark_method": method, "label": int(method != "human"),
                       "split": args.split, "token_length": len(ids)-1,
                       "original_input_token_length": length, "context_truncated": length > config["max_tokens"],
                       "text_sha256": hashlib.sha256(text_value.encode()).hexdigest()}
                records.append((ids, row))
        del data
        if args.smoke_rows:
            records.sort(key=lambda item: len(item[0]))
            sample_count = min(args.smoke_rows, len(records))
            positions = [round(i*(len(records)-1)/max(sample_count-1, 1)) for i in range(sample_count)]
            records = [records[position] for position in positions]
        records.sort(key=lambda item: len(item[0]))
        partial = output.with_name(output.name + ".partial." + run_id)
        parity, written, offset = [], 0, 0
        with partial.open("x") as handle:
            while offset < len(records):
                stop = min(offset + config["batch_size"], len(records))
                while stop > offset+1 and max(len(item[0]) for item in records[offset:stop])*(stop-offset) > config["max_batch_tokens"]:
                    stop -= 1
                batch = records[offset:stop]
                encoded = tokenizers[0].pad([{"input_ids": ids, "attention_mask": [1]*len(ids)} for ids, _ in batch], return_tensors="pt")
                tick = time.perf_counter()
                with torch.inference_mode():
                    lr = qm(**{key: value.to(qdevice) for key, value in encoded.items()}, use_cache=False).logits
                    ls = lr if same else pm(**{key: value.to(pdevice) for key, value in encoded.items()}, use_cache=False).logits
                torch.cuda.synchronize(qdevice)
                if not same:
                    torch.cuda.synchronize(pdevice)
                report["forward_seconds"] += time.perf_counter() - tick
                tick = time.perf_counter()
                for position, (ids, row) in enumerate(batch):
                    n = len(ids)-1
                    labels = encoded["input_ids"][position, 1:len(ids)].to(qdevice)
                    r, s = lr[position, :n], ls[position, :n]
                    computed = features(r, s, labels)
                    if written == 0 or args.smoke_rows:
                        from run_review_supplement_scores import features as old_features, tag
                        old = old_features(r, s, labels, [1.0, 1.2], [1.0, 0.8])
                        errors = {f"{endpoint_name(*key)}_{weight}": abs(old[tag(*key, weight)] - computed["stats"][weight]["endpoints"][endpoint_name(*key)]["R"])
                                  for key in ENDPOINTS for weight in ("uniform", "entropy")}
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
                    print(json.dumps({"worker": args.worker, "split": args.split, "cell": cell["cell"],
                                      "cell_rows": written, "total_rows": report["rows"]+written,
                                      "elapsed": round(time.perf_counter()-started, 1)}), flush=True)
            handle.flush()
            os.fsync(handle.fileno())
        # No process may overwrite a completed cell, even for matching content.
        if output.exists() or manifest.exists():
            raise RuntimeError(f"concurrent output collision; preserved attempt: {partial}")
        final_manifest = {"schema_version": SCHEMA_VERSION, "config_sha256": config_hash,
                          "script_sha256": code_hash, "sha256": sha(partial), "rows": written,
                          "input_sha256": cell["sha256"], "access": access,
                          "smoke_rows": args.smoke_rows, "proxy": worker["proxy"],
                          "parity_checks": parity, "run_id": run_id}
        # link() publishes without replace semantics if another worker raced us.
        os.link(partial, output)
        partial.unlink()
        atomic_json(manifest, final_manifest)
        report["rows"] += written
        report["cells"] += 1
        report["elapsed_seconds"] = time.perf_counter()-started
        atomic_json(progress_path, report)
        if args.smoke_rows:
            break
    report["status"] = "COMPLETE"
    report["elapsed_seconds"] = time.perf_counter()-started
    report["gpu_peak_allocated_bytes"] = {str(i): torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())}
    atomic_json(progress_path, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
