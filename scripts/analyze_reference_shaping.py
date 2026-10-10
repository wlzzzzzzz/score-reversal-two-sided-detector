#!/usr/bin/env python3
"""Frozen, domain-clustered reference-shaping analysis; never selects a lambda.

Only --stage test may open test moments, and only after all freeze checks pass.
The development and freeze stages do not open any test source or score file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

REPO = Path(__file__).resolve().parents[1]
SCORES = ("E00", "E10", "E01", "E11", "E_mix_half", "T_only", "C_only", "U00", "U11", "R0", "R1")
ENERGY_SCORES = SCORES[:9]
PROXIES = ("OPT", "Falcon")
ALPHAS = (0.01, 0.05)
VARIANCE_FLOOR = 1e-18
CODE_FILES = (
    "scripts/analyze_reference_shaping.py",
    "scripts/score_reference_moments.py",
    "scripts/audit_reference_split.py",
    "scripts/fastdetect_official_module_core.py",
    "scripts/model.py",
    "scripts/launch_reference_shaping.py",
)
QUEUE_CODE = "scripts/score_reference_queue.py"
ARTIFACTS = (
    "atomic.csv", "summary.csv", "by_domain.csv", "mechanism_features.csv",
    "hypothesis_results.json", "shared_thresholds.json", "shared_operating_points.csv",
    "split_half_features.csv", "human_bias.json", "REPORT.md", "run_manifest.json",
)
SUMMARY_METRICS = ("mean_auc", "p10", "cvar10", "flip_rate", "mean_tpr_at_1pct_fpr", "mean_tpr_at_5pct_fpr")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text())


def code_hashes():
    names = list(CODE_FILES)
    if (REPO / QUEUE_CODE).is_file():
        names.append(QUEUE_CODE)
    return {name: sha(REPO / name) for name in names}


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}


def read_config(path):
    config = load_json(path)
    if sha(config["protocol_path"]) != config["frozen_protocol_sha256"]:
        raise PermissionError("protocol no longer matches its frozen config digest")
    analysis = config["analysis"]
    required = {"lambda_values": [0, 0.5, 1], "bootstrap_draws": 2000,
                "bootstrap_seed": 20260910, "variance_floor": VARIANCE_FLOOR,
                "harmful_auc_delta": -0.02, "mean_noninferiority_margin": 0.005}
    for key, value in required.items():
        if analysis.get(key) != value:
            raise ValueError(f"fixed analysis option differs from protocol: {key}")
    cells = config["data_cells"]
    if len({(c["dataset"], c["generator"]) for c in cells}) != len(cells):
        raise ValueError("duplicate roster cell")
    domains = {stage: {c["dataset"] for c in cells if c["split"] == stage}
               for stage in ("development", "test")}
    if domains["development"] & domains["test"]:
        raise ValueError("development and test domains overlap")
    if len(domains["development"]) != 4 or len(domains["test"]) != 12:
        raise ValueError("protocol requires four development and twelve test domains")
    return config, sha(path)


def moment_paths(config, stage):
    """Explicit roster only; this function does not open paths."""
    root = Path(config["output_root"])
    for proxy in PROXIES:
        for cell in config["data_cells"]:
            if cell["split"] == stage:
                path = root / stage / proxy / f"{cell['dataset']}_{cell['generator']}.jsonl"
                yield proxy, cell, path


def verify_hash_map(hashes):
    if not isinstance(hashes, dict) or not hashes:
        raise PermissionError("missing frozen artifact hashes")
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_absolute():
            path = REPO / path
        if sha(path) != expected:
            raise PermissionError(f"frozen artifact changed: {path}")


def validate_test_gate(config, config_hash):
    """Must run BEFORE opening the test audit or any test moment manifest."""
    root = Path(config["output_root"])
    selection_path = root / "FROZEN_SELECTION.json"
    authorization_path = root / "TEST_ACCESS_AUTHORIZATION.json"
    if not selection_path.is_file() or not authorization_path.is_file():
        raise PermissionError("test is locked: freeze and authorization are required")
    frozen, authorization = load_json(selection_path), load_json(authorization_path)
    for key, expected in (("config_sha256", config_hash),
                          ("frozen_protocol_sha256", config["frozen_protocol_sha256"])):
        if frozen.get(key) != expected or authorization.get(key) != expected:
            raise PermissionError(f"test gate {key} mismatch")
    if authorization.get("authorized") is not True:
        raise PermissionError("test authorization must be explicit authorized=true")
    if authorization.get("frozen_selection_sha256") != sha(selection_path):
        raise PermissionError("authorization does not bind the current frozen selection")
    if frozen.get("code_sha256") != code_hashes():
        raise PermissionError("analysis/precompute/audit code differs from frozen versions")
    verify_hash_map(frozen.get("dev_artifact_sha256"))
    if frozen.get("software_versions") != versions():
        raise PermissionError("analysis software versions differ from development")
    return frozen


def derive_scores(stats):
    e, u = stats["entropy"], stats["uniform"]
    n0, n1 = float(e["N0"]), float(e["N1"])
    v0, v1, c01 = float(e["V0"]), float(e["V1"]), float(e["C01"])
    if min(v0, v1, c01, float(u["V0"]), float(u["V1"])) < 0:
        raise ValueError("negative reference variance or between-reference term")
    denominator0, denominator1 = max(v0, VARIANCE_FLOOR), max(v1, VARIANCE_FLOOR)
    scores = {
        "E00": n0 * n0 / denominator0,
        "E10": n1 * n1 / denominator0,
        "E01": n0 * n0 / denominator1,
        "E11": n1 * n1 / denominator1,
        "E_mix_half": ((n0 + n1) / 2) ** 2 / max((v0 + v1) / 2 + c01 / 4, VARIANCE_FLOOR),
        "T_only": float(e["R_T1p2_P1"]) ** 2,
        "C_only": float(e["R_T1_P0p8"]) ** 2,
        "U00": float(u["N0"]) ** 2 / max(float(u["V0"]), VARIANCE_FLOOR),
        "U11": float(u["N1"]) ** 2 / max(float(u["V1"]), VARIANCE_FLOOR),
        "R0": n0 / math.sqrt(denominator0), "R1": n1 / math.sqrt(denominator1),
    }
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("nonfinite derived score")
    return scores


def source_half(source_id):
    return hashlib.sha256(str(source_id).encode("utf-8")).digest()[0] % 2


def load_moments(config, config_hash, stage):
    """Read the requested stage only, validating every complete cell manifest."""
    if stage == "test":
        validate_test_gate(config, config_hash)
    frames, input_hashes = [], {}
    for proxy, cell, path in moment_paths(config, stage):
        manifest_path = path.with_suffix(".manifest.json")
        manifest = load_json(manifest_path)
        expected = {"config_sha256": config_hash, "script_sha256": sha(REPO / CODE_FILES[1]),
                    "input_sha256": cell["sha256"], "proxy": proxy, "smoke_rows": 0}
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError(f"moment manifest mismatch: {manifest_path}")
        if "runner_script_sha256" in manifest:
            if not (REPO / QUEUE_CODE).is_file() or manifest["runner_script_sha256"] != sha(REPO / QUEUE_CODE):
                raise ValueError(f"queue runner hash mismatch: {manifest_path}")
        access = manifest.get("access", {})
        if access.get("split") != stage or access.get("frozen_protocol_sha256") != config["frozen_protocol_sha256"]:
            raise PermissionError(f"moment access binding mismatch: {manifest_path}")
        if stage == "test":
            root = Path(config["output_root"])
            bindings = {"config_sha256": config_hash,
                        "frozen_selection_sha256": sha(root / "FROZEN_SELECTION.json"),
                        "test_authorization_sha256": sha(root / "TEST_ACCESS_AUTHORIZATION.json")}
            if any(access.get(k) != v for k, v in bindings.items()):
                raise PermissionError(f"test moment gate binding mismatch: {path}")
        digest = sha(path)
        if digest != manifest["sha256"]:
            raise ValueError(f"moment content hash mismatch: {path}")
        records = []
        with path.open() as handle:
            for line in handle:
                original = json.loads(line)
                if any(original[k] != v for k, v in (("proxy", proxy), ("dataset", cell["dataset"]),
                       ("generator", cell["generator"]), ("split", stage))):
                    raise ValueError(f"row metadata does not match roster: {path}")
                method = original["watermark_method"]
                if method not in ["human", *config["methods"]] or original["label"] != int(method != "human"):
                    raise ValueError("unexpected method or label")
                index = int(original["source_index"])
                if not 0 <= index < cell["source_count"] or original["source_id"] != f"{cell['dataset']}:{index}":
                    raise ValueError("invalid source identity")
                if int(original["token_length"]) < 1:
                    raise ValueError("empty scored text")
                record = {k: original[k] for k in ("source_id", "source_index", "dataset", "generator",
                          "proxy", "watermark_method", "label", "text_sha256", "token_length", "split")}
                record.update(derive_scores(original["stats"]))
                e = original["stats"]["entropy"]
                for score in ("R0", "R1"):
                    if not math.isfinite(float(e[score])):
                        raise ValueError("nonfinite stored signed response")
                    record[f"stored_{score}_abs_difference"] = abs(record[score] - float(e[score]))
                record.update({key: float(e[key]) for key in ("N0", "N1", "V0", "V1", "C01", "neff", "weighted_entropy")})
                if not all(math.isfinite(record[k]) for k in ("N0", "N1", "V0", "V1", "C01", "neff", "weighted_entropy")):
                    raise ValueError("nonfinite sufficient statistic")
                record["half"] = source_half(record["source_id"])
                records.append(record)
        frame = pd.DataFrame.from_records(records)
        wanted = cell["source_count"] * (1 + len(config["methods"]))
        if len(frame) != wanted or manifest["rows"] != wanted:
            raise ValueError(f"incomplete cell: {path}")
        if frame.duplicated(["source_id", "watermark_method"]).any():
            raise ValueError("duplicate source/method row")
        counts = frame.groupby("watermark_method").size()
        if not (counts == cell["source_count"]).all() or len(counts) != 1 + len(config["methods"]):
            raise ValueError("unbalanced source channels")
        frames.append(frame)
        input_hashes[str(path)] = digest
        input_hashes[str(manifest_path)] = sha(manifest_path)
    result = pd.concat(frames, ignore_index=True)
    # Both proxies must score identical text identities for every roster row.
    if result.groupby(["dataset", "generator", "source_id", "watermark_method"])["text_sha256"].nunique().max() != 1:
        raise ValueError("text hashes disagree between proxies")
    return result, input_hashes


def canonical_humans(frame):
    humans = frame.loc[frame.label.eq(0)].copy()
    if humans.groupby(["proxy", "dataset", "source_id"])["text_sha256"].nunique().max() != 1:
        raise ValueError("human sources differ between generators")
    return humans.sort_values(["proxy", "dataset", "source_id", "generator"]).drop_duplicates(["proxy", "dataset", "source_id"])


def estimate_bias(humans):
    domain = humans.groupby(["proxy", "dataset"]).N0.mean()
    full, lodo, means = {}, {}, {}
    for proxy in PROXIES:
        values = domain.loc[proxy]
        if len(values) < 2:
            raise ValueError("LODO bias requires multiple domains")
        means[proxy] = values.to_dict()
        full[proxy] = float(values.mean())
        lodo[proxy] = {name: float(values.drop(name).mean()) for name in values.index}
    return {"full_development_by_proxy": full, "lodo_by_proxy": lodo,
            "domain_means_by_proxy": means, "canonical_human_count": len(humans),
            "definition": "domain-equal mean N0; one lexicographically-first generator per proxy,dataset,source_id; never enters score"}


def threshold_spec(values, alpha):
    values = np.sort(np.asarray(values, dtype=np.float64))
    m = len(values)
    if not m:
        raise ValueError("cannot calibrate from no human texts")
    k = math.ceil((m + 1) * (1 - alpha))
    return {"value": float(values[k - 1]) if k <= m else None,
            "infinite": k > m, "m": m, "k": k, "alpha": alpha, "comparison": "strict >"}


def threshold_value(spec):
    return math.inf if spec["infinite"] else float(spec["value"])


def calibrate_thresholds(humans):
    full, lodo = {}, {}
    for proxy in PROXIES:
        part = humans.loc[humans.proxy.eq(proxy)]
        full[proxy] = {score: {str(alpha): threshold_spec(part[score], alpha) for alpha in ALPHAS} for score in ENERGY_SCORES}
        lodo[proxy] = {}
        for domain in sorted(part.dataset.unique()):
            training = part.loc[part.dataset.ne(domain)]
            lodo[proxy][domain] = {score: {str(alpha): threshold_spec(training[score], alpha) for alpha in ALPHAS} for score in ENERGY_SCORES}
    return {"full_development_by_proxy": full, "lodo_by_proxy": lodo,
            "definition": "unique development humans; ceil((m+1)*(1-alpha)); strict >; no score recentering"}


def attach_features(frame, bias, stage):
    result = frame.copy()
    if stage == "development":
        constants = np.asarray([bias["lodo_by_proxy"][p][d] for p, d in zip(result.proxy, result.dataset)])
    else:
        constants = result.proxy.map(bias["full_development_by_proxy"]).to_numpy(dtype=float)
    c = result.N0.to_numpy() - result.N1.to_numpy()
    v0 = np.maximum(result.V0.to_numpy(), VARIANCE_FLOOR)
    v1 = np.maximum(result.V1.to_numpy(), VARIANCE_FLOOR)
    result["analysis_bias"] = constants
    result["G"] = (2 * constants * c - c * c) / v0
    result["log_variance_ratio"] = np.log(v1) - np.log(v0)
    return result


def roc_metrics(human, machine):
    if not len(human) or not len(machine):
        return {"auc": math.nan, "tpr_at_1pct_fpr": math.nan, "tpr_at_5pct_fpr": math.nan}
    labels = np.r_[np.zeros(len(human), dtype=np.int8), np.ones(len(machine), dtype=np.int8)]
    values = np.r_[human, machine]
    fpr, tpr, _ = roc_curve(labels, values, drop_intermediate=False)
    return {"auc": float(roc_auc_score(labels, values)),
            "tpr_at_1pct_fpr": float(tpr[fpr <= .01 + 1e-12].max()),
            "tpr_at_5pct_fpr": float(tpr[fpr <= .05 + 1e-12].max())}


def task_features(human, machine):
    if not len(human) or not len(machine):
        return {"G_task": math.nan, "H_scale": math.nan, "G_human_median": math.nan, "G_machine_median": math.nan}
    if len(human) != len(machine):
        raise ValueError("pre-fixed pooled IQR requires equal human/machine counts")
    gh, gm = float(human.G.median()), float(machine.G.median())
    pooled = np.r_[human.log_variance_ratio, machine.log_variance_ratio]
    q25, q75 = np.quantile(pooled, [.25, .75])
    return {"G_task": (gh + gm) / 2, "H_scale": float(q75 - q25),
            "G_human_median": gh, "G_machine_median": gm}


def add_contrasts(row):
    row["delta_center"] = row["E10_auc"] - row["E00_auc"]
    row["delta_scale_given_center"] = row["E11_auc"] - row["E10_auc"]
    row["center_scale_interaction"] = row["E11_auc"] - row["E10_auc"] - row["E01_auc"] + row["E00_auc"]
    row["delta_hard_vs_base"] = row["E11_auc"] - row["E00_auc"]
    row["delta_half_vs_base"] = row["E_mix_half_auc"] - row["E00_auc"]
    row["delta_half_vs_hard"] = row["E_mix_half_auc"] - row["E11_auc"]


def build_atomic(frame, methods, split_halves=True):
    rows, half_rows = [], []
    for (proxy, domain, generator), cell in frame.groupby(["proxy", "dataset", "generator"], sort=True):
        human = cell.loc[cell.label.eq(0)].sort_values("source_id")
        for method in methods:
            machine = cell.loc[cell.watermark_method.eq(method)].sort_values("source_id")
            if human.source_id.tolist() != machine.source_id.tolist() or not len(human):
                raise ValueError(f"source bundles are incomplete: {proxy}/{domain}/{generator}/{method}")
            identity = {"proxy": proxy, "dataset": domain, "generator": generator, "method": method}
            row = {**identity, "human_count": len(human), "machine_count": len(machine),
                   "human_mean_length": float(human.token_length.mean()), "machine_mean_length": float(machine.token_length.mean()),
                   "analysis_bias": float(human.analysis_bias.iloc[0]), **task_features(human, machine)}
            for score in SCORES:
                row.update({f"{score}_{name}": value for name, value in roc_metrics(human[score], machine[score]).items()})
            add_contrasts(row)
            rows.append(row)
            if split_halves:
                for feature_half in (0, 1):
                    hf, mf = human.loc[human.half.eq(feature_half)], machine.loc[machine.half.eq(feature_half)]
                    ho, mo = human.loc[human.half.ne(feature_half)], machine.loc[machine.half.ne(feature_half)]
                    half_row = {**identity, "feature_half": feature_half, "outcome_half": 1 - feature_half,
                                "feature_human_count": len(hf), "outcome_human_count": len(ho), **task_features(hf, mf)}
                    aucs = {score: roc_metrics(ho[score], mo[score])["auc"] for score in ("E00", "E10", "E11")}
                    half_row.update(delta_center=aucs["E10"] - aucs["E00"],
                                    delta_scale_given_center=aucs["E11"] - aucs["E10"])
                    half_rows.append(half_row)
    return pd.DataFrame(rows), pd.DataFrame(half_rows)


def array_summary(auc, tpr1=None, tpr5=None):
    """Input [tasks,scores], output [scores,metrics]."""
    ordered = np.sort(np.asarray(auc, dtype=float), axis=0)
    k = max(1, math.ceil(.1 * len(ordered)))
    columns = [ordered.mean(axis=0), np.quantile(ordered, .1, axis=0), ordered[:k].mean(axis=0), (ordered < .5).mean(axis=0)]
    if tpr1 is not None:
        columns.extend([np.mean(tpr1, axis=0), np.mean(tpr5, axis=0)])
    return np.stack(columns, axis=1)


def bootstrap_domain_indices(domains, draws=2000, seed=20260910):
    return np.random.default_rng(seed).integers(0, len(domains), size=(draws, len(domains)))


def quantile_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"ci95_low": None, "ci95_high": None, "valid_draws": 0}
    low, high = np.quantile(values, [.025, .975])
    return {"ci95_low": float(low), "ci95_high": float(high), "valid_draws": len(values)}


def task_summary(atomic, domains, draws):
    summary, by_domain = [], []
    for proxy in PROXIES:
        part = atomic.loc[atomic.proxy.eq(proxy)].reset_index(drop=True)
        auc = part[[f"{s}_auc" for s in SCORES]].to_numpy()
        tpr1 = part[[f"{s}_tpr_at_1pct_fpr" for s in SCORES]].to_numpy()
        tpr5 = part[[f"{s}_tpr_at_5pct_fpr" for s in SCORES]].to_numpy()
        indices = [np.flatnonzero(part.dataset.eq(d).to_numpy()) for d in domains]
        if any(len(i) == 0 for i in indices):
            raise ValueError("all proxies must retain all domain clusters")
        sampled = np.empty((len(draws), len(SCORES), len(SUMMARY_METRICS)))
        for i, sampled_domains in enumerate(draws):
            selected = np.concatenate([indices[j] for j in sampled_domains])
            sampled[i] = array_summary(auc[selected], tpr1[selected], tpr5[selected])
        point = array_summary(auc, tpr1, tpr5)
        for j, score in enumerate(SCORES):
            row = {"proxy": proxy, "score": score, "task_count": len(part), "domain_count": len(domains)}
            for k, metric in enumerate(SUMMARY_METRICS):
                row[metric] = float(point[j, k])
                row.update({f"{metric}_{key}": value for key, value in quantile_ci(sampled[:, j, k]).items()})
                row[f"delta_vs_E00_{metric}"] = float(point[j, k] - point[0, k])
                row.update({f"delta_vs_E00_{metric}_{key}": value for key, value in quantile_ci(sampled[:, j, k] - sampled[:, 0, k]).items()})
            row["tpr_at_1pct_fpr"] = row["mean_tpr_at_1pct_fpr"]
            row["tpr_at_5pct_fpr"] = row["mean_tpr_at_5pct_fpr"]
            summary.append(row)
        for domain, selected in zip(domains, indices):
            result = array_summary(auc[selected], tpr1[selected], tpr5[selected])
            for j, score in enumerate(SCORES):
                by_domain.append({"proxy": proxy, "dataset": domain, "score": score,
                                  "task_count": len(selected), **dict(zip(SUMMARY_METRICS, result[j]))})
    return pd.DataFrame(summary), pd.DataFrame(by_domain)


def association(part, predictor, outcome, positive_only=False, minimum_domains=3):
    if positive_only:
        part = part.loc[part.G_task.gt(0)]
    part = part.loc[np.isfinite(part[predictor]) & np.isfinite(part[outcome])]
    result = {"task_count": len(part), "domain_count": int(part.dataset.nunique()), "rho": None, "reason": None}
    if result["domain_count"] < minimum_domains:
        result["reason"] = "fewer than three observed domains"
    elif part[predictor].nunique() < 2 or part[outcome].nunique() < 2:
        result["reason"] = "constant predictor or outcome"
    else:
        result["rho"] = float(spearmanr(part[predictor], part[outcome]).statistic)
    return result


def association_family(atomic, domains, draws, predictor, outcome, positive_only=False):
    per_proxy = {proxy: association(atomic.loc[atomic.proxy.eq(proxy)], predictor, outcome, positive_only) for proxy in PROXIES}
    boot = {proxy: [] for proxy in PROXIES}
    macro = []
    clusters = {(p, d): atomic.loc[atomic.proxy.eq(p) & atomic.dataset.eq(d)] for p in PROXIES for d in domains}
    for sampled in draws:
        current = []
        for proxy in PROXIES:
            if per_proxy[proxy]["rho"] is None:
                rho = None
            else:
                part = pd.concat([clusters[proxy, domains[j]] for j in sampled], ignore_index=True)
                # The >=3-domain requirement applies to the observed group above.
                # Repeated clusters are valid bootstrap draws, even if few unique
                # original domains remain; dropping them would truncate the CI.
                rho = association(part, predictor, outcome, positive_only, minimum_domains=1)["rho"]
            boot[proxy].append(math.nan if rho is None else rho)
            current.append(rho)
        macro.append(np.mean(current) if all(v is not None for v in current) else math.nan)
    for proxy in PROXIES:
        per_proxy[proxy].update(quantile_ci(boot[proxy]))
    points = [per_proxy[p]["rho"] for p in PROXIES]
    return {"by_proxy": per_proxy,
            "macro_within_proxy_rho": float(np.mean(points)) if all(v is not None for v in points) else None,
            **quantile_ci(macro), "bootstrap_draws": len(draws),
            "interval_note": "whole domains jointly resampled across proxies; invalid draws omitted and counted"}


def hypothesis_results(atomic, halves, domains, draws, stage):
    h1 = association_family(atomic, domains, draws, "G_task", "delta_center")
    h2 = association_family(atomic, domains, draws, "H_scale", "delta_scale_given_center", True)
    h2_all = association_family(atomic, domains, draws, "H_scale", "delta_scale_given_center")
    h1["supported"] = bool(all(h1["by_proxy"][p]["rho"] is not None and h1["by_proxy"][p]["rho"] > 0 for p in PROXIES)
                           and h1["ci95_low"] is not None and h1["ci95_low"] > 0)
    h2["supported"] = bool(all(h2["by_proxy"][p]["rho"] is not None and h2["by_proxy"][p]["rho"] < 0 for p in PROXIES)
                           and h2["ci95_high"] is not None and h2["ci95_high"] < 0)
    groups, h3 = [], {}
    for proxy in PROXIES:
        part = atomic.loc[atomic.proxy.eq(proxy)]
        for positive in (False, True):
            group = part.loc[part.G_task.gt(0) if positive else part.G_task.le(0)]
            groups.append({"proxy": proxy, "condition": "G>0" if positive else "G<=0",
                           "task_count": len(group), "domain_count": int(group.dataset.nunique()),
                           "mean_delta_center": float(group.delta_center.mean()),
                           "mean_delta_hard_vs_base": float(group.delta_hard_vs_base.mean())})
        hard = (part.delta_hard_vs_base.to_numpy() < -.02).astype(float)
        half = (part.delta_half_vs_base.to_numpy() < -.02).astype(float)
        delta = part.delta_half_vs_hard.to_numpy()
        indices = [np.flatnonzero(part.dataset.eq(d).to_numpy()) for d in domains]
        rate_draws, mean_draws = [], []
        for sampled in draws:
            selected = np.concatenate([indices[j] for j in sampled])
            rate_draws.append(float((half[selected] - hard[selected]).mean()))
            mean_draws.append(float(delta[selected].mean()))
        rate_ci, mean_ci = quantile_ci(rate_draws), quantile_ci(mean_draws)
        h3[proxy] = {"hard_harmful_rate": float(hard.mean()), "half_harmful_rate": float(half.mean()),
                     "harmful_rate_difference_half_minus_hard": {"point": float((half - hard).mean()), **rate_ci},
                     "mean_auc_difference_half_minus_hard": {"point": float(delta.mean()), **mean_ci},
                     "supported": bool(rate_ci["ci95_high"] < 0 and mean_ci["ci95_low"] > -.005)}
    sensitivity = {}
    for name, predictor, outcome, positive in (("H1", "G_task", "delta_center", False),
              ("H2_G_positive", "H_scale", "delta_scale_given_center", True),
              ("H2_all", "H_scale", "delta_scale_given_center", False)):
        sensitivity[name] = {}
        for proxy in PROXIES:
            folds = [association(halves.loc[halves.proxy.eq(proxy) & halves.feature_half.eq(i)], predictor, outcome, positive) for i in (0, 1)]
            defined = [fold["rho"] for fold in folds if fold["rho"] is not None]
            sensitivity[name][proxy] = {"directions": folds, "mean_rho": float(np.mean(defined)) if defined else None,
                                       "defined_directions": len(defined), "not_used_for_selection": True}
    foldwise = []
    for (proxy, domain), group in atomic.groupby(["proxy", "dataset"]):
        # Descriptive within-domain task correlation; not an independent-domain significance test.
        row = {"proxy": proxy, "dataset": domain, "task_count": len(group), "analysis_bias": float(group.analysis_bias.iloc[0])}
        for name, x, y, positive in (("H1", "G_task", "delta_center", False), ("H2", "H_scale", "delta_scale_given_center", True)):
            row[name] = association(group, x, y, positive, minimum_domains=1)
        foldwise.append(row)
    return {"stage": stage, "H1": h1, "H2_G_positive": h2, "H2_all_tasks": h2_all,
            "H3": {"by_proxy": h3, "supported": all(h3[p]["supported"] for p in PROXIES)},
            "G_sign_groups": groups, "foldwise_descriptive": foldwise, "split_half_sensitivity": sensitivity,
            "inference_note": ("Development OOF b; intervals condition on fixed overlapping-fold predictions. Four domains are unstable; these are descriptive, not held-domain confirmation."
                               if stage == "development" else "Frozen development b and thresholds; domain intervals condition on the fixed development set and observed within-domain samples."),
            "no_lambda_selection": True, "support_failure_is_not_proof_of_opposite": True}


def shared_operating_points(frame, humans, thresholds, stage):
    rows = []
    for proxy in PROXIES:
        data = frame.loc[frame.proxy.eq(proxy)]
        unique = humans.loc[humans.proxy.eq(proxy)]
        domains = sorted(data.dataset.unique())
        for score in ENERGY_SCORES:
            for alpha in ALPHAS:
                human_predictions, task_rates, all_machine_predictions, domain_human_rates = [], [], [], []
                for domain in domains:
                    spec = (thresholds["lodo_by_proxy"][proxy][domain] if stage == "development"
                            else thresholds["full_development_by_proxy"][proxy])[score][str(alpha)]
                    threshold = threshold_value(spec)
                    h = unique.loc[unique.dataset.eq(domain)]
                    m = data.loc[data.dataset.eq(domain) & data.label.eq(1)]
                    hp = h[score].to_numpy() > threshold
                    mp = m[score].to_numpy() > threshold
                    task = m.assign(decision=mp).groupby(["generator", "watermark_method"]).decision.mean().to_numpy()
                    human_predictions.extend(hp)
                    all_machine_predictions.extend(mp)
                    task_rates.extend(task)
                    domain_human_rates.append(float(hp.mean()))
                    rows.append({"proxy": proxy, "score": score, "nominal_development_alpha": alpha, "dataset": domain,
                                 "unique_human_count": len(h), "machine_count": len(m), "task_count": len(task),
                                 "actual_human_fpr": float(hp.mean()), "macro_machine_tpr": float(task.mean()),
                                 "pooled_machine_tpr": float(mp.mean()), "threshold": spec["value"],
                                 "threshold_infinite": spec["infinite"], "calibration_m": spec["m"],
                                 "threshold_mode": "LODO development" if stage == "development" else "frozen full development"})
                rows.append({"proxy": proxy, "score": score, "nominal_development_alpha": alpha, "dataset": "POOLED",
                             "unique_human_count": len(human_predictions), "machine_count": len(all_machine_predictions), "task_count": len(task_rates),
                             "actual_human_fpr": float(np.mean(human_predictions)), "macro_domain_human_fpr": float(np.mean(domain_human_rates)),
                             "macro_machine_tpr": float(np.mean(task_rates)), "pooled_machine_tpr": float(np.mean(all_machine_predictions)),
                             "threshold_mode": "LODO development" if stage == "development" else "frozen full development"})
    return pd.DataFrame(rows)


def report_text(stage, summary, hypotheses, frame, excluded):
    text = [f"# Fixed reference shaping: {stage}", "", f"Rows: {len(frame)}; domains: {frame.dataset.nunique()}; excluded source bundles: {len(excluded)}.",
            "", hypotheses["inference_note"], "", "All lambda values are fixed; no score or parameter is selected. G is an analysis predictor and never changes the detector score. Endpoints are recomputed in float64 from new matched sufficient statistics; tiny differences from stored float32 R can affect ties. The maximum differences are recorded in run_manifest.json.",
            "", "| Proxy | Score | Mean AUROC | P10 | CVaR10 | Flip rate |", "|---|---|---:|---:|---:|---:|"]
    for _, row in summary.iterrows():
        text.append(f"| {row['proxy']} | {row['score']} | {row.mean_auc:.4f} | {row.p10:.4f} | {row.cvar10:.4f} | {row.flip_rate:.4f} |")
    text.extend(["", f"H1 support criterion met: {hypotheses['H1']['supported']}; H2: {hypotheses['H2_G_positive']['supported']}; H3: {hypotheses['H3']['supported']}.",
                 "", "See hypothesis_results.json for directional correlations, domain counts, 2,000 paired domain-bootstrap intervals, sign groups and split-half sensitivity. Failure to meet a support criterion is inconclusive, not proof of its opposite.",
                 "", "atomic.csv contains threshold-exact task ROC operating points (no interpolation). shared_operating_points.csv separately reports actual human FPR and machine TPR under strict frozen/LODO thresholds; these are not mislabeled as test TPR at a fixed achieved FPR.",
                 "", "The original benchmark was inspected historically; this protocol enforces prospective separation for the new experiment, not historical blindness.", ""])
    return "\n".join(text)


def analysis_run(config, config_hash, stage):
    root = Path(config["output_root"])
    destination = root / "analysis" / stage
    if (destination / "run_manifest.json").exists():
        raise FileExistsError("completed analysis already exists; do not silently rerun a frozen evaluation")
    frozen, excluded, audit_hashes = None, set(), {}
    if stage == "test":
        frozen = validate_test_gate(config, config_hash)
        audit_path = root / "test_source_audit.json"
        audit = load_json(audit_path)
        expected = {"status": "PASS", "split": "test", "config_sha256": config_hash,
                    "frozen_protocol_sha256": config["frozen_protocol_sha256"],
                    "frozen_selection_sha256": sha(root / "FROZEN_SELECTION.json"),
                    "test_access_authorization_sha256": sha(root / "TEST_ACCESS_AUTHORIZATION.json"),
                    "development_source_manifest_sha256": sha(root / "development_source_manifest.json")}
        if any(audit.get(k) != v for k, v in expected.items()):
            raise PermissionError("test source audit does not match frozen access artifacts")
        excluded = set(audit["excluded_source_ids"])
        audit_hashes[str(audit_path)] = sha(audit_path)
    else:
        if (root / "FROZEN_SELECTION.json").exists():
            raise PermissionError("cannot change development analysis after freeze")
        audit_path = root / "development_source_manifest.json"
        audit = load_json(audit_path)
        if audit.get("status") != "PASS" or audit.get("config_sha256") != config_hash or audit.get("test_sources_opened") != 0:
            raise PermissionError("development source audit missing or invalid")
        audit_hashes[str(audit_path)] = sha(audit_path)
    raw, input_hashes = load_moments(config, config_hash, stage)
    unknown_exclusions = excluded - set(raw.source_id)
    if unknown_exclusions:
        raise ValueError("test audit excludes source IDs absent from the frozen roster")
    frame = raw.loc[~raw.source_id.isin(excluded)].copy()
    humans = canonical_humans(frame)
    if stage == "development":
        bias, thresholds = estimate_bias(humans), calibrate_thresholds(humans)
    else:
        bias = {"full_development_by_proxy": frozen["human_bias_by_proxy"], "definition": "unchanged frozen development constants"}
        thresholds = {"full_development_by_proxy": frozen["thresholds"], "definition": "unchanged frozen development energy thresholds"}
    frame = attach_features(frame, bias, stage)
    atomic, halves = build_atomic(frame, config["methods"])
    domains = sorted({c["dataset"] for c in config["data_cells"] if c["split"] == stage})
    draws = bootstrap_domain_indices(domains, config["analysis"]["bootstrap_draws"], config["analysis"]["bootstrap_seed"])
    summary, by_domain = task_summary(atomic, domains, draws)
    hypotheses = hypothesis_results(atomic, halves, domains, draws, stage)
    operating = shared_operating_points(frame, humans, thresholds, stage)
    destination.mkdir(parents=True, exist_ok=True)
    mechanism_columns = [name for name in atomic if not any(name.startswith(score + "_") for score in SCORES)]
    for name, value in (("atomic.csv", atomic), ("summary.csv", summary), ("by_domain.csv", by_domain),
                        ("mechanism_features.csv", atomic[mechanism_columns]), ("split_half_features.csv", halves),
                        ("shared_operating_points.csv", operating)):
        value.to_csv(destination / name, index=False)
    write_json(destination / "human_bias.json", bias)
    write_json(destination / "shared_thresholds.json", thresholds)
    write_json(destination / "hypothesis_results.json", hypotheses)
    (destination / "REPORT.md").write_text(report_text(stage, summary, hypotheses, frame, excluded))
    extra_artifacts = []
    if stage == "test" and excluded:
        # Same frozen constants, no recalibration or candidate selection.
        unfiltered, _ = build_atomic(attach_features(raw, bias, stage), config["methods"], split_halves=False)
        unfiltered_rows = []
        for proxy in PROXIES:
            part = unfiltered.loc[unfiltered.proxy.eq(proxy)]
            values = array_summary(part[[f"{s}_auc" for s in SCORES]].to_numpy())
            for j, score in enumerate(SCORES):
                unfiltered_rows.append({"proxy": proxy, "score": score, "task_count": len(part),
                                        **dict(zip(SUMMARY_METRICS[:4], values[j]))})
        pd.DataFrame(unfiltered_rows).to_csv(destination / "unfiltered_sensitivity_summary.csv", index=False)
        extra_artifacts.append("unfiltered_sensitivity_summary.csv")
    artifact_hashes = {str(destination / name): sha(destination / name) for name in ARTIFACTS if name != "run_manifest.json"}
    artifact_hashes.update({str(destination / name): sha(destination / name) for name in extra_artifacts})
    write_json(destination / "run_manifest.json", {
        "status": "COMPLETE", "stage": stage, "completed_at_utc": utc_now(), "config_sha256": config_hash,
        "frozen_protocol_sha256": config["frozen_protocol_sha256"], "code_sha256": code_hashes(),
        "software_versions": versions(), "rows": len(frame), "raw_rows": len(raw), "atomic_task_count": len(atomic),
        "domain_count": len(domains), "source_count_by_proxy": humans.groupby("proxy").size().to_dict(),
        "excluded_source_ids": sorted(excluded), "input_sha256": {**input_hashes, **audit_hashes},
        "signed_response_recomputation_audit": raw.groupby("proxy")[["stored_R0_abs_difference", "stored_R1_abs_difference"]].max().to_dict(orient="index"),
        "artifact_sha256": artifact_hashes, "no_lambda_selection": True,
        "test_files_opened": 0 if stage == "development" else len(input_hashes) // 2,
    })
    return destination


def freeze(config, config_hash):
    root = Path(config["output_root"])
    destination = root / "FROZEN_SELECTION.json"
    if destination.exists() or (root / "TEST_ACCESS_AUTHORIZATION.json").exists():
        raise FileExistsError("freeze or test authorization already exists")
    development = root / "analysis" / "development"
    for name in ARTIFACTS:
        if not (development / name).is_file():
            raise ValueError(f"development artifact missing: {name}")
    manifest = load_json(development / "run_manifest.json")
    expected = {"status": "COMPLETE", "stage": "development", "config_sha256": config_hash,
                "frozen_protocol_sha256": config["frozen_protocol_sha256"], "code_sha256": code_hashes(),
                "no_lambda_selection": True, "test_files_opened": 0}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise PermissionError("development report does not match current frozen configuration/code")
    expected_tasks = len(PROXIES) * sum(c["split"] == "development" for c in config["data_cells"]) * len(config["methods"])
    if manifest.get("atomic_task_count") != expected_tasks or manifest.get("domain_count") != 4:
        raise ValueError("development analysis is incomplete")
    verify_hash_map(manifest["input_sha256"])
    verify_hash_map(manifest["artifact_sha256"])
    required_inputs = {str(p) for _, _, path in moment_paths(config, "development") for p in (path, path.with_suffix(".manifest.json"))}
    if not required_inputs.issubset(manifest["input_sha256"]):
        raise ValueError("development moment roster is incomplete")
    bias = load_json(development / "human_bias.json")
    thresholds = load_json(development / "shared_thresholds.json")
    dev_hashes = {**manifest["input_sha256"], **manifest["artifact_sha256"], str(development / "run_manifest.json"): sha(development / "run_manifest.json")}
    write_json(destination, {
        "schema_version": "reference-shaping-freeze-v1", "frozen_at_utc": utc_now(),
        "config_sha256": config_hash, "frozen_protocol_sha256": config["frozen_protocol_sha256"],
        "code_sha256": code_hashes(), "software_versions": manifest["software_versions"],
        "dev_artifact_sha256": dev_hashes, "human_bias_by_proxy": bias["full_development_by_proxy"],
        "thresholds": thresholds["full_development_by_proxy"], "fixed_scores": SCORES,
        "fixed_lambda_values": [0, .5, 1], "selected_lambda": None, "no_lambda_selection": True,
        "test_sources_opened": 0,
    }, immutable=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("development", "freeze", "test"), required=True)
    args = parser.parse_args()
    config, config_hash = read_config(args.config)
    output = freeze(config, config_hash) if args.stage == "freeze" else analysis_run(config, config_hash, args.stage)
    print(json.dumps({"stage": args.stage, "output": str(output), "status": "COMPLETE"}), flush=True)


if __name__ == "__main__":
    main()
