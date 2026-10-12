#!/usr/bin/env python3
"""Read-only verification of the published HeavyWater prefix decomposition."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

from io_utils import sha256 as sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/runs/heavywater_prefix"

def check(previous_metrics=None):
    frozen = json.loads((OUT / "FROZEN.json").read_text())
    for item in frozen["source_metadata"]:
        assert item["data_sha256"] == sha(ROOT / item["datafile"]), item["datafile"]
    df = pd.read_csv(OUT / "prefix_metrics.csv")
    previous_equal = None
    if previous_metrics is not None:
        old = pd.read_csv(previous_metrics)
        key = ["dataset", "mechanism", "source_id", "position"]
        assert df[key].equals(old[key])
        baseline = ["delta_to_current_baseline", "delta_current_baseline_to_raw"]
        unaffected = [c for c in df.select_dtypes("number") if c not in baseline]
        assert np.array_equal(df[unaffected].to_numpy(), old[unaffected].to_numpy(), equal_nan=True)
        previous_equal = True
    assert len(df) == 25 and len(json.loads((OUT / "SKIPS.json").read_text())) == 7
    assert df.telescoping_error.abs().max() < 1e-12
    assert df.marginal_bound_slack.min() >= -1e-7
    assert df.selected_parity_l1.max() < 1e-6
    assert df.sinkhorn_warning.isna().all() and df.float64_warning.isna().all()
    cfg = json.loads((OUT / "BASELINE_RECONSTRUCTION.json").read_text())
    assert cfg["effective_top_k_reconstruction"] == 50
    assert cfg["effective_top_p_reconstruction"] == 1
    # delta(W - baseline) + delta(baseline - raw) + delta(raw - reference)
    # must recover delta(W - reference) at every identical prefix.
    err = (df.delta_to_current_baseline + df.delta_current_baseline_to_raw
           + df.delta_prompt_context - df.delta_total_to_detector_reference).abs().max()
    assert err < 1e-12
    components = ["prompt_context", "repetition", "ngram", "nucleus",
                  "ot_uniform_numerical", "selected_state", "total_to_detector_reference",
                  "current_baseline_to_raw", "to_current_baseline"]
    records = {}
    for mechanism, group in df.groupby("mechanism"):
        records[mechanism] = {
            "prefixes": len(group),
            "sources": int(group.source_id.nunique()),
            "mean_nats": {c: float(group["delta_" + c].mean()) for c in components},
            "negative_selected_state_prefixes": int((group.delta_selected_state < 0).sum()),
            "negative_repetition_prefixes": int((group.delta_repetition < 0).sum()),
            "selected_state_min": float(group.delta_selected_state.min()),
            "selected_state_max": float(group.delta_selected_state.max()),
            "observed_next_residual_mean": float(group.observed_next_residual.mean()),
        }
    result = {
        "checks_passed": True, "script_sha256": sha(__file__),
        "source_integrity": "frozen benchmark hashes",
        "historical_code_hashes": "original execution; current code parity not checked",
        "v1_v2_unaffected_numeric_columns_exactly_equal": previous_equal,
        "previous_version_comparison": "recomputed" if previous_equal else "not rerun; earlier CSV not supplied",
        "corrected_baseline_top_k": 50,
        "source_count": int(df.source_id.nunique()),
        "continuation_count": int(df[["source_id", "mechanism"]].drop_duplicates().shape[0]),
        "group_summary": records,
        "max_abs_numerical_mean_effect_nats": float(df.delta_ot_uniform_numerical.abs().max()),
        "max_abs_selected_float64_effect_nats": float(df.selected_float64_score_difference.abs().max()),
        "max_uniform_marginal_l1": float(df.uniform_l1.max()),
        "max_selected_float64_l1": float(df.selected_float64_l1.max()),
        "scope": "Descriptive same-prefix processor effects; correlated prefixes, reconstructed tokenization, no new generation or document AUROC inference.",
        "floating_point_note": "Selected probabilities are recovered by exp(processor log output); HF sampling additionally normalizes with softmax. Column parity is below 1e-6 L1; do not interpret rounded effects as exact sampler probabilities.",
    }
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-metrics", type=Path,
                        help="Optional earlier CSV for the historical version comparison")
    args = parser.parse_args()
    print(json.dumps(check(args.previous_metrics), indent=2, ensure_ascii=False))
