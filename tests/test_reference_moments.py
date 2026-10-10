import copy
import json
import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from score_reference_moments import (
    DEVELOPMENT_DOMAINS, ENDPOINTS, GENERATORS, TEST_DOMAINS,
    assigned_cells, cross_score, endpoint_name, features,
    mixture_score, sha, validate_access,
)
from fastdetect_official_module_core import _weights
from run_review_supplement_scores import features as old_features, tag


def fixture_logits():
    generator = torch.Generator().manual_seed(20260910)
    # Distinct q/p distributions and unequal padded output sizes emulate Falcon.
    ref = 2.0 * torch.randn(23, 41, generator=generator)
    score = 1.4 * torch.randn(23, 37, generator=generator)
    observed = torch.randint(37, (23,), generator=generator)
    return ref, score, observed


def distributions(ref):
    q0 = torch.softmax(ref, -1)
    q1_full = torch.softmax(ref / 1.2, -1)
    values, indices = q1_full.sort(descending=True, dim=-1)
    remove = values.cumsum(-1) > 0.8
    keep = torch.ones_like(remove)
    keep[:, 1:] = ~remove[:, :-1]
    values = values * keep
    values = values / values.sum(-1, keepdim=True)
    q1 = torch.zeros_like(q0).scatter(-1, indices, values)
    return q0, q1


def test_four_endpoint_parity_with_existing_supplement():
    ref, score, observed = fixture_logits()
    actual = features(ref, score, observed, chunk=7)
    reference = old_features(ref, score, observed, [1.0, 1.2], [1.0, 0.8], chunk=7)
    for key in ENDPOINTS:
        for weight in ("uniform", "entropy"):
            assert actual["stats"][weight]["endpoints"][endpoint_name(*key)]["R"] == reference[tag(*key, weight)]
    assert actual["diagnostics"]["reference_score_kl_mean"] > 0


@pytest.mark.parametrize("strength", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("weight_name", ["uniform", "entropy"])
def test_mixture_matches_direct_probability_calculation(strength, weight_name):
    ref, score, observed = fixture_logits()
    actual = features(ref, score, observed)
    ref = ref[:, :score.shape[-1]]
    q0, q1 = distributions(ref)
    mixture = (1.0-strength)*q0 + strength*q1
    lp = torch.log_softmax(score, -1)
    target = lp.gather(-1, observed[:, None]).squeeze(-1)
    mean = (mixture*lp).sum(-1)
    variance = (mixture*lp.square()).sum(-1)-mean.square()
    lq = torch.log_softmax(ref, -1)
    entropy = -(lq.exp()*lq).sum(-1)/math.log(ref.shape[-1])
    weight = _weights(entropy, "uniform" if weight_name == "uniform" else "linear_entropy", 1.5)
    numerator = (weight*target).sum()-(weight*mean).sum()
    denominator = (weight.square()*variance).sum().sqrt()
    expected = float(numerator/denominator)
    result = mixture_score(actual["stats"][weight_name], strength)
    assert result == pytest.approx(expected, abs=2e-5, rel=2e-5)
    if strength in (0.0, 1.0):
        assert result == pytest.approx(actual["stats"][weight_name][f"R{int(strength)}"], abs=1e-5)


def test_crossing_uses_requested_mean_and_scale_and_class_label_is_irrelevant():
    ref, score, observed = fixture_logits()
    stats = features(ref, score, observed)
    human = {"label": 0, "watermark_method": "human", **copy.deepcopy(stats)}
    machine = {"label": 1, "watermark_method": "EXP", **copy.deepcopy(stats)}
    assert human["stats"] == machine["stats"]
    for weight in ("uniform", "entropy"):
        for center in (0, 1):
            for scale in (0, 1):
                values = stats["stats"][weight]
                assert cross_score(values, center, scale) == values[f"N{center}"]/math.sqrt(values[f"V{scale}"])
        assert [mixture_score(human["stats"][weight], a) for a in (0, .5, 1)] == [mixture_score(machine["stats"][weight], a) for a in (0, .5, 1)]


def config_fixture(tmp_path):
    pairs = [(split, domain, generator)
             for split, domains in (("development", DEVELOPMENT_DOMAINS), ("test", TEST_DOMAINS))
             for domain in sorted(domains) for generator in sorted(GENERATORS)]
    return {"output_root": str(tmp_path), "frozen_protocol_sha256": "a"*64,
            "data_cells": [{"cell": index, "dataset": domain, "generator": generator,
                            "split": split,
                            "data_path": str(tmp_path / f"never_open_{index}.json")}
                           for index, (split, domain, generator) in enumerate(pairs)]}


def test_development_does_not_open_missing_test_or_raw_files(tmp_path):
    config = config_fixture(tmp_path)
    assert validate_access(config, "development")["split"] == "development"
    cells = assigned_cells(config, "development", {"cell_modulus": 3, "cell_remainder": 2})
    assert [cell["cell"] for cell in cells] == list(range(2, 20, 3))
    assert all(cell["split"] == "development" for cell in cells)


def test_test_gate_requires_selection_authorization_and_both_hashes(tmp_path):
    config = config_fixture(tmp_path)
    with pytest.raises(PermissionError):
        validate_access(config, "test")
    selection = tmp_path / "FROZEN_SELECTION.json"
    frozen = {"frozen_protocol_sha256": "a"*64, "lambda": 0.5, "config_sha256": "b"*64,
              "code_sha256": {"scripts/score_reference_moments.py":
                               sha(ROOT / "scripts/score_reference_moments.py")}}
    selection.write_text(json.dumps(frozen))
    authorization = tmp_path / "TEST_ACCESS_AUTHORIZATION.json"
    authorization.write_text(json.dumps({"authorized": True, "frozen_protocol_sha256": "a"*64,
                                         "config_sha256": "b"*64,
                                         "frozen_selection_sha256": sha(selection)}))
    assert validate_access(config, "test", "b"*64)["frozen_selection_sha256"] == sha(selection)
    with pytest.raises(PermissionError, match="config hash mismatch"):
        validate_access(config, "test", "c"*64)
    selection.write_text(json.dumps({**frozen, "lambda": 1.0}))
    with pytest.raises(PermissionError, match="current frozen selection"):
        validate_access(config, "test", "b"*64)


def test_split_duplicate_metadata_is_rejected_without_reading_inputs(tmp_path):
    config = config_fixture(tmp_path)
    config["data_cells"][20]["dataset"] = config["data_cells"][0]["dataset"]
    with pytest.raises(ValueError, match="whitelist"):
        validate_access(config, "development")


def test_swapped_domain_is_rejected_even_when_split_counts_match(tmp_path):
    config = config_fixture(tmp_path)
    config["data_cells"][0]["split"], config["data_cells"][20]["split"] = "test", "development"
    with pytest.raises(ValueError, match="whitelist"):
        validate_access(config, "development")


@pytest.mark.parametrize("chunk", [1, 7, 128])
@pytest.mark.parametrize("mass", [.5, .8, 1.0])
def test_chunked_shaped_score_matches_independent_endpoint(chunk, mass):
    from fastdetect_official_module_chunked import compute_official_module_features_token_chunked
    from fastdetect_official_module_core import OfficialModuleConfig

    ref, score, observed = fixture_logits()
    config = OfficialModuleConfig(upper_mass=mass)
    actual = compute_official_module_features_token_chunked(
        ref[None], score[None], observed[None], config, chunk)["nucleus_entropy"]
    expected = old_features(ref, score, observed, [1.2], [mass])[tag(1.2, mass, "entropy")]
    assert actual == pytest.approx(expected, abs=1e-6, rel=1e-6)


def test_chunked_shaped_score_rejects_invalid_reference():
    from fastdetect_official_module_chunked import compute_official_module_features_token_chunked

    ref, score, observed = fixture_logits()
    ref[0, 0] = float("nan")
    with pytest.raises(ValueError, match="invalid reference probabilities"):
        compute_official_module_features_token_chunked(ref[None], score[None], observed[None])


@pytest.mark.parametrize("direction,expected", [
    ("upper", [1., 1., .5, .5, .25]),
    ("lower", [.25, .75, .75, 1., 1.]),
])
def test_human_rank_calibration_keeps_ties_and_add_one_floor(direction, expected):
    import numpy as np
    from two_sided_core import rank_pvalues

    actual = rank_pvalues([0., 0., 1.], [-1., 0., .5, 1., 2.], direction)
    np.testing.assert_array_equal(actual, expected)
