"""Synthetic-only checks; this suite never opens the real held-out dataset."""
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_reference_shaping.py"
SPEC = importlib.util.spec_from_file_location("shaping_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def stats(n0=2., n1=1., v0=1., v1=2., c01=.2):
    e = {"N0": n0, "N1": n1, "V0": v0, "V1": v1, "C01": c01,
         "R0": n0 / math.sqrt(max(v0, 1e-18)), "R1": n1 / math.sqrt(max(v1, 1e-18)),
         "R_T1p2_P1": 3., "R_T1_P0p8": -4., "neff": 20., "weighted_entropy": .2}
    return {"entropy": e, "uniform": dict(e)}


def synthetic_frame():
    records = []
    for pi, proxy in enumerate(analysis.PROXIES):
        for di, domain in enumerate(("c4", "squad", "writing", "xsum")):
            for generator in ("a_generator", "z_generator"):
                for method in ("human", "a", "b"):
                    for i in range(12):
                        n0 = .3 + .05 * di + .03 * pi + .08 * math.sin(i)
                        if method == "a":
                            n0 += .12 + di * .015
                        if method == "b":
                            n0 -= .3 - di * .02
                        shift = .035 * (1 + di) + .003 * i + (.008 if method == "a" else 0.)
                        s = stats(n0, n0 - shift, .1 + .002 * i, (.1 + .002 * i) * (1.2 + .04 * di + .02 * i), .005 + .0001 * i)
                        source_id = f"{domain}:{i}"
                        row = {"proxy": proxy, "dataset": domain, "generator": generator,
                               "watermark_method": method, "label": int(method != "human"),
                               "source_id": source_id, "source_index": i, "text_sha256": f"text:{source_id}:{method}",
                               "token_length": 25 + i, "split": "development", "half": analysis.source_half(source_id),
                               **analysis.derive_scores(s)}
                        row.update({key: s["entropy"][key] for key in ("N0", "N1", "V0", "V1", "C01", "neff", "weighted_entropy")})
                        records.append(row)
    return pd.DataFrame(records)


def write_synthetic_moments(config, frame, stage, config_hash):
    root = Path(config["output_root"])
    for proxy, cell, path in analysis.moment_paths(config, stage):
        path.parent.mkdir(parents=True, exist_ok=True)
        selected = frame.loc[frame.proxy.eq(proxy) & frame.dataset.eq(cell["dataset"]) & frame.generator.eq(cell["generator"])]
        with path.open("w") as handle:
            for _, row in selected.iterrows():
                record = {key: row[key] for key in ("source_id", "source_index", "dataset", "generator", "proxy", "watermark_method", "label", "text_sha256", "token_length")}
                record["split"] = stage
                record["stats"] = stats(row.N0, row.N1, row.V0, row.V1, row.C01)
                handle.write(json.dumps(analysis.clean_json(record)) + "\n")
        access = {"split": stage, "frozen_protocol_sha256": config["frozen_protocol_sha256"]}
        if stage == "test":
            access.update(config_sha256=config_hash,
                          frozen_selection_sha256=analysis.sha(root / "FROZEN_SELECTION.json"),
                          test_authorization_sha256=analysis.sha(root / "TEST_ACCESS_AUTHORIZATION.json"))
        analysis.write_json(path.with_suffix(".manifest.json"), {
            "config_sha256": config_hash, "script_sha256": analysis.sha(analysis.REPO / analysis.CODE_FILES[1]),
            "input_sha256": cell["sha256"], "proxy": proxy, "smoke_rows": 0,
            "access": access, "sha256": analysis.sha(path), "rows": len(selected),
        })


class FormulaTests(unittest.TestCase):
    def test_mixture_uses_token_between_mean_variance_even_when_document_shift_cancels(self):
        # w=(.2,.8), delta_mu=(1,-.25): weighted center shift=0,
        # but sum(w^2*delta_mu^2)=.08, which MUST remain in mixture V.
        values = analysis.derive_scores(stats(2.8, 2.8, .68, .68, .08))
        self.assertAlmostEqual(values["E_mix_half"], 2.8 ** 2 / .70)
        self.assertNotEqual(values["E_mix_half"], values["E00"])

    def test_signed_and_energy_and_uniform_endpoints(self):
        values = analysis.derive_scores(stats(-2., 1., 4., 9., 1.))
        self.assertEqual(tuple(values), analysis.SCORES)
        self.assertEqual(values["R0"], -1.)
        self.assertEqual(values["E00"], 1.)
        self.assertAlmostEqual(values["E10"], .25)
        self.assertAlmostEqual(values["E01"], 4 / 9)
        self.assertEqual(values["T_only"], 9.)
        self.assertEqual(values["C_only"], 16.)

    def test_floor_and_nonfinite_input_rejected(self):
        self.assertTrue(math.isclose(analysis.derive_scores(stats(1., 0., 0., 0., 0.))["E00"], 1e18, rel_tol=1e-15))
        with self.assertRaises(ValueError):
            analysis.derive_scores(stats(v0=-1.))
        with self.assertRaises(ValueError):
            analysis.derive_scores(stats(n0=math.nan))

    def test_roc_ties_no_interpolation(self):
        metrics = analysis.roc_metrics(np.array([0., 2.]), np.array([1., 2.]))
        self.assertEqual(metrics["tpr_at_1pct_fpr"], 0.)
        self.assertEqual(metrics["tpr_at_5pct_fpr"], 0.)
        self.assertAlmostEqual(metrics["auc"], .625)

    def test_constant_variance_rescaling_cannot_change_auc(self):
        human = np.array([0., 2., 3.])
        machine = np.array([1., 2., 4.])
        self.assertEqual(analysis.roc_metrics(human, machine), analysis.roc_metrics(human / 19, machine / 19))

    def test_threshold_order_statistic_and_strict_comparison(self):
        spec = analysis.threshold_spec(np.arange(99), .01)
        self.assertEqual(spec["k"], 99)
        self.assertEqual(spec["value"], 98.)
        self.assertFalse(98. > analysis.threshold_value(spec))
        self.assertTrue(99. > analysis.threshold_value(spec))
        self.assertTrue(math.isinf(analysis.threshold_value(analysis.threshold_spec([0, 1], .01))))

    def test_cvar_uses_ceil_not_interpolated_tail(self):
        values = np.arange(11)[:, None] / 10
        result = analysis.array_summary(values)
        self.assertAlmostEqual(result[0, 2], .05)


class GroupAndFeatureTests(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_frame()

    def test_canonical_human_dedup_and_lodo_bias(self):
        humans = analysis.canonical_humans(self.frame)
        self.assertEqual(len(humans), 2 * 4 * 12)
        self.assertEqual(set(humans.generator), {"a_generator"})
        bias = analysis.estimate_bias(humans)
        for proxy in analysis.PROXIES:
            for domain in humans.dataset.unique():
                expected = humans.loc[humans.proxy.eq(proxy) & humans.dataset.ne(domain)].groupby("dataset").N0.mean().mean()
                self.assertAlmostEqual(bias["lodo_by_proxy"][proxy][domain], expected)
        altered = humans.copy()
        altered.loc[altered.dataset.eq("c4"), "N0"] = 999.
        changed = analysis.estimate_bias(altered)
        self.assertEqual(changed["lodo_by_proxy"]["OPT"]["c4"], bias["lodo_by_proxy"]["OPT"]["c4"])

    def test_bias_never_changes_detector_score(self):
        bias = analysis.estimate_bias(analysis.canonical_humans(self.frame))
        attached = analysis.attach_features(self.frame, bias, "development")
        np.testing.assert_array_equal(attached[list(analysis.SCORES)], self.frame[list(analysis.SCORES)])
        row = attached.iloc[0]
        self.assertAlmostEqual(row.G, (2 * row.analysis_bias * (row.N0 - row.N1) - (row.N0 - row.N1) ** 2) / row.V0)

    def test_atomic_keeps_generators_methods_and_source_halves(self):
        bias = analysis.estimate_bias(analysis.canonical_humans(self.frame))
        frame = analysis.attach_features(self.frame, bias, "development")
        atomic, halves = analysis.build_atomic(frame, ["a", "b"])
        self.assertEqual(len(atomic), 2 * 4 * 2 * 2)
        self.assertEqual(len(halves), 2 * len(atomic))
        self.assertTrue((halves.feature_human_count + halves.outcome_human_count).eq(12).all())
        self.assertTrue((halves.feature_half + halves.outcome_half).eq(1).all())
        self.assertAlmostEqual(atomic.iloc[0].delta_center, atomic.iloc[0].E10_auc - atomic.iloc[0].E00_auc)

    def test_source_exclusion_removes_entire_bundle(self):
        filtered = self.frame.loc[self.frame.source_id.ne("c4:0")]
        self.assertEqual(len(self.frame) - len(filtered), 2 * 2 * 3)
        bias = analysis.estimate_bias(analysis.canonical_humans(self.frame))
        atomic, _ = analysis.build_atomic(analysis.attach_features(filtered, bias, "development"), ["a", "b"])
        self.assertTrue(atomic.human_count.eq(atomic.machine_count).all())
        self.assertTrue(atomic.loc[atomic.dataset.eq("c4"), "human_count"].eq(11).all())

    def test_h2_empty_or_fewer_domains_is_inconclusive(self):
        frame = pd.DataFrame({"dataset": ["a", "b", "c"], "G_task": [-1., -2., -3.], "H_scale": [1., 2., 3.], "delta_scale_given_center": [3., 2., 1.]})
        self.assertIsNone(analysis.association(frame, "H_scale", "delta_scale_given_center", True)["rho"])
        frame["G_task"] = [1., 1., -1.]
        self.assertIsNone(analysis.association(frame, "H_scale", "delta_scale_given_center", True)["rho"])
        self.assertAlmostEqual(analysis.association(frame, "H_scale", "delta_scale_given_center")["rho"], -1.)

    def test_joint_domain_bootstrap_and_analysis_outputs(self):
        bias = analysis.estimate_bias(analysis.canonical_humans(self.frame))
        frame = analysis.attach_features(self.frame, bias, "development")
        atomic, halves = analysis.build_atomic(frame, ["a", "b"])
        domains = sorted(frame.dataset.unique())
        draws = analysis.bootstrap_domain_indices(domains, draws=20)
        np.testing.assert_array_equal(draws, analysis.bootstrap_domain_indices(domains, draws=20))
        summary, by_domain = analysis.task_summary(atomic, domains, draws)
        self.assertEqual(len(summary), 2 * len(analysis.SCORES))
        self.assertEqual(len(by_domain), 2 * 4 * len(analysis.SCORES))
        base = summary.loc[summary.score.eq("E00")]
        self.assertTrue(base.delta_vs_E00_mean_auc.eq(0).all())
        self.assertTrue(base.delta_vs_E00_mean_auc_ci95_low.eq(0).all())
        result = analysis.hypothesis_results(atomic, halves, domains, draws, "development")
        self.assertEqual(result["H1"]["bootstrap_draws"], 20)
        self.assertTrue(result["no_lambda_selection"])
        json.dumps(analysis.clean_json(result), allow_nan=False)

    def test_shared_operating_points_count_unique_humans(self):
        humans = analysis.canonical_humans(self.frame)
        thresholds = analysis.calibrate_thresholds(humans)
        points = analysis.shared_operating_points(self.frame, humans, thresholds, "development")
        pooled = points.loc[points.dataset.eq("POOLED")]
        self.assertTrue(pooled.unique_human_count.eq(48).all())
        self.assertTrue(pooled.task_count.eq(4 * 2 * 2).all())


class AccessTests(unittest.TestCase):
    def test_no_test_moment_open_when_gate_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"output_root": directory}
            with patch.object(analysis, "load_json", side_effect=AssertionError("no JSON may be opened before absent gate check")):
                with self.assertRaises(PermissionError):
                    analysis.load_moments(config, "digest", "test")

    def test_forged_authorization_rejected_before_artifact_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"output_root": directory, "frozen_protocol_sha256": "protocol"}
            frozen = {"config_sha256": "config", "frozen_protocol_sha256": "protocol"}
            analysis.write_json(root / "FROZEN_SELECTION.json", frozen)
            analysis.write_json(root / "TEST_ACCESS_AUTHORIZATION.json", {**frozen, "authorized": True, "frozen_selection_sha256": "forged"})
            with patch.object(analysis, "verify_hash_map", side_effect=AssertionError("must not inspect artifacts for forged gate")):
                with self.assertRaises(PermissionError):
                    analysis.validate_test_gate(config, "config")

    def test_freeze_cannot_proceed_without_complete_development(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(analysis, "load_moments", side_effect=AssertionError("freeze never loads test moments")):
                with self.assertRaises(ValueError):
                    analysis.freeze({"output_root": directory}, "digest")

    def test_synthetic_development_freeze_test_lifecycle(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_shaping_analysis_") as directory:
            root = Path(directory)
            frame = synthetic_frame()
            test = frame.copy()
            mapping = {name: f"synthetic_test_{i}" for i, name in enumerate(sorted(frame.dataset.unique()))}
            test["dataset"] = test.dataset.map(mapping)
            test["source_id"] = test.dataset + ":" + test.source_index.astype(str)
            test["text_sha256"] = "synthetic:" + test.source_id + ":" + test.watermark_method
            config = {"output_root": str(root), "frozen_protocol_sha256": "synthetic-protocol",
                      "methods": ["a", "b"], "analysis": {"bootstrap_draws": 10, "bootstrap_seed": 20260910},
                      "data_cells": []}
            for stage, values in (("development", frame), ("test", test)):
                for (domain, generator), _ in values.groupby(["dataset", "generator"]):
                    config["data_cells"].append({"dataset": domain, "generator": generator, "split": stage,
                                                 "source_count": 12, "sha256": "synthetic-input-hash"})
            config_hash = "synthetic-config"
            analysis.write_json(root / "development_source_manifest.json", {
                "status": "PASS", "config_sha256": config_hash, "test_sources_opened": 0})
            write_synthetic_moments(config, frame, "development", config_hash)
            development = analysis.analysis_run(config, config_hash, "development")
            self.assertTrue(all((development / name).is_file() for name in analysis.ARTIFACTS))
            frozen_path = analysis.freeze(config, config_hash)
            frozen = analysis.load_json(frozen_path)
            self.assertIsNone(frozen["selected_lambda"])
            with self.assertRaises(PermissionError):
                analysis.validate_test_gate(config, config_hash)
            analysis.write_json(root / "TEST_ACCESS_AUTHORIZATION.json", {
                "authorized": True, "config_sha256": config_hash,
                "frozen_protocol_sha256": config["frozen_protocol_sha256"],
                "frozen_selection_sha256": analysis.sha(frozen_path),
            })
            analysis.validate_test_gate(config, config_hash)
            write_synthetic_moments(config, test, "test", config_hash)
            analysis.write_json(root / "test_source_audit.json", {
                "status": "PASS", "split": "test", "config_sha256": config_hash,
                "frozen_protocol_sha256": config["frozen_protocol_sha256"],
                "frozen_selection_sha256": analysis.sha(frozen_path),
                "test_access_authorization_sha256": analysis.sha(root / "TEST_ACCESS_AUTHORIZATION.json"),
                "development_source_manifest_sha256": analysis.sha(root / "development_source_manifest.json"),
                "excluded_source_ids": ["synthetic_test_0:0"],
            })
            destination = analysis.analysis_run(config, config_hash, "test")
            manifest = analysis.load_json(destination / "run_manifest.json")
            self.assertEqual(manifest["excluded_source_ids"], ["synthetic_test_0:0"])
            self.assertEqual(manifest["raw_rows"] - manifest["rows"], 2 * 2 * 3)
            self.assertTrue((destination / "unfiltered_sensitivity_summary.csv").is_file())
            self.assertEqual(analysis.load_json(destination / "human_bias.json")["full_development_by_proxy"], frozen["human_bias_by_proxy"])
            with self.assertRaises(FileExistsError):
                analysis.analysis_run(config, config_hash, "test")
            # Tampering with a frozen development artifact blocks subsequent access.
            with (development / "REPORT.md").open("a") as handle:
                handle.write("changed")
            with self.assertRaises(PermissionError):
                analysis.validate_test_gate(config, config_hash)


if __name__ == "__main__":
    unittest.main()
