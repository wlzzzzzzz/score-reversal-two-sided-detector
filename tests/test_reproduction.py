"""Regression checks for paper reproduction and scorer construction."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import analyze_reference_shaping as analysis
import evaluate_detector as detector
import run_clean80_fulltext_strong_baselines as baselines


@pytest.mark.parametrize('script', ['scripts/evaluate_detector.py',
                                    'analysis/evidence/analyze_lower_branch.py',
                                    'analysis/evidence/analyze_reference_calibration.py'])
def test_paper_entrypoint_help(script):
    result = subprocess.run(
        [sys.executable, '-B', str(ROOT / script), '--help'],
        cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


def test_all_packaged_moment_paths_resolve():
    config = json.loads((ROOT / 'configs/original/reference_shaping.json').read_text())
    paths = [path for stage in ('development', 'test')
             for _, _, path in analysis.moment_paths(config, stage)]
    assert len(paths) == len(set(paths)) == 160
    assert all((ROOT / path).is_file() and (ROOT / path.with_suffix('.manifest.json')).is_file()
               for path in paths)


@pytest.mark.parametrize('name', ['../outside', '/tmp/outside', 'scratch/../../outside'])
def test_detector_rejects_paths_outside_repo(name):
    with pytest.raises(ValueError):
        detector.local(name)


def test_strong_scorer_initializes_both_models_and_witness():
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {'a': 0, 'b': 1}
    witness = object()
    with patch.object(baselines.AutoTokenizer, 'from_pretrained', return_value=tokenizer), \
            patch.object(baselines.AutoModelForCausalLM, 'from_pretrained') as load_model, \
            patch.object(baselines, 'load_witness', return_value=witness) as load_witness:
        scorer = baselines.Scorer(2048, 2, 4096)
    assert (scorer.max_tokens, scorer.batch_size, scorer.vocab_chunk_size) == (2048, 2, 4096)
    assert scorer.witness is witness
    assert load_model.call_count == 2
    assert {call.args[0] for call in load_model.call_args_list} == {baselines.OBSERVER, baselines.PERFORMER}
    load_witness.assert_called_once_with(baselines.DEFAULT_ADA_ROOT, scorer.performer_device)


def test_published_prefix_audit_runs_without_historical_workspace():
    import audit_heavywater_prefix as prefix
    report = prefix.check()
    archived = json.loads((ROOT / "results/runs/heavywater_prefix/FINAL_AUDIT.json").read_text())
    assert report["checks_passed"]
    assert report["source_count"] == archived["source_count"] == 4
    assert report["continuation_count"] == archived["continuation_count"] == 8
    assert report["v1_v2_unaffected_numeric_columns_exactly_equal"] is None
    for mechanism, group in report["group_summary"].items():
        old = archived["group_summary"][mechanism]
        assert group["prefixes"] == old["prefixes"]
        assert group["mean_nats"] == pytest.approx(old["mean_nats"], abs=1e-12)


@pytest.mark.parametrize("script,artifact", [
    ("check_asymmetric_reference.py", "REFERENCE_CHECKS.json"),
    ("check_mechanism_projection.py", "PROJECTION_CHECKS.json"),
    ("check_entropy_alignment.py", "ENTROPY_ALIGNMENT_CHECKS.json"),
])
def test_mathematical_checks_preserve_published_audits(script, artifact):
    audit = ROOT / "analysis/evidence/checks" / artifact
    before = audit.read_bytes()
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "analysis/evidence" / script)],
        cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "PASS"
    assert audit.read_bytes() == before


def test_reference_calibration_preserves_published_tables():
    tables = {p: p.read_bytes() for p in (ROOT / 'analysis/tables').glob('*.tex')}
    (ROOT / 'scratch').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as work:
        result = subprocess.run(
            [sys.executable, '-B', 'analysis/evidence/analyze_reference_calibration.py',
             '--output', str(Path(work) / 'calibration')], cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
                     OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2'),
            capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    assert all(path.read_bytes() == content for path, content in tables.items())


def test_table_replay_includes_every_paper_baseline(tmp_path):
    result = subprocess.run(
        [sys.executable, '-B', 'analysis/evidence/render_current_tables.py',
         '--output-dir', str(tmp_path)], cwd=ROOT,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    for name, columns in [('current_generators.tex', 5), ('current_datasets.tex', 12)]:
        rows = [line for line in (tmp_path / name).read_text().splitlines()
                if ' & ' in line and not line.startswith('Detector')]
        assert len(rows) == 10
        assert all(len(row.split('&')) == columns + 1 for row in rows)
        for method in ['Likelihood', 'Entropy', 'Rank', 'Log-Rank', 'Fast-DetectGPT',
                       'AdaDetectGPT', 'Binoculars', 'NTS', 'Ours (OPT)', 'Ours (Falcon pair)']:
            assert any(method in row.split('&')[0] for row in rows)
