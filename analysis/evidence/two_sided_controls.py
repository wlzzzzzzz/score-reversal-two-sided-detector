#!/usr/bin/env python3
"""Validate portable same-score controls and render their manuscript tables."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / 'data/two_sided_controls'
KEY = ['dataset', 'generator', 'watermark_method']


def read(name, file):
    return pd.read_csv(DATA / name / (file + '.csv'))


def summary_row(name, proxy, method):
    table = read(name, 'summary')
    return table[(table.proxy == proxy) & (table.method == method)].iloc[0]


def validate():
    provenance = json.loads((DATA / 'SOURCE_AUDIT.json').read_text())
    assert provenance['status'] == 'PASS'
    ledger = {r['path']: r['sha256'] for r in json.loads(
        (HERE.parent / 'manifest.json').read_text())['files']}
    for source, digest in provenance['source_sha256'].items():
        name = 'fd' if '/fd_two_tail_' in source else 'ada'
        copied = DATA / name / Path(source).name
        current = hashlib.sha256(copied.read_bytes()).hexdigest()
        assert current == ledger[copied.relative_to(HERE.parent).as_posix()], str(copied)
        # Historical JSON receipts predate portable path/metadata edits.
        # Numerical CSV exports must also match their execution-time digests.
        if copied.suffix == '.csv':
            assert current == digest, str(copied)
    expected = pd.read_csv(HERE / 'data/current/task_metrics.csv')
    deploy = pd.read_csv(HERE / 'data/current/deployment.csv')
    checks = []
    for name, seed in [('fd', 20260916), ('ada', 20260917)]:
        assert json.loads((DATA / name / 'AUDIT.json').read_text())['status'] == 'PASS'
        p = json.loads((DATA / name / 'PROTOCOL.json').read_text())
        assert p['beta'] == .5 and p['alphas'] == [.01, .05]
        assert p.get('prior_test_exposure', p.get('data_previously_inspected'))
        if name == 'fd':
            assert p['model_forwards'] == 0 and not p['parameter_selection']
        else:
            assert p['new_model_forwards'] == 0 and not p['parameter_search']
        tasks = read(name, 'by_task')
        summary = read(name, 'summary')
        for r in summary.itertuples():
            t = tasks[(tasks.proxy == r.proxy) & (tasks.method == r.method)]
            assert len(t) == r.tasks == 660
            assert not t.duplicated(KEY).any()
            assert (t.dataset.nunique(), t.generator.nunique(), t.watermark_method.nunique()) == (12, 5, 11)
            assert t.n_machine.sum() == r.test_machines == 129800
            assert r.calibration_humans == 800 and r.test_unique_humans == 2360
            np.testing.assert_allclose(t.auc.mean(), r.auc, atol=1e-12, rtol=0)
            np.testing.assert_allclose(t[t.watermark_method == 'unwatermarked'].auc.mean(), r.unwm_auc, atol=1e-12, rtol=0)
            for tag in ['a01', 'a05']:
                np.testing.assert_allclose(t['power_' + tag].mean(), getattr(r, 'power_' + tag), atol=1e-12, rtol=0)
                np.testing.assert_allclose(getattr(r, 'human_fp_' + tag) / 2360, getattr(r, 'human_fpr_' + tag), atol=1e-12, rtol=0)
            if r.method in ['Current_asymmetric', 'Ada_original_direction']:
                score = 'Ours_' + r.proxy if r.method == 'Current_asymmetric' else 'AdaDetectGPT'
                merged = t.merge(expected[expected.score == score], on=KEY, validate='one_to_one')
                assert len(merged) == 660
                np.testing.assert_allclose(merged.auc_x, merged.auc_y, atol=1e-12, rtol=0)
            if r.method == 'Current_asymmetric':
                for alpha, tag in [(.01, 'a01'), (.05, 'a05')]:
                    d = deploy[(deploy.proxy == r.proxy) & (deploy.alpha == alpha)].iloc[0]
                    np.testing.assert_allclose([getattr(r, 'power_' + tag), getattr(r, 'human_fpr_' + tag)], [d.power, d.fpr], atol=1e-12, rtol=0)
        draws = np.random.RandomState(seed).randint(0, 12, size=(10000, 12))
        intervals = read(name, 'paired_domain_intervals')
        for r in intervals.itertuples():
            t = tasks if name == 'ada' else tasks[tasks.proxy == r.proxy]
            if r.cohort != 'all':
                t = t[t.watermark_method == r.cohort]
            means = t.groupby(['dataset', 'method'])[r.metric].mean().unstack().sort_index()
            if name == 'ada':
                left, right = r.comparison.split(' minus ')
            else:
                left, right = 'Current_asymmetric', 'FD_same_score_two_tails'
            values = (means[left] - means[right]).to_numpy()
            lo, hi = np.quantile(values[draws].mean(1), [.025, .975])
            np.testing.assert_allclose([values.mean(), lo, hi], [r.delta, r.ci_low, r.ci_high], atol=1e-12, rtol=0)
        checks.append(dict(source=name, task_rows=len(tasks), configurations=len(summary), paired_intervals=len(intervals)))
    # The overlapping Falcon controls must be identical across the two exports.
    a = read('ada', 'by_task')
    f = read('fd', 'by_task')
    methods = ['FD_same_score_two_tails', 'Current_asymmetric']
    a = a[a.method.isin(methods)].set_index(['proxy', 'method'] + KEY).sort_index()
    f = f[(f.proxy == 'Falcon') & f.method.isin(methods)].set_index(['proxy', 'method'] + KEY).sort_index()
    pd.testing.assert_frame_equal(a, f, check_exact=False, atol=1e-12, rtol=0)
    return dict(status='PASS', source_integrity='current release ledger',
                historical_numerical_exports_unchanged=True,
                source_checks=checks, same_falcon_controls=True,
                original_paper_results_unchanged=True, paired_interval_replay=True)


def tabular(align, header, rows):
    return '\n'.join([r'\begin{tabular}{@{}' + align + r'@{}}', r'\toprule',
                       header + r' \\', r'\midrule', *rows, r'\bottomrule', r'\end{tabular}']) + '\n'


def render():
    rows = []
    choices = [('fd', 'OPT', 'FD_same_score_two_tails', 'FD + two tails (OPT)'),
               ('fd', 'OPT', 'Current_asymmetric', 'Ours (OPT)'),
               ('fd', 'Falcon', 'FD_same_score_two_tails', 'FD + two tails (Falcon)'),
               ('ada', 'Falcon', 'Ada_same_score_two_tails', r'Ada + two tails (Falcon)$^\dagger$'),
               ('fd', 'Falcon', 'Current_asymmetric', 'Ours (Falcon pair)')]
    for i, (name, proxy, method, label) in enumerate(choices):
        if i == 2:
            rows.append(r'\midrule')
        r = summary_row(name, proxy, method)
        values = [f'{r.auc:.4f}'] + [f'{100 * r[k]:.2f}' for k in ['power_a01', 'human_fpr_a01', 'power_a05', 'human_fpr_a05']]
        rows.append(label + ' & ' + ' & '.join(values) + r' \\')
    tables = {'two_tail_main.tex': tabular('lrrrrr',
        'Configuration' + r' & AUROC & Power$_1$ & FPR$_1$ & Power$_5$ & FPR$_5$', rows)}
    rows = []
    for name, proxy, comparator in [('fd', 'OPT', 'FD'), ('fd', 'Falcon', 'FD'), ('ada', 'Falcon', 'Ada')]:
        ci = read(name, 'paired_domain_intervals')
        if name == 'fd':
            ci = ci[(ci.proxy == proxy) & (ci.cohort == 'all')]
        else:
            ci = ci[(ci.cohort == 'all') & ci.comparison.str.startswith('Current_asymmetric')]
        label = proxy + ' / ' + comparator + (r'$^\dagger$' if comparator == 'Ada' else '')
        vals = []
        for metric, scale in [('auc', 1), ('power_a01', 100)]:
            r = ci[ci.metric == metric].iloc[0]
            precision = 4 if metric == 'auc' else 2
            vals.append(f'{scale*r.delta:+.{precision}f} [{scale*r.ci_low:.{precision}f}, {scale*r.ci_high:.{precision}f}]')
        rows.append(label + ' & ' + ' & '.join(vals) + r' \\')
    tables['two_tail_intervals.tex'] = tabular('lrr',
        'Proxy / control' + r' & $\Delta$AUROC [95\% CI] & $\Delta$Power$_1$ [95\% CI]', rows)
    rows = []
    order = ['unwatermarked', 'KGW', 'SWEET', 'Unigram', 'EXP', 'SynthID', 'DIP', 'ITSEdit', 'CGW', 'HeavyWater', 'SemStamp']
    for wm in order:
        values = []
        for name, proxy, method, _ in choices:
            t = read(name, 'by_task')
            values.append(t[(t.proxy == proxy) & (t.method == method) & (t.watermark_method == wm)].auc.mean())
        label = 'Unwatermarked' if wm == 'unwatermarked' else wm
        rows.append(label + ' & ' + ' & '.join(f'{v:.4f}' for v in values) + r' \\')
    tables['two_tail_mechanisms.tex'] = tabular('lrrrrr',
        r'Condition & FD OPT & Ours OPT & FD Falcon & Ada Falcon$^\dagger$ & Ours Falcon', rows)
    from table_emphasis import format_table
    tables = {name: format_table(name, text) for name, text in tables.items()}
    return tables


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    audit = validate()
    for name, text in render().items():
        path = HERE / 'tables' / name
        if args.write:
            path.write_text(text)
        else:
            assert path.read_text() == text, str(path)
    print(json.dumps(audit, indent=2))
