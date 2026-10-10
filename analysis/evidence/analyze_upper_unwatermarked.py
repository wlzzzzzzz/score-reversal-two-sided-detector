#!/usr/bin/env python3
"""Replay the current upper branch on the fixed 60-cell unwatermarked cohort."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parents[1]
DEST = HERE / 'data/upper_unwatermarked'
ORDER = ['Fast-DetectGPT', 'AdaDetectGPT', 'NTS', 'Upper_OPT', 'Upper_Falcon', 'Ours_OPT', 'Ours_Falcon']
KEY = ['dataset', 'generator', 'watermark_method']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute():
    original = pd.read_csv(HERE / 'data/current/task_metrics.csv')
    source = original[original.watermark_method == 'unwatermarked'].copy()
    rows, diagnostics, audit = [], [], {}
    for proxy in ['OPT', 'Falcon']:
        path = HERE / 'data/sensitivity' / f'cached_{proxy}.npz'
        audit[str(path.relative_to(HERE))] = sha(path)
        with np.load(path) as z:
            cal = z['cal_upper'][:, 2]
            query = z['query_upper'][:, 2]
            assert cal.shape == (800,) and query.shape == (141600,)
            p = (1 + len(cal) - np.searchsorted(np.sort(cal), query, side='left')) / (len(cal) + 1)
            score = -np.log(p)
            lower = (1 + np.searchsorted(np.sort(z['cal_lower']), z['query_lower'], side='right')) / 801
            joint = -np.log(np.minimum(p / .5, lower / .5))
            q = pd.DataFrame({k: z[k] for k in ['row_id', 'source_id', *KEY, 'label']})
        assert np.isfinite(score).all()
        unique_humans = q[q.label == 0].drop_duplicates(['dataset', 'source_id'])
        assert len(unique_humans) == 2360
        machines = q[(q.label == 1) & (q.watermark_method == 'unwatermarked')]
        assert len(machines) == 11800
        complete = source[source.score == 'Ours_' + proxy].set_index(['dataset', 'generator'])
        n_tasks = 0
        for (ds, gen), m in machines.groupby(['dataset', 'generator'], sort=True):
            h = q[(q.label == 0) & (q.dataset == ds) & (q.generator == gen)]
            assert h.source_id.is_unique and m.source_id.is_unique and set(h.source_id) == set(m.source_id)
            ids = np.r_[h.index, m.index]
            labels = np.r_[np.zeros(len(h)), np.ones(len(m))]
            auc = roc_auc_score(labels, score[ids])
            complete_auc = roc_auc_score(labels, joint[ids])
            np.testing.assert_allclose(complete_auc, complete.loc[(ds, gen), 'auc'], atol=1e-12, rtol=0)
            rows.append(dict(score='Upper_' + proxy, dataset=ds, generator=gen, watermark_method='unwatermarked',
                             n_human=len(h), n_machine=len(m), auc=auc))
            diagnostics.append(dict(proxy=proxy, dataset=ds, generator=gen, raw_upper_auc=roc_auc_score(labels, query[ids]),
                                    calibrated_upper_auc=auc, complete_auc=complete_auc))
            n_tasks += 1
        assert n_tasks == 60
    keep = ['score', *KEY, 'n_human', 'n_machine', 'auc']
    exported = source[source.score.isin(ORDER)][keep]
    tasks = pd.concat([exported, pd.DataFrame(rows)], ignore_index=True)
    assert len(tasks) == 420 and not tasks.duplicated(['score', *KEY]).any()
    reference = tasks[tasks.score == 'Upper_OPT'].set_index(KEY)
    summaries = []
    for method in ORDER:
        block = tasks[tasks.score == method]
        aligned = block.set_index(KEY).sort_index()
        assert len(block) == 60
        np.testing.assert_array_equal(aligned[['n_human', 'n_machine']], reference.sort_index()[['n_human', 'n_machine']])
        summaries.append(dict(score=method, tasks=len(block), n_machine=int(block.n_machine.sum()),
                              unique_humans=2360, auc=float(block.auc.mean())))
    tasks = tasks.sort_values(['score', *KEY]).reset_index(drop=True)
    summary = pd.DataFrame(summaries)
    audit['data/current/task_metrics.csv'] = sha(HERE / 'data/current/task_metrics.csv')
    protocol = dict(status='PASS', score='J_plus = -log(p_plus); inclusive add-one upper ranks',
                    T=1.2, c=.8, kappa=1.5, calibration_unique_humans=800, calibration_cells=20,
                    test_cells=60, test_unique_humans=2360, test_unwatermarked_machine_texts=11800,
                    aggregation='Equal-weight mean of 60 task AUROCs', parameter_selection=False,
                    new_model_forwards=False, baseline_scores_rerun=False,
                    baseline_source='Same-cohort current task export; AdaDetectGPT uses the documented adapted configuration',
                    full_test_auc_parity_tolerance=1e-12, source_sha256=audit)
    return tasks, summary, pd.DataFrame(diagnostics), protocol


def validate():
    tasks, summary, diagnostics, protocol = compute()
    for name, frame in [('task_metrics.csv', tasks), ('summary.csv', summary), ('rank_diagnostics.csv', diagnostics)]:
        stored = pd.read_csv(DEST / name)
        pd.testing.assert_frame_equal(stored, frame, check_exact=False, atol=1e-12, rtol=0)
    assert json.loads((DEST / 'PROTOCOL.json').read_text()) == protocol
    return dict(status='PASS', tasks_per_score=60, scores=ORDER,
                upper_opt_auc=float(summary[summary.score == 'Upper_OPT'].auc.iloc[0]),
                upper_falcon_auc=float(summary[summary.score == 'Upper_Falcon'].auc.iloc[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(validate(), indent=2))
    else:
        tasks, summary, diagnostics, protocol = compute()
        DEST.mkdir(parents=True, exist_ok=True)
        for name, frame in [('task_metrics.csv', tasks), ('summary.csv', summary), ('rank_diagnostics.csv', diagnostics)]:
            frame.to_csv(DEST / name, index=False, float_format='%.17g')
        (DEST / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2) + '\n')
        print(summary.to_string(index=False))
