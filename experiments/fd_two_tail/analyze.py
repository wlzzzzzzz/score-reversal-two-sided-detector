#!/usr/bin/env python3
"""Matched single-score Fast-DetectGPT two-tail control; no model forwards or tuning."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'analysis'
OUT = ROOT / 'results/runs/fd_control'
BASE = ROOT / 'results/runs/calibrated_scores'
sys.path.insert(0, str(PAPER / 'evidence'))
from analyze_current_supplement import rank, tasks, human_indices, FIELDS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, data):
    pd.DataFrame(data).to_csv(OUT / (name + '.csv'), index=False, float_format='%.17g')


def main():
    OUT.mkdir(exist_ok=False)
    source = {}
    task_rows, summary_rows, transition_rows, audits = [], [], [], []
    old_tasks = pd.read_csv(PAPER / 'data/current/task_metrics.csv')
    old_deploy = pd.read_csv(PAPER / 'data/current/deployment.csv')
    source['data/current/task_metrics.csv'] = sha(PAPER / 'data/current/task_metrics.csv')
    source['data/current/deployment.csv'] = sha(PAPER / 'data/current/deployment.csv')
    # Independent rank check includes ties, empirical-tail asymmetry and the add-one correction.
    bank_check = np.array([-3., -1., -1., 2., 6.])
    query_check = np.array([-4., -1., 0., 2., 7.])
    for upper in [True, False]:
        comparison = bank_check[None, :] >= query_check[:, None] if upper else bank_check[None, :] <= query_check[:, None]
        np.testing.assert_array_equal(rank(bank_check, query_check, upper), (1 + comparison.sum(1)) / 6.)
    for proxy in ['OPT', 'Falcon']:
        path = PAPER / 'data/sensitivity' / ('cached_' + proxy + '.npz')
        source[str(path.relative_to(PAPER))] = sha(path)
        with np.load(path) as z:
            a = {k: z[k] for k in z.files}
        q = pd.DataFrame({k: a[k] for k in FIELDS})
        hi = human_indices(q)
        machine = np.flatnonzero(q.label.to_numpy() == 1)
        assert len(hi) == 2360 and len(machine) == 129800
        bank_path = BASE / ('calibration_' + proxy + '.pkl')
        bank = pd.read_pickle(bank_path)
        source[str(bank_path.relative_to(ROOT))] = sha(bank_path)
        assert len(bank) == 800 and bank.source_id.is_unique
        assert not set(bank.source_id) & set(q.source_id)
        rbank = bank.uniform_N0.to_numpy() / np.maximum(np.sqrt(bank.uniform_V0.to_numpy()), 1e-9)
        np.testing.assert_allclose(np.sort(rbank), np.sort(a['cal_lower']), rtol=0, atol=1e-12)
        lower = rank(a['cal_lower'], a['query_lower'], False)
        upper_fd = rank(a['cal_lower'], a['query_lower'], True)
        upper_ours = rank(a['cal_upper'][:, 2], a['query_upper'][:, 2], True)
        settings = {
            'FD_original_direction': (a['query_lower'], upper_fd, False),
            'FD_same_score_two_tails': (-np.log(np.minimum(upper_fd / .5, lower / .5)), upper_fd, True),
            'Current_asymmetric': (-np.log(np.minimum(upper_ours / .5, lower / .5)), upper_ours, True),
        }
        # Analytical response relation: both matched methods have the identical lower branch.
        masks = {}
        raw_fd_differences = []
        for method, (score, upper, two_tails) in settings.items():
            assert np.isfinite(score).all()
            for alpha, tag in [(.01, 'a01'), (.05, 'a05')]:
                if two_tails:
                    reject = (upper <= alpha / 2) | (lower <= alpha / 2)
                    np.testing.assert_array_equal(reject, score >= -np.log(alpha))
                else:
                    reject = upper <= alpha
                masks[(method, tag)] = reject
            block_rows = []
            for ds, gen, wm, hh, mm in tasks(q):
                assert len(hh) == len(mm)
                labels = np.r_[np.zeros(len(hh)), np.ones(len(mm))]
                auc = roc_auc_score(labels, np.r_[score[hh], score[mm]])
                row = dict(proxy=proxy, method=method, dataset=ds, generator=gen,
                           watermark_method=wm, n_human=len(hh), n_machine=len(mm), auc=auc)
                for tag in ['a01', 'a05']:
                    row['power_' + tag] = masks[(method, tag)][mm].mean()
                old = old_tasks[(old_tasks.dataset == ds) & (old_tasks.generator == gen) &
                                (old_tasks.watermark_method == wm)]
                if method == 'Current_asymmetric':
                    ref = old[old.score == 'Ours_' + proxy].auc.iloc[0]
                    assert abs(auc - ref) < 1e-12
                elif method == 'FD_original_direction' and proxy == 'Falcon':
                    raw_fd_differences.append(auc - old[old.score == 'Fast-DetectGPT'].auc.iloc[0])
                block_rows.append(row)
            frame = pd.DataFrame(block_rows)
            assert len(frame) == 660
            r = dict(proxy=proxy, method=method, tasks=660, calibration_humans=800,
                     test_unique_humans=2360, test_machines=129800, auc=frame.auc.mean(),
                     unwm_auc=frame[frame.watermark_method == 'unwatermarked'].auc.mean(),
                     wm_auc=frame[frame.watermark_method != 'unwatermarked'].auc.mean(),
                     reversed_tasks=int((frame.auc < .5).sum()))
            for alpha, tag in [(.01, 'a01'), (.05, 'a05')]:
                reject = masks[(method, tag)]
                r['power_' + tag] = frame['power_' + tag].mean()
                r['pooled_power_' + tag] = reject[machine].mean()
                r['human_fp_' + tag] = int(reject[hi].sum())
                r['human_fpr_' + tag] = reject[hi].mean()
                if method == 'Current_asymmetric':
                    ref = old_deploy[(old_deploy.proxy == proxy) & (old_deploy.alpha == alpha)].iloc[0]
                    np.testing.assert_allclose([r['power_' + tag], r['human_fpr_' + tag]],
                                               [ref.power, ref.fpr], atol=1e-12, rtol=0)
            task_rows.extend(block_rows)
            summary_rows.append(r)
        for alpha, tag in [(.01, 'a01'), (.05, 'a05')]:
            fd = masks[('FD_same_score_two_tails', tag)]
            ours = masks[('Current_asymmetric', tag)]
            for label, ids in [('human', hi), ('machine', machine)]:
                gained = int(((~fd) & ours)[ids].sum())
                lost = int((fd & (~ours))[ids].sum())
                assert int(ours[ids].sum() - fd[ids].sum()) == gained - lost
                transition_rows.append(dict(proxy=proxy, alpha=alpha, label=label, n=len(ids),
                                            fd_rejects=int(fd[ids].sum()), current_rejects=int(ours[ids].sum()),
                                            gained=gained, lost=lost))
        audit = dict(proxy=proxy, current_task_auc_and_threshold_parity=True,
                     disjoint_calibration_test_sources=True, identical_lower=True)
        if raw_fd_differences:
            audit['falcon_raw_mean_delta_from_paper_baseline'] = float(np.mean(raw_fd_differences))
            audit['falcon_raw_max_task_delta_from_paper_baseline'] = float(np.max(np.abs(raw_fd_differences)))
            assert abs(audit['falcon_raw_mean_delta_from_paper_baseline']) < 1e-6
            assert audit['falcon_raw_max_task_delta_from_paper_baseline'] < 3e-5
        audits.append(audit)
    t = pd.DataFrame(task_rows)
    s = pd.DataFrame(summary_rows)
    save('by_task', t)
    save('summary', s)
    save('by_mechanism', t.groupby(['proxy', 'method', 'watermark_method'])[['auc', 'power_a01', 'power_a05']].mean().reset_index())
    save('decision_transitions', transition_rows)
    draws = np.random.RandomState(20260916).randint(0, 12, size=(10000, 12))
    intervals = []
    for proxy in ['OPT', 'Falcon']:
        part = t[t.proxy == proxy]
        for cohort in ['all', 'unwatermarked']:
            block = part if cohort == 'all' else part[part.watermark_method == cohort]
            for metric in ['auc', 'power_a01', 'power_a05']:
                means = block.groupby(['dataset', 'method'])[metric].mean().unstack().sort_index()
                diff = (means.Current_asymmetric - means.FD_same_score_two_tails).to_numpy()
                lo, high = np.quantile(diff[draws].mean(1), [.025, .975])
                intervals.append(dict(proxy=proxy, cohort=cohort, metric=metric,
                                      delta=float(diff.mean()), ci_low=float(lo), ci_high=float(high)))
    save('paired_domain_intervals', intervals)
    protocol = dict(status='PASS', experiment='matched Fast-DetectGPT same-score two-tail control',
                    calibration='20-cell; 800 unique humans', test='60-cell; 660 tasks per proxy',
                    data_previously_inspected=True, parameter_selection=False, model_forwards=0,
                    beta=.5, alphas=[.01, .05], ranks='inclusive add-one empirical human tails',
                    two_tail_score='-log(min(p_upper / beta, p_lower / (1-beta))); no score clipping',
                    fd_score='original-reference uniform discrepancy R_0^U',
                    opt_model='OPT-2.7B for both reference and scoring',
                    falcon_models='Falcon-7B reference; Falcon-7B-Instruct scoring',
                    current_params=dict(T=1.2, c=.8, kappa=1.5),
                    aggregation='task-macro AUROC/power; unique-human FPR; pooled power separately exported',
                    intervals='10000 paired whole-domain resamples of 12 domains, conditional on fixed rules and bank',
                    raw_baseline_note='Matched moment-cache implementation; Falcon raw AUROC differs slightly from the paper baseline due to floating-point/canonical-cache evaluation. Original-direction score uses raw discrepancy, fixed thresholds use upper ranks.',
                    source_sha256=source, script_sha256=sha(Path(__file__)))
    (OUT / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2) + '\n')
    (OUT / 'AUDIT.json').write_text(json.dumps(dict(status='PASS', rows=len(t), checks=audits,
        tied_rank_check=True, threshold_score_equivalence=True, transition_identities=True), indent=2) + '\n')
    print(s[['proxy', 'method', 'auc', 'unwm_auc', 'power_a01', 'human_fpr_a01']].to_string(index=False))
    print(pd.DataFrame(intervals).to_string(index=False))


if __name__ == '__main__':
    main()
