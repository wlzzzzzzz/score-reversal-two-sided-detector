#!/usr/bin/env python3
"""Frozen same-score AdaDetectGPT two-tail evaluation on the paper cohort."""
import os
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '1')
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'analysis'
OUT = ROOT / 'results/runs/ada_control'
BASE = ROOT / 'results/runs/calibrated_scores'
STRONG = ROOT / 'results/baselines/benchmark/strong_scores'
FD = ROOT / 'results/runs/fd_control'
sys.path.insert(0, str(PAPER / 'evidence'))
from analyze_current_supplement import rank, tasks, human_indices


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, data):
    pd.DataFrame(data).to_csv(OUT / (name + '.csv'), index=False, float_format='%.17g')


def main():
    OUT.mkdir(exist_ok=False)
    sources = {}
    def source(path):
        sources[str(path.relative_to(ROOT))] = sha(path)
        return path
    cal = pd.read_pickle(source(BASE / 'calibration_Falcon.pkl'))
    q = pd.read_pickle(source(BASE / 'test/Falcon/scoring_frame.pkl')).reset_index(drop=True)
    assert len(cal) == 800 and cal.source_id.is_unique
    assert not set(cal.source_id) & set(q.source_id)
    records = []
    files = sorted(STRONG.glob('*.json'))
    assert len(files) == 80
    for path in files:
        obj = json.loads(source(path).read_text())
        config = obj['config']
        assert config['max_tokens'] == 2048
        assert config['observer'].endswith('/falcon-7b')
        assert config['performer'].endswith('/falcon-7b-instruct')
        for row in obj['rows']:
            records.append({k: row[k] for k in ['sample_id', 'source_id', 'text_sha256', 'adadetectgpt_pretrained_full']})
    scores = pd.DataFrame(records).set_index('sample_id')
    assert scores.index.is_unique
    for part in [cal, q]:
        selected = scores.loc[part.row_id]
        np.testing.assert_array_equal(selected.text_sha256.to_numpy(), part.text_sha256.to_numpy())
        np.testing.assert_array_equal(selected.source_id.to_numpy(), part.source_id.to_numpy())
    bank = scores.loc[cal.row_id, 'adadetectgpt_pretrained_full'].to_numpy()
    raw = scores.loc[q.row_id, 'adadetectgpt_pretrained_full'].to_numpy().copy()
    assert np.isfinite(bank).all() and np.isfinite(raw).all()
    hh = human_indices(q)
    mm = np.flatnonzero(q.label.to_numpy() == 1)
    assert len(hh) == 2360 and len(mm) == 129800
    canonical = dict(zip(q.loc[hh, 'source_id'], hh))
    repeated = np.flatnonzero(q.label.to_numpy() == 0)
    origin = np.array([canonical[s] for s in q.loc[repeated, 'source_id']])
    repeated_difference = float(np.max(np.abs(raw[repeated] - raw[origin])))
    raw[repeated] = raw[origin]
    upper = rank(bank, raw, True)
    lower = rank(bank, raw, False)
    # Independently verify inclusive ranks, including arbitrary bank ties.
    for start in range(0, len(raw), 1024):
        value = raw[start:start + 1024, None]
        np.testing.assert_array_equal(upper[start:start + 1024], (1 + (bank[None, :] >= value).sum(1)) / 801.)
        np.testing.assert_array_equal(lower[start:start + 1024], (1 + (bank[None, :] <= value).sum(1)) / 801.)
    joint = -np.log(np.minimum(upper / .5, lower / .5))
    expected = pd.read_csv(source(PAPER / 'data/current/task_metrics.csv'))
    expected = expected[expected.score == 'AdaDetectGPT'].set_index(['dataset', 'generator', 'watermark_method'])
    rows, summaries = [], []
    parity_errors = []
    for method, values in [('Ada_original_direction', raw), ('Ada_same_score_two_tails', joint)]:
        decisions = {}
        for alpha, tag in [(.01, 'a01'), (.05, 'a05')]:
            decision = upper <= alpha if method == 'Ada_original_direction' else ((upper <= alpha / 2) | (lower <= alpha / 2))
            if method == 'Ada_same_score_two_tails':
                np.testing.assert_array_equal(decision, values >= -np.log(alpha))
            decisions[tag] = decision
        part = []
        for ds, gen, wm, hi, mi in tasks(q):
            assert len(hi) == len(mi)
            auc = roc_auc_score(np.r_[np.zeros(len(hi)), np.ones(len(mi))], np.r_[values[hi], values[mi]])
            r = dict(proxy='Falcon', method=method, dataset=ds, generator=gen,
                     watermark_method=wm, n_human=len(hi), n_machine=len(mi), auc=auc)
            for tag in ['a01', 'a05']:
                r['power_' + tag] = float(decisions[tag][mi].mean())
            if method == 'Ada_original_direction':
                parity_errors.append(auc - expected.loc[(ds, gen, wm), 'auc'])
            part.append(r)
        block = pd.DataFrame(part)
        assert len(block) == 660
        r = dict(proxy='Falcon', method=method, tasks=660, calibration_humans=800,
                 test_unique_humans=2360, test_machines=129800, auc=float(block.auc.mean()),
                 unwm_auc=float(block[block.watermark_method == 'unwatermarked'].auc.mean()),
                 wm_auc=float(block[block.watermark_method != 'unwatermarked'].auc.mean()),
                 reversed_tasks=int((block.auc < .5).sum()))
        for tag in ['a01', 'a05']:
            r['power_' + tag] = float(block['power_' + tag].mean())
            r['pooled_power_' + tag] = float(decisions[tag][mm].mean())
            r['human_fp_' + tag] = int(decisions[tag][hh].sum())
            r['human_fpr_' + tag] = float(decisions[tag][hh].mean())
        rows.extend(part)
        summaries.append(r)
    assert max(np.abs(parity_errors)) < 1e-12
    fd_tasks = pd.read_csv(source(FD / 'by_task.csv'))
    fd_summary = pd.read_csv(source(FD / 'summary.csv'))
    keep = ['FD_same_score_two_tails', 'Current_asymmetric']
    fd_tasks = fd_tasks[(fd_tasks.proxy == 'Falcon') & fd_tasks.method.isin(keep)]
    fd_summary = fd_summary[(fd_summary.proxy == 'Falcon') & fd_summary.method.isin(keep)]
    t = pd.concat([pd.DataFrame(rows), fd_tasks], ignore_index=True)
    s = pd.concat([pd.DataFrame(summaries), fd_summary], ignore_index=True)
    keys = ['dataset', 'generator', 'watermark_method']
    ref = t[t.method == 'Ada_same_score_two_tails'].set_index(keys).sort_index()
    for method, block in t.groupby('method'):
        aligned = block.set_index(keys).sort_index()
        assert aligned.index.equals(ref.index)
        np.testing.assert_array_equal(aligned[['n_human', 'n_machine']], ref[['n_human', 'n_machine']])
    save('by_task', t)
    save('summary', s)
    save('by_mechanism', t.groupby(['method', 'watermark_method'])[['auc', 'power_a01', 'power_a05']].mean().reset_index())
    draws = np.random.RandomState(20260917).randint(0, 12, size=(10000, 12))
    ci = []
    for cohort in ['all', 'unwatermarked']:
        part = t if cohort == 'all' else t[t.watermark_method == 'unwatermarked']
        for metric in ['auc', 'power_a01', 'power_a05']:
            means = part.groupby(['dataset', 'method'])[metric].mean().unstack().sort_index()
            for alternative in keep:
                diff = (means[alternative] - means.Ada_same_score_two_tails).to_numpy()
                lo, hi = np.quantile(diff[draws].mean(1), [.025, .975])
                ci.append(dict(comparison=alternative + ' minus Ada_same_score_two_tails', cohort=cohort,
                               metric=metric, delta=float(diff.mean()), ci_low=float(lo), ci_high=float(hi)))
    save('paired_domain_intervals', ci)
    protocol = dict(status='PASS', detector='paper adapted AdaDetectGPT', model_pair='Falcon-7B / Falcon-7B-Instruct',
                    witness='fixed pretrained B-spline; no benchmark fitting', scoring='same cached BF16-forward/FP32-reduction scores as paper',
                    calibration_cells=20, calibration_humans=800, test_cells=60, tasks=660,
                    test_unique_humans=2360, machine_texts=129800, beta=.5, alphas=[.01, .05],
                    two_tail_score='-log(min(2*p_upper,2*p_lower)); inclusive add-one human ranks; unclipped score',
                    new_model_forwards=0, parameter_search=False, prior_test_exposure=True,
                    aggregation='task-macro AUROC/power; unique-human FPR; pooled power separate',
                    intervals='10000 paired whole-domain resamples of 12 domains; fixed rules and bank',
                    source_sha256=sources, script_sha256=sha(Path(__file__)))
    (OUT / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2) + '\n')
    audit = dict(status='PASS', matched_task_rows=len(t), original_ada_max_task_auc_error=float(np.max(np.abs(parity_errors))),
                 repeated_human_max_score_difference=repeated_difference, all_text_hashes_matched=True,
                 calibration_test_sources_disjoint=True, same_800_humans_as_fd_control=True,
                 inclusive_rank_direct_checks=2 * len(raw), joint_threshold_equivalence=True,
                 equal_comparison_task_counts=True)
    (OUT / 'AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(s[['method', 'auc', 'unwm_auc', 'power_a01', 'human_fpr_a01', 'power_a05', 'human_fpr_a05']].to_string(index=False))
    print(pd.DataFrame(ci).query("cohort == 'all' and metric == 'auc'").to_string(index=False))


if __name__ == '__main__':
    main()
