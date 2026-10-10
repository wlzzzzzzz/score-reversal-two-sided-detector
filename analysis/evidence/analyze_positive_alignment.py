#!/usr/bin/env python3
"""Export/replay positive numerator retention from current matched moments."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / 'data/positive_alignment'


def calculate(proxy, a):
    q = pd.DataFrame({k: a[k] for k in a.files})
    assert q.row_id.is_unique
    assert len(q) == 23600 and int(q.label.sum()) == 11800
    hu = q[q.label == 0].drop_duplicates('source_id')
    assert len(hu) == 2360
    rows = []
    for (ds, gen), g in q.groupby(['dataset', 'generator'], sort=True):
        m, h = g[g.label == 1], g[g.label == 0]
        assert set(m.source_id) == set(h.source_id)
        du, de = m.N_U.mean() - h.N_U.mean(), m.N_E.mean() - h.N_E.mean()
        rows.append(dict(proxy=proxy, dataset=ds, generator=gen,
                         n_machine=len(m), n_human=len(h),
                         gap_uniform=du, gap_entropy=de, entropy_gain=de-du,
                         human_covariance_term=float((h.N_E-h.N_U).mean()),
                         machine_covariance_term=float((m.N_E-m.N_U).mean())))
    d = pd.DataFrame(rows)
    assert len(d) == 60
    np.testing.assert_allclose(d.entropy_gain,
                               d.machine_covariance_term-d.human_covariance_term, atol=1e-12)
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path)
    args = parser.parse_args()
    if args.source_dir:
        OUT.mkdir(parents=True, exist_ok=True)
        sources = {}
        for proxy in ['OPT', 'Falcon']:
            path = args.source_dir / 'test' / proxy / 'scoring_frame.pkl'
            sources[proxy] = dict(filename=path.name,
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            q = pd.read_pickle(path)
            q = q[(q.label == 0) | (q.watermark_method == 'unwatermarked')]
            np.savez_compressed(OUT / f'moments_{proxy}.npz',
                                **{k: q[k].to_numpy(dtype=str) for k in
                                   ['row_id', 'source_id', 'dataset', 'generator']},
                                label=q.label.to_numpy(), N_U=q.uniform_N1.to_numpy(),
                                N_E=q.entropy_N1.to_numpy())
        protocol = dict(status='PASS', source_frames=sources, source_cohort='current 60-cell test',
                        method='Difference of mean shaped-reference numerator residuals; current uniform and entropy weights.',
                        aggregation='60-task macro mean for each proxy; generator-paired human copies used once per task.',
                        calibration='No thresholds refitted; this is a numerator diagnostic, not a new detector.',
                        interpretation='Observable human-relative retention; not an estimate of each conditional upward strength or of standardized efficiency.',
                        current_parameters=dict(T=1.2, c=0.8, kappa=1.5))
        (OUT/'PROTOCOL.json').write_text(json.dumps(protocol, indent=2)+'\n')
    tables = []
    for proxy in ['OPT', 'Falcon']:
        with np.load(OUT/f'moments_{proxy}.npz') as a:
            tables.append(calculate(proxy, a))
    d = pd.concat(tables, ignore_index=True)
    summary = d.groupby('proxy', sort=False).agg(
        tasks=('dataset', 'size'), gap_uniform=('gap_uniform', 'mean'),
        gap_entropy=('gap_entropy', 'mean'), entropy_gain=('entropy_gain', 'mean'),
        positive_gain_tasks=('entropy_gain', lambda x: int((x > 0).sum()))).reset_index()
    if args.source_dir:
        d.to_csv(OUT/'by_task.csv', index=False)
        summary.to_csv(OUT/'summary.csv', index=False)
    else:
        pd.testing.assert_frame_equal(d, pd.read_csv(OUT/'by_task.csv'), atol=1e-12, rtol=0)
        pd.testing.assert_frame_equal(summary, pd.read_csv(OUT/'summary.csv'), atol=1e-12, rtol=0)
    print(json.dumps(dict(status='PASS', task_rows=len(d), summary=summary.to_dict('records')), indent=2))


if __name__ == '__main__':
    main()
