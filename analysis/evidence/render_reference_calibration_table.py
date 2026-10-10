#!/usr/bin/env python3
"""Render Table 14: the two calibrated reference-shaping gains reported in the paper."""
import argparse
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parents[1]


def render():
    values = pd.read_csv(HERE / 'data/reference_calibration/summary.csv').set_index(['proxy', 'cohort'])
    cohorts = [('unwatermarked', 'Unwatermarked'), ('positive6', 'Six upward WMs'),
               ('CGW_HeavyWater', 'CGW / HeavyWater'), ('ITSEdit_SemStamp', 'ITSEdit / SemStamp')]
    rows = []
    for proxy in ('OPT', 'Falcon'):
        for cohort, label in cohorts:
            record = values.loc[(proxy, cohort)]
            cells = []
            for field in ('upper_rank_delta', 'joint_delta'):
                text = f'{record[field]:+.4f}'
                cells.append(r'\textbf{' + text + '}' if record[field] > 0 else text)
            rows.append(('Falcon pair' if proxy == 'Falcon' else proxy) + ' & ' + label + ' & ' + ' & '.join(cells) + r' \\')
    return '\n'.join([r'\begin{tabular}{llrr}', r'\toprule',
                      r'Proxy & Conditions & $\Delta A_+$ & $\Delta A$ \\', r'\midrule',
                      *rows, r'\bottomrule', r'\end{tabular}', ''])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    target = HERE / 'tables/reference_calibration.tex'
    result = render()
    if args.check_only:
        assert target.read_text() == result
    else:
        target.write_text(result)
    print('PASS: Table 14 matches the calibrated reference-shaping results.')
