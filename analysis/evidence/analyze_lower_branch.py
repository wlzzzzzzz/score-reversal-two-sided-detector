#!/usr/bin/env python3
"""Recompute Table 6 from the final detector's document moments (no model inference)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
KEYS = ['proxy', 'dataset', 'generator', 'watermark_method']
METRICS = ['gap_uniform', 'gap_entropy', 'uniform_minus_entropy_gap',
           'cov_machine_minus_human', 'lower_auc', 'center_repairs',
           'center_damage', 'r_human_ge_machine_fraction', 'center_loss']


def lower_credit(human, machine):
    """Pairwise lower-tail AUROC credits, assigning one half to ties."""
    return (machine[:, None] < human[None, :]).astype(float) + .5 * (machine[:, None] == human[None, :])


def recompute():
    records = []
    for proxy in ('OPT', 'Falcon'):
        frame = pd.read_pickle(ROOT / 'results/runs/calibrated_scores/test' / proxy / 'scoring_frame.pkl')
        assert set(frame.label) == {0, 1}
        assert 'wp:11' not in set(frame.source_id)
        for (dataset, generator), cell in frame.groupby(['dataset', 'generator']):
            human = cell[cell.label == 0]
            for method in ('CGW', 'HeavyWater'):
                machine = cell[(cell.label == 1) & (cell.watermark_method == method)]
                assert len(human) == len(machine) > 0
                response = []
                shifts = []
                for group in (human, machine):
                    denominator = np.maximum(np.sqrt(group.uniform_V0.to_numpy()), 1e-9)
                    response.append((group.uniform_N0.to_numpy() / denominator,
                                     group.uniform_N1.to_numpy() / denominator))
                    shifts.append((group.uniform_N0.to_numpy() - group.uniform_N1.to_numpy()) / denominator)
                original = lower_credit(response[0][0], response[1][0])
                changed = lower_credit(response[0][1], response[1][1])
                delta = changed - original
                gap_uniform = float(human.uniform_N0.mean() - machine.uniform_N0.mean())
                gap_entropy = float(human.entropy_N0.mean() - machine.entropy_N0.mean())
                records.append(dict(
                    proxy=proxy, dataset=dataset, generator=generator, watermark_method=method,
                    gap_uniform=gap_uniform, gap_entropy=gap_entropy,
                    uniform_minus_entropy_gap=gap_uniform-gap_entropy,
                    cov_machine_minus_human=gap_entropy-gap_uniform,
                    lower_auc=float(original.mean()),
                    center_repairs=float(np.maximum(delta, 0).mean()),
                    center_damage=float(np.maximum(-delta, 0).mean()),
                    r_human_ge_machine_fraction=float((shifts[0][None, :] >= shifts[1][:, None]).mean()),
                    center_loss=float(-delta.mean())))
    tasks = pd.DataFrame(records)
    assert len(tasks) == 240
    summary = tasks.groupby(['proxy', 'watermark_method'])[METRICS].mean().reset_index()
    return tasks, summary


def check(actual, path, keys):
    expected = pd.read_csv(path).sort_values(keys).reset_index(drop=True)
    actual = actual.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[keys], expected[keys])
    np.testing.assert_allclose(actual[METRICS], expected[METRICS], rtol=0, atol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='verify all saved task and aggregate fields')
    parser.add_argument('--output', type=Path, default=Path('scratch/lower_branch'))
    args = parser.parse_args()
    tasks, summary = recompute()
    source = ROOT / 'analysis/data/mechanism_evidence'
    check(tasks, source / 'by_task.csv', KEYS)
    check(summary, source / 'summary.csv', ['proxy', 'watermark_method'])
    if not args.check_only:
        output = (ROOT / args.output).resolve()
        if (ROOT / 'scratch').resolve() not in output.parents:
            raise ValueError('Output must be a new directory below scratch/')
        output.mkdir(parents=True, exist_ok=False)
        tasks.to_csv(output / 'by_task.csv', index=False, float_format='%.17g')
        summary.to_csv(output / 'summary.csv', index=False, float_format='%.17g')
    print('PASS: Table 6; 240 task rows and all four summary rows match at absolute tolerance 1e-12.')


if __name__ == '__main__':
    main()
