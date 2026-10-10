#!/usr/bin/env python3
"""Check the four externally implemented baselines against final paper task AUROCs."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SCORES = {'Fast-DetectGPT': 'FastDetect_Falcon_native', 'AdaDetectGPT': 'AdaDetectGPT',
          'Binoculars': 'Binoculars', 'NTS': 'NTS'}


def main():
    frame = pd.read_pickle(ROOT / 'results/baselines/review_supplement/analysis/aligned_scores.pkl')
    query = frame[(frame.split == 'test') & (frame.source_id != 'wp:11')]
    # The main paper counts each human source once, choosing the same deterministic generator order.
    human = query[query.label == 0].sort_values(['dataset', 'source_id', 'generator']).drop_duplicates(['dataset', 'source_id'])
    assert len(human) == 2360 and (query.label == 1).sum() == 129800
    paper = pd.read_csv(ROOT / 'analysis/data/current/task_metrics.csv').set_index(['score', 'dataset', 'generator', 'watermark_method'])
    rows = 0
    for (dataset, generator, method), machine in query[query.label == 1].groupby(['dataset', 'generator', 'watermark_method']):
        humans = human[human.dataset == dataset]
        assert len(humans) == len(machine)
        labels = np.r_[np.zeros(len(humans)), np.ones(len(machine))]
        for name, column in SCORES.items():
            value = roc_auc_score(labels, np.r_[humans[column], machine[column]])
            expected = paper.loc[(name, dataset, generator, method), 'auc']
            np.testing.assert_allclose(value, expected, rtol=0, atol=1e-12)
            rows += 1
    assert rows == 2640
    print('PASS: all 2,640 baseline task AUROCs match the paper at absolute tolerance 1e-12.')


if __name__ == '__main__':
    main()
