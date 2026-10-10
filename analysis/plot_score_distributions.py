#!/usr/bin/env python3
"""Draw the single-column native Fast-DetectGPT score boxplot.

Dependencies: matplotlib (no model inference required).
Input: the verified empirical quantiles exported from the current test scores.
Boxes span Q25--Q75; whiskers span P10--P90; white lines mark medians.
Human scores count each of the 2,360 sources once.
Horizontal axis: generation mechanism. Vertical axis: native score.
One panel, 5.50 by 1.60 inches; 10 pt text matches the manuscript scale.
Full category names use 8 pt text, angled 30 degrees to keep the panel compact.
The two-line vertical axis title uses 9 pt text, one point below the body.
The published PDF uses LaTeX fonts; pass --tex when that optional stack is installed.

Example:
    python analysis/plot_score_distributions.py --output-dir scratch/figures
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ORDER = ["human", "unwatermarked", "KGW", "SWEET", "Unigram", "EXP",
         "SynthID", "DIP", "ITSEdit", "CGW", "HeavyWater", "SemStamp"]
LABELS = ["Human", "Unwatermarked", "KGW", "SWEET", "Unigram", "EXP",
          "SynthID", "DIP", "ITSEdit", "CGW", "HeavyWater", "SemStamp"]
QUANTILES = ["p10", "q25", "median", "q75", "p90"]


def read_statistics(path: Path) -> dict:
    """Read empirical boxplot statistics; do not synthesize sample scores."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(ORDER) or {row["method"] for row in rows} != set(ORDER):
        raise ValueError("Expected one row per human/unwatermarked/watermark group.")
    result = {}
    for row in rows:
        values = [float(row[key]) for key in QUANTILES]
        if not all(math.isfinite(value) for value in values) or values != sorted(values):
            raise ValueError(f"Invalid empirical quantiles: {row['method']}")
        result[row["method"]] = dict(zip(QUANTILES, values))
    return result


def draw(summary: dict, output_dir: Path, dpi: int, usetex: bool = False) -> None:
    from paper_figure_style import configure, save_figure
    configure(usetex=usetex)
    fig, ax = plt.subplots(figsize=(5.5, 1.60))
    # Reserve only the space needed for the angled category names.
    fig.subplots_adjust(left=.13, right=.99, bottom=.32 if usetex else .41, top=.95)
    human = summary["human"]
    ax.axhspan(human["q25"], human["q75"], color="#566575", alpha=.075, zorder=0)
    ax.axhline(human["median"], color="#65717D", lw=.7, ls=(0, (3, 3)), zorder=1)
    # bxp draws the CSV's empirical quantiles directly; no sample fabrication.
    stats = [{"med": summary[group]["median"], "q1": summary[group]["q25"],
              "q3": summary[group]["q75"], "whislo": summary[group]["p10"],
              "whishi": summary[group]["p90"]} for group in ORDER]
    positions = list(range(len(ORDER)))
    artists = ax.bxp(stats, positions=positions, vert=True, widths=.55,
        patch_artist=True, showfliers=False, manage_ticks=False,
        medianprops={"color": "white", "linewidth": 1},
        whiskerprops={"linewidth": .8}, capprops={"linewidth": .8})
    for i, (group, box) in enumerate(zip(ORDER, artists["boxes"])):
        color = "#65717D" if group == "human" else ("#6954A5" if group == "unwatermarked" else "#C87634")
        box.set(facecolor=color, edgecolor=color, alpha=.93)
        for line in artists["whiskers"][2*i:2*i+2] + artists["caps"][2*i:2*i+2]:
            line.set_color(color)
    ax.set(xticks=positions, xticklabels=LABELS,
           xlim=(-.65, len(ORDER)-.35), ylim=(-6.3, 6.05), yticks=[-6, -3, 0, 3, 6],
           ylabel="Score by\nFast-DetectGPT")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor", fontsize=8)
    ax.yaxis.label.set_fontsize(9)
    ax.yaxis.label.set_linespacing(1.05)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.axvline(1.5, color="#C8CDD2", linewidth=.6)
    ax.grid(axis="y", alpha=.14)
    save_figure(fig, "score_distributions", output_dir, dpi,
                allowed_font_sizes=(8, 9, 10))


def verify_statistics(summary: dict) -> None:
    """Rebuild Figure 1 quantiles and Table 7 AUROCs from document scores."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    frame = pd.read_pickle(HERE.parent / 'results/runs/calibrated_scores/test/OPT/scoring_frame.pkl')
    human = frame[frame.label == 0].sort_values(
        ['dataset', 'source_id', 'generator']).drop_duplicates(['dataset', 'source_id'])
    assert len(human) == 2360
    query = pd.concat([human, frame[frame.label == 1]], ignore_index=True)
    query['native_score'] = query.uniform_N0 / np.maximum(np.sqrt(query.uniform_V0), 1e-9)
    for method in ORDER:
        block = query[query.label == 0] if method == 'human' else query[query.watermark_method == method]
        assert len(block) == (2360 if method == 'human' else 11800)
        actual = np.quantile(block.native_score, [.1, .25, .5, .75, .9])
        np.testing.assert_allclose(actual, [summary[method][key] for key in QUANTILES], rtol=0, atol=1e-12)
    expected = pd.read_csv(HERE / 'data/two_sided_controls/fd/by_task.csv')
    expected = expected[(expected.proxy == 'OPT') & (expected.method == 'FD_original_direction')]
    expected = expected.set_index(['dataset', 'generator', 'watermark_method'])
    rows = 0
    for key, machine in query[query.label == 1].groupby(['dataset', 'generator', 'watermark_method']):
        h = query[(query.label == 0) & (query.dataset == key[0])]
        value = roc_auc_score(np.r_[np.zeros(len(h)), np.ones(len(machine))],
                              np.r_[h.native_score, machine.native_score])
        np.testing.assert_allclose(value, expected.loc[key, 'auc'], rtol=0, atol=1e-12)
        rows += 1
    assert rows == 660
    print('PASS: Figure 1 quantiles and all 660 Table 7 task AUROCs match document scores.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=HERE / "data/current/score_distributions.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "scratch/figures")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--check-only", action="store_true", help="Verify quantiles and native OPT task scores without plotting")
    parser.add_argument("--tex", action="store_true", help="Use the optional LaTeX font stack of the published PDF")
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    summary = read_statistics(args.data)
    if args.check_only:
        verify_statistics(summary)
    else:
        draw(summary, args.output_dir, args.dpi, usetex=args.tex)


if __name__ == "__main__":
    main()
