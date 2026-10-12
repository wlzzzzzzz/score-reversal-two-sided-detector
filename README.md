# Resolving Score-Ordering Reversal in Watermarked Text

Code, data, and results for **“Resolving Score-Ordering
Reversal in Watermarked Text: An Asymmetric Two-Sided Test for LLM Text
Detection”**.

[Installation](#installation) |
[Reproducing reported artifacts](#reproducing-reported-artifacts) |
[Detailed reproduction guide](docs/REPRODUCE.md) |
[Citation](#citation)

## Overview

Watermarked machine-generated text can receive lower Fast-DetectGPT scores
than human-written text. We study this score reversal and develop a detector
that uses both unusually high and unusually low scores as evidence of machine
generation.

![Figure 1: Fast-DetectGPT score distributions for human, unwatermarked, and watermarked text](analysis/figures/score_distributions.png)

**Figure 1.** Fast-DetectGPT score distributions using OPT-2.7B. Some
watermarking methods shift machine-text scores below human-text scores.
[PDF](analysis/figures/score_distributions.pdf).

We evaluate on **12 test datasets, five generators, and 11 generation
conditions** (unwatermarked text and ten watermarking methods), giving
660 combinations for each detector. Calibration uses 800 human-written texts;
the test set contains 2,360 human-written texts and 129,800 machine-generated
texts. Our detector achieves the following mean AUROC across the 660 combinations:

| Reference model | Scoring model | Mean AUROC |
| --- | --- | --- |
| OPT-2.7B | OPT-2.7B | 0.8219 |
| Falcon-7B | Falcon-7B-Instruct | 0.8317 |

## Method

The detector has two branches:

- **Upper tail:** an entropy-weighted discrepancy score computed with a
  reshaped reference distribution.
- **Lower tail:** a uniformly weighted discrepancy score computed with the
  original reference distribution.

We calibrate each branch using human-written texts and combine the two
p-values into a detection score. Higher scores indicate stronger evidence of
machine generation. The calibration and evaluation code is in
[`scripts/evaluate_detector.py`](scripts/evaluate_detector.py).

![Figure 2: Architecture of the asymmetric two-sided detector](analysis/figures/overview.png)

**Figure 2.** Scoring, human calibration, and the final two-sided decision.
[PDF](analysis/figures/overview.pdf).

## Installation

Use **Python 3.10 or 3.11**. We tested the CPU workflow with Python 3.11.
Clone or download this repository, open a terminal in its root directory,
and install the analysis dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-analysis.txt
```

The repository is approximately **2.1 GiB**, including benchmark texts and
saved model outputs. These files support the CPU workflow below without
downloading model checkpoints.

For GPU scoring, see the [model setup instructions](docs/REPRODUCE.md#3-prepare-for-fresh-model-scoring-of-all-80-benchmark-cells)
for checkpoint locations and `requirements-gpu.txt`. Watermark generation
also requires `requirements-generation.txt` and the
[generation dependencies](docs/REPRODUCE.md#optional-generation-and-baseline-dependencies).

## Reproducing reported artifacts

Choose either workflow below. Run commands from the repository root with the
environment activated. For individual tables and figures, see the
[reproduction guide](docs/REPRODUCE.md) and
[paper result index](docs/RESULTS_INDEX.md).

### 1. Reproduce results from saved model outputs (CPU)

```bash
python3 scripts/evaluate_detector.py --output scratch/detector --check-paper
```

This command uses the saved per-document statistics to recompute human
calibration, both tail p-values, and final detection scores. The
`--check-paper` flag compares all **1,320 evaluation rows** (660 combinations
for each model setup), aggregate results, and calibrated false-positive rates
and detection power with the paper's recorded values. The absolute tolerance
is `1e-12`.

The command creates `scratch/detector/` with these main outputs:

| File | Contents |
| --- | --- |
| `summary.csv` | Aggregate detection metrics for each model setup |
| `task_metrics.csv` | Metrics for each dataset, generator, and generation condition |
| `deployment.csv` | False-positive rates and detection power at the specified significance levels |
| `audit.json` | Input records, sample counts, and the result of the comparison with the paper |

For another run, choose a new directory under `scratch/` with `--output`.

### 2. Recompute model outputs and detector results (GPU)

To start from the benchmark texts, follow the
[GPU reproduction guide](docs/REPRODUCE.md#3-prepare-for-fresh-model-scoring-of-all-80-benchmark-cells).
It covers all 80 dataset–generator pairs: 20 for development and 60 for
testing. Score the development split and freeze its analysis, then score the
test split and evaluate the detector. Numerical results may vary with
hardware, CUDA, and model checkpoint revisions.

Regenerating the benchmark texts and rerunning external baselines require
additional source data, code, or checkpoints. See
[additional resources and reproduction limits](docs/REPRODUCE.md#limits-of-a-full-historical-rerun).

## Repository structure

| Directory | Contents |
| --- | --- |
| `scripts/` | Detector evaluation, model scoring, generation, baselines, and numerical diagnostics |
| `experiments/` | Matched two-sided controls and HeavyWater generation controls |
| `configs/` | Dataset splits, input file lists, and experiment settings |
| `data/` | Benchmark texts for the 80 dataset–generator pairs and their source information |
| `results/` | Saved model outputs, baseline scores, and experiment records |
| `analysis/` | Source data and scripts for the paper's tables and figures, with the final exports |
| `third_party/` | Watermark implementations required by the paper experiments and their original licenses |
| `tests/` | Regression tests for the detector, scoring, generation, and paper reproduction |
| `docs/` | Reproduction instructions, table and figure index, experimental protocols, and data provenance |

## Citation

The paper is under anonymous review. We will add a BibTeX entry and a preprint
link when the paper is public.

## Acknowledgments and license

This repository builds on [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt)
and the watermark implementations listed in [third-party components](third_party/README.md).
Original project code and accompanying software documentation use the
[MIT License](LICENSE). Third-party components retain their own licenses.
Source datasets, model checkpoints, and the manuscript are outside the scope
of this code license. See [data sources and terms](data/README.md) and
[provenance](docs/PROVENANCE.md) for attribution and redistribution status.
