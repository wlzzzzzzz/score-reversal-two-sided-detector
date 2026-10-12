# Data and experiment provenance

## Source of truth

The released numerical exports, result tables, and figures under `analysis/`
correspond to the manuscript's main text and appendices. Their mapping is
documented in [the paper result index](RESULTS_INDEX.md). Publication status
and citation information are in the root [README](../README.md#citation).

`configs/benchmark_cells.json` records the 80 benchmark cells and their
input hashes. Each document-moment cache has an experiment manifest recording
its source, configuration, row count, and content hash. The evaluator checks
these inputs as part of reproduction. Historical code/configuration hashes
describe the original execution and may differ from the current source files.
Fresh runs create new receipts and follow the fixed protocol in
`docs/protocols/reference_shaping.md`.

## Included data and exclusions

The 80 dataset-generator files contain source-aligned human texts and all
eleven machine-generation conditions. Source counts vary by cell and are
listed in `configs/benchmark_cells.json`: development has 4,000 generator-paired
human rows and test has 11,805 before exclusion. The documented source-overlap
rule removes one test WritingPrompts source (`wp:11`) and its generations
from primary metrics. Hence the main test has 2,360 unique human sources and
129,800 machine texts. Original dataset files are unmodified; source
exclusion is recorded in the score audits and result manifests.

The numerical `analysis/data/` tree contains the full result exports for all
19 paper tables. `results/runs/` retains the scored-text caches and raw
results for the main method, two-sided controls, reference shaping,
calibration, lower-branch analysis, HeavyWater controls, and sensitivity.
`results/baselines/` contains the corresponding per-cell baseline scores and
aggregates. Its alignment frame has 189,660 rows and 17 columns of metadata
and detector scores.

`scripts/evaluate_detector.py` rebuilds the final calibrated frames and detector
metrics from the archived document moments or a freshly scored 80-cell run.
Archived CPU replay has been checked against all 1,320 final-detector task
rows and the aggregate/deployment exports. Earlier source-corpus generation
and some historical baselines still need external resources; see
`REPRODUCE.md`.

## External dependencies and redistribution

The original code and accompanying software documentation use the MIT license
in [`LICENSE`](../LICENSE). Third-party components retain their own notices
and terms. This code license does not license the source datasets, model
checkpoints, or manuscript.

Model checkpoints are not in this folder. The paper uses OPT-2.7B,
Falcon-7B, Falcon-7B-Instruct, and five generators; their model identifiers
and numerical configurations are recorded in the paper and frozen run
manifests. Obtain model files from their original providers subject to their
terms. GPU reruns also require a compatible CUDA/PyTorch/Transformers stack.

`third_party/README.md` lists the included components, their paper use,
upstream projects and retained MIT/Apache-2.0 licenses. It also describes
changes to package imports and algorithm registries.

SemStamp source is not included: no redistribution license was found in the
inspected upstream tree. Its generation worker accepts a separately obtained
checkout under `external_code/SemStamp/`; see
[optional dependencies](REPRODUCE.md#optional-generation-and-baseline-dependencies)
for requirements and the limits of matching the historical implementation.
The SemStamp benchmark texts, score caches and paper results remain included.
Source dataset redistribution remains governed by the original datasets'
terms; the root MIT license does not grant rights to those texts.
[data/README.md](../data/README.md) records the known source routes and
provider terms, including inconsistent MAGE license labels and source versions
that cannot be established from the retained manifests alone. The rights to
redistribute every source text have not been verified.

## Scientific audit scope

The retained audits record inputs, seeds, model/checkpoint and implementation
hashes, precision, independent numerical checks, source overlap/exclusions,
calibration/test separation and the original frozen decisions. The original
development observations are retained because they document prior data exposure.
Historical code/input digests identify the original execution. Recorded code
paths and hashes can differ from the current public scripts; use
`docs/REPRODUCE.md` for executable commands.
