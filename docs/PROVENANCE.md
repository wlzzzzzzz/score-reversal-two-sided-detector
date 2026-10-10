# Provenance and release notes

## Source of truth

`paper/paper.pdf` contains the complete manuscript in preprint format.
Its scientific content matches the final submitted manuscript. Numerical
exports and result tables are under `analysis/`.

`manifest.json` records repository-relative paths, byte counts, and SHA-256
checksums for every published file except the manifest itself. Locally installed
environments, model checkpoints, external inputs, and generated `scratch/`
outputs are excluded. `scripts/check_release.py`
verifies those hashes and checks the 80-cell cohort against its recorded
input checksums. Published code and metadata include portability fixes and
may differ from their historical versions. Nested code/configuration hashes
in archived run receipts describe the original execution; the root manifest
describes this release. Fresh runs create new hashes and their own
development/test gate. The fixed protocol is in
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
calibration, lower-branch analysis, HeavyWater controls, and sensitivity. `results/baselines/` retains the matching baseline score caches and
aggregates. Benchmark text files, main moment caches, calibrated arrays and paper result
CSVs retain their recorded numerical contents. The baseline alignment frame
contains the 17 metadata/paper-score columns used in this release; all 189,660
original rows and every retained value are preserved exactly. The source
per-cell baseline caches are also retained.
Path-bearing cohort and audit metadata use repository-relative locations.

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
upstream projects and retained MIT/Apache-2.0 licenses. The vendored trees
contain the imported paper algorithms and utilities. Unused algorithms,
attack/demo frontends and unrelated environment dumps are omitted. Changes
to package imports and algorithm registries are identified in those files.

SemStamp source is not included: no redistribution license was found in the
inspected upstream tree. Its generation worker accepts a separately obtained
checkout under `external_code/SemStamp/`; see `external_code/README.md` for
requirements and the limits of matching the historical implementation.
The SemStamp benchmark texts, score caches and paper results remain included.
Source dataset redistribution remains governed by the original datasets'
terms; the root MIT license does not grant rights to those texts.
[data/README.md](../data/README.md) records the known source routes and
provider terms, including inconsistent MAGE license labels and source versions
that cannot be established from the retained manifests alone. The rights to
redistribute every source text have not been verified.

The repository is approximately 2.1 GiB uncompressed. Its files are below
GitHub's 100 MiB single-file limit. Use Git rather than browser upload for the
larger caches, and account for the 2 GiB per-push limit described in the root
README. `manifest.json` and `scripts/check_release.py` verify the payload and
reject files exceeding the single-file limit.

## Scientific audit scope

The retained audits record inputs, seeds, model/checkpoint and implementation
hashes, precision, independent numerical checks, source overlap/exclusions,
calibration/test separation and the original frozen decisions. The original
development observations are retained because they document prior data exposure.
Historical code/input digests identify the original execution; portability and
scope edits are covered by the current root manifest. They should not be used
as current executable paths or current code hashes.

The result index maps all retained analyses to paper tables, figures, appendix
text or these scientific dependencies. Manuscript editing, translation, build
and delivery records are outside the release scope.
