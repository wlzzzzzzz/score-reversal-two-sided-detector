# Reproduce the repository results

Run every command below from the repository root. There are two supported
starting points:

1. **Verify the published numerical results** from the included benchmark,
   score caches, and analysis exports. This checks the complete release and
   rebuilds final detector scores and regenerates the supported tables without a GPU.
2. **Recompute the main detector and its reference moments** from all 80 included text
   cells. This performs fresh OPT and Falcon model forwards, development
   analysis, a frozen transition to test, and final two-sided
   calibration and evaluation. It requires local
   model checkpoints and GPUs. Its output goes under `scratch/` and does not
   replace the recorded results.

The starting input for a model rerun is the included final benchmark in
`data/benchmark/`. Recreating that benchmark from the original source corpora
is a separate generation task: some source corpora and model checkpoints are
not distributed here. See [Limits of a full historical rerun](#limits-of-a-full-historical-rerun).

## 1. Inspect and verify the included release

Use Python 3.10 or 3.11 for a new environment. The checks and analysis were
also verified in the original Python 3.8.10 environment; the CPU replay runs
in an isolated Python 3.11 environment with `requirements-analysis.txt`. No third-party
package is needed for the two integrity checks below.

```bash
python3 scripts/check_release.py
python3 scripts/check_paths.py
```

`check_release.py` verifies the recorded payload hashes, the 80 benchmark
input hashes, the main reported AUROCs, and the paper table/figure inventory.
`check_paths.py` checks that stored paths are portable. A failed hash check
means the local release differs from the recorded one; resolve that before
comparing numerical results.

## 2. Rebuild the final detector and check the tables without GPU inference

Create an environment and install the analysis dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-analysis.txt
```

Rebuild human calibration, both directional p-values, detector scores, all
660 task metrics per proxy, and deployment FPR/power directly from the 160
included document-moment caches:

```bash
python3 scripts/evaluate_detector.py --output scratch/detector --check-paper
```

The command validates benchmark text hashes, per-cell completeness, and the
current release hashes. It uses 800 unique calibration humans, applies the
recorded `wp:11` test exclusion, and evaluates 2,360 unique test humans and
129,800 machine texts. The upper branch uses entropy-weighted shaped-reference
moments; the lower branch uses uniform original-reference moments. Both use
inclusive add-one human ranks with allocation `beta=0.5`.

`--check-paper` checks every final-detector task and the aggregate/deployment
metrics within absolute tolerance `1e-12`. The expected macro AUROCs are:

| Proxy | AUROC |
| --- | --- |
| OPT | 0.8219095663414562 |
| Falcon | 0.8316661796123974 |

Inspect `scratch/detector/summary.csv`, `task_metrics.csv`, `deployment.csv`,
and `audit.json`. Per-proxy calibration frames, test frames, row indexes,
and prediction arrays are also saved. The command refuses to overwrite an
existing directory; choose another path below `scratch/` for a repeat run.

Check the main-method and supplementary numerical tables supported by the
renderers against the included exports, then verify the supporting analyses:

```bash
python3 analysis/evidence/render_current_tables.py --check-only
python3 analysis/evidence/render_supplement_tables.py --check-only
python3 analysis/evidence/render_reference_calibration_table.py --check-only
python3 analysis/evidence/analyze_lower_branch.py --check-only
python3 analysis/evidence/check_baselines.py
python3 analysis/evidence/two_sided_controls.py
python3 analysis/evidence/check_heavywater_controls.py
python3 scripts/audit_heavywater_prefix.py
python3 analysis/evidence/validate_supplement.py
python3 analysis/plot_score_distributions.py --check-only
```

These commands use `analysis/data/` and the recorded score summaries. They
compare generated table bodies to `analysis/tables/` and do not run a language
model. The current-table renderer covers Tables 1, 2, 3, 5, 6, 7 and 10, plus
supporting exports; `two_sided_controls.py` covers Table 4 and its appendix
controls. Tables 2 and 10 contain all ten detectors. Other table exports
are checked for integrity by `check_release.py`;
some also have separate validators in `analysis/evidence/`.
Write the current-table exports to a separate directory with:

```bash
python3 analysis/evidence/render_current_tables.py --output-dir scratch/tables
```

For renderers without an output-directory option, omit `--check-only` only
**in a separate copy** of the repository because this replaces recorded tables. Use `docs/RESULTS_INDEX.md` to trace
each of the 19 paper tables and two figures to its source files. This replay
checks available numerical exports, not the wording of captions in
`paper/paper.pdf`.

### Figures

Verify Figure 1 and Table 7 directly from the saved native OPT document
scores with `python3 analysis/plot_score_distributions.py --check-only`.
This compares all five empirical quantiles for each group and all 660 task
AUROCs at absolute tolerance `1e-12`.

```bash
python3 analysis/plot_score_distributions.py --output-dir scratch/figures
```

This writes `score_distributions.pdf`, `.png` and `.svg` using Matplotlib.
The default requires no LaTeX or Poppler installation. It plots the same empirical quantiles; fonts and label spacing can differ
from the published master.
For the original LaTeX font setup, add `--tex` after installing a working
LaTeX distribution and the Matplotlib TeX font packages, including `type1cm`
and `type1ec`. The published Figure 1 PDF remains included.

Figure 2 is a conceptual diagram rather than a numerical experiment.
`analysis/figures/overview.pdf` is the vector master from the final manuscript;
the PNG and SVG are exports of that same master. No generative script for this
final diagram is included. The earlier schematic builder is not used by the
final paper.

The following mathematical checks use synthetic inputs and print JSON reports
without changing the published audit records:

```bash
python3 analysis/evidence/check_asymmetric_reference.py
python3 analysis/evidence/check_mechanism_projection.py
python3 analysis/evidence/check_entropy_alignment.py
```

## 3. Prepare for fresh model scoring of all 80 benchmark cells

The document-moment scorer uses OPT-2.7B as both reference and scoring
model, and Falcon-7B plus Falcon-7B-Instruct as a model pair. Obtain those
checkpoints from their providers and put compatible local model directories
at these repository-relative paths:

```text
models/opt-2.7B/
models/falcon-7b/
models/falcon-7b-instruct/
```

Install a CUDA-compatible PyTorch build for the machine, then install the
remaining GPU dependencies:

```bash
python3 -m pip install -r requirements-gpu.txt
```

The commands below use two visible GPUs **sequentially**: OPT uses GPU 0;
the Falcon reference and scoring models use GPUs 0 and 1. The original run
used more parallel workers. GPU memory must accommodate the local models and
the fixed 2048-token, FP16 scoring protocol. Keep the same Python environment,
model files, and code between development and test: the frozen test gate
checks code hashes and analysis package versions. The original analysis
versions are recorded in
`results/runs/reference_shaping/analysis/development/run_manifest.json`.
The analysis packages and core GPU packages are pinned to the available
experiment environment. Fresh floating-point values can still differ with
hardware, CUDA libraries, and checkpoint revisions. `sentencepiece` supports
optional generator tokenizers; it is not needed for this OPT/Falcon replay.

Create a new configuration and output directory under ignored `scratch/`.
This leaves `configs/original/` and `results/runs/` intact. The script stops
if a previous replay configuration or output exists:

```bash
python3 - <<'PY'
import json
from pathlib import Path

source = Path('configs/original/reference_shaping.json')
target = Path('scratch/reference_shaping.json')
output = Path('scratch/reference_shaping_run')
if target.exists() or output.exists():
    raise SystemExit('Replay already exists; inspect it or choose a new scratch path')
config = json.loads(source.read_text())
config['output_root'] = output.as_posix()
workers = []
for proxy, gpus in [('OPT', [0]), ('Falcon', [0, 1])]:
    worker = next(w.copy() for w in config['workers'] if w['proxy'] == proxy)
    worker.update(name=proxy.lower(), gpus=gpus,
                  cell_modulus=1, cell_remainder=0)
    workers.append(worker)
config['workers'] = workers
target.write_text(json.dumps(config, indent=2) + '\n')
print(target)
PY
```

The copied configuration retains all 80 input paths, their SHA-256 values,
the 20-development/60-test split, numerical settings, and the frozen protocol
hash. The output is new; never point this replay at
`results/runs/reference_shaping/` because that directory contains the
archived run.

## 4. Audit and score the development split

First validate only the 20 development cells. The test files are not opened:

```bash
python3 scripts/audit_reference_split.py \
  --config scratch/reference_shaping.json --split development
```

Check that both model configurations fit and return finite scores for a
synthetic 2048-token input before processing the benchmark:

```bash
python3 scripts/launch_reference_shaping.py \
  --config scratch/reference_shaping.json --split development \
  --resource-smoke OPT --gpus 0
python3 scripts/launch_reference_shaping.py \
  --config scratch/reference_shaping.json --split development \
  --resource-smoke Falcon --gpus 0,1
```

Score the complete development split, one proxy at a time so that the two
workers can reuse the same GPUs:

```bash
python3 scripts/score_reference_moments.py \
  --config scratch/reference_shaping.json --split development --worker 0
python3 scripts/score_reference_moments.py \
  --config scratch/reference_shaping.json --split development --worker 1
python3 scripts/analyze_reference_shaping.py \
  --config scratch/reference_shaping.json --stage development
python3 scripts/analyze_reference_shaping.py \
  --config scratch/reference_shaping.json --stage freeze
```

The scorers write document-level `.jsonl` files and per-cell manifests under
`scratch/reference_shaping_run/development/{OPT,Falcon}/`. The analysis writes
`analysis/development/` and then `FROZEN_SELECTION.json`. Each scorer checks
the input hash before reading a cell. The development analysis checks the
complete score roster; the freeze step checks its code, input, and output
hashes. A failed or partial stage must be resolved before continuing.

## 5. Open, audit, and score the test split

The test scripts require an authorization file bound to the new
configuration and frozen development results. Create it only after the
development and freeze commands have succeeded:

```bash
python3 - <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

config_path = Path('scratch/reference_shaping.json')
config = json.loads(config_path.read_text())
root = Path(config['output_root'])
frozen = root / 'FROZEN_SELECTION.json'
if not frozen.is_file():
    raise SystemExit('Development freeze is missing')
authorization = {
    'authorized': True,
    'config_sha256': sha256(config_path),
    'frozen_protocol_sha256': config['frozen_protocol_sha256'],
    'frozen_selection_sha256': sha256(frozen),
    'opened_at_utc': datetime.now(timezone.utc).isoformat(),
}
with (root / 'TEST_ACCESS_AUTHORIZATION.json').open('x') as stream:
    json.dump(authorization, stream, indent=2)
    stream.write('\n')
PY
python3 scripts/audit_reference_split.py \
  --config scratch/reference_shaping.json --split test
python3 scripts/score_reference_moments.py \
  --config scratch/reference_shaping.json --split test --worker 0
python3 scripts/score_reference_moments.py \
  --config scratch/reference_shaping.json --split test --worker 1
```

The source audit applies the frozen source-overlap rule, including `wp:11`.
The document caches are saved under `scratch/reference_shaping_run/test/`.
The scorer and final evaluator enforce the configuration, code, software,
development-artifact and test-authorization hashes.

The development analysis and frozen receipts preserve the scientific access
history required by this scoring protocol. The final paper's metrics are
computed by the next command.

## 6. Calibrate and evaluate the final detector on the fresh scores

Complete the end-to-end main detector rerun using the new moment caches:

```bash
python3 scripts/evaluate_detector.py \
  --config scratch/reference_shaping.json --output scratch/detector_fresh
```

This uses the same calibration and evaluation rules as section 2. It first
checks the new development freeze, test authorization, source audit, and
per-cell scoring hashes. The outputs have the same layout as section 2;
compare the two `summary.csv` files and inspect per-task differences.
Exact paper parity is required for the archived CPU replay. A new model
forward run can produce numerical differences, so `--check-paper` is optional
for fresh scores and fails if the differences exceed `1e-12`.

For regression checks of the formulas, test-access gate, packaged paths, and
scorer initialization, install `requirements-dev.txt` and run:

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest -q tests
```

## 7. Recompute the reference-shaping comparison from included moments

```bash
python3 analysis/evidence/analyze_reference_calibration.py \
  --output scratch/reference_calibration
```

This rebuilds the Table 5/Table 14 comparison, all 1,320 task rows, threshold
changes and paired intervals. It writes only to the requested scratch directory;
published table files remain unchanged. It checks each task against the recorded paper
exports and writes a new directory under `scratch/`.

## Limits of a full historical rerun

Sections 1–2 verify the included release, rebuild the final detector from
archived moments, and regenerate the supported tables. Sections 3–6 rerun
the main detector from the included benchmark texts through model scoring,
human calibration, and final evaluation.

The repository does not currently contain a single executable pipeline that
starts with the original upstream human corpora and rebuilds every generation
condition, every external baseline, and all
19 paper tables from new GPU outputs. Earlier generation may require source
corpora under `external_data/`; some baseline reruns require upstream code
under `external_code/`; all GPU reruns require checkpoints under `models/`.
For token-watermark, CGW and HeavyWater generation, install the additional
imports with:

```bash
python3 -m pip install -r requirements-generation.txt
```

The vendored source subset and optional ITSEdit verification acceleration are
described in `third_party/README.md`. ITSEdit generation also works with the
Python fallback and does not require compiling that extension. SemStamp
requires a separately obtained source checkout and embedding dependencies as
described in `external_code/README.md`. These dependency instructions do not
replace the earlier source corpora or establish exact historical generation
parity. The recorded final benchmark,
baseline scores, calibrated score frames, and table exports are included so
that every reported result remains inspectable and checkable without those
external resources. Use `docs/RESULTS_INDEX.md` for the source of each
figure and table, and `docs/PROVENANCE.md` for the release boundaries.


### HeavyWater prefix diagnostic

The read-only command `python3 scripts/audit_heavywater_prefix.py` checks the
published prefix decomposition against the release manifest, original benchmark
hashes and numerical identities. It preserves `results/runs/heavywater_prefix/`.
The old-versus-corrected replay comparison is not rerun unless the earlier CSV
is explicitly supplied with `--previous-metrics`; its historical audit remains
included. Historical code hashes are not hashes of the edited public scripts.

A new GPU diagnostic requires the earlier prepared source workspace, including
`extension/generation_requests.json` and `dataset/complete_prompt_free/`.
After obtaining that workspace and installing the generation dependencies:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/diagnose_heavywater_prefix.py \
  --base scratch/clean80_english20_test60_plus3wm \
  --output scratch/heavywater_prefix
```

Use a new output directory for each run. The source record named
`results/runs/heavywater_prefix/diagnostic_source.py` is preserved for historical
inspection; use `scripts/diagnose_heavywater_prefix.py` as the executable entrypoint.
