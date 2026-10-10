# Resolving Score-Ordering Reversal in Watermarked Text

Experiment code, benchmark texts, document-level scores, and complete table
exports for **“Resolving Score-Ordering Reversal in Watermarked Text: An
Asymmetric Two-Sided Test for LLM Text Detection”**.

The manuscript is available in [preprint format](paper/paper.pdf).
The work is under review; author information is currently anonymized.
The repository covers the final paper, its appendices, and the code and
scientific provenance needed to reproduce those results.
No model checkpoint or Python virtual environment is included.

## Main result

The detector combines an entropy-weighted, shaped-reference upper statistic
with an original-reference, uniformly weighted lower statistic. Human
calibration scores turn them into an asymmetric two-sided test. The main
evaluation has 660 settings, 800 calibration humans, 2,360 test humans, and
129,800 machine texts. Mean AUROC is **0.8219** with OPT-2.7B as a single
proxy and **0.8317** with the Falcon-7B/Falcon-7B-Instruct pair. The exact
underlying figures are in `analysis/data/current/summary.csv`.

## Repository layout

| Path | Contents |
| --- | --- |
| `paper/` | Complete manuscript PDF in preprint format |
| `analysis/data/` | Derived CSV/NPZ results underlying all 19 paper tables and both figures |
| `analysis/evidence/` | Analysis, table-rendering, numerical-check, and controlled-generation code |
| `analysis/tables/`, `analysis/figures/` | Recorded result tables and figures |
| `data/README.md` | Source attribution and data-term status |
| `data/benchmark/complete_prompt_free/` | All 80 development/test dataset-generator cells with human and eleven generation conditions |
| `configs/benchmark_cells.json` | Portable relative-path roster and original SHA-256 values for those 80 cells |
| `configs/original/` | Portable copies of frozen experiment configurations |
| `scripts/` | Final detector evaluation, release checks, and generation, scoring, calibration, baseline and sensitivity implementations |
| `experiments/` | Matched FD/Ada two-tail controls and HeavyWater generation controls |
| `results/baselines/` | Baseline score caches, benchmark summaries, and audits |
| `results/runs/` | Document-level scores and experiment results in short, descriptive directories |
| `third_party/` | Paper watermark source dependencies with their original licenses; see `third_party/README.md` |
| `manifest.json` | Repository-relative path, byte count, and SHA-256 of every release artifact |
| `models/`, `external_data/`, `external_code/` | Local locations for optional checkpoints and inputs needed for full reruns |

See [docs/RESULTS_INDEX.md](docs/RESULTS_INDEX.md) for a table-by-table map,
[docs/REPRODUCE.md](docs/REPRODUCE.md) for step-by-step table replay and fresh
80-cell model scoring through final detector evaluation,
[docs/PROVENANCE.md](docs/PROVENANCE.md) for provenance and release limitations,
and [docs/PATHS.md](docs/PATHS.md) for path conventions.

## Verify this release

Start from the repository root. The first two checks need only Python:

```bash
python3 scripts/check_release.py
python3 scripts/check_paths.py
```

The verifier checks the full release inventory, all input hashes, the main
result values, and the 19 table/2 figure inventory. To rebuild the final
detector's calibration and scores from the included document moments, then
check the paper tables without GPU inference:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-analysis.txt
python3 scripts/evaluate_detector.py --output scratch/detector --check-paper
python3 analysis/evidence/render_current_tables.py --check-only
python3 analysis/evidence/render_supplement_tables.py --check-only
python3 analysis/evidence/render_reference_calibration_table.py --check-only
python3 analysis/plot_score_distributions.py --check-only
```

Use Python 3.10 or 3.11 for a new environment. `--check-paper` compares all
1,320 final-detector task rows, aggregate metrics, and deployment metrics
against the recorded results. The table renderer includes all ten detectors
in the final paper's generator and dataset comparisons. The figure check
recomputes the empirical quantiles and all 660 native OPT task AUROCs.
Outputs are written to a new directory under
`scratch/`; choose a new name for each repeat run.

To save the main table exports and regenerate Figure 1 without changing the
published artifacts:

```bash
python3 analysis/evidence/render_current_tables.py --output-dir scratch/tables
python3 analysis/plot_score_distributions.py --output-dir scratch/figures
```

Figure 1 uses the included empirical quantiles. Default rendering requires only
the analysis dependencies; fonts may differ from the published PDF. Figure 2
is a conceptual schematic, provided as the final manuscript's vector PDF in
`analysis/figures/overview.pdf`. See the reproduction guide for the remaining
main-text tables and the optional LaTeX font setting.

For a fresh model run from the included benchmark texts, follow
[the complete staged procedure](docs/REPRODUCE.md#3-prepare-for-fresh-model-scoring-of-all-80-benchmark-cells).
It covers checkpoint placement, a new replay configuration, development and
test scoring, source audits, the frozen test gate, and result reports. The
guide continues through human calibration and final detection metrics, and
identifies the external inputs needed to regenerate the benchmark
and every historical baseline from earlier source material.

## Repository size

The included release is approximately 2.1 GiB before Git compression.
Document-level caches are included for numerical reproduction and scientific
audits. Use Git or GitHub Desktop to upload the complete repository: some
included files exceed the browser's 25 MiB limit. Every file is below GitHub's
100 MiB single-file limit, so these files do not require Git LFS.
GitHub also limits each push to 2 GiB; this applies to the Git transfer, not
the uncompressed working directory. See the official
[file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
and [push limit](https://docs.github.com/en/get-started/using-git/troubleshooting-the-2-gb-push-limit).
Run the release checks before creating your first commit.

## Citation and license

See [CITATION.bib](CITATION.bib) for the provisional manuscript citation.
Authors are currently anonymized; an arXiv identifier will be added when available.
The original code and accompanying software documentation are released under
the [MIT License](LICENSE). Third-party components retain their own copyright
and license terms, including the Fast-DetectGPT notice in
[`third_party/fast-detect-gpt-LICENSE`](third_party/fast-detect-gpt-LICENSE).
Source datasets, model checkpoints, and the manuscript are outside the scope
of this code license. See [data source and terms](data/README.md) for the
remaining redistribution questions affecting the included text corpus.
