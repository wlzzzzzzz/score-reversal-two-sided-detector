# Paper result index

The manuscript has 19 tables and 2 figures. This index maps each to the
numeric exports and recorded experiment runs. A table
may combine several exports; the retained result tables show recorded values.

| Paper item | Primary result files | Experiment/code |
| --- | --- | --- |
| Figure 1, score reversal | `analysis/data/current/score_distributions.csv`; `analysis/figures/score_distributions.pdf` | `analysis/plot_score_distributions.py` |
| Figure 2, detector | `analysis/figures/overview.pdf` | final vector schematic supplied with the manuscript; `analysis/evidence/asymmetric_reference.py` specifies the test |
| Table 1, overall baselines | `analysis/data/current/summary.csv`, `task_metrics.csv` | `results/baselines/review_supplement/`; `results/runs/calibrated_scores/` |
| Table 2, generators | `analysis/data/current/task_metrics.csv` | `analysis/evidence/render_current_tables.py`; all ten detectors |
| Table 3, watermark methods | `analysis/data/upper_all_conditions/task_metrics.csv`; `analysis/data/current/task_metrics.csv` | upper and joint score exports |
| Table 4, two-sided controls | `analysis/data/two_sided_controls/fd/summary.csv`, `ada/summary.csv`; `analysis/data/current/deployment.csv` | `analysis/evidence/two_sided_controls.py`; `results/runs/fd_control/`, `results/runs/ada_control/` |
| Table 5, staged ablation | `analysis/data/reference_calibration/summary.csv`; `analysis/data/two_sided_controls/fd/summary.csv` | `analysis/evidence/analyze_reference_calibration.py` |
| Table 6, lower branch | `analysis/data/mechanism_evidence/summary.csv` | `analysis/evidence/analyze_lower_branch.py`; `results/runs/calibrated_scores/test/` |
| Table 7, OPT score ordering | `analysis/data/current/score_distributions.csv`; `analysis/data/two_sided_controls/fd/by_task.csv` | `analysis/plot_score_distributions.py --check-only`; `analysis/evidence/render_current_tables.py` |
| Table 8, HeavyWater controls | `analysis/data/heavywater_controls/`; `results/runs/heavywater_controls/` | `experiments/heavywater_causal/`; `analysis/evidence/check_heavywater_controls.py` |
| Table 9, human sources | `analysis/tables/evaluation_sources.tex`; `data/benchmark/complete_prompt_free/`; `configs/benchmark_cells.json` | `results/baselines/review_supplement/main_human_source_overlap_audit.json` |
| Table 10, datasets | `analysis/data/current/task_metrics.csv` | `analysis/evidence/render_current_tables.py`; all ten detectors |
| Table 11, paired comparisons | `analysis/data/two_sided_controls/*/paired_domain_intervals.csv` | `analysis/evidence/two_sided_controls.py` |
| Table 12, controls by condition | `analysis/data/two_sided_controls/*/by_mechanism.csv` | same matched controls |
| Table 13, entropy cap | `analysis/data/sensitivity/kappa.csv`, `kappa_tasks.csv` | `results/runs/entropy_cap/`; `analysis/evidence/measure_kappa.py` |
| Table 14, shaped reference | `analysis/data/reference_calibration/by_mechanism.csv`, `paired_intervals.csv` | `analysis/evidence/analyze_reference_calibration.py` |
| Table 15, temperature and mass | `analysis/data/sensitivity/temperature_mass.csv` | `results/runs/reference_shaping/` |
| Table 16, local settings | `analysis/data/sensitivity/local_construction.csv` | `results/runs/local_parameters/`; `analysis/evidence/measure_construction.py` |
| Table 17, branch attribution | `analysis/data/directional/attribution.csv` | `analysis/evidence/analyze_current_supplement.py` |
| Table 18, calibration bank | `analysis/data/calibration/bank_summary.csv`, `bank_replicates.csv` | `analysis/evidence/analyze_current_supplement.py` |
| Table 19, natural length | `analysis/data/length_strata/summary.csv`, `task_auc.csv` | `analysis/evidence/analyze_current_supplement.py` |

The original document-level moments for the complete test are in
`results/runs/reference_shaping/` (20 development
and 60 test cells for each proxy). The final scoring frames and calibration
banks are in `results/runs/calibrated_scores/`.
Original machine and human texts are the 80 JSON cells in `data/benchmark/`.
The `analysis/data/` exports are sufficient to audit and rebuild the paper
tables without running the language models again.

## Appendix text and scientific provenance

- Appendix E.1: `analysis/data/positive_alignment/` and
  `analysis/evidence/analyze_positive_alignment.py` preserve the entropy-alignment
  measurements; `analysis/data/upper_unwatermarked/` preserves the corresponding
  upper-branch comparisons.
- Appendix F.1: `analysis/data/calibration/allocation.csv` records the three
  fixed tail allocations; the bank-size experiment is listed under Table 18.
- Appendix A.8: `results/runs/heavywater_controls/` includes all 360 generated
  documents, source prompts, generation/scoring metadata, first-attempt checks,
  and original-record matches. `results/runs/heavywater_prefix/` preserves
  the transport/context decomposition audit underlying the mechanism analysis.
- Numerical checks: `analysis/evidence/checks/` and `REFERENCE_THEORY_AUDIT.json`
  retain formula, cache, cohort and rank-identity checks. HeavyWater validation
  is stored alongside its results in `results/runs/heavywater_controls/`.
- Scoring provenance: `results/runs/reference_shaping/` retains the original
  development analysis, observation lock, source-overlap audit, freeze and test
  access receipts. They document the original selection and access history and
  are dependencies of the fresh-run gate.
- Baseline provenance: `results/baselines/review_supplement/` retains checkpoint,
  precision, source-overlap and model-pair audits. Its `analysis/aligned_scores.pkl`
  contains metadata and the paper baseline scores for every original row.

CPU checks of the lower branch and the four external baseline score columns:

```bash
python3 analysis/evidence/analyze_lower_branch.py --check-only
python3 analysis/evidence/check_baselines.py
```
