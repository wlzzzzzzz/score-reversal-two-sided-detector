# Calibration sensitivity protocol

Both document-response formulas and their model moments remain fixed.

- Allocation: beta in {0.25, 0.5, 0.75}, full 800-human bank; both model configurations. Report joint overall/unwatermarked AUROC and power/actual FPR at nominal total 1% and 5%.
- Bank size: m in {199, 400, 800}, beta=0.5. Use 20 stratified subsets without replacement for each smaller bank, with per-domain counts differing by at most one. Seed 2026091306 plus replicate index; identical seed schedule across model configurations. Evaluate the full bank once.
- The 20-subset min/max ranges are conditional bank-perturbation ranges, not new-population confidence intervals. At m=199 an equal-allocation 1% test only just admits rejection.
- Retain default beta=0.5 and m=800. Default results agree with the main-table exports to 1e-12. No model inference is needed for these fixed-response measurements.

Portable results are in `data/calibration/`; source export hashes are in `data/current/SOURCE_AUDIT.json`.
