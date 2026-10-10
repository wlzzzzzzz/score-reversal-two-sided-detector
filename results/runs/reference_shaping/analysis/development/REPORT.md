# Fixed reference shaping: development

Rows: 96000; domains: 4; excluded source bundles: 0.

Development OOF b; intervals condition on fixed overlapping-fold predictions. Four domains are unstable; these are descriptive, not held-domain confirmation.

All lambda values are fixed; no score or parameter is selected. G is an analysis predictor and never changes the detector score. Endpoints are recomputed in float64 from new matched sufficient statistics; tiny differences from stored float32 R can affect ties. The maximum differences are recorded in run_manifest.json.

| Proxy | Score | Mean AUROC | P10 | CVaR10 | Flip rate |
|---|---|---:|---:|---:|---:|
| OPT | E00 | 0.7241 | 0.5187 | 0.4841 | 0.0682 |
| OPT | E10 | 0.7531 | 0.5398 | 0.5119 | 0.0318 |
| OPT | E01 | 0.7224 | 0.5170 | 0.4806 | 0.0773 |
| OPT | E11 | 0.7514 | 0.5350 | 0.5074 | 0.0318 |
| OPT | E_mix_half | 0.7425 | 0.5327 | 0.5006 | 0.0455 |
| OPT | T_only | 0.7159 | 0.3629 | 0.2769 | 0.2091 |
| OPT | C_only | 0.2436 | 0.0012 | 0.0004 | 0.7727 |
| OPT | U00 | 0.7115 | 0.5084 | 0.4717 | 0.0773 |
| OPT | U11 | 0.7111 | 0.5145 | 0.4843 | 0.0636 |
| OPT | R0 | 0.7340 | 0.3372 | 0.2711 | 0.2682 |
| OPT | R1 | 0.7553 | 0.4090 | 0.3406 | 0.1909 |
| Falcon | E00 | 0.7706 | 0.5081 | 0.4849 | 0.0909 |
| Falcon | E10 | 0.7981 | 0.5430 | 0.5086 | 0.0318 |
| Falcon | E01 | 0.7675 | 0.5015 | 0.4787 | 0.0909 |
| Falcon | E11 | 0.7958 | 0.5353 | 0.5022 | 0.0364 |
| Falcon | E_mix_half | 0.7816 | 0.5121 | 0.4817 | 0.0773 |
| Falcon | T_only | 0.7862 | 0.5813 | 0.4514 | 0.0500 |
| Falcon | C_only | 0.2237 | 0.0033 | 0.0020 | 0.8091 |
| Falcon | U00 | 0.7507 | 0.5177 | 0.4876 | 0.0682 |
| Falcon | U11 | 0.7547 | 0.5254 | 0.4898 | 0.0636 |
| Falcon | R0 | 0.7724 | 0.4235 | 0.3704 | 0.2000 |
| Falcon | R1 | 0.8101 | 0.5404 | 0.4871 | 0.0500 |

H1 support criterion met: False; H2: False; H3: False.

See hypothesis_results.json for directional correlations, domain counts, 2,000 paired domain-bootstrap intervals, sign groups and split-half sensitivity. Failure to meet a support criterion is inconclusive, not proof of its opposite.

atomic.csv contains threshold-exact task ROC operating points (no interpolation). shared_operating_points.csv separately reports actual human FPR and machine TPR under strict frozen/LODO thresholds; these are not mislabeled as test TPR at a fixed achieved FPR.

The original benchmark was inspected historically; this protocol enforces prospective separation for the new experiment, not historical blindness.
