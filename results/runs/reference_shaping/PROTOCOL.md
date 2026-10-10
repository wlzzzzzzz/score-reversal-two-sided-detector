# Reference shaping: development on 20 cells, held-domain testing on 60 cells

Protocol version: 20260910-v1. Written before the new development experiment and before any new test access. The original 60-cell benchmark has been inspected in earlier work and in prior exploratory analyses. This experiment enforces prospective separation from this point; it does not erase that history or describe the test as historically unseen.

## Scientific question and hypotheses

When does a shaped reference improve a two-sided conditional discrepancy detector?

The scoring proxy is frozen. OPT uses OPT-2.7B for both reference and observed log probabilities; Falcon uses Falcon-7B for reference and Falcon-7B-Instruct for observed log probabilities. Both configurations are fixed before development observations. This is not a search over proxy models.

Let q0 be the untempered, full-vocabulary reference and q1 the T=1.2, retained-mass=0.8 reference. The observed log probabilities remain unchanged. Entropy weights are computed from the untempered reference with cap=1.5, exactly as in the existing algorithm. Define N_j=sum_t w_t(ell_t-mu_jt), V_j=sum_t w_t^2 nu_jt, and C01=sum_t w_t^2(mu_0t-mu_1t)^2. The score E_ab=N_a^2/V_b separates reference center and conditional scale. Denominators use the existing standard-deviation floor 1e-9. E10 and E01 are diagnostic hybrids, without claiming that their denominator is the variance of their numerator's reference experiment.

H1 (direction and magnitude of center correction): a reference-only prediction of beneficial human centering is positively associated with the held-domain AUROC gain of E10 over E00. Let b be the development-human mean N0, calculated separately for each proxy, giving each of the four development domains equal weight and deduplicating human sources. Let C=N0-N1=sum_t w_t(mu_1t-mu_0t). Define G(x)=(2*b*C-C^2)/V0. G>0 predicts a center shift in the appropriate direction without excessive displacement. This is an approximation to human bias correction, not an identity for classification performance. It may fail because human bias does not transfer or because machine separation changes differently. The predictor never enters the detector score.

For each dataset-generator-mechanism task, G_task is the average of the human median G and the machine median G. This uses reference quantities plus the frozen development constant, not machine observed residuals or shaped machine detection outcomes. Primary H1 outcome is Delta_center=AUROC(E10)-AUROC(E00). Report the within-proxy Spearman correlations and their arithmetic mean. Support requires positive correlation for both proxies and a positive lower 95% domain-bootstrap bound for the mean correlation. Also report the pre-fixed G>0 and G<=0 groups, including task/domain counts. No threshold changes after seeing test results.

H2 (heterogeneous scale changes can erase a center gain): among tasks with G_task>0, greater H_scale=IQR(log(V1/V0)) is negatively associated with Delta_scale_given_center=AUROC(E11)-AUROC(E10). H_scale pools equal numbers of human and corresponding machine documents. The primary association is the mean of within-proxy Spearman correlations, with the same joint-domain bootstrap. Support requires negative correlations for both proxies and an upper 95% bound below zero. The all-task association is always reported. A shared multiplicative scale cannot change AUROC, so a mean variance ratio alone is not evidence of a scale penalty. Empty groups, fewer than three domains or constant predictors produce an inconclusive result, not a revised hypothesis.

H3 (softening the intervention reduces downside): the fixed mixture q_half=(q0+q1)/2 reduces the frequency of tasks losing more than 0.02 AUROC relative to E00, compared with hard shaping E11, while losing no more than 0.005 macro AUROC against E11. These margins are fixed before development. Report both proxies separately. This is a candidate hypothesis, not a guarantee or a parameter selected using the test.

H3 support requires, for both proxies, an upper 95% paired domain-bootstrap bound below zero for the harmful-task-rate difference (half minus hard) and a lower 95% bound above -0.005 for the macro AUROC difference (half minus hard). Failure to meet a support criterion is not automatically proof of its opposite.

The mixture is computed exactly from N_lambda=(1-lambda)N0+lambda*N1 and V_lambda=(1-lambda)V0+lambda*V1+lambda*(1-lambda)*C01. The between-reference mean term is mandatory. Only lambda in {0,0.5,1} is evaluated. No best lambda is selected. All three are fixed scientific comparisons.

## Fixed experiment family

Primary entropy-weighted scores: E00, E10, E01, E11 and E_mix_half. Report the full center/scale interaction I=AUC11-AUC10-AUC01+AUC00. Secondary diagnostics: temperature-only energy, truncation-only energy, uniformly weighted E00 and E11, and signed R0 and R1. These controls are fixed now and cannot become new tuned candidates after testing.

All input texts use their complete prompt-free form with the existing first-2048-token cap, special-token handling and at least one predicted token. Forward passes use FP16 and token moments use FP32. Moments and weights are identical within each matched comparison. Source IDs, input hashes, batch settings and numerical versions are recorded. Both reference configurations reuse the same model forward passes. Fresh sufficient-statistic recomputation is necessary because existing scalar caches do not retain N0,V0 or C01. Numerical changes relative to older FP16 batch caches are audited on development only.

## Development and test boundaries

Development domains: c4, squad, writing, xsum; each crossed with GLM-4-9B-0414, falcon, llama-2-7b, opt-2.7B, opt-6.7B. This is 20 cells but only FOUR independent dataset groups, with 800 unique human sources. Every machine stream contains unwatermarked, KGW, SWEET, Unigram, EXP, SynthID, DIP, ITSEdit, CGW, HeavyWater and SemStamp. Development includes all 200 sources per cell, not the first 20 documents of test cells.

Test domains: cmv, cnn, dialogsum, eli5, hswag, imdb, pubmed, roct, sci, tldr, wp, yelp, crossed with the same five generators. This is 60 cells but TWELVE dataset groups. The test role is unseen datasets in this experiment; generators and watermark mechanisms are shared. Test rows, scores and outcomes cannot be read until development reports, code, predictor constants and thresholds have been frozen. Path rosters and schemas may be inspected for implementation without reading test outcomes.

Development H1/H2 observations use leave-one-domain-out estimates of b: only the other three domains estimate b for each held development domain. Development association intervals, if reported, condition on these fixed OOF predictions and do not establish generalization with four domains. Foldwise signs and counts are reported. After development, b is computed once from all four domains and frozen for test analysis. No human recentering, learned scoring head, runtime routing, or test-label-dependent score reversal is introduced.

Test integrity checks execute only after the test gate opens. SHA hashes and source counts must match the fixed source roster. Exact text duplication across splits, or human 5-word-shingle containment at least 0.8 with at least 20 unique shingles on both sides, excludes the complete affected TEST source bundle from primary analysis. Development is unchanged. Exclusions are based only on text integrity and the pre-fixed rule; original text and raw score files are preserved. Report excluded IDs and counts, and unfiltered sensitivity separately. Cross-domain near matches below this fixed rule are not searched adaptively.

## Metrics and uncertainty

All AUROCs use higher-is-machine fixed orientation and identical human/machine rows within matched comparisons. Atomic tasks are dataset x generator x machine mechanism. Report macro task AUROC, P10, CVaR10 (mean of the lowest ceil(10% * task_count) tasks), AUROC<0.5 rate, and task-specific ROC TPR at 1% and 5% FPR. A task is harmful for H3 if its paired AUROC change from E00 is strictly below -0.02.

Separately, estimate shared 1% and 5% energy thresholds only from unique development humans using k=ceil((m+1)*(1-alpha)) and strict score>threshold. Development uses thresholds from the other three domains for OOF decisions. Test uses the frozen full-development thresholds; report unique-human FPR by domain and pooled, and macro machine TPR. These are distinct from task-specific ROC operating points. Human calibration selects thresholds only, not the score origin or reference configuration.

Main confidence intervals use 2,000 paired whole-domain bootstrap draws (seed 20260910), sampling the same domain indices for both proxies and all candidate scores. Every domain carries all its generators, sources and mechanisms, preserving shared-human dependence. They measure variation across these sampled domains, conditional on fixed development constants and observed within-domain samples. They are not intervals treating 60 cells or 660 tasks as independent. Four-domain development intervals are descriptive. Hypothesis verdicts use the directional interval criteria above; all three hypotheses and all predefined outcomes are reported without post-test hypothesis selection. Subgroup analyses are secondary and not evidence of universal benefit.

An additional fixed split-half sensitivity separates condition measurement and outcome measurement within a domain: source_id SHA256 parity assigns complete source bundles to halves A/B; estimate G,H on A and AUROC changes on B, then exchange them. Average the two directional within-proxy correlations when defined. This is secondary and never selects a score or revises a hypothesis. G is a robust proxy for predicted bias correction, not an exact expected energy reduction or sufficient condition. Conditional H2 groups can have unequal numbers of tasks by domain; their association is descriptive of those selected tasks and domain counts are disclosed.

## Execution gate and artifacts

1. Write this protocol, fixed config and data roster; hash them.
2. Validate development sources only, run synthetic formula/parity tests and a development-only smoke check.
3. Compute full development moments for both proxies and finish the development report, OOF observations and thresholds.
4. Write FROZEN_SELECTION.json (the historical filename denotes a freeze manifest; no lambda is selected), including config/protocol/code hashes, development artifact hashes, b and thresholds.
5. Only then write TEST_ACCESS_AUTHORIZATION.json containing all matching hashes and the UTC opening time. Both the source-audit and scoring executables refuse test access without these files.
6. Run the text-integrity audit, compute test moments, and run the frozen evaluator once. It must verify the code/config/development hashes before evaluation.
7. Produce a Chinese report with distinct hypotheses, development observations, held-domain test outcomes, negative results and limitations. No test result is used to amend the current experiment. Any later hypothesis is a separate experiment and cannot reuse this test as fresh confirmation.

This work does not modify the existing default detector or manuscript tables.
