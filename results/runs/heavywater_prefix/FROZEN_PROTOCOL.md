# Frozen no-tilt HeavyWater fixed-prefix decomposition

2026-09-13. This protocol precedes calculation of the new prefix effects. The benchmark reversal and current implementation audit are already known. This is a bounded, retrospective mechanism diagnostic, not an untouched benchmark or a full regeneration study.

Use only local OPT-2.7B and GPU 6. Select source slot 0 in cmv, cnn, hswag and sci from the HeavyWater extension generation requests. For each source, take its saved unwatermarked and HeavyWater continuation from the final complete_prompt_free data. Check request prompt hash. Tokenize the prompt and continuation separately with the checkpoint tokenizer; reconstruct prefixes by concatenation. These reconstructed text prefixes are not certified original generation token IDs. The same fixed prefix is used for all probability interventions.

Freeze continuation-prefix lengths at 64, 96, 128 and 160 scored tokens: at most 32 snapshots. Skip only if the continuation lacks the next token or the combined context exceeds model capacity; record every skip. These positions avoid uncertain historical early EOS minimum settings. Do not replace skipped texts or select positions using outcome scores.

At each prefix obtain prompt-conditioned raw generator logits and continuation-only OPT logits (with the tokenizer's normal BOS). The latter log probabilities define the detector feature ell. Evaluate:

1. raw prompt-conditioned softmax;
2. a current-baseline reconstruction with the loaded GenerationConfig top-k/top-p, explicit temperature 1, and no repetition/ngram penalties;
3. repetition penalty 1.1 on the same raw logits;
4. repetition penalty plus no-repeat 4-gram;
5. HeavyWater's internal top-p=.999;
6. the original no-tilt HeavyTailLogitsProcessor, k=1024, lognormal fixed cost, key15485863, dynamic_seed=markov_1, reg=.05, threshold1e-5, max2000 iterations.

Call the actual processor and intercept its POT call only to request diagnostic logs and capture the plan. Do not modify source code, scores, selection rules or output dataset. Compute every state-column conditional distribution and their uniform average; compare the actual chosen state to that average. Save actual state, mean/range/quantiles of all-state score changes, row/column residuals, L1 drift, and implementation parity.

Also solve each captured OT problem with float64 normalized marginals, the identical captured cost, reg=.05, threshold1e-10 and max5000 iterations. This is a numerical sensitivity control, not a replacement output. Record warnings and nonconvergence without dropping cases. Compare actual selected-column distribution and detector expectation against this tighter solution; small average marginal error alone does not certify selected-column accuracy.

Exact finite-prefix telescoping decomposition relative to the continuation-only raw reference:

  selected minus detector-reference expectation
  = prompt/context difference
  + repetition effect
  + ngram effect
  + internal nucleus effect
  + uniform-state numerical marginal effect
  + selected-state minus uniform-state effect.

All expectations use the same detector ell. Also report selected minus reconstructed baseline. The two final OT terms are an exact numerical-plan decomposition; call the last state-selection deviation, without claiming it is purified of solver error. Float64 sensitivity addresses part of that issue.

Save model/tokenizer/config/source hashes, exact selected IDs, logits snapshots, per-prefix metrics, all-state shifts, group summaries and warnings to outputs/heavywater_no_tilt_prefix_20260913_v1. No generated continuations, no detector fitting or calibration, no test parameter selection. Report fixed-prefix effects separately from observed document-level response and length/quality-selection effects. Historical resolved unwatermarked decoding parameters were not fully recorded; current defaults are a sensitivity reconstruction, not proof of every historical baseline call.

## Implementation correction before the v2 repeat

The v1 run completed 25 eligible prefixes and recorded seven length skips. Inspection found that instantiating GenerationConfig() still leaves top_k and top_p as None in Transformers 5.0.0. Consequently its three baseline-related columns describe raw softmax, not the promised current-baseline reconstruction. They must not be used as a top-k comparison. The core six-term decomposition, actual processor output, numerical control and selected prefixes are unaffected.

Preserve v1 with its exact source and protocol. Repeat the same frozen inputs/positions/parameters into outputs/heavywater_no_tilt_prefix_20260913_v2, resolving None using GenerationConfig._get_default_generation_params() and update(defaults_only=True), the rule actually used by the local generate implementation. For this OPT checkpoint the result is top_k=50, top_p=1.0. No parameters or selected cases change in response to observed scores. Suppress only the model-load progress bar. The v2 result is the reported result; compare its unaffected quantities with v1 as a regression check.
