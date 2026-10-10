# No-tilt HeavyWater: fixed CPU implementation and sequence-moment checks

Written before computing new numerical results on 2026-09-13. This is a bounded synthetic mechanism check of the local HeavyWater implementation, not an LLM benchmark or a claim about which mechanism caused historical score reversal. No GPU, LM forward, parameter selection or seed selection after results.

## Actual code and fixed settings

Import `third_party/external_watermarks/HeavyWater_SimplexWater/watermark/heavy_tail_randscore.py` and its actual `top_p_indices` dependency. Create `HeavyTailLogitsProcessor` via `__new__`, set the fields required by `tilt_q_HT`, and call that exact method on CPU. Use its original normalized lognormal/gamma cost generator. Set tilt=False, top_p=0.999, temperature=1, sinkhorn_reg=0.05, sinkhorn_thresh=1e-5 and num_iter=2000. Do not replace the production solver with a stabilized solver if it fails; preserve warnings and residuals.

Fixed grid: (vocabulary size,state count) in {(16,64),(32,128),(64,1024)}, cost distribution in {lognormal,gamma}, seeds in {15485863,20260913}, and four strictly positive input profiles:

- uniform: p_i=1/V;
- geometric: softmax(-0.4*i), i=0,...,V-1;
- peaked: p_0=0.97, p_i=0.03/(V-1) otherwise;
- singleton_control: p_0=0.9995, p_i=0.0005/(V-1) otherwise.

The grid contains 48 cases. For singleton support the actual implementation returns the original input distribution; record this as a separate bypass path, not as an OT marginal test. For other cases the target marginal is the top-p-retained input renormalized over its support. Report differences against both processed and original input so that truncation is not attributed to the transport step.

## Capture and parity

Wrap the module's `ot.sinkhorn` only to capture its input tensors and returned plan while calling the original solver unchanged. The production state-1 call is the primary computation. Obtain all-state conditional distributions by column normalization of that captured plan, and verify states1, floor(k/2), k through additional direct calls of the actual method. Independently call the same POT solver with log=True to record iteration/error diagnostics and verify parity with the captured plan. Do not alter cost normalization, data type or solver tolerances. Record import/source hashes, package versions, actual tensor devices and warnings.

## Marginal and score diagnostics

For the captured plan P with desired row marginal a and column marginal u=1/k, report L1 row and column residuals. With Q_s=P[:,s]/sum_iP[i,s], the exact-arithmetic bound is

    || mean_s Q_s - a ||_1 <= ||P1-a||_1 + ||P^T1-u||_1.

Compute the bound in float64 from the returned FP32 plan, and separately measure FP32 column-normalization rounding. Report any discrepancy rather than concealing nonconvergence. All-state mean drift is compared against the processed marginal; the original-input difference also contains top-p truncation. Record signed drifts for two fixed scores:

- aligned score ell_i=log(p_i), based on the original input;
- mismatched proxy score logsoftmax(sin(0.7*i)+0.4*cos(0.13*i)).

For each score report baseline expectation, all-state mean expectation and drift, state1 drift, minimum/maximum state drift and positive/negative/near-zero state counts (absolute tolerance1e-8). No state or seed is chosen for a desired drift. Uniform aligned scores are constant and need not exhibit both signs. Bound score drift by max_i|ell_i| times marginal L1 discrepancy; also report the centered-range bound including the total-mass mismatch term. Bounds are numerical audits, not a human false-positive guarantee.

## Fixed versus fresh state: exact expectations

With a fixed input distribution and fixed Q_s at every position, evaluate document lengths n in {1,16,256}, without sampling:

- fresh_uniform: independently choose a uniform state at each position, then sample a token from its conditional distribution;
- repeated_state1: always use the predeclared state1;
- repeated_uniform_state: draw a state uniformly once per document and reuse it for all tokens.

Let mu_s and v_s be the conditional token score mean and variance. Fresh-state document means have expectation mean_s(mu_s) and variance [mean_s(v_s)+Var_s(mu_s)]/n. Reusing a uniformly drawn state has the same expectation but variance mean_s(v_s)/n+Var_s(mu_s). Fixed state1 has expectation mu_1 and variance v_1/n. Verify these identities and the nonnegative excess variance (1-1/n)Var_s(mu_s). This fixed-context conditional-independence toy is not the actual autoregressive/hash process; it only demonstrates that one-token marginal preservation does not determine document-level dependence or variance.

## Hand-checkable counterexample

Use p=(0.9,0.1), two equally probable states with Q1=(0.82,0.18), Q2=(0.98,0.02). Verify their mean equals p exactly, while the aligned log-probability mean drifts by opposite signs in the two fixed states. This is a valid no-tilt marginal-preserving coupling, not a claim that a particular generated cost matrix yields these exact columns. Also verify its fresh/reused-state variance identities.

## Output and scope

Write `scratch/heavywater_no_tilt_numerics/` with full case, score, sequence-moment and solver tables, saved small matrices, source hashes, all numerical checks and a short English report. Preserve nonconverged cases and adverse signs. No comparison of two toy cost distributions proves that HeavyWater is uniquely vulnerable, that gamma/lognormal always reverses scores, or that the actual secret state is fresh uniform conditional on every text prefix.
