# HeavyWater: no-tilt processor

This directory contains the paper-specific source subset described in
`../../README.md`. The original license is in `LICENSE`.
Use the root requirement files and generation wrappers for this release.
The upstream project and local modifications are listed in the third-party index.

Both modules postpone type annotations for compatibility with the archived
Python 3.8 analysis environment; numerical routines are unchanged.

Only the generation kernel, cost generator, top-p helper and required base
processor are retained. Standalone watermark detectors and unrelated dataset,
fluency and evaluation helpers are omitted. The retained transport calculations
and base initialization follow the supplied implementation.
