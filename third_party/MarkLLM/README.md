# MarkLLM: seven paper token-watermark algorithms

This directory contains the paper-specific source subset described in
`../README.md`. The original license is in `LICENSE`.
Use the root requirement files and generation wrappers for this release.
The upstream project and local modifications are listed in the third-party index.

The subset retains all seven paper generation algorithms. The unused vLLM
adapter, experimental rank-tracking base implementation and its RNG helpers
are omitted. Default configuration paths resolve relative to this package;
explicit configuration paths and overrides are supported. Invalid or missing
configuration files raise their original errors.
