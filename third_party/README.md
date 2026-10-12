# Third-party components

This tree contains the source dependencies used by the paper generation
workers. Each component retains its original license and source copyright
notices. The root MIT license covers the original project code, not a
relicensing of these components.

| Directory/file | Paper use | Upstream | License |
| --- | --- | --- | --- |
| `MarkLLM/` | KGW, SWEET, Unigram, EXP, SynthID, DIP, ITSEdit | [THU-BPM/MarkLLM](https://github.com/THU-BPM/MarkLLM) | Apache-2.0; `MarkLLM/LICENSE` |
| `external_watermarks/MarkMyWords/` | Binary/CGW generation and its imported utilities | [wagner-group/MarkMyWords](https://github.com/wagner-group/MarkMyWords) | Apache-2.0; component `LICENSE` |
| `external_watermarks/MarkMyWords_apps/` | `ConfigCode` type imported by MarkMyWords | [hendrycks/apps](https://github.com/hendrycks/apps) | MIT; component `LICENSE` |
| `external_watermarks/HeavyWater_SimplexWater/` | No-tilt HeavyWater processor and base class | [DorTsur/CC_WM_LLMs](https://github.com/DorTsur/CC_WM_LLMs) | MIT; component `LICENSE` |
| `fast-detect-gpt-LICENSE` | Analytic baseline functions in `scripts/fast_detect_gpt.py` | [baoguangsheng/fast-detect-gpt](https://github.com/baoguangsheng/fast-detect-gpt) | MIT |

These are paper-specific source subsets. Unused algorithms, attacks,
benchmark frontends and upstream environment dumps are omitted. Local changes
include portability/compatibility fixes, reduced algorithm registries and
package imports, and retention of only the C4 loader required by SynthID.
The recorded execution hashes identify the implementations used in the
original experiments.

Install the main analysis/model dependencies from the root requirement files.
For the generation workers, additionally install `requirements-generation.txt`.
For CGW, `scripts/markmywords_compat/hash_cpp.py` provides the required hash
interface without the upstream CUDA extension. ITSEdit generation has a Python
fallback for its verification import; its optional extension can be built from
the repository root with:

```bash
python3 third_party/MarkLLM/watermark/its_edit/cython_files/setup.py build_ext --inplace
```

SemStamp is an optional checkout under `external_code/SemStamp/`; see
[generation dependencies](../docs/REPRODUCE.md#optional-generation-and-baseline-dependencies).
Its source is excluded because a redistribution
license was not found in the inspected upstream tree. Its paper benchmark
texts and results remain included.
