# Path conventions

Run commands from the repository root. Paths recorded in code, configurations,
cohort manifests, and release metadata are relative to that root.

The 80 included benchmark cells live under `data/benchmark/`; recorded scores,
audits, and tables live under `results/` and `analysis/`. Model checkpoints are
not bundled. Put locally obtained checkpoints under `models/` using the model
directory names referenced by the scripts. Earlier generation stages may
require separate source corpora under `external_data/` or external baseline
code under `external_code/`. These optional locations are excluded from Git.

Historical execution receipts may contain relative source identifiers from
the original experiment layout. Their digests document the original execution.
Use `docs/RESULTS_INDEX.md` for current paper-artifact locations and
`docs/REPRODUCE.md` for executable paths in a fresh run. The root manifest
verifies the current published files.
Fresh workflow outputs use repository-relative directories under `scratch/`;
these intermediate outputs are not bundled. Release checks skip local model,
external-input, environment, and scratch contents so that installing the
dependencies does not change the published inventory.

The benchmark text itself is preserved exactly. A quoted path inside a human
or machine-generated document is part of that source text, not a repository
dependency. Standard Python shebangs and web links also retain their normal
forms.

## Exact bytes in Git

The root `.gitattributes` disables automatic line-ending conversion. Some
frozen CSV exports use CRLF while code and other exports use LF. Both forms
are intentional inputs to the recorded SHA-256 checks. Keep this file when
creating the repository so that Git settings on another operating system do
not change the published bytes.
