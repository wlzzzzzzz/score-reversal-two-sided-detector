# Configurations

- `benchmark_cells.json`: the 80 included text cells, development/test split,
  generation conditions, sample counts and input hashes.
- `original/reference_shaping.json`: the fixed document-moment protocol and
  starting configuration for the complete main-detector replay.
- `original/review_supplement.json` and `original/opt_factorial.json`: the
  original OPT/Falcon baseline and reference-score cache configurations.

For a new model run, follow `docs/REPRODUCE.md`, copy the configuration below
`scratch/`, select local GPU/model locations and a new output directory, then
create new source, code and freeze receipts. Historical digests record the
original execution; `manifest.json` verifies the published release.
