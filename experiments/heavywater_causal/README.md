# HeavyWater generation controls

The four scripts in this directory implement the fixed generation controls
and provenance analyses used in paper Table 8:

| Script | Purpose |
| --- | --- |
| `study.py` | Prepare 40 calibration-domain prompts and generate nine control arms with OPT |
| `analyze.py` | Aggregate all saved control outcomes and compute paired intervals |
| `audit.py` | Match historical generation records and analyze saved prefix states |
| `first_attempt.py` | Compare existing test scores with the subset accepted on the first generation attempt |

Recorded outcomes and audit inputs are in `results/runs/heavywater_controls/`.
Table exports are in `analysis/data/heavywater_controls/`. To validate the
included results without model inference, run from the repository root:

```bash
python3 analysis/evidence/check_heavywater_controls.py
```

Generation requires `models/opt-2.7B/`, the source registry, and earlier
generation manifests referenced by `study.py`. Historical record matching
also requires the earlier generation records referenced by `audit.py`.
These earlier inputs are not all distributed in this release. The study
scripts write under `results/runs/heavywater_controls/`; use a separate copy
for a fresh study to preserve the published outputs. The main detector rerun
is documented in `docs/REPRODUCE.md`.
