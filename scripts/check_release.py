#!/usr/bin/env python3
"""Check the copied artifact ledger and the final paper's central counts."""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from release_utils import release_files, sha256

ROOT = Path(__file__).resolve().parents[1]


def check():
    ledger = json.loads((ROOT / 'manifest.json').read_text())['files']
    errors = []
    seen = set()
    total_bytes = 0
    date_pattern = re.compile(r'(?:19|20)\d{6}|iclr20\d{2}', re.IGNORECASE)
    inventory = {p.relative_to(ROOT).as_posix() for p in release_files(ROOT)} - {'manifest.json'}
    for name in sorted(inventory):
        path = ROOT / name
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            errors.append(f'unsupported release symlink: {relative}')
        if (path.suffix == '.lock' or '.queue_claims' in relative.parts or
                path.name in {'SESSION_STATE.json', 'SCHEDULER_HANDOFF.json', 'RUNTIME_SCHEDULING_NOTE.md',
                              'LAUNCH.json', 'LAUNCH_EXIT.json', 'PROGRESS.json'}):
            errors.append(f'operational file in release: {relative}')
        for part in relative.parts:
            if date_pattern.search(part):
                errors.append(f'dated path component: {relative}')
                break
            if len(part) > 48:
                errors.append(f'long path component: {relative}')
                break
    for record in ledger:
        name = record['path']
        if Path(name).is_absolute() or '..' in Path(name).parts:
            errors.append(f'non-relative manifest path: {name}')
            continue
        if name in seen:
            errors.append(f'duplicate manifest path: {name}')
            continue
        seen.add(name)
        path = ROOT / name
        if not path.is_file():
            errors.append(f'missing: {name}')
            continue
        size = path.stat().st_size
        total_bytes += size
        if size > 100 * 2**20:
            errors.append(f'file exceeds GitHub 100 MiB limit: {name}')
        if size != record['bytes']:
            errors.append(f'size changed: {name}')
        elif sha256(path) != record['sha256']:
            errors.append(f'content changed: {name}')
    for name in sorted(inventory - seen):
        errors.append(f'unlisted release file: {name}')
    for name in sorted(seen - inventory):
        errors.append(f'non-release manifest entry: {name}')

    roster = json.loads((ROOT / 'configs/benchmark_cells.json').read_text())
    cells = roster['cells']
    if len(cells) != 80:
        errors.append(f'benchmark has {len(cells)} cells; expected 80')
    if sum(c['split'] == 'development' for c in cells) != 20:
        errors.append('expected 20 development cells')
    if sum(c['split'] == 'test' for c in cells) != 60:
        errors.append('expected 60 test cells')
    if sum(c['source_count'] for c in cells if c['split'] == 'development') != 4000:
        errors.append('development generator-paired human count differs')
    if sum(c['source_count'] for c in cells if c['split'] == 'test') != 11805:
        errors.append('raw test generator-paired human count differs')
    methods = set(roster['methods'])
    if len(methods) != 11:
        errors.append('expected 11 generation conditions')
    # The recorded overlap exclusion removes wp:11 across five generators.
    # This yields 11,800 paired human rows and 129,800 machine rows.
    if (sum(c['source_count'] for c in cells if c['split'] == 'test') - 5) * 11 != 129800:
        errors.append('machine-text count after the recorded exclusion differs')
    for cell in cells:
        path = ROOT / cell['path']
        if not path.is_file() or sha256(path) != cell['sha256']:
            errors.append(f'benchmark hash mismatch: {cell["cell"]}')
            continue
        data = json.loads(path.read_text())
        if set(data) != methods | {'original'}:
            errors.append(f'benchmark method list differs: {cell["cell"]}')
        if any(len(data.get(k, [])) != cell['source_count'] for k in methods | {'original'}):
            errors.append(f'benchmark sample count differs: {cell["cell"]}')

    with (ROOT / 'analysis/data/current/summary.csv').open(newline='') as handle:
        summary = {row['score']: row for row in csv.DictReader(handle)}
    for name, target in [('Ours_OPT', 0.8219095663414562),
                         ('Ours_Falcon', 0.8316661796123974)]:
        value = float(summary[name]['auc'])
        if not math.isclose(value, target, rel_tol=0, abs_tol=1e-12):
            errors.append(f'main AUROC differs for {name}: {value}')

    table_files = [
        'current_main.tex', 'current_generators.tex', 'current_mechanisms.tex',
        'two_tail_main.tex', 'component_ablation.tex', 'negative_evidence.tex',
        'watermark_mechanism_observations.tex', 'heavywater_generation_controls.tex',
        'evaluation_sources.tex', 'current_datasets.tex', 'two_tail_intervals.tex',
        'two_tail_mechanisms.tex', 'kappa_sensitivity.tex',
        'reference_calibration.tex', 'temperature_mass.tex',
        'local_construction.tex', 'tail_attribution_a01.tex', 'review_bank.tex',
        'length_performance.tex', 'length_calibration.tex',
    ]
    for name in table_files:
        if not (ROOT / 'analysis/tables' / name).is_file():
            errors.append(f'paper table input absent: {name}')
    for name in ('score_distributions.pdf', 'overview.pdf'):
        if not (ROOT / 'analysis/figures' / name).is_file():
            errors.append(f'paper figure absent: {name}')

    print(f'Checked {len(ledger)} payload files ({total_bytes / 2**30:.2f} GiB), '
          f'{len(cells)} benchmark cells, {len(table_files)} table exports, and 2 figures.')
    if errors:
        for error in errors:
            print('FAIL:', error)
        raise SystemExit(1)
    print('PASS: all payload hashes and paper checks match.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    check()
