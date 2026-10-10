#!/usr/bin/env python3
"""Check that release files contain no machine-specific project paths."""

import json
from pathlib import Path
from release_utils import release_files, sha256


ROOT = Path(__file__).resolve().parents[1]
# Match local storage roots without retaining usernames or server mount names.
FORBIDDEN = tuple(b"/" + root + b"/" for root in (
    b"home", b"Users", b"mnt", b"media", b"inspire",
    *(f"data{index}".encode() for index in range(10)),
))


def check():
    errors = []
    # These files contain benchmark documents, including literal paths quoted
    # by their authors/models. They contain no repository dependency metadata.
    roster = json.loads((ROOT / 'configs/benchmark_cells.json').read_text())
    text_cells = {cell['path'] for cell in roster['cells']}
    checked = 0
    overlap = max(map(len, FORBIDDEN)) - 1
    for path in release_files(ROOT):
        relative = path.relative_to(ROOT).as_posix()
        if any(marker.decode() in relative for marker in FORBIDDEN):
            errors.append('machine path in file name: ' + relative)
        if not path.is_file():
            continue
        checked += 1
        if relative in text_cells:
            continue
        with path.open('rb') as stream:
            tail = b''
            while True:
                block = stream.read(8 << 20)
                if not block:
                    break
                current = tail + block
                if any(marker in current for marker in FORBIDDEN):
                    errors.append('machine path in file: ' + relative)
                    break
                tail = current[-overlap:]

    input_hashes = {}
    cohort_rows = 0
    for name in ('entropy_cap', 'local_parameters'):
        cohort = ROOT / 'results/runs' / name / 'cohort.jsonl'
        with cohort.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                cohort_rows += 1
                name = row['input_path']
                if Path(name).is_absolute() or '..' in Path(name).parts:
                    errors.append('non-relative cohort input: ' + name)
                    continue
                path = ROOT / name
                if not path.is_file():
                    errors.append('missing cohort input: ' + name)
                    continue
                expected = row['input_sha256']
                if name in input_hashes and input_hashes[name] != expected:
                    errors.append('inconsistent cohort input hash: ' + name)
                input_hashes[name] = expected
    for name, expected in input_hashes.items():
        if sha256(ROOT / name) != expected:
            errors.append('cohort input hash mismatch: ' + name)

    config = json.loads((ROOT / 'configs/original/reference_shaping.json').read_text())
    required = [config['protocol_path']]
    caches = 0
    for cell in config['data_cells']:
        required.append(cell['data_path'])
        for proxy in ('OPT', 'Falcon'):
            stem = Path(config['output_root']) / cell['split'] / proxy / (cell['dataset'] + '_' + cell['generator'])
            required.extend([str(stem) + '.jsonl', str(stem) + '.manifest.json'])
            caches += 1
    for name in required:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            errors.append('non-relative main workflow input: ' + name)
        elif not (ROOT / path).is_file():
            errors.append('missing main workflow input: ' + name)
    protocol = ROOT / config['protocol_path']
    if protocol.is_file() and sha256(protocol) != config['frozen_protocol_sha256']:
        errors.append('main protocol digest differs from configuration')

    print(f'Checked {checked} files and {cohort_rows} cohort rows; '
          f'{len(input_hashes)} unique input files; {caches} moment caches.')
    if errors:
        for error in errors[:20]:
            print('FAIL:', error)
        raise SystemExit(f'{len(errors)} path errors')
    print('PASS: no machine-specific paths; cohort inputs resolve in the repository.')


if __name__ == '__main__':
    check()
