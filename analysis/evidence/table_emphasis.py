"""Presentation-only table emphasis; original values and precision are preserved.

Comparisons use displayed values (all displayed ties are bold). Red indicates
absolute AUROC, including worst-decile mean AUROC, below 0.5, never a delta.
Bold marks metric optima, positive contrasts whose intervals exclude zero,
and branch contributions as specified in captions.
"""
from pathlib import Path
from decimal import Decimal
import argparse
import re

HERE = Path(__file__).resolve().parents[1]
NUMBER = re.compile(r'^([+-]?\d+(?:\.\d+)?)(?=\s|$)')
WRAPPER = re.compile(r'\\(?:textbf|reversal)\{([^{}]*)\}')


def plain(text):
    while WRAPPER.search(text):
        text = WRAPPER.sub(r'\1', text)
    return text


def format_table(name, text):
    # Column indices are zero-based. Diagnostic emphasis is distinct from
    # performance optima and is explained separately in each table caption.
    policies = {
        'current_main.tex': dict(max=[1, 2, 4, 5], min=[3], auc=[1, 2]),
        'current_mechanisms.tex': dict(rows=[[1, 2, 3, 4, 5, 6, 7]], auc=[1, 2, 3, 4, 5, 6, 7]),
        'current_datasets.tex': dict(max=list(range(1, 13)), auc=list(range(1, 13))),
        'current_generators.tex': dict(max=list(range(1, 6)), auc=list(range(1, 6))),
        'two_tail_main.tex': dict(max=[1, 2, 4], min=[3, 5], auc=[1], group='proxy_label'),
        'two_tail_mechanisms.tex': dict(rows=[[1, 2], [3, 4, 5]], auc=[1, 2, 3, 4, 5]),
        'upper_unwatermarked.tex': dict(max=[1, 2, 3], auc=[1, 2, 3]),
        'negative_evidence.tex': dict(auc=[2]),
        'component_ablation.tex': dict(max=[2, 5]),
        'reference_calibration.tex': dict(),
        'two_tail_intervals.tex': dict(positive_ci=[1, 2]),
        'temperature_mass.tex': dict(max=[3, 4, 5, 7], min=[6, 8], auc=[3, 4], group='first', params=[1, 2], primary=3),
        'local_construction.tex': dict(max=[2, 3, 4, 6], min=[5, 7], auc=[2, 3], params=[0, 1], primary=2),
        'kappa_sensitivity.tex': dict(max=[1, 2, 3, 5], min=[4, 6], auc=[1, 2], params=[0], primary=1),
        'kappa_effects.tex': dict(max=[1]),
        'review_allocation.tex': dict(max=[2, 3, 4, 6], min=[5, 7], auc=[2, 3], group='first', params=[1], primary=2),
        'review_bank.tex': dict(max=[2], min=[3], group='first'),
        'length_performance.tex': dict(max=[4, 5, 6], auc=[4], group='block'),
        'length_calibration.tex': dict(min=[2, 4], group='block'),
        'heavywater_generation_controls.tex': dict(max=[2], auc=[2]),
        'watermark_mechanism_observations.tex': dict(max=[2], auc=[2]),
        'tail_attribution_a01.tex': dict(union=True, branches=[[1, 2], [5, 6]]),
    }
    if name not in policies:
        return text
    spec = policies[name]
    text = plain(text)
    lines = text.splitlines(keepends=True)
    records = []
    block = 0
    for i, line in enumerate(lines):
        if line.startswith(r'\multicolumn'):
            block += 1
        if '&' not in line or not line.rstrip().endswith(r'\\') or line.lstrip().startswith('%'):
            continue
        cells = line.rstrip()[:-2].split('&')
        vals = {}
        for j, cell in enumerate(cells):
            match = NUMBER.match(cell.strip())
            if match:
                vals[j] = Decimal(match.group(1))
        if not vals:
            continue
        group = spec.get('group')
        key = (cells[0].strip() if group == 'first' else
               cells[1].strip() if group == 'mechanism' else
               ('OPT' if 'OPT' in cells[0] else 'Falcon') if group == 'proxy_label' else
               block if group == 'block' else 'all')
        records.append(dict(line=i, cells=cells, vals=vals, key=key, bold=set()))
    for key in {r['key'] for r in records}:
        group = [r for r in records if r['key'] == key]
        for direction in ['max', 'min']:
            for col in spec.get(direction, []):
                values = [r['vals'][col] for r in group if col in r['vals']]
                if not values:
                    continue
                best = (max if direction == 'max' else min)(values)
                for r in group:
                    if r['vals'].get(col) == best:
                        r['bold'].add(col)
                        if col == spec.get('primary'):
                            r['bold'].update(spec.get('params', []))
    for r in records:
        if spec.get('external') and r['cells'][0].strip().startswith('Ours'):
            baseline = [b for b in records if not plain(b['cells'][0]).strip().startswith('Ours')]
            for direction in ['max', 'min']:
                for col in spec.get(direction, []):
                    best = (max if direction == 'max' else min)(b['vals'][col] for b in baseline)
                    if (r['vals'][col] > best if direction == 'max' else r['vals'][col] < best):
                        r['bold'].add(col)
        if spec.get('ours_cols'):
            best = max(r['vals'][c] for c in spec['baseline_cols'])
            r['bold'].update(c for c in spec['ours_cols'] if r['vals'][c] > best)
        if spec.get('upper_external') and r['cells'][0].strip().startswith('Ours'):
            best = max(b['vals'][1] for b in records if 1 in b['vals'])
            if r['vals'][2] > best:
                r['bold'].add(2)
        r['bold'].update(c for c in spec.get('positive', []) if r['vals'].get(c, 0) > 0)
        r['bold'].update(c for c in spec.get('above_chance', []) if r['vals'].get(c, 0) > Decimal('0.5'))
        for c in spec.get('positive_ci', []):
            interval = re.search(r'\[([+-]?\d+(?:\.\d+)?),\s*([+-]?\d+(?:\.\d+)?)\]', r['cells'][c])
            if r['vals'].get(c, 0) > 0 and interval and Decimal(interval.group(1)) > 0:
                r['bold'].add(c)
        if 'Human' not in r['cells'][0]:
            for cols in spec.get('branches', []):
                values = [r['vals'][c] for c in cols if c in r['vals']]
                if values:
                    r['bold'].update(c for c in cols if r['vals'].get(c, 0) > 0 and
                                    ('All machine' in r['cells'][0] or r['vals'][c] == max(values)))
        for cols in spec.get('rows', []):
            values = [r['vals'][c] for c in cols if c in r['vals']]
            if values:
                r['bold'].update(c for c in cols if r['vals'].get(c) == max(values))
        if spec.get('union'):
            cols = [4, 8]
            values = [r['vals'][c] for c in cols if c in r['vals']]
            if len(values) == 2:
                best = (min if 'Human' in r['cells'][0] else max)(values)
                r['bold'].update(c for c in cols if r['vals'][c] == best)
        for j, cell in enumerate(r['cells']):
            if j == 0 and cell.strip().startswith('Ours'):
                r['cells'][j] = cell.replace(cell.strip(), r'\textbf{' + cell.strip() + '}')
                continue
            red = j in spec.get('auc', []) and r['vals'].get(j, Decimal(1)) < Decimal('0.5')
            if j not in r['bold'] and not red:
                continue
            # Emphasize the point estimate, leaving any interval unchanged.
            match = re.search(r'[+-]?\d+(?:\.\d+)?', cell)
            value = match.group(0)
            if j in r['bold']:
                value = r'\textbf{' + value + '}'
            if red:
                value = r'\reversal{' + value + '}'
            r['cells'][j] = cell[:match.start()] + value + cell[match.end():]
        ending = '\n' if lines[r['line']].endswith('\n') else ''
        lines[r['line']] = '&'.join(r['cells']) + r'\\' + ending
    result = ''.join(lines)
    assert plain(result) == text, name + ': formatting changed table contents'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    changed = []
    for path in sorted((HERE / 'tables').glob('*.tex')):
        old = path.read_text()
        new = format_table(path.name, old)
        if args.check_only:
            assert old == new, path.name
        elif old != new:
            path.write_text(new)
            changed.append(path.name)
    print('PASS: table emphasis verified' if args.check_only else 'Styled: ' + ', '.join(changed))
