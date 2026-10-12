#!/usr/bin/env python3
"""Rebuild the final detector from saved or freshly computed document moments.

Inputs are checked against benchmark and per-cell hashes. Fresh inputs must
also pass the development freeze and test-access checks.
Historical code/configuration digests are never rewritten to authorize a run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from io_utils import sha256 as sha
from two_sided_core import rank_pvalues

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'results/runs/reference_shaping'
FIELDS = ['row_id', 'source_id', 'dataset', 'generator', 'watermark_method',
          'label', 'split', 'text_sha256', 'token_length']
MOMENTS = [w + '_' + k for w in ('entropy', 'uniform') for k in ('N0', 'N1', 'V0', 'V1')]


def local(path):
    path = Path(path)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Use repository-relative paths without parent traversal')
    resolved = (ROOT / path).resolve()
    resolved.relative_to(ROOT)
    return resolved


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def humans(frame):
    return frame[frame.label == 0].sort_values(
        ['dataset', 'source_id', 'generator', 'row_id']).drop_duplicates(['dataset', 'source_id'])


def load_frame(config, proxy, split, inputs, *, fresh):
    root = local(config['output_root'])
    records = []
    for cell in config['data_cells']:
        if cell['split'] != split:
            continue
        path = root / split / proxy / (cell['dataset'] + '_' + cell['generator'] + '.jsonl')
        manifest_path = path.with_suffix('.manifest.json')
        manifest = json.loads(manifest_path.read_text())
        digest = sha(path)
        if digest != manifest['sha256'] or manifest['input_sha256'] != cell['sha256']:
            raise ValueError('Cell hash mismatch: ' + relative(path))
        if manifest['proxy'] != proxy or manifest['access']['split'] != split or manifest['smoke_rows'] != 0:
            raise ValueError('Incomplete or incompatible cell: ' + relative(path))
        inputs[relative(path)] = digest
        inputs[relative(manifest_path)] = sha(manifest_path)
        if fresh:
            if manifest['config_sha256'] != config['_config_sha256']:
                raise ValueError('Fresh scores use a different configuration')
            if manifest['script_sha256'] != sha(ROOT / 'scripts/score_reference_moments.py'):
                raise ValueError('Fresh scores use a different scoring implementation')
            if split == 'test':
                for key, expected in {
                    'config_sha256': config['_config_sha256'],
                    'frozen_selection_sha256': sha(root / 'FROZEN_SELECTION.json'),
                    'test_authorization_sha256': sha(root / 'TEST_ACCESS_AUTHORIZATION.json'),
                }.items():
                    if manifest['access'].get(key) != expected:
                        raise ValueError('Fresh test scores have incompatible access bindings')
        data_path = local(cell['data_path'])
        if sha(data_path) != cell['sha256']:
            raise ValueError('Benchmark hash mismatch: ' + relative(data_path))
        inputs[relative(data_path)] = cell['sha256']
        source = json.loads(data_path.read_text())
        identities = set()
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                method, index = row['watermark_method'], row['source_index']
                identity = (method, index)
                if (method not in ['human'] + config['methods'] or
                        not 0 <= index < cell['source_count'] or identity in identities):
                    raise ValueError('Invalid or duplicate cell row')
                identities.add(identity)
                expected_id = '{}:{}:{}:{}'.format(cell['dataset'], cell['generator'], index, method)
                if (row['row_id'] != expected_id or row['source_id'] != cell['dataset'] + ':' + str(index)
                        or row['label'] != int(method != 'human') or row['split'] != split
                        or row['proxy'] != proxy or row['dataset'] != cell['dataset']
                        or row['generator'] != cell['generator'] or row['token_length'] < 1):
                    raise ValueError('Row metadata differs from the benchmark roster')
                text = source['original' if method == 'human' else method][index]
                if hashlib.sha256(text.encode()).hexdigest() != row['text_sha256']:
                    raise ValueError('Scored text differs from the benchmark')
                if split == 'test' and row['source_id'] == 'wp:11':
                    continue
                result = {key: row[key] for key in FIELDS}
                result.update({w + '_' + k: float(row['stats'][w][k])
                               for w in ('entropy', 'uniform') for k in ('N0', 'N1', 'V0', 'V1')})
                records.append(result)
        expected = cell['source_count'] * (len(config['methods']) + 1)
        if len(identities) != expected or manifest['rows'] != expected:
            raise ValueError('Incomplete scored cell: ' + relative(path))
    frame = pd.DataFrame(records)
    if frame.empty or frame.row_id.duplicated().any() or not np.isfinite(frame[MOMENTS]).all().all():
        raise ValueError('Invalid moment frame')
    if (frame[[name for name in MOMENTS if '_V' in name]] < 0).any().any():
        raise ValueError('Negative variance')
    grouped = frame[frame.label == 0].groupby('source_id')
    if grouped[['text_sha256', 'token_length']].nunique().max().max() != 1:
        raise ValueError('Human identity differs across generators')
    # Use the same lexicographic human forward for every generator, as in the paper.
    canonical = humans(frame).set_index('source_id')
    mask = frame.label == 0
    for column in MOMENTS:
        frame.loc[mask, column] = frame.loc[mask, 'source_id'].map(canonical[column]).to_numpy()
    expected_rows, expected_humans = (48000, 800) if split == 'development' else (141600, 2360)
    if len(frame) != expected_rows or len(humans(frame)) != expected_humans:
        raise ValueError('Unexpected cohort size')
    return frame


def evaluate(calibration, query, proxy):
    def response(frame, weight, endpoint):
        return frame[weight + '_N' + endpoint].to_numpy() / np.maximum(
            np.sqrt(frame[weight + '_V' + endpoint].to_numpy()), 1e-9)
    upper = rank_pvalues(response(calibration, 'entropy', '1'), response(query, 'entropy', '1'), 'upper')
    lower = rank_pvalues(response(calibration, 'uniform', '0'), response(query, 'uniform', '0'), 'lower')
    score = -np.log(np.minimum(upper / .5, lower / .5))
    predictions = dict(score=score, p_upper=upper, p_lower=lower)
    for alpha, tag in [(.01, 'a01'), (.05, 'a05')]:
        predictions['reject_' + tag] = (upper <= alpha / 2) | (lower <= alpha / 2)
    by_cell = {key: block.index.to_numpy() for key, block in
               query[query.label == 0].groupby(['dataset', 'generator'])}
    tasks = []
    for (dataset, generator, method), block in query[query.label == 1].groupby(
            ['dataset', 'generator', 'watermark_method'], sort=True):
        hi, mi = by_cell[(dataset, generator)], block.index.to_numpy()
        if set(query.loc[hi, 'source_id']) != set(block.source_id) or len(hi) != len(mi):
            raise ValueError('Incomplete paired task')
        labels = np.r_[np.zeros(len(hi)), np.ones(len(mi))]
        values = np.r_[score[hi], score[mi]]
        fpr, tpr, _ = roc_curve(labels, values, drop_intermediate=False)
        row = dict(score='Ours_' + proxy, dataset=dataset, generator=generator,
                   watermark_method=method, n_human=len(hi), n_machine=len(mi),
                   auc=roc_auc_score(labels, values))
        for alpha, tag in [(.01, 'a01'), (.05, 'a05')]:
            row['roc_tpr_' + tag] = float(tpr[fpr <= alpha].max())
            row['power_' + tag] = float(predictions['reject_' + tag][mi].mean())
        tasks.append(row)
    tasks = pd.DataFrame(tasks)
    if len(tasks) != 660 or int(tasks.n_machine.sum()) != 129800:
        raise ValueError('Expected all 660 test settings')
    aucs = tasks.auc.to_numpy()
    summary = dict(score='Ours_' + proxy, auc=float(aucs.mean()), p10=float(np.quantile(aucs, .1)),
                   cvar10=float(np.sort(aucs)[:66].mean()), flips=int((aucs < .5).sum()),
                   unwm_auc=float(tasks[tasks.watermark_method == 'unwatermarked'].auc.mean()),
                   wm_auc=float(tasks[tasks.watermark_method != 'unwatermarked'].auc.mean()))
    deployment = []
    for alpha, tag in [(.01, 'a01'), (.05, 'a05')]:
        summary['roc_tpr_' + tag] = float(tasks['roc_tpr_' + tag].mean())
        deployment.append(dict(proxy=proxy, alpha=alpha, power=float(tasks['power_' + tag].mean()),
                               fpr=float(predictions['reject_' + tag][humans(query).index].mean())))
    return tasks, summary, deployment, predictions


def check_paper(tasks, summary, deployment):
    keys = ['score', 'dataset', 'generator', 'watermark_method']
    expected = pd.read_csv(ROOT / 'analysis/data/current/task_metrics.csv')
    names = list(summary.score)
    expected = expected[expected.score.isin(names)].set_index(keys).sort_index()
    actual = tasks.set_index(keys).sort_index()
    pd.testing.assert_index_equal(actual.index, expected.index)
    np.testing.assert_allclose(actual[expected.columns], expected, rtol=0, atol=1e-12)
    expected = pd.read_csv(ROOT / 'analysis/data/current/summary.csv').set_index('score').loc[names]
    np.testing.assert_allclose(summary.set_index('score').loc[names, expected.columns], expected,
                               rtol=0, atol=1e-12)
    expected = pd.read_csv(ROOT / 'analysis/data/current/deployment.csv').set_index(['proxy', 'alpha'])
    actual = deployment.set_index(['proxy', 'alpha']).sort_index()
    np.testing.assert_allclose(actual[['power', 'fpr']], expected.loc[actual.index, ['power', 'fpr']],
                               rtol=0, atol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/original/reference_shaping.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-paper', action='store_true', help='Compare every final-detector task with the paper exports')
    args = parser.parse_args()
    config_path, output = local(args.config), local(args.output)
    # Require fresh ignored output; archived experiment records cannot be replaced.
    if output == ROOT / 'scratch' or ROOT / 'scratch' not in output.parents or output.exists():
        raise ValueError('Choose a new output directory below scratch/')
    config = json.loads(config_path.read_text())
    archived = local(config['output_root']) == ARCHIVE
    if not archived:
        from analyze_reference_shaping import read_config, validate_test_gate
        config, digest = read_config(config_path)
        validate_test_gate(config, digest)
        config['_config_sha256'] = digest
        root = local(config['output_root'])
        audit = json.loads((root / 'test_source_audit.json').read_text())
        expected = {
            'status': 'PASS', 'split': 'test', 'config_sha256': digest,
            'frozen_protocol_sha256': config['frozen_protocol_sha256'],
            'frozen_selection_sha256': sha(root / 'FROZEN_SELECTION.json'),
            'test_access_authorization_sha256': sha(root / 'TEST_ACCESS_AUTHORIZATION.json'),
            'development_source_manifest_sha256': sha(root / 'development_source_manifest.json'),
            'excluded_source_ids': ['wp:11'],
        }
        if any(audit.get(key) != value for key, value in expected.items()):
            raise PermissionError('Test source audit does not match the frozen benchmark protocol')
    output.mkdir(parents=True)
    inputs = {relative(config_path): sha(config_path)}
    all_tasks, summaries, deployment = [], [], []
    for proxy in ('OPT', 'Falcon'):
        calibration = humans(load_frame(config, proxy, 'development', inputs, fresh=not archived)).reset_index(drop=True)
        query = load_frame(config, proxy, 'test', inputs, fresh=not archived).reset_index(drop=True)
        if set(calibration.source_id) & set(query.source_id) or set(calibration.text_sha256) & set(query.text_sha256):
            raise ValueError('Calibration and test inputs overlap')
        tasks, summary, deploy, predictions = evaluate(calibration, query, proxy)
        destination = output / 'test' / proxy
        destination.mkdir(parents=True)
        calibration.to_pickle(output / ('calibration_' + proxy + '.pkl'))
        query.to_pickle(destination / 'scoring_frame.pkl')
        query[FIELDS].to_csv(destination / 'row_index.csv.gz', index=False)
        np.savez_compressed(destination / 'predictions.npz', **predictions)
        tasks.to_csv(destination / 'task_metrics.csv', index=False)
        all_tasks.append(tasks); summaries.append(summary); deployment.extend(deploy)
        print(proxy, 'AUROC', summary['auc'], flush=True)
    tasks = pd.concat(all_tasks, ignore_index=True)
    summary, deployment = pd.DataFrame(summaries), pd.DataFrame(deployment)
    if args.check_paper:
        check_paper(tasks, summary, deployment)
    for name, frame in [('task_metrics', tasks), ('summary', summary), ('deployment', deployment)]:
        frame.to_csv(output / (name + '.csv'), index=False)
    audit = dict(status='PASS', input_mode='archived' if archived else 'fresh',
                 calibration_humans=800, test_humans=2360, test_machines=129800, tasks_per_proxy=660,
                 excluded_source_ids=['wp:11'], beta=.5, historical_test_exposure=True,
                 paper_parity_checked=args.check_paper, input_sha256=inputs,
                 output_sha256={relative(p): sha(p) for p in output.rglob('*') if p.is_file()})
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('PASS:', relative(output), flush=True)


if __name__ == '__main__':
    main()
