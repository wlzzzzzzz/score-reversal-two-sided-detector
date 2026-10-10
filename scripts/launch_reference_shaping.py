#!/usr/bin/env python3
"""Run the fixed GPU workers for one gated experiment stage."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--split', choices=['development', 'test'], required=True)
    parser.add_argument('--smoke-rows', type=int, default=0)
    parser.add_argument('--queue', action='store_true', help='one OPT GPU and three Falcon GPU pairs')
    parser.add_argument('--resource-smoke', choices=['OPT', 'Falcon'], help='pre-freeze synthetic maximum-length resource check')
    parser.add_argument('--gpus', help='physical GPUs for the synthetic resource check only')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    # The scorer checks the same gate before loading a model or any raw data.
    from score_reference_moments import validate_access
    validate_access(config, args.split, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
    if args.resource_smoke:
        if args.split != 'development' or args.queue or args.smoke_rows or not args.gpus:
            raise ValueError('resource smoke requires development, --gpus, and no dataset execution options')
        if (Path(config['output_root']) / 'FROZEN_SELECTION.json').exists():
            raise PermissionError('synthetic resource check must finish before the experiment is frozen')
        environment = dict(os.environ)
        environment.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES=args.gpus)
        return subprocess.call([sys.executable, '-u', str(root / 'scripts/check_reference_resources.py'),
                                '--config', str(config_path), '--proxy', args.resource_smoke, '--gpus', args.gpus],
                               cwd=root, env=environment)
    if args.gpus:
        raise ValueError('--gpus is only available for a synthetic resource smoke check')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    logs = Path(config['output_root']) / 'logs'
    logs.mkdir(exist_ok=True)
    queue_workers = [
        {'name': 'opt_queue_0', 'proxy': 'OPT', 'gpus': [0]},
        {'name': 'falcon_queue_12', 'proxy': 'Falcon', 'gpus': [1, 2]},
        {'name': 'falcon_queue_34', 'proxy': 'Falcon', 'gpus': [3, 4]},
        {'name': 'falcon_queue_56', 'proxy': 'Falcon', 'gpus': [5, 6]},
    ]
    if args.queue and args.smoke_rows:
        raise ValueError('queue mode reuses the completed separate numerical smoke checks')
    workers = queue_workers if args.queue else config['workers']
    indices = list(range(len(workers)))
    if args.smoke_rows:
        indices = [next(i for i, w in enumerate(config['workers']) if w['proxy'] == proxy)
                   for proxy in ['OPT', 'Falcon']]
    started = time.monotonic()

    def run(index):
        worker = workers[index]
        log = logs / f'{run_id}_{args.split}_smoke{args.smoke_rows}_{worker["name"]}.log'
        if args.queue:
            command = [sys.executable, '-u', str(root / 'scripts/score_reference_queue.py'),
                       '--config', str(config_path), '--split', args.split, '--proxy', worker['proxy'],
                       '--gpus', ','.join(map(str, worker['gpus']))]
        else:
            command = [sys.executable, '-u', str(root / 'scripts/score_reference_moments.py'),
                       '--config', str(config_path), '--split', args.split, '--worker', str(index)]
        if args.smoke_rows:
            command += ['--smoke-rows', str(args.smoke_rows)]
        environment = dict(os.environ)
        environment.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           PYTHONHASHSEED='20260910', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                           CUDA_VISIBLE_DEVICES=','.join(map(str, worker['gpus'])))
        with log.open('x') as handle:
            process = subprocess.Popen(command, cwd=root, env=environment, stdout=handle,
                                       stderr=subprocess.STDOUT)
            print(json.dumps({'event': 'started', 'worker': index, 'proxy': worker['proxy'],
                              'pid': process.pid, 'log': str(log)}), flush=True)
            code = process.wait()
        result = {'worker': index, 'proxy': worker['proxy'], 'exit_code': code, 'log': str(log)}
        print(json.dumps({'event': 'finished', **result}), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=len(indices)) as pool:
        results = [future.result() for future in as_completed([pool.submit(run, i) for i in indices])]
    report = {'split': args.split, 'smoke_rows': args.smoke_rows, 'queue': args.queue, 'run_id': run_id,
              'completed_at_utc': datetime.now(timezone.utc).isoformat(),
              'elapsed_seconds': time.monotonic() - started, 'workers': results}
    (logs / f'{run_id}_launcher.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    return int(any(item['exit_code'] for item in results))


if __name__ == '__main__':
    raise SystemExit(main())
