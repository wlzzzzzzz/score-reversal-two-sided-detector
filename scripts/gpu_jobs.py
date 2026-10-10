"""Run one queued generation or quality-audit job per visible GPU."""
import os
import subprocess
import time
from collections import deque

from io_utils import atomic_json


def parse_gpus(value):
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("--gpus must contain distinct, nonempty device IDs")
    return gpus


def run_jobs(jobs, gpus, audit, seed, *, progress=False):
    """Return job outcomes; retain logs and stop active children on interruption."""
    queue = deque(jobs)
    active, completed = {}, []
    started = time.time()
    audit.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONHASHSEED=str(seed),
               TRANSFORMERS_VERBOSITY="error", HF_HUB_DISABLE_PROGRESS_BARS="1")
    try:
        while queue or active:
            for gpu in gpus:
                if gpu in active or not queue:
                    continue
                job = queue.popleft()
                log = audit / f"{job['name']}.log"
                with log.open("a", encoding="utf-8") as handle:
                    process = subprocess.Popen(job["command"], cwd=".",
                        env=dict(env, CUDA_VISIBLE_DEVICES=gpu),
                        stdout=handle, stderr=subprocess.STDOUT)
                row = dict(job, gpu=gpu, pid=process.pid, log=str(log), started=time.time())
                active[gpu] = (process, row)
            time.sleep(1)
            for gpu, (process, row) in list(active.items()):
                code = process.poll()
                if code is not None:
                    completed.append(dict(row, returncode=code,
                                          elapsed_seconds=time.time() - row["started"]))
                    del active[gpu]
            if progress:
                atomic_json(audit / "progress.json", {
                    "status": "RUNNING", "queued": list(queue),
                    "active": [row for _, row in active.values()],
                    "completed": completed, "elapsed_seconds": time.time() - started,
                }, sort_keys=False)
    finally:
        for process, _ in active.values():
            if process.poll() is None:
                process.terminate()
        for process, _ in active.values():
            process.wait()
    return completed
