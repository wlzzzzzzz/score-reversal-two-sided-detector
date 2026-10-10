"""Checks for deterministic generation, quality rules, and GPU job scheduling."""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generation_utils import seed_everything, stable_seed
from gpu_jobs import parse_gpus, run_jobs
from io_utils import atomic_json
from text_utils import normalize_text, sanitize_text, strict_text_quality_reasons, text_metrics
import retry_broad_watermark_failed_records as recovery


def test_seed_is_stable_and_resets_all_rngs():
    seed = stable_seed(23, "squad", "source:4", "CGW", 2)
    assert seed != stable_seed(23, "squad", "source:4", "CGW", 3)
    seed_everything(seed)
    first = (random.random(), np.random.random(), torch.rand(3))
    seed_everything(seed)
    assert random.random() == first[0]
    assert np.random.random() == first[1]
    assert torch.equal(torch.rand(3), first[2])


def test_text_cleanup_preserves_punctuation_and_whitespace():
    raw = "Cafe\u0301?\n\tGood\x00\ufffd\ud800\x01!"
    clean, removed = sanitize_text(raw)
    assert clean == "Cafe\u0301?\n\tGood!" and removed == 4
    assert normalize_text(clean) == "Caf\u00e9? Good!"
    assert text_metrics("One two three four")[0:4] == (4, 1.0, 0.0, 0.0)
    assert strict_text_quality_reasons("One two three four") == []
    assert strict_text_quality_reasons("echo " * 32) == [
        "UNIQUE_LEXICAL_RATIO_BELOW_0.15", "REPEATED_4GRAM_RATIO_ABOVE_0.35"]
    assert "CJK_RATIO_ABOVE_0.05" in strict_text_quality_reasons("\u4e00\u4e01\u4e02\u4e03")
    assert text_metrics("a\x00b")[-1] == 1


@pytest.mark.parametrize("sort_keys", [False, True])
def test_atomic_json_preserves_record_format(tmp_path, sort_keys):
    path = tmp_path / "records" / "record.json"
    value = {"z": "\u00e9", "a": [1, None, True]}
    atomic_json(path, value, sort_keys=sort_keys)
    assert path.read_bytes() == (json.dumps(value, ensure_ascii=False, indent=2,
                                          sort_keys=sort_keys) + "\n").encode()
    assert not path.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("value", ["", " , ", "0,0"])
def test_gpu_selection_rejects_empty_or_duplicate_devices(value):
    with pytest.raises(ValueError):
        parse_gpus(value)


def test_queue_runs_all_jobs_and_records_failures(tmp_path):
    # A lock catches concurrent use of the same GPU without requiring GPU hardware.
    program = tmp_path / "job.py"
    program.write_text("""import fcntl, json, os, sys, time
from pathlib import Path
folder = Path(sys.argv[1])
gpu = os.environ['CUDA_VISIBLE_DEVICES']
with (folder / ('gpu_' + gpu + '.lock')).open('w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    time.sleep(.05)
    print(json.dumps({k: os.environ[k] for k in
        ('CUDA_VISIBLE_DEVICES', 'PYTHONHASHSEED', 'OMP_NUM_THREADS')}), flush=True)
    sys.exit(int(sys.argv[2]))
""")
    jobs = [{"name": f"job_{i}", "command": [sys.executable, str(program), str(tmp_path),
                                             "2" if i == 1 else "0"]} for i in range(5)]
    outcomes = run_jobs(jobs, parse_gpus("0, 1"), tmp_path / "logs", 23, progress=True)
    assert len(outcomes) == 5
    assert sum(row["returncode"] != 0 for row in outcomes) == 1
    assert {row["gpu"] for row in outcomes} == {"0", "1"}
    for row in outcomes:
        assert row["returncode"] == (2 if row["name"] == "job_1" else 0)
        env = json.loads(Path(row["log"]).read_text())
        assert env == {"CUDA_VISIBLE_DEVICES": row["gpu"], "PYTHONHASHSEED": "23", "OMP_NUM_THREADS": "1"}
    progress = json.loads((tmp_path / "logs/progress.json").read_text())
    assert progress["queued"] == progress["active"] == []
    assert len(progress["completed"]) == 5
    assert len(jobs) == 5


def test_resume_requires_matching_record_and_prompt(tmp_path):
    from io_utils import sha256_text
    row = dict(dataset="c4", generator="falcon", method="CGW", source_id="c4:0",
               slot_index=0, prompt="The prompt")
    path = recovery.record_path(tmp_path, row)
    assert not recovery.is_success(tmp_path, row)
    path.parent.mkdir(parents=True)
    for invalid in ("{", "[]", "null"):
        path.write_text(invalid)
        assert not recovery.is_success(tmp_path, row)
    record = dict(row, status="SUCCESS", prompt_sha256=sha256_text(row["prompt"]))
    atomic_json(path, record)
    assert recovery.is_success(tmp_path, row)
    assert not recovery.is_success(tmp_path, dict(row, prompt="Another prompt"))
    assert not recovery.is_success(tmp_path, dict(row, source_id="c4:1"))


def test_resume_does_not_hide_permission_errors(tmp_path, monkeypatch):
    row = dict(dataset="c4", generator="falcon", method="CGW", slot_index=0)
    def denied(*args, **kwargs):
        raise PermissionError("record cannot be read")
    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(PermissionError):
        recovery.is_success(tmp_path, row)


def test_sparse_recovery_preserves_failures_and_attempt_rules(tmp_path, monkeypatch):
    from io_utils import sha256_text
    requests = []
    for i, method in enumerate(("CGW", "HeavyWater", "SemStamp")):
        requests.append(dict(dataset="c4", generator="falcon", method=method,
                             source_id=f"c4:{i}", slot_index=i, prompt=f"Prompt {i}"))
    output = tmp_path / "run"
    request_path = tmp_path / "requests.json"
    atomic_json(request_path, requests)
    originals = {}
    for row in requests[1:]:
        path = recovery.record_path(output, row)
        atomic_json(path, dict(row, status="FAILED", attempts=[{"attempt": 1, "reason": "short"}]))
        originals[row["method"]] = path.read_bytes()
    launched = []
    def finish_jobs(jobs, gpus, audit, seed):
        assert gpus == ["0"] and len(jobs) == 3 and seed == 30
        outcomes = []
        for job in jobs:
            command = job["command"]
            launched.append(job)
            missing = job["recovery_mode"] == "MISSING_RECORD"
            assert command[command.index("--base-seed") + 1] == ("23" if missing else "31")
            assert "--native-falcon" in command and "--strict-text-quality" in command
            if job["method"] == "SemStamp":
                assert command[command.index("--semstamp-min-new-tokens") + 1] == "64"
            rows = json.loads(Path(command[command.index("--request-manifest") + 1]).read_text())
            for row in rows:
                atomic_json(recovery.record_path(output, row),
                            dict(row, status="SUCCESS", prompt_sha256=sha256_text(row["prompt"])))
            outcomes.append(dict(job, gpu="0", returncode=0))
        return outcomes
    monkeypatch.setattr(recovery, "run_jobs", finish_jobs)
    monkeypatch.setattr(sys, "argv", ["retry", "--output-dir", str(output),
                        "--request-manifest", str(request_path), "--gpus", "0",
                        "--seed", "30", "--first-pass-seed", "23"])
    assert recovery.main() == 0
    summary = json.loads((output / "sparse_recovery_audit/summary.json").read_text())
    assert summary["status"] == "PASS" and summary["remaining_failures"] == 0
    assert summary["archived_pre_retry_record_count"] == 2 and len(launched) == 3
    for record in summary["archived_pre_retry_records"]:
        if record["status"] == "ARCHIVED":
            assert Path(record["path"]).read_bytes() == originals[record["method"]]
        else:
            assert record["method"] == "CGW" and record["status"] == "MISSING_BEFORE_RETRY"
