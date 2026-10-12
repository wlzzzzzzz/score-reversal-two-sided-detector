"""Shared inventory rules for the published repository and its checksums."""
import hashlib
import os
from pathlib import Path

LOCAL_ROOTS = {'scratch', 'models', 'external_data', 'external_code'}
IGNORED_DIRS = {'.git', '__pycache__', 'pycache', '.pytest_cache', '.mypy_cache',
                '.ruff_cache', '.venv', 'venv', 'env', 'build'}


def release_files(root):
    """Exclude local environments, optional inputs, and generated scratch files."""
    root = Path(root)
    for directory, dirs, files in os.walk(root):
        base = Path(directory)
        relative = base.relative_to(root)
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS and not d.endswith('.egg-info'))
        if not relative.parts:
            dirs[:] = [d for d in dirs if d not in LOCAL_ROOTS]
        for name in sorted(files):
            if name.endswith(('.pyc', '.pyo', '.swp', '.swo')) or name == '.DS_Store':
                continue
            yield base / name


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()
