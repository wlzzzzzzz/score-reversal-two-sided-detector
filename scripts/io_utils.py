"""Hashing and atomic JSON output shared by generation and audit scripts."""
import hashlib
import json
import os
from pathlib import Path

__all__ = ["sha256", "sha256_text", "atomic_json"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: object, *, sort_keys: bool = True) -> None:
    """Replace a complete JSON file; callers coordinate concurrent writers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
