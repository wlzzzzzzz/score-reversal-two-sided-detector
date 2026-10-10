"""CPU-compatible drop-in for MarkMyWords' ``hash_cpp`` extension.

The upstream extension implements SHA-1 hashing in C++/CUDA.  This module
matches its byte layout exactly for the small ``index_hash`` calls used by the
CGW Binary generator. It supports CPU and CUDA tensors without compiling
an additional CUDA extension. Generation and verification use the upstream
MarkMyWords algorithms.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Iterable

import torch


MAX_UINT32 = 0xFFFFFFFF


def _one(seed: str, index: int) -> float:
    first = hashlib.sha1(str(seed).encode("utf-8")).digest()
    material = first + struct.pack("<I", int(index) & MAX_UINT32)
    second = hashlib.sha1(material).digest()
    value = struct.unpack("<I", second[16:20])[0]
    return float(value) / float(MAX_UINT32)


def index_hash(seeds: Iterable[str], token: int) -> torch.Tensor:
    return torch.tensor([_one(seed, token) for seed in seeds], dtype=torch.float32)


def all_index_hash(seeds: Iterable[str], target: torch.Tensor) -> torch.Tensor:
    seeds = list(seeds)
    values = torch.tensor(
        [[_one(seed, index) for index in range(target.shape[1])] for seed in seeds],
        dtype=target.dtype,
        device="cpu",
    )
    target.copy_(values.to(target.device))
    return target


def levenshtein(scores: torch.Tensor, output: torch.Tensor, gamma: float) -> torch.Tensor:
    # MarkMyWords uses this only for edit-distance verification, not generation.
    # Keep a faithful CPU fallback so importing the official verifier is safe.
    local_scores = scores.detach().cpu()
    local_output = output.detach().cpu().clone()
    key_len = local_scores.shape[0]
    seq_len = local_output.shape[1] - 1
    for batch in range(local_output.shape[0]):
        for i in range(1, seq_len + 1):
            for j in range(1, seq_len + 1):
                cost = float(local_scores[(batch + i - 1) % key_len, j - 1])
                local_output[batch, i, j] = min(
                    float(local_output[batch, i - 1, j]) + gamma,
                    float(local_output[batch, i, j - 1]) + gamma,
                    float(local_output[batch, i - 1, j - 1]) + cost,
                )
    output.copy_(local_output.to(output.device))
    return output
