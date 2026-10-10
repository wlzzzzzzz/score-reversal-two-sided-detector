"""Shared analytic document scores and normalized entropy weights."""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class OfficialModuleConfig:
    """Fixed endpoint settings used by the independent scoring check."""
    upper_mass: float = 0.8
    response_temperature: float = 1.2
    entropy_cap_multiplier: float = 1.5
    epsilon: float = 1e-9

    def __post_init__(self):
        if not 0 < self.upper_mass <= 1 or self.response_temperature <= 0:
            raise ValueError("Invalid reference temperature or nucleus mass")



def _document_score(
    target: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    weights: torch.Tensor,
    epsilon: float,
) -> float:
    target_document = (weights * target).sum()
    mean_document = (weights * mean).sum()
    variance_document = (weights.square() * variance).sum()
    return float(((target_document - mean_document) / variance_document.sqrt().clamp_min(epsilon)).item())

def _weights(entropy: torch.Tensor, mode: str, cap_multiplier: float) -> torch.Tensor:
    if mode == "uniform":
        raw = torch.ones_like(entropy)
    elif mode == "linear_entropy":
        cap = max(float(entropy.mean().item()) * cap_multiplier, 1e-6)
        raw = entropy.clamp(max=cap)
    else:
        raise ValueError(mode)
    return raw / raw.sum().clamp_min(1e-8)
