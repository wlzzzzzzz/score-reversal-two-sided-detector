"""Independent bounded-memory check of the entropy-weighted shaped score."""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import torch
from fastdetect_official_module_core import OfficialModuleConfig, _document_score, _weights


def compute_official_module_features_token_chunked(
    logits_ref: torch.Tensor,
    logits_score: torch.Tensor,
    labels: torch.Tensor,
    config: OfficialModuleConfig = OfficialModuleConfig(),
    token_chunk_size: int = 32,
) -> Dict[str, Any]:
    """Check the paper shaped-reference endpoint using independent token chunks."""
    if logits_ref.ndim != 3 or logits_score.ndim != 3 or labels.ndim != 2:
        raise ValueError("expected [batch, tokens, vocab] logits and [batch, tokens] labels")
    if logits_ref.shape[:2] != logits_score.shape[:2] or logits_ref.shape[:2] != labels.shape:
        raise ValueError("sampling/scoring logits and labels must align exactly")
    if logits_ref.shape[0] != 1 or token_chunk_size < 1:
        raise ValueError("batch must be one and token_chunk_size positive")
    common_vocab = min(logits_ref.shape[-1], logits_score.shape[-1])
    if common_vocab < 2 or bool((labels >= common_vocab).any()):
        raise ValueError("labels fall outside the common official vocabulary")
    target_parts, entropy_parts, mean_parts, variance_parts = [], [], [], []

    for start in range(0, labels.shape[1], token_chunk_size):
        stop = min(start + token_chunk_size, labels.shape[1])
        ref = logits_ref[0, start:stop, :common_vocab].float()
        score = logits_score[0, start:stop, :common_vocab].float()
        chunk_labels = labels[0, start:stop]
        reference_log = torch.log_softmax(ref, dim=-1)
        score_log = torch.log_softmax(score, dim=-1)
        target = score_log.gather(-1, chunk_labels.unsqueeze(-1)).squeeze(-1)
        entropy = -(reference_log.exp() * reference_log).sum(-1) / np.log(common_vocab)
        response_probs = torch.softmax(ref / config.response_temperature, dim=-1)
        sorted_probs, sorted_indices = torch.sort(response_probs, dim=-1, descending=True)
        score_sorted = score_log.gather(-1, sorted_indices)
        remove = sorted_probs.cumsum(-1) > config.upper_mass
        allowed = torch.ones_like(remove)
        allowed[:, 1:] = ~remove[:, :-1]
        retained = sorted_probs * allowed
        mass = retained.sum(-1)
        if not bool(torch.isfinite(mass).all()) or bool((mass <= 0).any()):
            raise ValueError("invalid reference probabilities")
        distribution = retained / mass.unsqueeze(-1)
        mean = (distribution * score_sorted).sum(-1)
        variance = ((distribution * score_sorted.square()).sum(-1) - mean.square()).clamp_min(0)
        target_parts.append(target.cpu())
        entropy_parts.append(entropy.cpu())
        mean_parts.append(mean.cpu())
        variance_parts.append(variance.cpu())

    target, entropy = torch.cat(target_parts), torch.cat(entropy_parts)
    weights = _weights(entropy, "linear_entropy", config.entropy_cap_multiplier)
    return {"nucleus_entropy": _document_score(
        target, torch.cat(mean_parts), torch.cat(variance_parts), weights, config.epsilon)}
