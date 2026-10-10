"""Inclusive add-one human-rank calibration for the final detector."""

import numpy as np


def _finite_array(values, name, ndim=None):
    result = np.asarray(values, dtype=np.float64)
    if (ndim is not None and result.ndim != ndim) or not result.size:
        raise ValueError(f"{name} has invalid dimensions or is empty")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def rank_pvalues(calibration, query, direction):
    """Conservative one-sided rank p-values with +1 and inclusive ties."""
    bank = np.sort(_finite_array(calibration, "calibration", 1))
    values = _finite_array(query, "query")
    if direction == "upper":
        count = bank.size - np.searchsorted(bank, values, side="left")
    elif direction == "lower":
        count = np.searchsorted(bank, values, side="right")
    else:
        raise ValueError("direction must be upper or lower")
    return (1.0 + count) / (bank.size + 1.0)

