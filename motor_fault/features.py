"""Feature engineering used by the trained models."""

from __future__ import annotations

import numpy as np


FEATURE_NAMES = (
    "I1_rms",
    "I2_rms",
    "I3_rms",
    "I_sum",
    "I_mean",
    "I_max_abs",
    "I_min_abs",
    "I_range_abs",
    "I_std",
    "imbalance_percent",
    "I1_to_mean",
    "I2_to_mean",
    "I3_to_mean",
    "I1_minus_I2",
    "I2_minus_I3",
    "I3_minus_I1",
)


def build_feature_vector(i1_rms: float, i2_rms: float, i3_rms: float) -> np.ndarray:
    """Build the V2 RMS feature vector expected by the saved scaler/models."""
    values = np.array([i1_rms, i2_rms, i3_rms], dtype=np.float32)
    abs_values = np.abs(values)
    i_sum = float(np.sum(values))
    i_mean = i_sum / 3.0
    i_max = float(np.max(abs_values))
    i_min = float(np.min(abs_values))
    i_range = i_max - i_min
    i_std = float(np.std(values))
    eps = 1e-6
    imbalance_percent = i_range / (abs(i_mean) + eps)
    i1_to_mean = i1_rms / (i_mean + eps)
    i2_to_mean = i2_rms / (i_mean + eps)
    i3_to_mean = i3_rms / (i_mean + eps)
    i1_minus_i2 = i1_rms - i2_rms
    i2_minus_i3 = i2_rms - i3_rms
    i3_minus_i1 = i3_rms - i1_rms

    return np.array(
        [[
            i1_rms,
            i2_rms,
            i3_rms,
            i_sum,
            i_mean,
            i_max,
            i_min,
            i_range,
            i_std,
            imbalance_percent,
            i1_to_mean,
            i2_to_mean,
            i3_to_mean,
            i1_minus_i2,
            i2_minus_i3,
            i3_minus_i1,
        ]],
        dtype=np.float32,
    )
