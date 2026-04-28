"""
features.py
===========

Per-window feature extraction for inter-turn stator fault detection.

A window is shape (278, 3) sampled at ~1388.89 Hz (200.16 ms = ~10 line
cycles at 50 Hz). The rate is set by the production ADS1256 capture
ceiling, not chosen freely. For each window we compute 28 features:

  Per-phase (x3 phases = 18):
    1-3   RMS(I_a), RMS(I_b), RMS(I_c)
    4-6   skew(I_a), skew(I_b), skew(I_c)
    7-9   kurt(I_a), kurt(I_b), kurt(I_c)        (Fisher / excess)
    10-12 |H3|/|H1| per phase
    13-15 |H5|/|H1| per phase
    16-18 |H7|/|H1| per phase

  Three-phase / sequence-domain (10):
    19    |I_neg|         negative-sequence magnitude at 50 Hz
    20    |I_pos|         positive-sequence magnitude at 50 Hz
    21    NPSR = |I_neg| / |I_pos|       <- expected to dominate importance
    22    var(I_d)        Park d-axis variance
    23    var(I_q)        Park q-axis variance
    24    mean(|I_dq|)
    25    var(|I_dq|)
    26    ellipticity = std(|I_dq|) / mean(|I_dq|)
    27    RMS(I1 + I2 + I3)              residual / zero-sequence RMS
    28    phase_imbalance = (max - min)/mean of per-phase RMS

The rfft for each phase is computed exactly once per window, on a Hann-
windowed signal, and reused across the harmonic and sequence features.
"""

from __future__ import annotations

import json
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Constants for 278-sample / ~1388.89 Hz windows. Bin resolution is
# fs / N = 1388.89 / 278 ≈ 4.996 Hz. The fundamental at 50 Hz therefore
# lands on bin 10 of the rfft (50 / 4.996 ≈ 10.008). Harmonics:
#   150 Hz -> bin 30   (offset 0.024 bin = 0.12 Hz)
#   250 Hz -> bin 50   (offset 0.040 bin = 0.20 Hz)
#   350 Hz -> bin 70   (offset 0.056 bin = 0.28 Hz)
# All within 0.5 Hz of the centre, well inside the ±1-bin search.
# ---------------------------------------------------------------------------
WINDOW_SIZE = 278
SAMPLE_RATE_HZ = 50_000 / 36  # 1388.888... — see data.py downsample_to_target
LINE_FREQ_HZ = 50

BIN_FUND = int(round(LINE_FREQ_HZ * WINDOW_SIZE / SAMPLE_RATE_HZ))    # 10
BIN_H3 = int(round(3 * LINE_FREQ_HZ * WINDOW_SIZE / SAMPLE_RATE_HZ))  # 30
BIN_H5 = int(round(5 * LINE_FREQ_HZ * WINDOW_SIZE / SAMPLE_RATE_HZ))  # 50
BIN_H7 = int(round(7 * LINE_FREQ_HZ * WINDOW_SIZE / SAMPLE_RATE_HZ))  # 70
assert (BIN_FUND, BIN_H3, BIN_H5, BIN_H7) == (10, 30, 50, 70), (
    "harmonic bin layout changed; spec assumed 10/30/50/70 for 278 @ 1388.89 Hz"
)
# Each bin must hit its harmonic to within 0.5 Hz.
_BIN_HZ = SAMPLE_RATE_HZ / WINDOW_SIZE
for _h, _b in [(50, BIN_FUND), (150, BIN_H3), (250, BIN_H5), (350, BIN_H7)]:
    assert abs(_b * _BIN_HZ - _h) < 0.5, (
        f"bin {_b} for {_h} Hz off by {abs(_b * _BIN_HZ - _h):.3f} Hz"
    )

# ±1 bin (≈ ±5 Hz here) tolerates small line-frequency drift.
BIN_HALF_WIDTH = 1

# Park transform constants (amplitude-invariant Concordia).
_INV_SQRT_6 = 1.0 / np.sqrt(6.0)
_INV_SQRT_2 = 1.0 / np.sqrt(2.0)

# Symmetrical-component rotation operators for the 50 Hz fundamental.
_A = np.exp(1j * 2 * np.pi / 3)        # 120 deg
_A2 = np.exp(1j * 4 * np.pi / 3)       # 240 deg

# Hann window for spectral leakage reduction. Precompute once at import.
_HANN = np.hanning(WINDOW_SIZE).astype(np.float64)
# rfft scale factor: amplitude-correct output = (2 / sum(window)) * rfft.
# This makes sequence magnitudes interpretable as currents (in A peak).
_RFFT_SCALE = 2.0 / _HANN.sum()


FEATURE_NAMES: list[str] = [
    "rms_a", "rms_b", "rms_c",
    "skew_a", "skew_b", "skew_c",
    "kurt_a", "kurt_b", "kurt_c",
    "h3_ratio_a", "h3_ratio_b", "h3_ratio_c",
    "h5_ratio_a", "h5_ratio_b", "h5_ratio_c",
    "h7_ratio_a", "h7_ratio_b", "h7_ratio_c",
    "neg_seq_mag",
    "pos_seq_mag",
    "neg_pos_ratio",
    "park_d_var",
    "park_q_var",
    "park_mag_mean",
    "park_mag_var",
    "park_ellipticity",
    "zero_seq_rms",
    "phase_imbalance",
]


def feature_names() -> list[str]:
    """Public accessor — order matches extract_features() output."""
    return list(FEATURE_NAMES)


def save_feature_names(path: str) -> None:
    with open(path, "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)


def _peak_complex(spec: np.ndarray, bin_center: int) -> complex:
    """Pick the strongest bin in [center-w .. center+w] (handles small line
    drift) and return its complex value. Keeps phase for the sequence
    transform — symmetric-component magnitudes need both magnitude and phase
    of each phase's 50 Hz component.
    """
    lo = max(0, bin_center - BIN_HALF_WIDTH)
    hi = min(spec.shape[-1], bin_center + BIN_HALF_WIDTH + 1)
    chunk = spec[lo:hi]
    idx = int(np.argmax(np.abs(chunk)))
    return complex(chunk[idx])


def extract_features(window: np.ndarray) -> np.ndarray:
    """Return a 28-element float32 feature vector for one (200, 3) window."""
    if window.shape != (WINDOW_SIZE, 3):
        raise ValueError(f"expected ({WINDOW_SIZE}, 3), got {window.shape}")

    x = window.astype(np.float64, copy=False)
    ia, ib, ic = x[:, 0], x[:, 1], x[:, 2]

    # ---- one Hann-windowed rfft per phase, reused for harmonics + sequence ----
    Sa = np.fft.rfft(ia * _HANN) * _RFFT_SCALE
    Sb = np.fft.rfft(ib * _HANN) * _RFFT_SCALE
    Sc = np.fft.rfft(ic * _HANN) * _RFFT_SCALE

    # ---- per-phase moments (3 features × 3 phases = 9) ----
    rms_a = float(np.sqrt(np.mean(ia * ia)))
    rms_b = float(np.sqrt(np.mean(ib * ib)))
    rms_c = float(np.sqrt(np.mean(ic * ic)))
    skew_a = float(stats.skew(ia, bias=False))
    skew_b = float(stats.skew(ib, bias=False))
    skew_c = float(stats.skew(ic, bias=False))
    kurt_a = float(stats.kurtosis(ia, fisher=True, bias=False))
    kurt_b = float(stats.kurtosis(ib, fisher=True, bias=False))
    kurt_c = float(stats.kurtosis(ic, fisher=True, bias=False))

    # ---- per-phase harmonic ratios (3 features × 3 phases = 9) ----
    def harm_ratios(spec: np.ndarray) -> tuple[float, float, float]:
        mag_h1 = abs(_peak_complex(spec, BIN_FUND))
        if mag_h1 < 1e-9:
            return 0.0, 0.0, 0.0
        return (
            abs(_peak_complex(spec, BIN_H3)) / mag_h1,
            abs(_peak_complex(spec, BIN_H5)) / mag_h1,
            abs(_peak_complex(spec, BIN_H7)) / mag_h1,
        )

    h3_a, h5_a, h7_a = harm_ratios(Sa)
    h3_b, h5_b, h7_b = harm_ratios(Sb)
    h3_c, h5_c, h7_c = harm_ratios(Sc)

    # ---- symmetrical components at 50 Hz (3 features) ----
    Ia_f = _peak_complex(Sa, BIN_FUND)
    Ib_f = _peak_complex(Sb, BIN_FUND)
    Ic_f = _peak_complex(Sc, BIN_FUND)
    I_pos = (Ia_f + _A * Ib_f + _A2 * Ic_f) / 3.0
    I_neg = (Ia_f + _A2 * Ib_f + _A * Ic_f) / 3.0
    pos_mag = float(abs(I_pos))
    neg_mag = float(abs(I_neg))
    npsr = neg_mag / pos_mag if pos_mag > 1e-9 else 0.0

    # ---- Park (Concordia) transform features (5) ----
    i_d = (2.0 * ia - ib - ic) * _INV_SQRT_6
    i_q = (ib - ic) * _INV_SQRT_2
    park_d_var = float(np.var(i_d))
    park_q_var = float(np.var(i_q))
    mag = np.sqrt(i_d * i_d + i_q * i_q)
    park_mag_mean = float(mag.mean())
    park_mag_var = float(mag.var())
    park_ellip = float(mag.std() / park_mag_mean) if park_mag_mean > 1e-9 else 0.0

    # ---- residual (zero-sequence) and imbalance (2) ----
    zero_rms = float(np.sqrt(np.mean((ia + ib + ic) ** 2)))
    rms_arr = np.array([rms_a, rms_b, rms_c])
    rms_mean = rms_arr.mean()
    imb = float((rms_arr.max() - rms_arr.min()) / rms_mean) if rms_mean > 1e-9 else 0.0

    return np.array(
        [
            rms_a, rms_b, rms_c,
            skew_a, skew_b, skew_c,
            kurt_a, kurt_b, kurt_c,
            h3_a, h3_b, h3_c,
            h5_a, h5_b, h5_c,
            h7_a, h7_b, h7_c,
            neg_mag, pos_mag, npsr,
            park_d_var, park_q_var, park_mag_mean, park_mag_var, park_ellip,
            zero_rms, imb,
        ],
        dtype=np.float32,
    )


def extract_features_batch(windows: np.ndarray) -> np.ndarray:
    """(W, WINDOW_SIZE, 3) -> (W, 28). Plain Python loop — feature work per
    window is dominated by the rfft, which is already vectorised."""
    if windows.ndim != 3 or windows.shape[1:] != (WINDOW_SIZE, 3):
        raise ValueError(f"expected (W, {WINDOW_SIZE}, 3), got {windows.shape}")
    out = np.empty((windows.shape[0], len(FEATURE_NAMES)), dtype=np.float32)
    for i in range(windows.shape[0]):
        out[i] = extract_features(windows[i])
    return out


if __name__ == "__main__":
    # Self-benchmark: ensure feature extraction is fast enough for the Pi.
    import time
    rng = np.random.default_rng(0)
    win = rng.standard_normal((WINDOW_SIZE, 3)).astype(np.float32)
    for _ in range(50):
        extract_features(win)
    N = 1000
    t0 = time.perf_counter()
    for _ in range(N):
        extract_features(win)
    t1 = time.perf_counter()
    per_window_ms = (t1 - t0) / N * 1000
    print(f"feature extraction: {per_window_ms:.3f} ms / window  (target <5 ms laptop)")
    print(f"# features: {len(FEATURE_NAMES)}  (expected 28)")
    if per_window_ms > 5:
        print("WARNING: feature extraction is slower than target.")
