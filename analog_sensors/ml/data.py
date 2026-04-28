"""
data.py
=======

Loading, downsampling, windowing, and labelling for the inter-turn fault
training set.

Pipeline contract:
  - Training CSVs are sampled at 50 kHz (header tag "ts@2.0E-5").
  - Production capture (capture_adc.py) runs on the ADS1256 at DRATE=30000
    with a 3-channel multiplexer cycle. Per the chip's datasheet (Table 14)
    that's a hard ceiling of 4374/3 = 1458 Hz/channel; production captures
    at ~1392 Hz, i.e. 95% of ceiling. We cannot reach 2 kHz.
  - Both training and inference must therefore feed the feature pipeline
    at the same rate. We integer-decimate 50 kHz training data by 36 ->
    1388.89 Hz, the closest integer factor to the production rate. The
    model only needs train and inference to *match* — exact 1389 Hz is
    not achievable from a 50 kHz source by integer decimation.

Only I1, I2, I3 are returned to downstream code. I4 is read briefly during
load_csv() purely as a label sanity check (faulty files must show I4
residual current > 0.05 A; healthy files must be below).
"""

from __future__ import annotations

import os
import re
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from scipy import signal


SOURCE_RATE_HZ = 50_000
TARGET_RATE_HZ_DEFAULT = 1389  # production runs at ~1392 Hz; closest integer-decim is 1388.89
DECIM_FACTOR_DEFAULT = round(SOURCE_RATE_HZ / TARGET_RATE_HZ_DEFAULT)  # 36
ACTUAL_RATE_HZ_DEFAULT = SOURCE_RATE_HZ / DECIM_FACTOR_DEFAULT          # 1388.888...

# Window defaults at ~1388.89 Hz: 278 samples = 200.16 ms = ~10 line cycles
# at 50 Hz, 50% overlap. The longer window (vs the previous 100 ms one)
# halves the FFT bin width to ~5 Hz and lowers variance on per-window
# moments — both help separate 1% / 3% mu faults from healthy.
WINDOW_SIZE_DEFAULT = 278
HOP_DEFAULT = 139

# Drop the first second to skip motor inrush / startup transients.
TRIM_SECONDS = 1.0
TRIM_SAMPLES_50KHZ = int(TRIM_SECONDS * SOURCE_RATE_HZ)

# I4 RMS thresholds for the label sanity check.
I4_HEALTHY_MAX = 0.05
I4_FAULTY_MIN = 0.05


def label_from_filename(filename: str) -> int:
    """0 = healthy, 1 = any fault file. Matches name case-insensitively."""
    base = os.path.basename(filename).lower()
    if "healthy" in base:
        return 0
    # Match "<digits> mu" or "<digits>%mu" or "<digits>%" — covers Halfload_5%_rf3.csv too.
    if re.search(r"\d+\s*%?\s*mu", base):
        return 1
    if re.search(r"\d+\s*%", base):
        return 1
    return 0


def mu_level_from_filename(filename: str) -> str:
    """Return 'healthy' | '1%' | '3%' | '5%' | 'unknown'.

    Diagnostic only — the model is binary and never sees this. Used by the
    training scripts to break down test-fold accuracy per fault severity.
    """
    base = os.path.basename(filename).lower()
    if "healthy" in base:
        return "healthy"
    m = re.search(r"(\d+)\s*%", base)
    if m:
        return f"{m.group(1)}%"
    return "unknown"


def i4_rms(path: str) -> float:
    """Read just the I4 column once and return its RMS in amperes.

    Used for the startup ground-truth check. Loads the full file so the
    answer matches what's actually in the CSV; full read is ~1 s/file.
    """
    df = pd.read_csv(path, dtype=np.float32, engine="c")
    cols = {c.split("[")[0].strip(): c for c in df.columns}
    if "I4" not in cols:
        raise ValueError(f"{path}: I4 column missing")
    i4 = df[cols["I4"]].to_numpy(dtype=np.float32)
    return float(np.sqrt(np.mean(i4.astype(np.float64) ** 2)))


def assert_label_consistent(path: str, i4_rms_value: float | None = None) -> None:
    """Halt if filename label disagrees with I4 RMS.

    Per spec: faulty files must have I4 RMS > 0.05 A; healthy < 0.05 A. A
    mismatch indicates a relabeled or copied file and is fail-fast.
    """
    label = label_from_filename(path)
    rms = i4_rms_value if i4_rms_value is not None else i4_rms(path)
    base = os.path.basename(path)
    if label == 0 and rms > I4_HEALTHY_MAX:
        raise SystemExit(
            f"[label sanity] {base}: filename HEALTHY but I4 RMS={rms:.4f} A "
            f"(>{I4_HEALTHY_MAX}). Halting — fix the labels or filename."
        )
    if label == 1 and rms < I4_FAULTY_MIN:
        raise SystemExit(
            f"[label sanity] {base}: filename FAULTY but I4 RMS={rms:.4f} A "
            f"(<{I4_FAULTY_MIN}). Halting — fix the labels or filename."
        )


def load_csv(path: str) -> np.ndarray:
    """Read one training CSV.

    Returns float32 array of shape (N, 3) at 50 kHz containing [I1, I2, I3]
    with the first TRIM_SECONDS of samples dropped. I4 is consumed for the
    label sanity check and then discarded — a mismatch halts the program.
    """
    # pandas is ~10x faster than csv.reader for million-row files.
    df = pd.read_csv(path, dtype=np.float32, engine="c")
    cols = {c.split("[")[0].strip(): c for c in df.columns}
    needed = ["I1", "I2", "I3", "I4"]
    for n in needed:
        if n not in cols:
            raise ValueError(f"{path}: column {n!r} missing. Got {list(df.columns)}")

    i123 = df[[cols["I1"], cols["I2"], cols["I3"]]].to_numpy(dtype=np.float32)
    i4 = df[cols["I4"]].to_numpy(dtype=np.float32)

    rms = float(np.sqrt(np.mean(i4.astype(np.float64) ** 2)))
    assert_label_consistent(path, rms)

    if i123.shape[0] <= TRIM_SAMPLES_50KHZ:
        raise ValueError(f"{path}: only {i123.shape[0]} rows, need >{TRIM_SAMPLES_50KHZ}")
    return i123[TRIM_SAMPLES_50KHZ:]


def downsample_to_target(
    x_50khz: np.ndarray, target_hz: int = TARGET_RATE_HZ_DEFAULT
) -> np.ndarray:
    """50 kHz -> ~target_hz with anti-alias filter, per channel.

    Uses ``scipy.signal.decimate`` with an 8th-order IIR Butterworth and
    zero-phase filtering. Decimation factor is the nearest integer to
    ``50_000 / target_hz``; the achieved rate is therefore
    ``50_000 / round(50_000 / target_hz)``, which for target_hz=1389 gives
    1388.888... Hz. Train and inference run at the same achieved rate, so
    this small offset is not visible to the model.
    """
    if x_50khz.ndim != 2 or x_50khz.shape[1] != 3:
        raise ValueError(f"expected (N, 3), got {x_50khz.shape}")
    decim = round(SOURCE_RATE_HZ / target_hz)
    if decim < 2:
        raise ValueError(f"target_hz {target_hz} too high vs source {SOURCE_RATE_HZ}")
    out = signal.decimate(
        x_50khz.astype(np.float64), decim, n=8, ftype="iir", axis=0, zero_phase=True
    )
    return out.astype(np.float32)


def make_windows(
    x: np.ndarray, window_size: int = WINDOW_SIZE_DEFAULT, hop: int = HOP_DEFAULT
) -> Iterator[np.ndarray]:
    """Yield 50%-overlap-by-default windows of shape (window_size, 3)."""
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"expected (N, 3), got {x.shape}")
    n = x.shape[0]
    for start in range(0, n - window_size + 1, hop):
        yield x[start : start + window_size]


def windows_array(
    x: np.ndarray, window_size: int = WINDOW_SIZE_DEFAULT, hop: int = HOP_DEFAULT
) -> np.ndarray:
    """Same as make_windows but materialised as one (W, window_size, 3) array."""
    wins = list(make_windows(x, window_size, hop))
    if not wins:
        return np.empty((0, window_size, 3), dtype=np.float32)
    return np.stack(wins, axis=0)


def load_load_class(filename: str) -> str:
    """'noload' | 'halfload' | 'fullload' — used for leave-one-load-out CV."""
    base = os.path.basename(filename).lower()
    if base.startswith("noload"):
        return "noload"
    if base.startswith("halfload"):
        return "halfload"
    if base.startswith("fullload"):
        return "fullload"
    raise ValueError(f"cannot determine load class from {filename!r}")


def discover_files(training_dir: str) -> list[str]:
    """Return absolute paths of all *.csv in training_dir, sorted."""
    if not os.path.isdir(training_dir):
        raise FileNotFoundError(training_dir)
    files = sorted(
        os.path.join(training_dir, f)
        for f in os.listdir(training_dir)
        if f.lower().endswith(".csv")
    )
    return files


def load_and_prepare(
    path: str,
    window_size: int = WINDOW_SIZE_DEFAULT,
    hop: int = HOP_DEFAULT,
    target_hz: int = TARGET_RATE_HZ_DEFAULT,
) -> Tuple[np.ndarray, int, str]:
    """Load -> downsample -> window. Returns (W, window_size, 3) plus the
    file's binary label and load class."""
    x50 = load_csv(path)
    x = downsample_to_target(x50, target_hz=target_hz)
    wins = windows_array(x, window_size=window_size, hop=hop)
    return wins, label_from_filename(path), load_load_class(path)


def load_all_with_labels(
    data_dir: str,
    window_size: int = WINDOW_SIZE_DEFAULT,
    hop: int = HOP_DEFAULT,
    target_hz: int = TARGET_RATE_HZ_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """End-to-end loader.

    Returns (X_windows, y_labels, file_groups, load_idx, file_paths) where:
      X_windows:  (W, window_size, 3) float32 at the achieved decim rate
      y_labels:   (W,) int32 in {0, 1}
      file_groups: (W,) int32 — file ID per window for grouped CV
      load_idx:   (W,) int32 — 0=noload 1=halfload 2=fullload
      file_paths: list of absolute paths in fileid order
    """
    files = discover_files(data_dir)
    if not files:
        raise SystemExit(f"No CSVs found in {data_dir}")

    decim = round(SOURCE_RATE_HZ / target_hz)
    actual_hz = SOURCE_RATE_HZ / decim
    win_ms = window_size / actual_hz * 1000.0
    print(
        f"Decimation: 50 kHz -> {actual_hz:.2f} Hz (factor {decim}); "
        f"window={window_size} samples = {win_ms:.2f} ms = "
        f"~{win_ms / 20:.1f} line cycles @ 50 Hz"
    )

    LOAD_TO_IDX = {"noload": 0, "halfload": 1, "fullload": 2}
    Xs, ys, gs, ls = [], [], [], []
    print(f"Loading {len(files)} files from {data_dir}")
    print(f"{'file':42s}  {'label':7s}  {'load':9s}  {'I4 RMS':>8s}  {'wins':>5s}")
    for fi, path in enumerate(files):
        # ground-truth check before any processing
        rms = i4_rms(path)
        assert_label_consistent(path, rms)

        wins, label, load_cls = load_and_prepare(
            path, window_size=window_size, hop=hop, target_hz=target_hz
        )
        if wins.shape[0] == 0:
            print(f"  {os.path.basename(path):42s}  (no windows produced — skipped)")
            continue
        Xs.append(wins.astype(np.float32))
        ys.append(np.full(wins.shape[0], label, dtype=np.int32))
        gs.append(np.full(wins.shape[0], fi, dtype=np.int32))
        ls.append(np.full(wins.shape[0], LOAD_TO_IDX[load_cls], dtype=np.int32))
        print(
            f"  {os.path.basename(path):42s}  "
            f"{'FAULTY' if label else 'HEALTHY':7s}  {load_cls:9s}  "
            f"{rms:8.4f}  {wins.shape[0]:5d}"
        )
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    g = np.concatenate(gs, axis=0)
    li = np.concatenate(ls, axis=0)
    print(f"\nTotal windows: {X.shape[0]}  shape: {X.shape}  "
          f"healthy={int((y == 0).sum())}  faulty={int((y == 1).sum())}\n")
    return X, y, g, li, files
