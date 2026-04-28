"""
inference.py
============

Production-side fault detector for the live capture loop.

Loads the trained XGBoost and TFLite-MLP models once at startup. Each call
to `predict()` takes a (278, 3) window of I1/I2/I3 at ~1389 Hz (the
ADS1256 capture ceiling), extracts the 28 engineered features, and runs
both classifiers. Returns labels, probabilities, agreement flag, and a
consensus label.

Consensus rule: report FAULTY only if both models agree. This biases the
detector toward false negatives over false positives — appropriate for a
trip detector where false trips are operationally costly. To trade
sensitivity vs. specificity, callers can use `xgb_label` or `mlp_label`
directly instead of `consensus_label`.

The capture script writes raw ~1389 Hz samples; the intended integration
is something like:

    detector = FaultDetector()             # one-time, slow
    while True:
        window = grab_last_278_samples()   # (278, 3) at ~1389 Hz
        verdict = detector.predict(window)
        if verdict["consensus_label"] == "FAULTY":
            alert()

A `--demo` mode replays each training file (downsampled to ~1389 Hz)
through the same inference path so the user can verify it works on the Pi
without needing a live motor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from features import extract_features, FEATURE_NAMES, WINDOW_SIZE  # noqa: E402

N_CHANNELS = 3
N_FEAT = len(FEATURE_NAMES)

# XGBoost is required.
try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

# Prefer the slim tflite_runtime on the Pi; fall back to full TF on a laptop.
try:
    import tflite_runtime.interpreter as tflite_rt  # type: ignore
    _TFLITE_BACKEND = "tflite_runtime"
except ImportError:  # pragma: no cover
    try:
        import tensorflow as tf  # type: ignore
        tflite_rt = tf.lite
        _TFLITE_BACKEND = "tensorflow"
    except ImportError:
        tflite_rt = None
        _TFLITE_BACKEND = "none"


class FaultDetector:
    """One-shot init, many-call inference. Both models share the 28-feature
    vector — feature extraction runs once per `predict()` call."""

    def __init__(self, model_dir: str | os.PathLike | None = None) -> None:
        if xgb is None:
            raise RuntimeError("xgboost is not installed")
        if tflite_rt is None:
            raise RuntimeError(
                "neither tflite_runtime nor tensorflow are installed; "
                "install one of them to run the MLP"
            )
        d = Path(model_dir) if model_dir else THIS_DIR

        # ---- XGBoost + threshold ----
        self.clf = xgb.XGBClassifier()
        self.clf.load_model(str(d / "model_xgb.json"))
        with open(d / "model_xgb_threshold.json") as f:
            self.xgb_threshold = float(json.load(f)["threshold"])

        # ---- MLP scaler ----
        with open(d / "mlp_scaler.json") as f:
            sc = json.load(f)
        self.mean = np.array(sc["mean"], dtype=np.float32)
        self.scale = np.array(sc["scale"], dtype=np.float32)
        if self.mean.shape != (N_FEAT,) or self.scale.shape != (N_FEAT,):
            raise ValueError(
                f"scaler shape mismatch: expected ({N_FEAT},), "
                f"got mean={self.mean.shape}, scale={self.scale.shape}"
            )

        # ---- MLP TFLite + threshold ----
        self.interp = tflite_rt.Interpreter(model_path=str(d / "model_mlp.tflite"))
        self.interp.allocate_tensors()
        self._in = self.interp.get_input_details()[0]
        self._out = self.interp.get_output_details()[0]
        with open(d / "model_mlp_threshold.json") as f:
            self.mlp_threshold = float(json.load(f)["threshold"])

        # Feature-name sanity check: catch silent retraining drift.
        try:
            with open(d / "feature_names.json") as f:
                names = json.load(f)
            if names != FEATURE_NAMES:
                raise ValueError(
                    "feature_names.json on disk does not match features.py — "
                    "retrain with the current features module"
                )
        except FileNotFoundError:
            pass  # not fatal: only a sanity check

    # ---- internals ----
    def _xgb_proba(self, feats: np.ndarray) -> float:
        return float(self.clf.predict_proba(feats.reshape(1, -1))[0, 1])

    def _mlp_proba(self, feats: np.ndarray) -> float:
        x = ((feats - self.mean) / self.scale).astype(self._in["dtype"]).reshape(1, -1)
        self.interp.set_tensor(self._in["index"], x)
        self.interp.invoke()
        return float(self.interp.get_tensor(self._out["index"]).ravel()[0])

    @staticmethod
    def _label(p: float, thr: float) -> str:
        return "FAULTY" if p >= thr else "HEALTHY"

    def predict(self, window: np.ndarray) -> dict[str, Any]:
        """Run both models on a (278, 3) window and report verdicts.

        Window is expected to be I1/I2/I3 at ~1389 Hz, ~200 ms long. No
        normalisation needed beforehand — features are absolute, MLP
        normalises internally.
        """
        if window.shape != (WINDOW_SIZE, N_CHANNELS):
            raise ValueError(f"expected ({WINDOW_SIZE}, {N_CHANNELS}), got {window.shape}")
        feats = extract_features(window.astype(np.float32, copy=False))
        p_xgb = self._xgb_proba(feats)
        p_mlp = self._mlp_proba(feats)
        l_xgb = self._label(p_xgb, self.xgb_threshold)
        l_mlp = self._label(p_mlp, self.mlp_threshold)
        agreed = l_xgb == l_mlp
        # consensus = FAULTY only if both call FAULTY; healthy otherwise.
        consensus = "FAULTY" if (l_xgb == "FAULTY" and l_mlp == "FAULTY") else "HEALTHY"
        return {
            "xgb_label": l_xgb,
            "xgb_proba": p_xgb,
            "mlp_label": l_mlp,
            "mlp_proba": p_mlp,
            "agreed": agreed,
            "consensus_label": consensus,
        }


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------
def _demo(args: argparse.Namespace) -> int:
    """Run the detector over every CSV in training_data/ and print stats."""
    from data import discover_files, load_and_prepare

    training_dir = args.training_dir or str(THIS_DIR.parent / "training_data")
    files = discover_files(training_dir)
    if not files:
        print(f"No CSVs in {training_dir}", file=sys.stderr)
        return 2

    print(f"FaultDetector demo  (tflite backend: {_TFLITE_BACKEND})")
    det = FaultDetector()
    print(f"Loaded models from {THIS_DIR}")
    print(f"  XGB threshold: {det.xgb_threshold:.3f}   "
          f"MLP threshold: {det.mlp_threshold:.3f}\n")

    header = (
        f"{'file':42s}  {'truth':7s}  "
        f"{'xgb':7s} {'p_xgb':>6s}  {'mlp':7s} {'p_mlp':>6s}  "
        f"{'cons':7s}  {'wins':>5s}  acc_xgb  acc_mlp  acc_cons"
    )
    print(header)
    print("-" * len(header))

    accs_xgb, accs_mlp, accs_cons = [], [], []
    total_lat_ms = 0.0
    total_pred = 0
    for path in files:
        wins, label, _ = load_and_prepare(path)
        if wins.shape[0] == 0:
            continue
        xgb_correct = mlp_correct = cons_correct = 0
        xgb_p_sum = mlp_p_sum = 0.0
        truth = "FAULTY" if label == 1 else "HEALTHY"
        t0 = time.perf_counter()
        for i in range(wins.shape[0]):
            v = det.predict(wins[i])
            xgb_p_sum += v["xgb_proba"]
            mlp_p_sum += v["mlp_proba"]
            xgb_correct += int(v["xgb_label"] == truth)
            mlp_correct += int(v["mlp_label"] == truth)
            cons_correct += int(v["consensus_label"] == truth)
        dt = time.perf_counter() - t0
        total_lat_ms += dt * 1000
        total_pred += wins.shape[0]

        n = wins.shape[0]
        xgb_acc = xgb_correct / n
        mlp_acc = mlp_correct / n
        cons_acc = cons_correct / n
        accs_xgb.append((xgb_acc, n))
        accs_mlp.append((mlp_acc, n))
        accs_cons.append((cons_acc, n))
        avg_p_xgb = xgb_p_sum / n
        avg_p_mlp = mlp_p_sum / n
        xgb_pred = "FAULTY" if avg_p_xgb >= det.xgb_threshold else "HEALTHY"
        mlp_pred = "FAULTY" if avg_p_mlp >= det.mlp_threshold else "HEALTHY"
        cons_pred = "FAULTY" if (xgb_pred == "FAULTY" and mlp_pred == "FAULTY") else "HEALTHY"
        print(
            f"{os.path.basename(path):42s}  {truth:7s}  "
            f"{xgb_pred:7s} {avg_p_xgb:6.3f}  {mlp_pred:7s} {avg_p_mlp:6.3f}  "
            f"{cons_pred:7s}  {n:5d}  {xgb_acc:7.3f}  {mlp_acc:7.3f}  {cons_acc:7.3f}"
        )

    def _wmean(pairs):
        if not pairs:
            return float("nan")
        s = sum(a * w for a, w in pairs)
        n = sum(w for _, w in pairs)
        return s / n if n else float("nan")

    print()
    print(
        f"window-level accuracy across all files:  "
        f"XGB={_wmean(accs_xgb):.4f}   "
        f"MLP={_wmean(accs_mlp):.4f}   "
        f"CONS={_wmean(accs_cons):.4f}"
    )
    if total_pred:
        print(f"end-to-end predict() latency: {total_lat_ms / total_pred:.2f} ms / window")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Inter-turn fault detector")
    p.add_argument("--demo", action="store_true", help="run on training_data/")
    p.add_argument("--training-dir", default=None, help="override training data dir")
    args = p.parse_args(argv)
    if args.demo:
        return _demo(args)
    print("Use --demo to test against the training set.")
    print("In production:")
    print("    from inference import FaultDetector")
    print("    det = FaultDetector()")
    print("    print(det.predict(window))   # window: (278, 3) float32 at ~1389 Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
