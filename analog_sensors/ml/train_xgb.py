"""
train_xgb.py
============

Train a small XGBoost classifier on the 28 engineered features.

CV strategy: leave-one-load-out (LOLO). 3 folds:
  - train {Halfload, Fullload} -> test Noload
  - train {Noload,   Fullload} -> test Halfload
  - train {Noload,   Halfload} -> test Fullload

Threshold calibration: per-fold and final, the decision threshold is tuned
on a small validation slice from the *training* files (GroupShuffleSplit
by file, 10%) — not on the test fold, which would leak.

Final model: train on all 12 files, save to model_xgb.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

import xgboost as xgb

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from data import load_all_with_labels, mu_level_from_filename  # noqa: E402
from features import (  # noqa: E402
    FEATURE_NAMES,
    extract_features,
    extract_features_batch,
    save_feature_names,
)


TRAINING_DIR = THIS_DIR.parent / "training_data"
MODEL_PATH = THIS_DIR / "model_xgb.json"
THRESHOLD_PATH = THIS_DIR / "model_xgb_threshold.json"
FEATURE_NAMES_PATH = THIS_DIR / "feature_names.json"
METRICS_PATH = THIS_DIR / "xgb_metrics.json"


def _scale_pos_weight(y: np.ndarray) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos == 0:
        return 1.0
    return neg / pos


def _make_clf(y_train: np.ndarray) -> xgb.XGBClassifier:
    """Spec hyperparameters: max_depth=4, n_est=200, lr=0.1, scale_pos_weight=auto.

    A small min_child_weight + reg_lambda damps the over-confident leaves
    that otherwise saturate predict_proba near 1.0 on held-out load — too
    much regularisation hurts AUC, too little makes thresholds meaningless.
    """
    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        min_child_weight=3,
        scale_pos_weight=_scale_pos_weight(y_train),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=0,
        random_state=42,
    )


def _eval(name: str, y_true: np.ndarray, p: np.ndarray, thr: float) -> dict:
    y_pred = (p >= thr).astype(np.int32)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, p)
    except ValueError:
        auc = float("nan")  # only one class
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    print(f"  [{name}] acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}  thr={thr:.3f}  cm={cm}")
    return {
        "name": name, "accuracy": float(acc), "f1": float(f1),
        "auc": float(auc), "cm": cm, "threshold": float(thr),
    }


def calibrate_threshold_f1(y_true: np.ndarray, p: np.ndarray) -> float:
    """Return the threshold in [0.05, 0.95] that maximises F1 on (y, p).

    Tiebreak: when F1 ties (often the case when probabilities saturate near
    1.0 on held-out domains), prefer the threshold with higher balanced
    accuracy, then the threshold closest to the midpoint between the
    healthy and faulty mean probability — that's the most stable place to
    cut and beats the grid's left-most-tie default of 0.05.

    Sanity guard: if mean(p|faulty) <= mean(p|healthy) on this val slice,
    the model is *inverted* here — the val files happen to fall on the
    wrong side of the decision surface. F1 is then maximised by
    "predict-everything-positive" (thr=0.05), which is a degenerate
    calibration. Return 0.5 instead so the deployed threshold is not
    pathologically low.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5

    p_h_mean = float(p[y_true == 0].mean())
    p_f_mean = float(p[y_true == 1].mean())
    if p_f_mean <= p_h_mean:
        # Inverted/uninformative val — refuse to calibrate.
        print(f"  [warn] val is inverted (H_mean={p_h_mean:.3f} >= "
              f"F_mean={p_f_mean:.3f}); using thr=0.5")
        return 0.5

    grid = np.linspace(0.05, 0.95, 91)
    best = (-1.0, -1.0, 0.0, 0.5)  # (f1, balanced_acc, -dist_to_mid, thr)
    mid = 0.5 * (p_h_mean + p_f_mean)

    for thr in grid:
        pred = (p >= thr).astype(np.int32)
        f1 = f1_score(y_true, pred, zero_division=0)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fn = int((y_true == 1).sum()) - tp
        fp = int((y_true == 0).sum()) - tn
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        bal = 0.5 * (tpr + tnr)
        # composite key so we get a deterministic best threshold.
        key = (f1, bal, -abs(thr - mid), float(thr))
        if key > best:
            best = key
    return best[3]


def _val_split(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, val_frac: float = 0.1, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Group-aware (file-aware) train/val split. Returns (train_mask, val_mask).

    With only 2-3 healthy files and 6-9 faulty files per LOLO fold, a strict
    10% group split lands on 1 file — almost always faulty. To keep
    threshold calibration valid we explicitly pick at least one healthy +
    one faulty file for val. Tries GroupShuffleSplit first (random, honors
    val_frac), falls back to picking exactly 1 file per class.
    """
    rng = np.random.default_rng(seed)
    n = len(y)

    # Try GroupShuffleSplit at the requested fraction; accept only if both
    # classes land in val and train.
    for k in range(20):
        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed + k)
        idx_tr, idx_val = next(gss.split(X, y, groups))
        if len(np.unique(y[idx_val])) >= 2 and len(np.unique(y[idx_tr])) >= 2:
            tr_mask = np.zeros(n, dtype=bool); tr_mask[idx_tr] = True
            val_mask = np.zeros(n, dtype=bool); val_mask[idx_val] = True
            return tr_mask, val_mask

    # Fallback: pick 1 healthy + 1 faulty file. Minimum viable val that
    # satisfies the spec's "GroupShuffleSplit-style" intent (file-aware,
    # ~10% of data) without leaking and without collapsing to 1 class.
    files = np.unique(groups)
    file_label = np.array([y[groups == fi][0] for fi in files])
    healthy = files[file_label == 0]; faulty = files[file_label == 1]
    if len(healthy) == 0 or len(faulty) == 0:
        raise RuntimeError("training set has only one class")
    rng.shuffle(healthy); rng.shuffle(faulty)
    val_files = {int(healthy[0]), int(faulty[0])}
    val_mask = np.isin(groups, list(val_files))
    return ~val_mask, val_mask


def _per_mu_breakdown(
    test_mask: np.ndarray, groups: np.ndarray, files: list[str],
    p_test: np.ndarray, thr: float, label_name: str,
) -> dict[str, dict]:
    """Diagnostic only. For each fault severity (healthy / 1% / 3% / 5%),
    report what fraction of windows in this test fold the model labels
    FAULTY. The model is binary and never sees mu — this just tells us
    whether failures cluster on the low-severity files or are spread.
    """
    mu_per_file = [mu_level_from_filename(p) for p in files]
    test_groups = groups[test_mask]
    pred = (p_test >= thr).astype(np.int32)
    out: dict[str, dict] = {}
    print(f"  [{label_name}] per-fault-level (faulty-rate on test fold):")
    for mu in ["healthy", "1%", "3%", "5%"]:
        file_ids = [fi for fi, m in enumerate(mu_per_file) if m == mu]
        sel = np.isin(test_groups, file_ids)
        if not sel.any():
            continue
        n = int(sel.sum())
        n_faulty = int(pred[sel].sum())
        frac = n_faulty / n if n else 0.0
        # For 'healthy' the desired faulty rate is 0%; for fault levels it's 100%.
        target = "FAULTY" if mu != "healthy" else "HEALTHY"
        correct = (n - n_faulty) if mu == "healthy" else n_faulty
        print(f"      {mu:8s}: {correct:4d}/{n:4d} = {correct / n * 100:5.1f}% "
              f"correct ({target}); FAULTY-rate {frac * 100:5.1f}%")
        out[mu] = {"n": n, "faulty_pred": n_faulty, "faulty_rate": float(frac)}
    return out


def lolo_cv(
    X: np.ndarray, y: np.ndarray, load_idx: np.ndarray,
    groups: np.ndarray, files: list[str],
) -> list[dict]:
    """Leave-one-load-out CV with file-aware threshold calibration."""
    LOADS = [(0, "noload"), (1, "halfload"), (2, "fullload")]
    out = []
    for held_idx, held_name in LOADS:
        train_mask_full = load_idx != held_idx
        test_mask = load_idx == held_idx
        if test_mask.sum() == 0 or train_mask_full.sum() == 0:
            print(f"  [LOLO {held_name}] skipped (no data)")
            continue

        Xtr_full = X[train_mask_full]
        ytr_full = y[train_mask_full]
        gtr_full = groups[train_mask_full]
        # Train/val carve-out from the train set (file-aware, never touches test)
        tr_m, val_m = _val_split(Xtr_full, ytr_full, gtr_full, val_frac=0.1, seed=42)

        clf = _make_clf(ytr_full[tr_m])
        clf.fit(Xtr_full[tr_m], ytr_full[tr_m], verbose=False)

        # Threshold calibration on the held-out validation slice (training files)
        p_val = clf.predict_proba(Xtr_full[val_m])[:, 1]
        thr = calibrate_threshold_f1(ytr_full[val_m], p_val)

        # Evaluate on the truly-held-out load
        p_test = clf.predict_proba(X[test_mask])[:, 1]
        r = _eval(f"LOLO test={held_name}", y[test_mask], p_test, thr=thr)
        r["per_mu"] = _per_mu_breakdown(
            test_mask, groups, files, p_test, thr,
            label_name=f"XGB LOLO test={held_name}",
        )
        out.append(r)
    return out


def train_final(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[xgb.XGBClassifier, float, dict]:
    """All-data fit; threshold tuned on a 10% file-aware holdout."""
    print("\n--- Final XGBoost (all files, 10% file-aware val for threshold) ---")
    tr_m, val_m = _val_split(X, y, groups, val_frac=0.1, seed=7)
    clf = _make_clf(y[tr_m])
    t0 = time.perf_counter()
    clf.fit(X[tr_m], y[tr_m], verbose=False)
    print(f"  fit time: {time.perf_counter() - t0:.2f}s")

    p_val = clf.predict_proba(X[val_m])[:, 1]
    thr = calibrate_threshold_f1(y[val_m], p_val)
    print(f"  calibrated threshold: {thr:.3f}")

    # Refit on all data using the same hyperparameters now that the threshold
    # is fixed. Keeps the deployed model from leaving 10% on the cutting-room
    # floor.
    final_clf = _make_clf(y)
    final_clf.fit(X, y, verbose=False)
    p_all = final_clf.predict_proba(X)[:, 1]
    metrics_in = _eval("final XGB (in-sample, all files)", y, p_all, thr=thr)
    metrics_in["holdout_val_threshold"] = thr
    return final_clf, thr, metrics_in


def benchmark_inference(clf: xgb.XGBClassifier) -> tuple[float, float]:
    """Return (feature_ms_per_window, predict_only_ms_per_window)."""
    from features import WINDOW_SIZE as _WIN
    rng = np.random.default_rng(1)
    win = rng.standard_normal((_WIN, 3)).astype(np.float32)

    # warm-up
    feats = extract_features(win).reshape(1, -1)
    for _ in range(40):
        clf.predict_proba(feats)

    # feature extraction alone
    N = 500
    t0 = time.perf_counter()
    for _ in range(N):
        extract_features(win)
    feat_ms = (time.perf_counter() - t0) / N * 1000

    # predict only
    N = 500
    t0 = time.perf_counter()
    for _ in range(N):
        clf.predict_proba(feats)
    pred_ms = (time.perf_counter() - t0) / N * 1000

    print(f"\n  features alone: {feat_ms:.3f} ms / window")
    print(f"  XGB predict alone: {pred_ms:.3f} ms / window")
    return feat_ms, pred_ms


def feature_importance_report(clf: xgb.XGBClassifier) -> list[tuple[str, float]]:
    booster = clf.get_booster()
    score = booster.get_score(importance_type="gain")
    rows: list[tuple[str, float]] = []
    for k, v in score.items():
        idx = int(k[1:])  # XGBoost names features f0..fN
        if 0 <= idx < len(FEATURE_NAMES):
            rows.append((FEATURE_NAMES[idx], float(v)))
    rows.sort(key=lambda kv: kv[1], reverse=True)
    print("\nTop feature importances (by gain):")
    for name, val in rows[:10]:
        print(f"  {name:25s}  {val:.3f}")
    if not rows:
        return rows
    top5 = {name for name, _ in rows[:5]}
    if "neg_pos_ratio" not in top5:
        warnings.warn(
            "neg_pos_ratio is NOT in the top-5 features by gain. "
            "Inter-turn fault theory says it should be — investigate the "
            "feature implementation before declaring the run done.",
            stacklevel=2,
        )
    return rows


def main() -> int:
    overall_t0 = time.perf_counter()
    print("=" * 60)
    print("XGBoost training")
    print("=" * 60)

    X_wins, y_wins, groups, load_idx, files = load_all_with_labels(str(TRAINING_DIR))
    print(f"Extracting features for {X_wins.shape[0]} windows...")
    t0 = time.perf_counter()
    X = extract_features_batch(X_wins).astype(np.float32)
    print(f"  feature extraction: {time.perf_counter() - t0:.2f}s "
          f"({(time.perf_counter() - t0) / X_wins.shape[0] * 1000:.2f} ms/window)")
    y = y_wins

    print("\n--- Leave-one-load-out CV ---")
    lolo_results = lolo_cv(X, y, load_idx, groups, files)

    final_clf, thr, final_metrics = train_final(X, y, groups)
    importances = feature_importance_report(final_clf)

    final_clf.save_model(str(MODEL_PATH))
    save_feature_names(str(FEATURE_NAMES_PATH))
    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": float(thr)}, f, indent=2)

    feat_ms, pred_ms = benchmark_inference(final_clf)
    model_kb = MODEL_PATH.stat().st_size / 1024.0
    print(f"\nModel saved to {MODEL_PATH}  ({model_kb:.1f} KB)")
    print(f"Threshold saved to {THRESHOLD_PATH}")
    print(f"Feature names saved to {FEATURE_NAMES_PATH}")

    summary = {
        "lolo": lolo_results,
        "final": final_metrics,
        "threshold": float(thr),
        "feature_extraction_ms_per_window": feat_ms,
        "predict_only_ms_per_window": pred_ms,
        "model_kb": model_kb,
        "n_features": len(FEATURE_NAMES),
        "top_features": importances[:10],
        "total_train_seconds": time.perf_counter() - overall_t0,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    final_acc = final_metrics["accuracy"]
    if final_acc < 0.85:
        print("\n!!! WARNING: final XGBoost accuracy < 0.85 !!!")
        print("   Possible causes: feature bug, label/file mismatch, or genuine")
        print("   data overlap between healthy and faulty windows.")

    print(f"\nTotal pipeline time: {summary['total_train_seconds']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
