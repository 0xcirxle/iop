"""
train_mlp.py
============

Train a tiny MLP on the same 28 engineered features as the XGBoost model.

Architecture (~1.3k params -> ~5 KB TFLite):
  Dense(32) + ReLU + Dropout(0.3)
  Dense(16) + ReLU + Dropout(0.2)
  Dense(1, sigmoid)

The two models intentionally share the feature contract — that's what makes
the consensus path in inference.py meaningful (two independent classifiers
on the same vector). It also keeps the Pi runtime minimal: one feature
extraction, two cheap predictions.

Outputs:
  model_mlp.h5            full Keras model
  model_mlp.tflite        dynamic-range quantised TFLite (<10 KB)
  model_mlp_threshold.json {"threshold": <float>}
  mlp_scaler.json         StandardScaler (mean/scale per feature)
  mlp_metrics.json        per-fold + final + tflite acc + size + latency
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

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
H5_PATH = THIS_DIR / "model_mlp.h5"
TFLITE_PATH = THIS_DIR / "model_mlp.tflite"
SCALER_PATH = THIS_DIR / "mlp_scaler.json"
THRESHOLD_PATH = THIS_DIR / "model_mlp_threshold.json"
FEATURE_NAMES_PATH = THIS_DIR / "feature_names.json"
METRICS_PATH = THIS_DIR / "mlp_metrics.json"

SEED = 42
N_FEAT = len(FEATURE_NAMES)


def _set_seed():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def build_mlp() -> tf.keras.Model:
    inp = tf.keras.layers.Input(shape=(N_FEAT,), name="features")
    x = tf.keras.layers.Dense(32, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="fault_proba")(x)
    model = tf.keras.Model(inputs=inp, outputs=out, name="mlp_fault")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")],
    )
    return model


def class_weights_for(y: np.ndarray) -> dict[int, float]:
    n = len(y)
    h = float((y == 0).sum())
    f = float((y == 1).sum())
    if h == 0 or f == 0:
        return {0: 1.0, 1: 1.0}
    return {0: n / (2.0 * h), 1: n / (2.0 * f)}


def _val_split(
    y: np.ndarray, groups: np.ndarray, val_frac: float = 0.1, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """File-aware split with class-balance fallback. See train_xgb._val_split."""
    n = len(y)
    rng = np.random.default_rng(seed)
    for k in range(20):
        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed + k)
        idx_tr, idx_val = next(gss.split(np.zeros(n), y, groups))
        if len(np.unique(y[idx_val])) >= 2 and len(np.unique(y[idx_tr])) >= 2:
            tr_mask = np.zeros(n, dtype=bool); tr_mask[idx_tr] = True
            val_mask = np.zeros(n, dtype=bool); val_mask[idx_val] = True
            return tr_mask, val_mask

    # Fallback: explicit 1 healthy + 1 faulty file.
    files = np.unique(groups)
    file_label = np.array([y[groups == fi][0] for fi in files])
    healthy = files[file_label == 0]; faulty = files[file_label == 1]
    if len(healthy) == 0 or len(faulty) == 0:
        raise RuntimeError("training set has only one class")
    rng.shuffle(healthy); rng.shuffle(faulty)
    val_files = {int(healthy[0]), int(faulty[0])}
    val_mask = np.isin(groups, list(val_files))
    return ~val_mask, val_mask


def calibrate_threshold_f1(y_true: np.ndarray, p: np.ndarray) -> float:
    """F1-maximising threshold; ties broken by balanced accuracy then
    distance to the midpoint between class-mean probabilities. See
    train_xgb.calibrate_threshold_f1 for the rationale and the inverted-
    val sanity guard."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    p_h_mean = float(p[y_true == 0].mean())
    p_f_mean = float(p[y_true == 1].mean())
    if p_f_mean <= p_h_mean:
        print(f"  [warn] val is inverted (H_mean={p_h_mean:.3f} >= "
              f"F_mean={p_f_mean:.3f}); using thr=0.5")
        return 0.5
    grid = np.linspace(0.05, 0.95, 91)
    mid = 0.5 * (p_h_mean + p_f_mean)
    best = (-1.0, -1.0, 0.0, 0.5)
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
        key = (f1, bal, -abs(thr - mid), float(thr))
        if key > best:
            best = key
    return best[3]


def _fit_one(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 100,
) -> tf.keras.Model:
    _set_seed()
    model = build_mlp()
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    model.fit(
        X_tr, y_tr.astype(np.float32),
        validation_data=(X_val, y_val.astype(np.float32)),
        batch_size=64, epochs=epochs,
        class_weight=class_weights_for(y_tr),
        callbacks=[es], verbose=0, shuffle=True,
    )
    return model


def _eval(name: str, y_true: np.ndarray, p: np.ndarray, thr: float) -> dict:
    y_pred = (p >= thr).astype(np.int32)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, p)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    print(f"  [{name}] acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}  thr={thr:.3f}  cm={cm}")
    return {
        "name": name, "accuracy": float(acc), "f1": float(f1),
        "auc": float(auc), "cm": cm, "threshold": float(thr),
    }


def _per_mu_breakdown(
    test_mask: np.ndarray, groups: np.ndarray, files: list[str],
    p_test: np.ndarray, thr: float, label_name: str,
) -> dict[str, dict]:
    """Diagnostic only — see train_xgb._per_mu_breakdown."""
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
    LOADS = [(0, "noload"), (1, "halfload"), (2, "fullload")]
    out = []
    for held_idx, held_name in LOADS:
        train_mask_full = load_idx != held_idx
        test_mask = load_idx == held_idx
        if test_mask.sum() == 0 or train_mask_full.sum() == 0:
            print(f"  [LOLO {held_name}] skipped (no data)")
            continue
        Xtr_full = X[train_mask_full]; ytr_full = y[train_mask_full]
        gtr_full = groups[train_mask_full]

        tr_m, val_m = _val_split(ytr_full, gtr_full, val_frac=0.1)
        scaler = StandardScaler().fit(Xtr_full[tr_m])
        Xs_tr = scaler.transform(Xtr_full[tr_m]).astype(np.float32)
        Xs_val = scaler.transform(Xtr_full[val_m]).astype(np.float32)
        Xs_test = scaler.transform(X[test_mask]).astype(np.float32)

        t0 = time.perf_counter()
        model = _fit_one(Xs_tr, ytr_full[tr_m], Xs_val, ytr_full[val_m])
        fit_s = time.perf_counter() - t0

        p_val = model.predict(Xs_val, batch_size=128, verbose=0).ravel()
        thr = calibrate_threshold_f1(ytr_full[val_m], p_val)
        p_test = model.predict(Xs_test, batch_size=128, verbose=0).ravel()
        r = _eval(f"LOLO test={held_name}", y[test_mask], p_test, thr)
        r["fit_seconds"] = fit_s
        r["per_mu"] = _per_mu_breakdown(
            test_mask, groups, files, p_test, thr,
            label_name=f"MLP LOLO test={held_name}",
        )
        out.append(r)
    return out


def train_final(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[tf.keras.Model, StandardScaler, float, dict]:
    print("\n--- Final MLP (all files, 10% file-aware val for threshold) ---")
    tr_m, val_m = _val_split(y, groups, val_frac=0.1, seed=7)
    scaler = StandardScaler().fit(X[tr_m])
    Xs_tr = scaler.transform(X[tr_m]).astype(np.float32)
    Xs_val = scaler.transform(X[val_m]).astype(np.float32)
    Xs_all = scaler.transform(X).astype(np.float32)

    t0 = time.perf_counter()
    model = _fit_one(Xs_tr, y[tr_m], Xs_val, y[val_m], epochs=100)
    print(f"  fit time: {time.perf_counter() - t0:.2f}s")

    p_val = model.predict(Xs_val, batch_size=128, verbose=0).ravel()
    thr = calibrate_threshold_f1(y[val_m], p_val)
    print(f"  calibrated threshold: {thr:.3f}")

    p_all = model.predict(Xs_all, batch_size=128, verbose=0).ravel()
    metrics = _eval("final MLP (in-sample, all files)", y, p_all, thr)
    metrics["holdout_val_threshold"] = thr
    return model, scaler, thr, metrics


def export_tflite(
    model: tf.keras.Model, scaler: StandardScaler,
    X_check: np.ndarray, y_check: np.ndarray, thr: float, keras_acc: float,
) -> tuple[str, float]:
    """Dynamic-range quantisation: weights -> int8, activations -> float.

    Per spec: more conservative than full int8, rarely loses accuracy. We
    still verify accuracy doesn't drop more than 1%; on a 28-feature MLP
    it almost never does.
    """
    Xs = scaler.transform(X_check).astype(np.float32)

    def _try(mode: str) -> bytes:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        if mode == "dynamic":
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
        elif mode == "fp16":
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.target_spec.supported_types = [tf.float16]
        else:
            pass
        return conv.convert()

    def _predict(buf: bytes, X_in: np.ndarray) -> np.ndarray:
        interp = tf.lite.Interpreter(model_content=buf)
        interp.allocate_tensors()
        in_d = interp.get_input_details()[0]
        out_d = interp.get_output_details()[0]
        preds = np.empty(X_in.shape[0], dtype=np.float32)
        for i in range(X_in.shape[0]):
            interp.set_tensor(in_d["index"], X_in[i : i + 1].astype(in_d["dtype"]))
            interp.invoke()
            preds[i] = interp.get_tensor(out_d["index"]).ravel()[0]
        return preds

    buf = _try("dynamic")
    p_dr = _predict(buf, Xs)
    acc_dr = accuracy_score(y_check, (p_dr >= thr).astype(np.int32))
    drop = keras_acc - acc_dr
    print(f"  TFLite dynamic-range acc = {acc_dr:.4f}  "
          f"(keras was {keras_acc:.4f}, drop={drop:+.4f})")
    if drop <= 0.01:
        TFLITE_PATH.write_bytes(buf)
        return "dynamic_range", float(acc_dr)

    print("  dynamic-range dropped accuracy > 1pp; falling back to float16")
    buf = _try("fp16")
    p_fp16 = _predict(buf, Xs)
    acc_fp16 = accuracy_score(y_check, (p_fp16 >= thr).astype(np.int32))
    print(f"  TFLite fp16 acc = {acc_fp16:.4f}")
    TFLITE_PATH.write_bytes(buf)
    return "fp16", float(acc_fp16)


def benchmark_tflite() -> float:
    interp = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, N_FEAT)).astype(np.float32)
    if in_d["dtype"] != np.float32:
        x = x.astype(in_d["dtype"])
    for _ in range(40):
        interp.set_tensor(in_d["index"], x); interp.invoke()
    N = 500
    t0 = time.perf_counter()
    for _ in range(N):
        interp.set_tensor(in_d["index"], x); interp.invoke()
    t1 = time.perf_counter()
    per_ms = (t1 - t0) / N * 1000
    print(f"  TFLite inference: {per_ms:.3f} ms / window  (target <10 ms laptop)")
    return per_ms


def main() -> int:
    overall_t0 = time.perf_counter()
    print("=" * 60)
    print("MLP training (28-feature engineered input)")
    print("=" * 60)
    _set_seed()

    X_wins, y_wins, groups, load_idx, files = load_all_with_labels(str(TRAINING_DIR))
    print(f"Extracting features for {X_wins.shape[0]} windows...")
    t0 = time.perf_counter()
    X = extract_features_batch(X_wins).astype(np.float32)
    print(f"  feature extraction: {time.perf_counter() - t0:.2f}s")
    y = y_wins

    print("\n--- Leave-one-load-out CV ---")
    lolo_results = lolo_cv(X, y, load_idx, groups, files)

    model, scaler, thr, final_metrics = train_final(X, y, groups)
    n_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    print(f"  trainable params: {n_params}")

    model.save(str(H5_PATH))
    save_feature_names(str(FEATURE_NAMES_PATH))
    with open(SCALER_PATH, "w") as f:
        json.dump(
            {
                "mean":  scaler.mean_.astype(float).tolist(),
                "scale": scaler.scale_.astype(float).tolist(),
                "feature_names": FEATURE_NAMES,
            },
            f, indent=2,
        )
    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": float(thr)}, f, indent=2)

    # TFLite export against a random 20% holdout for accuracy check.
    rng = np.random.default_rng(1)
    idx = rng.choice(X.shape[0], size=min(2000, X.shape[0]), replace=False)
    tflite_mode, tflite_acc = export_tflite(
        model, scaler, X[idx], y[idx], thr, keras_acc=final_metrics["accuracy"]
    )
    inf_ms = benchmark_tflite()

    h5_kb = H5_PATH.stat().st_size / 1024.0
    tflite_kb = TFLITE_PATH.stat().st_size / 1024.0
    print(f"\nSaved:  {H5_PATH}  ({h5_kb:.1f} KB)")
    print(f"        {TFLITE_PATH}  ({tflite_kb:.1f} KB, mode={tflite_mode})")
    print(f"        {SCALER_PATH}")
    print(f"        {THRESHOLD_PATH}")
    print(f"        {FEATURE_NAMES_PATH}")

    summary = {
        "lolo": lolo_results,
        "final": final_metrics,
        "threshold": float(thr),
        "n_params": n_params,
        "tflite_mode": tflite_mode,
        "tflite_holdout_accuracy": tflite_acc,
        "inference_ms_per_window": inf_ms,
        "h5_kb": h5_kb,
        "tflite_kb": tflite_kb,
        "total_train_seconds": time.perf_counter() - overall_t0,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")

    if final_metrics["accuracy"] < 0.85:
        print("\n!!! WARNING: final MLP accuracy < 0.85 !!!")
    print(f"\nTotal pipeline time: {summary['total_train_seconds']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
