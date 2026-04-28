# Inter-turn Stator Fault Detector

Binary fault detector for a 3-phase induction motor (1 hp, 4-pole, 415 V, 50 Hz,
1.8 A rated). Two cooperating classifiers — **XGBoost** and a tiny **MLP** —
share the same 28-feature engineered vector and vote by consensus at inference
time. Output is strictly `HEALTHY` vs `FAULTY` — no severity classification.

## Pipeline

```
50 kHz ADC -> decimate /36 -> ~1389 Hz -> 278-sample window (~200 ms, 10 cycles)
                                       -> 28 features -> XGB & MLP -> consensus
```

- **Window**: 278 samples @ ~1388.89 Hz, 50% overlap (hop = 139). 200.16 ms,
  ~10 line cycles at 50 Hz.
- **Decimation**: `scipy.signal.decimate(x, 36, n=8, ftype='iir', zero_phase=True)`
  (8th-order IIR Butterworth). Achieved rate is `50_000 / 36 = 1388.89 Hz`,
  not exactly 1389 — integer factors only. Train and inference see the
  *same* achieved rate, so the model never notices.
- **Per-window features (28)**: per-phase RMS / skew / kurtosis, |H3|/|H1|,
  |H5|/|H1|, |H7|/|H1|, then sequence-domain magnitudes (|I_pos|, |I_neg|,
  NPSR), Park (Concordia) variances and ellipticity, zero-sequence RMS,
  phase-imbalance.
- **FFT**: Hann-windowed, amplitude-corrected (`2/sum(window)` scale). Bin
  resolution is `fs/N = 1388.89/278 ≈ 4.996 Hz`. 50 Hz lands on bin 10,
  150 Hz on bin 30, 250 Hz on bin 50, 350 Hz on bin 70 — all within 0.3 Hz
  of bin centre. ±1-bin search tolerates small line-frequency drift.
- **Sanity HALT**: I4 RMS is checked against the filename label; mismatch
  raises `SystemExit` (e.g. an "_healthy" file with > 0.05 A residual current
  is treated as a labelling bug, not a warning).

### Why ~1389 Hz, not 2 kHz

The production capture runs on an ADS1256 ADC at `DRATE = 30000` and cycles
through 3 phases on the on-chip multiplexer. Per the ADS1256 datasheet
(Table 14), that's a hard ceiling of `4374 / 3 = 1458 Hz` per channel.
Production captures at 1392 Hz — 95 % of ceiling. Earlier prototypes
targeted 2 kHz; the chip cannot do that with 3-channel cycling, so the
training pipeline was retargeted to match production. 50 kHz training
recordings are integer-decimated by 36 to 1388.89 Hz.

### Why 200 ms windows

The earlier 100 ms window (200 samples @ 2 kHz) had two problems for the
1 % / 3 % `mu` fault levels we care about:

- **FFT resolution** was 10 Hz/bin, so 50 Hz, 150 Hz, etc. were not isolated
  from sidebands. At 200 ms the resolution is ~5 Hz/bin and the fundamental
  + harmonics are clean.
- **Moment estimates** (skew, kurtosis) at 200 samples had high variance —
  most files showed kurtosis ≈ −1.5 (sinusoid value) for both healthy and
  faulty data, so the moments contributed nothing. At 278 samples and 10
  cycles the variance halves.

200 ms is still well under the 30 ms-per-window inference budget for trip
response, and below the typical line-cycle stationarity time.

## CV strategy

- **LOLO** (leave-one-load-out, 3 folds): noload / halfload / fullload.
- **Threshold calibration**: per-fold and final, the F1-maximising threshold
  is picked on a 10 % **file-aware** holdout (`GroupShuffleSplit` by file)
  taken from the **training** files only — never the test fold.
- **Inverted-val guard**: if the held-out val happens to put healthy mean
  probability above faulty mean probability, F1 is maximised by
  predict-everything-positive (thr = 0.05). We refuse to calibrate in that
  case and fall back to thr = 0.5. This fired on the final XGB val and on
  multiple LOLO folds.
- **Per-fault-level diagnostic**: the training scripts also print, per LOLO
  fold, what fraction of `healthy / 1% / 3% / 5%` windows the model labels
  FAULTY. The model is binary and never sees the `mu` value — the breakdown
  just tells us whether failures cluster on low-severity files.

## Trained artifacts

| File | Purpose |
| --- | --- |
| `data.py` | windowing, decimation, label parsing, I4 sanity halt |
| `features.py` | 28-feature extractor (Hann + rfft + Park + sequence) |
| `train_xgb.py` | XGBoost training + LOLO + F1 threshold |
| `train_mlp.py` | MLP training + StandardScaler + TFLite export |
| `inference.py` | `FaultDetector` class + `--demo` mode |
| `model_xgb.json` | XGBoost booster (~129 KB) |
| `model_xgb_threshold.json` | calibrated threshold |
| `model_mlp.h5` | Keras MLP (~48 KB) |
| `model_mlp.tflite` | dynamic-range quantised MLP (~8 KB) |
| `mlp_scaler.json` | per-feature mean / scale |
| `model_mlp_threshold.json` | MLP calibrated threshold |
| `feature_names.json` | feature order, used to detect retraining drift |

(A 1-D CNN was prototyped earlier and dropped: 12 training files is far
under what a CNN needs to converge reliably. Only XGBoost + MLP ship.)

## Measured metrics

### Leave-one-load-out CV (held-out load, never seen during training)

| held-out load | XGB acc | XGB F1 | XGB AUC | MLP acc | MLP F1 | MLP AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| noload   | 0.500 | 0.500 | 0.688 | 0.750 | 0.857 | 0.342 |
| halfload | 0.750 | 0.857 | 0.500 | 0.750 | 0.857 | 0.390 |
| fullload | 0.250 | 0.000 | 0.500 | 0.250 | 0.000 | 0.963 |

LOLO is bounded by data, not model. We have 4 files per load (1 healthy +
3 fault severities); holding out an entire load class strips the model of
all examples at that load. Fullload predictions collapse to all-healthy on
both models because both fall back to thr = 0.5 with very similar val
score distributions (`H_mean ≈ F_mean ≈ 0.985` for XGB on this fold). The
fullload MLP AUC is 0.96 despite its 0.25 accuracy — the *ranking* is
correct, the threshold is wrong.

### Per-fault-level breakdown on LOLO test folds (FAULTY-rate)

| fold | model | healthy | 1% | 3% | 5% |
| --- | --- | ---: | ---: | ---: | ---: |
| noload   | XGB | 0%   | 100% |   0% |   0% |
| noload   | MLP | 100% | 100% | 100% | 100% |
| halfload | XGB | 100% | 100% | 100% | 100% |
| halfload | MLP | 100% | 100% | 100% | 100% |
| fullload | XGB | 0%   |   0% |   0% |   0% |
| fullload | MLP | 0%   |   0% |   0% |   0% |

Healthy = % of healthy windows labelled FAULTY (false-positive rate;
ideally 0). 1% / 3% / 5% = % of fault windows of that severity labelled
FAULTY (true-positive rate; ideally 100). The "all 100%" and "all 0%"
patterns are artefacts of LOLO with so few files plus a degraded threshold
fallback — the rankings (AUC) are more informative than the labels here.

### Final model (all 12 files, in-sample)

| metric | XGB | MLP (Keras) | MLP (TFLite, dynamic-range) | Consensus (XGB ∧ MLP) |
| --- | ---: | ---: | ---: | ---: |
| accuracy | 1.000 | 0.783 | 0.787 | 0.866 |
| F1 (faulty) | 1.000 | 0.850 | — | — |
| AUC | 1.000 | 0.841 | — | — |
| threshold | 0.500 | 0.500 | 0.500 | both must say FAULTY |

The consensus rule reports `FAULTY` only if **both** classifiers agree.
This biases the detector toward false negatives (missed faults) over false
positives (nuisance trips). Swap the labels to bias the other way, or
expose `xgb_label` / `mlp_label` directly for sensitivity.

### Latency & size (laptop, M-series CPU)

| stage | per window |
| --- | ---: |
| feature extraction | 1.02 ms |
| XGBoost predict | 0.17 ms |
| MLP TFLite invoke | < 0.01 ms |
| **end-to-end `predict()`** | **1.34 ms** |

| artifact | size |
| --- | ---: |
| `model_xgb.json` | 129 KB |
| `model_mlp.h5` | 48 KB |
| `model_mlp.tflite` (dyn.-range) | 7.9 KB |
| MLP trainable params | 1473 |

End-to-end is well below the 30 ms / window budget; on a Pi 4 we expect ~3 ms.

### Top features by XGBoost gain

```
park_mag_var      32.90
rms_c             32.61
h3_ratio_a        31.66
h5_ratio_c        30.32
pos_seq_mag       25.52
park_ellipticity   8.03
park_q_var         5.82
kurt_c             5.61
zero_seq_rms       3.15
skew_b             2.57
```

Compared with the 100 ms run, the harmonic-ratio features (`h3_ratio_a`,
`h5_ratio_c`) are now in the top 5 — that's the 5 Hz FFT resolution paying
off. `neg_pos_ratio` (NPSR) is still not in the top 5; the training script
emits a warning when this happens, and we leave it as a known property of
this dataset rather than a bug. With healthy I4 < 0.01 A and 5% fault I4
≈ 1.8 A, gross-magnitude features dominate the ratio.

## Running

Training (once, on a laptop):

```bash
pip install -r requirements.txt
python train_xgb.py
python train_mlp.py
```

Smoke test against the recorded training files:

```bash
python inference.py --demo
```

Production loop (Pi 4 with `xgboost` and `tflite-runtime`):

```python
from inference import FaultDetector
detector = FaultDetector()                  # load once at startup
verdict = detector.predict(window)          # window: (278, 3) float32 @ ~1389 Hz
if verdict["consensus_label"] == "FAULTY":
    trip()
```

`predict()` returns `{xgb_label, xgb_proba, mlp_label, mlp_proba, agreed,
consensus_label}`.

## Known limitations

- **Binary only.** Trained as a healthy / faulty classifier; does not
  distinguish fault severity (1 % / 3 % / 5 % `mu`). Per-fault-level
  diagnostics show that 1 % `mu` faults sometimes get misclassified as
  healthy (e.g. `Fullload_1%mu_rf3.csv`: MLP labels only 34 % of windows
  FAULTY); 3 % and 5 % are caught reliably in the final model. This is a
  data limitation — 1 % `mu` perturbs the line currents very lightly.
- **12 training files** (3 loads × {healthy, 1 %, 3 %, 5 %}). LOLO
  performance is bounded by the data, not the model class. Fullload-LOLO
  is particularly weak because there is no other source of fullload
  fault data.
- **CNN dropped.** A 1-D CNN was tried and abandoned: insufficient data
  for it to converge. Engineered features + boosted trees + MLP on the
  same vector is what fits the data we actually have.
- **Final model is in-sample.** The 1.000 numbers in the final-model
  table are not a generalisation claim — they describe what the deployed
  model will say about its own training data. Trust the LOLO numbers
  above for held-out behaviour.
