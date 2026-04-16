"""Model loading, live inference, and rolling decision helpers."""

from __future__ import annotations

import warnings
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, Optional

import joblib
import numpy as np

from .features import build_feature_vector
from .labels import TASK_LABELS, TASK_ORDER

warnings.filterwarnings(
    "ignore",
    message=".*Changing updater from `grow_gpu_hist` to `grow_quantile_histmaker`.*",
    category=UserWarning,
    module="xgboost.core",
)


@dataclass(frozen=True)
class PredictionResult:
    task: str
    class_id: int
    label: str
    probabilities: Dict[int, float]
    confidence: float | None = None


class MotorFaultPredictor:
    """Loads the saved scaler and task models and runs V2 RMS inference."""

    def __init__(
        self,
        model_dir: Path | str,
        confidence_threshold: float = 0.60,
        max_safe_current: float | None = None,
    ):
        self.model_dir = Path(model_dir)
        self.confidence_threshold = confidence_threshold
        self.max_safe_current = max_safe_current
        self.scaler = None
        self.models: Dict[str, object] = {}
        self._load()

    def _load(self) -> None:
        self.scaler = joblib.load(self.model_dir / "scaler.joblib")
        self.models = {
            task: joblib.load(self.model_dir / f"model_{task}.joblib")
            for task in TASK_ORDER
        }

    def available_tasks(self) -> Iterable[str]:
        return self.models.keys()

    def predict(self, i1_rms: float, i2_rms: float, i3_rms: float) -> Dict[str, PredictionResult]:
        features = build_feature_vector(i1_rms, i2_rms, i3_rms)
        scaled = self.scaler.transform(features)
        results: Dict[str, PredictionResult] = {}

        for task, model in self.models.items():
            probabilities = self._predict_probabilities(model, scaled)
            class_id = max(probabilities, key=probabilities.get)
            results[task] = PredictionResult(
                task=task,
                class_id=class_id,
                label=TASK_LABELS[task][class_id],
                probabilities=probabilities,
                confidence=probabilities[class_id],
            )

        return results

    def summarize_prediction(
        self,
        results: Dict[str, PredictionResult],
        i1_rms: float,
        i2_rms: float,
        i3_rms: float,
    ) -> Dict[str, object]:
        currents = np.array([i1_rms, i2_rms, i3_rms], dtype=np.float32)

        if self.max_safe_current is not None and np.max(np.abs(currents)) > self.max_safe_current:
            return {
                "safety_status": "TRIP_IMMEDIATE_OVERCURRENT",
                "binary": "Faulty",
                "binary_confidence": 1.0,
                "severity": "Unknown",
                "severity_confidence": None,
                "phase": "Unknown",
                "phase_confidence": None,
                "load": "Unknown",
                "load_confidence": None,
            }

        binary_result = results["binary"]
        load_result = results["load"]

        if binary_result.confidence is not None and binary_result.confidence < self.confidence_threshold:
            return {
                "safety_status": "ML_UNCERTAIN",
                "binary": "Uncertain",
                "binary_confidence": binary_result.confidence,
                "severity": "N/A",
                "severity_confidence": None,
                "phase": "N/A",
                "phase_confidence": None,
                "load": load_result.label,
                "load_confidence": load_result.confidence,
            }

        if binary_result.label == "Healthy":
            return {
                "safety_status": "OK",
                "binary": binary_result.label,
                "binary_confidence": binary_result.confidence,
                "severity": "N/A",
                "severity_confidence": None,
                "phase": "N/A",
                "phase_confidence": None,
                "load": load_result.label,
                "load_confidence": load_result.confidence,
            }

        severity_result = results["severity"]
        phase_result = results["phase"]
        severity_label = severity_result.label
        phase_label = phase_result.label

        if severity_result.confidence is not None and severity_result.confidence < self.confidence_threshold:
            severity_label = "Uncertain"
        if phase_result.confidence is not None and phase_result.confidence < self.confidence_threshold:
            phase_label = "Uncertain"

        return {
            "safety_status": "OK",
            "binary": binary_result.label,
            "binary_confidence": binary_result.confidence,
            "severity": severity_label,
            "severity_confidence": severity_result.confidence,
            "phase": phase_label,
            "phase_confidence": phase_result.confidence,
            "load": load_result.label,
            "load_confidence": load_result.confidence,
        }

    def predict_live(self, i1_rms: float, i2_rms: float, i3_rms: float) -> Dict[str, object]:
        results = self.predict(i1_rms, i2_rms, i3_rms)
        return self.summarize_prediction(results, i1_rms, i2_rms, i3_rms)

    @staticmethod
    def _predict_probabilities(model: object, scaled_features: np.ndarray) -> Dict[int, float]:
        if hasattr(model, "predict_proba"):
            raw = MotorFaultPredictor._run_quietly(model.predict_proba, scaled_features)[0]
            return {index: float(value) for index, value in enumerate(raw)}
        predicted = int(MotorFaultPredictor._run_quietly(model.predict, scaled_features)[0])
        return {predicted: 1.0}

    @staticmethod
    def _run_quietly(func, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Changing updater from `grow_gpu_hist` to `grow_quantile_histmaker`.*",
                category=UserWarning,
            )
            return func(*args, **kwargs)


class RollingMotorDecision:
    """Stabilize live predictions using a short rolling vote window."""

    def __init__(
        self,
        window_size: int = 5,
        fault_votes_required: int = 3,
    ):
        self.window_size = window_size
        self.fault_votes_required = fault_votes_required
        self.history: Deque[Dict[str, object]] = deque(maxlen=window_size)

    def update(self, current_prediction: Dict[str, object]) -> Dict[str, Dict[str, object]]:
        if current_prediction["safety_status"] == "TRIP_IMMEDIATE_OVERCURRENT":
            return {
                "current_prediction": current_prediction,
                "rolling_decision": dict(current_prediction),
            }

        self.history.append(current_prediction)
        fault_votes = sum(1 for item in self.history if item["binary"] == "Faulty")
        healthy_votes = sum(1 for item in self.history if item["binary"] == "Healthy")

        if fault_votes >= self.fault_votes_required:
            binary = "Faulty"
        elif len(self.history) == self.window_size and healthy_votes > fault_votes:
            binary = "Healthy"
        else:
            binary = str(current_prediction["binary"])

        rolling_decision = {
            "binary": binary,
            "severity": self._majority_value("severity") if binary == "Faulty" else "N/A",
            "phase": self._majority_value("phase") if binary == "Faulty" else "N/A",
            "load": self._majority_value("load"),
            "window_filled": len(self.history),
            "fault_votes": fault_votes,
            "healthy_votes": healthy_votes,
        }

        return {
            "current_prediction": current_prediction,
            "rolling_decision": rolling_decision,
        }

    def _majority_value(self, key: str) -> str:
        values = [
            str(item[key])
            for item in self.history
            if item[key] not in {"N/A", "Uncertain", "Unknown"}
        ]
        if not values:
            return "N/A"
        return Counter(values).most_common(1)[0][0]
