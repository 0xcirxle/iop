"""Model loading and inference helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import joblib
import numpy as np

from .features import build_feature_vector
from .labels import TASK_ORDER, get_task_labels

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


class MotorFaultPredictor:
    """Loads the saved scaler and task models and runs inference."""

    def __init__(self, model_dir: Path | str, binary_labels_swapped: bool = False):
        self.model_dir = Path(model_dir)
        self.task_labels = get_task_labels(binary_labels_swapped=binary_labels_swapped)
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

    def predict(self, i1: float, i2: float, i3: float) -> Dict[str, PredictionResult]:
        features = build_feature_vector(i1, i2, i3)
        scaled = self.scaler.transform(features)
        results: Dict[str, PredictionResult] = {}

        for task, model in self.models.items():
            class_id = int(self._run_quietly(model.predict, scaled)[0])
            probabilities = self._predict_probabilities(model, scaled)
            results[task] = PredictionResult(
                task=task,
                class_id=class_id,
                label=self.task_labels[task][class_id],
                probabilities=probabilities,
            )

        return results

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
