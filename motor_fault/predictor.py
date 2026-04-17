"""Live inference helpers for the threshold-based motor fault model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from motor_fault_model import BUFFER_N, LiveInferencer


LABELS = {
    0: "Healthy",
    1: "Faulty",
}


@dataclass(frozen=True)
class PredictionResult:
    label_id: int | None
    label: str | None
    proba_fault: float | None
    ready: bool
    reason: str | None
    buffer_fill: int
    buffer_size: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "label_id": self.label_id,
            "label": self.label,
            "proba_fault": self.proba_fault,
            "ready": self.ready,
            "reason": self.reason,
            "buffer_fill": self.buffer_fill,
            "buffer_size": self.buffer_size,
        }


class MotorFaultPredictor:
    """Wrap the integrated rolling inferencer used by the new threshold model."""

    def __init__(self, model_path: Path | str, buffer_n: int = BUFFER_N):
        self.model_path = Path(model_path)
        self.inferencer = LiveInferencer(self.model_path, buffer_n=buffer_n)

    @property
    def buffer_size(self) -> int:
        return self.inferencer.buffer_n

    def update(self, i1_rms: float, i2_rms: float, i3_rms: float) -> PredictionResult:
        raw = self.inferencer.update(i1_rms, i2_rms, i3_rms)
        label_id = None if raw["label"] is None else int(raw["label"])
        return PredictionResult(
            label_id=label_id,
            label=LABELS.get(label_id),
            proba_fault=raw["proba_fault"],
            ready=bool(raw["ready"]),
            reason=raw.get("reason"),
            buffer_fill=len(self.inferencer.rolling_buffer),
            buffer_size=self.inferencer.buffer_n,
        )
