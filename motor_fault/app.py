"""Application orchestration for the motor fault monitor."""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, Optional

from .cloud import ThingSpeakUploader
from .config import AppConfig
from .predictor import MotorFaultPredictor, PredictionResult, RollingMotorDecision
from .sensors import CurrentSample, SensorReadError, build_sensor_reader


LOGGER = logging.getLogger("motor_fault")
REQUIRED_PREDICTION_SENSORS = ("I1", "I2", "I3")


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


class MonitorConfigurationError(RuntimeError):
    """Raised when monitor mode is used without the required sensor inputs."""


class MotorFaultMonitor:
    """Coordinates sensors, model inference, and optional cloud upload."""

    def __init__(self, config: AppConfig):
        self.config = config
        missing = self.config.missing_sensor_names(REQUIRED_PREDICTION_SENSORS)
        if missing:
            raise MonitorConfigurationError(
                "Full monitor mode still requires three configured sensors "
                f"({', '.join(REQUIRED_PREDICTION_SENSORS)}). Missing: {', '.join(missing)}. "
                "Use `python test_sensors.py` to validate the currently connected sensors."
            )
        self.reader = build_sensor_reader(config)
        self.predictor = MotorFaultPredictor(
            config.model_dir,
            confidence_threshold=config.prediction_confidence_threshold,
            max_safe_current=config.max_safe_current,
        )
        self.rolling_decision = RollingMotorDecision(
            window_size=config.rolling_window_size,
            fault_votes_required=config.fault_votes_required,
        )
        self.uploader: Optional[ThingSpeakUploader] = None
        if config.thingspeak_enabled:
            self.uploader = ThingSpeakUploader(
                api_key=config.thingspeak_api_key,
                url=config.thingspeak_url,
            )

    def open(self) -> None:
        LOGGER.info("Using model directory: %s", self.config.model_dir)
        self.reader.open()

    def close(self) -> None:
        self.reader.close()

    def run_once(self) -> Dict[str, object]:
        sample = self.reader.read_currents()
        predictions = self.predictor.predict(
            sample.currents["I1"],
            sample.currents["I2"],
            sample.currents["I3"],
        )
        current_prediction = self.predictor.summarize_prediction(
            predictions,
            sample.currents["I1"],
            sample.currents["I2"],
            sample.currents["I3"],
        )
        rolling_result = self.rolling_decision.update(current_prediction)
        payload = self._build_payload(
            sample,
            predictions,
            rolling_result["current_prediction"],
            rolling_result["rolling_decision"],
        )
        if self.uploader is not None:
            uploaded = self.uploader.upload(sample.currents, predictions)
            payload["thingspeak_uploaded"] = uploaded
        return payload

    def run_forever(self) -> None:
        while True:
            started = time.time()
            try:
                result = self.run_once()
                LOGGER.info(format_monitor_result(result))
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(json.dumps(result, ensure_ascii=True))
            except SensorReadError as exc:
                LOGGER.error("Sensor read failed: %s", exc)
            elapsed = time.time() - started
            time.sleep(max(0.0, self.config.sample_interval - elapsed))

    @staticmethod
    def _build_payload(
        sample: CurrentSample,
        predictions: Dict[str, PredictionResult],
        current_prediction: Dict[str, object],
        rolling_decision: Dict[str, object],
    ) -> Dict[str, object]:
        return {
            "timestamp": sample.timestamp,
            "currents": sample.currents,
            "current_prediction": current_prediction,
            "rolling_decision": rolling_decision,
            "predictions": {
                task: {
                    "class_id": result.class_id,
                    "label": result.label,
                    "confidence": result.confidence,
                    "probabilities": result.probabilities,
                }
                for task, result in predictions.items()
            },
        }


def format_monitor_result(payload: Dict[str, object]) -> str:
    currents = payload["currents"]
    current_text = " ".join(
        f"{name}={float(currents[name]):.3f}"
        for name in REQUIRED_PREDICTION_SENSORS
        if name in currents
    )
    current_prediction = payload["current_prediction"]
    rolling_decision = payload["rolling_decision"]

    return (
        f"currents: {current_text} | "
        f"current: {format_prediction_summary(current_prediction)} | "
        f"rolling: {format_prediction_summary(rolling_decision, include_votes=True)}"
    )


def format_prediction_summary(
    prediction: Dict[str, object],
    *,
    include_votes: bool = False,
) -> str:
    if prediction.get("safety_status") == "TRIP_IMMEDIATE_OVERCURRENT":
        return "TRIP_IMMEDIATE_OVERCURRENT"

    parts = [str(prediction["binary"])]
    binary_confidence = prediction.get("binary_confidence")
    if binary_confidence is not None and not include_votes:
        parts[0] = f"{parts[0]}({float(binary_confidence):.2f})"

    for key in ("severity", "phase", "load"):
        value = prediction.get(key)
        if value and value not in {"N/A", "Unknown", "Uncertain"}:
            parts.append(str(value))

    if include_votes:
        parts.append(
            f"[F={int(prediction.get('fault_votes', 0))} "
            f"H={int(prediction.get('healthy_votes', 0))} "
            f"W={int(prediction.get('window_filled', 0))}]"
        )

    return " ".join(parts)
