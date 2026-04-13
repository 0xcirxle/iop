"""Application orchestration for the motor fault monitor."""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, Optional

from .cloud import ThingSpeakUploader
from .config import AppConfig
from .predictor import MotorFaultPredictor, PredictionResult
from .sensors import CurrentSample, SensorReadError, build_sensor_reader


LOGGER = logging.getLogger("motor_fault")
REQUIRED_PREDICTION_SENSORS = ("I1", "I2", "I3")
PREDICTION_TASK_ORDER = ("binary", "severity", "phase", "load")


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
        self.predictor = MotorFaultPredictor(config.model_dir)
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
        payload = self._build_payload(sample, predictions)
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
    ) -> Dict[str, object]:
        return {
            "timestamp": sample.timestamp,
            "currents": sample.currents,
            "prediction_summary": build_prediction_summary(predictions),
            "predictions": {
                task: {
                    "class_id": result.class_id,
                    "label": result.label,
                    "probabilities": result.probabilities,
                }
                for task, result in predictions.items()
            },
        }


def build_prediction_summary(
    predictions: Dict[str, PredictionResult],
) -> Dict[str, str]:
    return {
        task: predictions[task].label
        for task in PREDICTION_TASK_ORDER
        if task in predictions
    }


def format_monitor_result(payload: Dict[str, object]) -> str:
    currents = payload["currents"]
    prediction_summary = payload.get("prediction_summary", {})

    current_text = " ".join(
        f"{name}={float(currents[name]):.3f}"
        for name in REQUIRED_PREDICTION_SENSORS
        if name in currents
    )
    prediction_text = " ".join(
        f"{task}={prediction_summary[task]}"
        for task in PREDICTION_TASK_ORDER
        if task in prediction_summary
    )

    if prediction_text:
        return f"currents: {current_text} | predictions: {prediction_text}"
    return f"currents: {current_text}"
