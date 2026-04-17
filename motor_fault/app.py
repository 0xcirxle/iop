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


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


class MonitorConfigurationError(RuntimeError):
    """Raised when monitor mode is used without the required sensor inputs."""


class MotorFaultMonitor:
    """Coordinates sensors, rolling model inference, and optional cloud upload."""

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
            config.model_path,
            buffer_n=config.rolling_buffer_size,
        )
        self.uploader: Optional[ThingSpeakUploader] = None
        if config.thingspeak_enabled:
            self.uploader = ThingSpeakUploader(
                api_key=config.thingspeak_api_key,
                url=config.thingspeak_url,
            )

    def open(self) -> None:
        LOGGER.info("Using model path: %s", self.config.model_path)
        self.reader.open()

    def close(self) -> None:
        self.reader.close()

    def run_once(self) -> Dict[str, object]:
        sample = self.reader.read_currents()
        prediction = self.predictor.update(
            sample.currents["I1"],
            sample.currents["I2"],
            sample.currents["I3"],
        )
        payload = self._build_payload(sample, prediction)
        if self.uploader is not None:
            uploaded = self.uploader.upload(sample.currents, prediction)
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
        prediction: PredictionResult,
    ) -> Dict[str, object]:
        return {
            "timestamp": sample.timestamp,
            "currents": sample.currents,
            "prediction": prediction.as_dict(),
        }


def format_monitor_result(payload: Dict[str, object]) -> str:
    currents = payload["currents"]
    current_text = " ".join(
        f"{name}={float(currents[name]):.3f}"
        for name in REQUIRED_PREDICTION_SENSORS
        if name in currents
    )
    prediction = payload["prediction"]

    return f"currents: {current_text} | prediction: {format_prediction_summary(prediction)}"


def format_prediction_summary(prediction: Dict[str, object]) -> str:
    if not prediction.get("ready"):
        reason = prediction.get("reason")
        if reason == "warmup":
            return (
                f"warmup [{int(prediction.get('buffer_fill', 0))}/"
                f"{int(prediction.get('buffer_size', 0))}]"
            )
        if reason == "motor_off":
            return "motor_off"
        return str(reason or "not_ready")

    return (
        f"{prediction['label']} "
        f"p_fault={float(prediction['proba_fault']):.2f}"
    )
