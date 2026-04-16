"""Configuration helpers for local and Raspberry Pi deployment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


DEFAULT_SENSOR_PORTS = {
    "I1": "/dev/serial0",
    "I2": "/dev/ttyUSB0",
    "I3": "/dev/ttyUSB1",
}
SENSOR_ORDER = ("I1", "I2", "I3")


def _load_env_file() -> None:
    """Load a local .env file without requiring python-dotenv."""
    root = Path(__file__).resolve().parent.parent
    candidates = (Path.cwd() / ".env", root / ".env")

    for env_path in candidates:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            os.environ.setdefault(key, value)
        break


_load_env_file()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return float(raw)


def _build_sensor_ports() -> Dict[str, str]:
    ports: Dict[str, str] = {}
    for name in SENSOR_ORDER:
        default = DEFAULT_SENSOR_PORTS[name]
        raw_value = os.getenv(f"{name}_PORT")
        if raw_value is None and name == "I1":
            # Keep supporting the older UART_PORT variable for the first sensor.
            raw_value = os.getenv("UART_PORT")
        port = (raw_value if raw_value is not None else default).strip()
        if port:
            ports[name] = port
    return ports


def _detect_model_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    candidates = (
        Path(os.getenv("MODEL_DIR", "")) if os.getenv("MODEL_DIR") else None,
        root / "trained_models_v2",
        root / "trained_models",
        root,
    )
    for candidate in candidates:
        if candidate and (candidate / "scaler.joblib").exists():
            return candidate
    return root


@dataclass(frozen=True)
class AppConfig:
    baud_rate: int = _env_int("BAUD_RATE", 9600)
    serial_timeout: float = _env_float("SERIAL_TIMEOUT", 1.0)
    sample_interval: float = _env_float("SAMPLE_INTERVAL", 1.0)
    warmup_seconds: float = _env_float("SENSOR_WARMUP_SECONDS", 0.1)
    read_attempts: int = _env_int("SENSOR_READ_ATTEMPTS", 5)
    prediction_confidence_threshold: float = _env_float(
        "PREDICTION_CONFIDENCE_THRESHOLD",
        0.60,
    )
    rolling_window_size: int = _env_int("ROLLING_WINDOW_SIZE", 5)
    fault_votes_required: int = _env_int("FAULT_VOTES_REQUIRED", 3)
    max_safe_current: float | None = field(
        default_factory=lambda: _env_optional_float("MAX_SAFE_CURRENT")
    )
    sensor_read_fallback_enabled: bool = _env_bool(
        "SENSOR_READ_FALLBACK_ENABLED",
        False,
    )
    sensor_read_fallback_value: float = _env_float("SENSOR_READ_FALLBACK_VALUE", 0.0)
    sensor_ports: Dict[str, str] = field(default_factory=_build_sensor_ports)
    model_dir: Path = field(default_factory=_detect_model_dir)
    thingspeak_api_key: str = os.getenv(
        "THINGSPEAK_API_KEY",
        "REPLACE_WITH_YOUR_THINGSPEAK_WRITE_API_KEY",
    )
    thingspeak_url: str = os.getenv(
        "THINGSPEAK_URL",
        "https://api.thingspeak.com/update",
    )
    thingspeak_enabled: bool = os.getenv("THINGSPEAK_ENABLED", "false").lower() == "true"

    def ordered_sensor_names(self) -> Tuple[str, ...]:
        return tuple(name for name in SENSOR_ORDER if name in self.sensor_ports)

    def missing_sensor_names(
        self,
        required: Tuple[str, ...] = SENSOR_ORDER,
    ) -> Tuple[str, ...]:
        return tuple(name for name in required if name not in self.sensor_ports)
