"""Sensor abstractions for Raspberry Pi deployment."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

from .config import AppConfig

try:
    import serial  # type: ignore
except ImportError:  # pragma: no cover - optional on non-Pi machines
    serial = None


def parse_sensor_value(raw: str) -> Optional[float]:
    raw = raw.strip()
    if not raw:
        return None
    if raw[0] in "~+-":
        value = float(raw[1:])
        return -value if raw[0] == "-" else value
    return float(raw)


@dataclass
class CurrentSample:
    currents: Dict[str, float]
    timestamp: float


class SensorReadError(RuntimeError):
    """Raised when the hardware path is reachable but no valid reading is obtained."""


def _fallback_current(config: AppConfig) -> float:
    return float(config.sensor_read_fallback_value)


class SerialSensorReader:
    """Reads one sensor per configured serial device."""

    def __init__(self, config: AppConfig):
        if serial is None:
            raise RuntimeError("pyserial is required to use the sensor readers")
        self.config = config
        self.connections: Dict[str, object] = {}

    def open(self) -> None:
        if not self.config.sensor_ports:
            raise RuntimeError(
                "No sensor ports are configured. Set at least one of I1_PORT, I2_PORT, or I3_PORT."
            )
        for name, port in self.config.sensor_ports.items():
            self.connections[name] = serial.Serial(
                port=port,
                baudrate=self.config.baud_rate,
                timeout=self.config.serial_timeout,
            )
            time.sleep(self.config.warmup_seconds)

    def close(self) -> None:
        for connection in self.connections.values():
            connection.close()
        self.connections.clear()

    def read_currents(self) -> CurrentSample:
        values = {}
        for name, connection in self.connections.items():
            values[name] = self._read_sensor(name, connection)
        return CurrentSample(currents=values, timestamp=time.time())

    def _read_sensor(self, name: str, connection: object) -> float:
        port = self.config.sensor_ports[name]
        connection.reset_input_buffer()
        last_error = None
        raw_samples = []
        for _ in range(self.config.read_attempts):
            line = connection.readline().decode("ascii", errors="ignore")
            raw_samples.append(line.strip())
            try:
                value = parse_sensor_value(line)
            except ValueError as exc:
                last_error = exc
                value = None
            if value is not None:
                return value
        hint = (
            f"No valid serial data received for {name} on {port} "
            f"after {self.config.read_attempts} attempts. "
            "Check sensor power, GND, TX-to-RX wiring, port mapping, and the "
            "sensor output format."
        )
        if any(raw_samples):
            hint += f" Raw samples: {raw_samples!r}."
        if last_error is not None:
            hint += f" Last parse error: {last_error}."
        if self.config.sensor_read_fallback_enabled:
            return _fallback_current(self.config)
        raise SensorReadError(hint)


def build_sensor_reader(config: AppConfig):
    return SerialSensorReader(config)
