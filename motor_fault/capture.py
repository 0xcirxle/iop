"""CSV capture helpers for collecting timestamped sensor readings."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, TextIO

from .config import AppConfig
from .sensors import CurrentSample, build_sensor_reader


REQUIRED_CAPTURE_SENSORS = ("I1", "I2", "I3")
CAPTURE_FIELDNAMES = (
    "sample_index",
    "timestamp_epoch",
    "timestamp_iso",
    "I1",
    "I2",
    "I3",
)


class CaptureConfigurationError(RuntimeError):
    """Raised when CSV capture is started without all three sensors configured."""


@dataclass(frozen=True)
class CapturedRow:
    sample_index: int
    timestamp_epoch: float
    timestamp_iso: str
    currents: Dict[str, float]

    def as_dict(self) -> Dict[str, object]:
        return {
            "sample_index": self.sample_index,
            "timestamp_epoch": f"{self.timestamp_epoch:.6f}",
            "timestamp_iso": self.timestamp_iso,
            "I1": f"{float(self.currents['I1']):.6f}",
            "I2": f"{float(self.currents['I2']):.6f}",
            "I3": f"{float(self.currents['I3']):.6f}",
        }


class CurrentCsvCapture:
    """Continuously writes timestamped current samples to a CSV file."""

    def __init__(self, config: AppConfig):
        self.config = config
        missing = self.config.missing_sensor_names(REQUIRED_CAPTURE_SENSORS)
        if missing:
            raise CaptureConfigurationError(
                "CSV capture requires all three configured sensors "
                f"({', '.join(REQUIRED_CAPTURE_SENSORS)}). Missing: {', '.join(missing)}."
            )
        self.output_path = config.capture_file_path
        self.reader = build_sensor_reader(config)
        self._file: TextIO | None = None
        self._writer: csv.DictWriter | None = None
        self._sample_index = 0

    def open(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.output_path.exists()
        self._file = self.output_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CAPTURE_FIELDNAMES)
        if (not file_exists) or self.output_path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()
        self.reader.open()

    def close(self) -> None:
        self.reader.close()
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None

    def capture_once(self) -> CapturedRow:
        if self._writer is None or self._file is None:
            raise RuntimeError("CurrentCsvCapture must be opened before capture_once()")

        sample = self.reader.read_currents()
        self._sample_index += 1
        row = build_captured_row(sample, self._sample_index)
        self._writer.writerow(row.as_dict())
        self._file.flush()
        return row

    def run_forever(self) -> None:
        while True:
            started = time.time()
            self.capture_once()
            elapsed = time.time() - started
            time.sleep(max(0.0, self.config.sample_interval - elapsed))


def build_captured_row(sample: CurrentSample, sample_index: int) -> CapturedRow:
    timestamp_epoch = float(sample.timestamp)
    timestamp_iso = datetime.fromtimestamp(timestamp_epoch, tz=timezone.utc).isoformat()
    return CapturedRow(
        sample_index=sample_index,
        timestamp_epoch=timestamp_epoch,
        timestamp_iso=timestamp_iso,
        currents=sample.currents,
    )


def format_captured_row(row: CapturedRow, output_path: Path) -> str:
    return (
        f"saved #{row.sample_index} "
        f"{row.timestamp_iso} "
        f"I1={float(row.currents['I1']):.3f} "
        f"I2={float(row.currents['I2']):.3f} "
        f"I3={float(row.currents['I3']):.3f} "
        f"-> {output_path}"
    )
