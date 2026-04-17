import pytest

import motor_fault.capture as capture_module
from motor_fault.capture import CaptureConfigurationError, CurrentCsvCapture, build_captured_row
from motor_fault.config import AppConfig
from motor_fault.sensors import CurrentSample


class FakeReader:
    def __init__(self, sample: CurrentSample):
        self.sample = sample
        self.open_called = False
        self.close_called = False

    def open(self):
        self.open_called = True

    def close(self):
        self.close_called = True

    def read_currents(self):
        return self.sample


def test_build_captured_row_includes_timestamp_and_currents():
    sample = CurrentSample(
        currents={"I1": 1.1, "I2": 2.2, "I3": 3.3},
        timestamp=1712345678.25,
    )

    row = build_captured_row(sample, sample_index=4)

    assert row.sample_index == 4
    assert row.timestamp_epoch == 1712345678.25
    assert row.timestamp_iso == "2024-04-05T19:34:38.250000+00:00"
    assert row.currents == {"I1": 1.1, "I2": 2.2, "I3": 3.3}


def test_current_csv_capture_writes_header_and_row(tmp_path, monkeypatch):
    sample = CurrentSample(
        currents={"I1": 1.25, "I2": 1.35, "I3": 1.45},
        timestamp=1712345678.5,
    )
    fake_reader = FakeReader(sample)
    monkeypatch.setattr(capture_module, "build_sensor_reader", lambda config: fake_reader)
    output_path = tmp_path / "Noload_healthy.csv"

    capture = CurrentCsvCapture(
        AppConfig(
            sensor_ports={"I1": "/dev/serial0", "I2": "/dev/ttyUSB0", "I3": "/dev/ttyUSB1"},
            capture_file_name=str(output_path),
            warmup_seconds=0.0,
            read_attempts=1,
        )
    )

    capture.open()
    row = capture.capture_once()
    capture.close()

    assert fake_reader.open_called is True
    assert fake_reader.close_called is True
    assert row.sample_index == 1

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "sample_index,timestamp_epoch,timestamp_iso,I1,I2,I3"
    assert lines[1] == "1,1712345678.500000,2024-04-05T19:34:38.500000+00:00,1.250000,1.350000,1.450000"


def test_current_csv_capture_requires_all_three_sensors():
    with pytest.raises(CaptureConfigurationError, match="Missing: I3"):
        CurrentCsvCapture(
            AppConfig(
                sensor_ports={"I1": "/dev/serial0", "I2": "/dev/ttyUSB0"},
            )
        )
