import pytest

from motor_fault.app import MonitorConfigurationError, MotorFaultMonitor
from motor_fault.config import AppConfig
from motor_fault.sensors import SerialSensorReader
import motor_fault.sensors as sensors_module


class FakeSerialConnection:
    def __init__(self, lines):
        self.lines = [line.encode("ascii") for line in lines]
        self.closed = False

    def reset_input_buffer(self):
        return None

    def readline(self):
        if self.lines:
            return self.lines.pop(0)
        return b""

    def close(self):
        self.closed = True


def test_serial_sensor_reader_reads_uart_and_usb_ports(monkeypatch):
    opened = {}

    def fake_serial_ctor(port, baudrate, timeout):
        data_by_port = {
            "/dev/serial0": ["~0.123\n"],
            "/dev/ttyUSB0": ["+1.500\n"],
        }
        connection = FakeSerialConnection(data_by_port[port])
        opened[port] = {
            "baudrate": baudrate,
            "timeout": timeout,
            "connection": connection,
        }
        return connection

    monkeypatch.setattr(sensors_module, "LenientSerial", fake_serial_ctor)

    config = AppConfig(
        sensor_ports={
            "I1": "/dev/serial0",
            "I2": "/dev/ttyUSB0",
        },
        warmup_seconds=0.0,
        read_attempts=1,
    )
    reader = SerialSensorReader(config)
    reader.open()

    sample = reader.read_currents()

    assert sample.currents == pytest.approx(
        {
            "I1": 0.123,
            "I2": 1.5,
        }
    )
    assert opened["/dev/serial0"]["baudrate"] == 9600
    assert opened["/dev/ttyUSB0"]["timeout"] == 1.0

    reader.close()

    assert opened["/dev/serial0"]["connection"].closed is True
    assert opened["/dev/ttyUSB0"]["connection"].closed is True


def test_monitor_requires_all_three_sensor_ports_for_inference():
    config = AppConfig(
        sensor_ports={
            "I1": "/dev/serial0",
            "I2": "/dev/ttyUSB0",
        }
    )

    with pytest.raises(MonitorConfigurationError, match="Missing: I3"):
        MotorFaultMonitor(config)
