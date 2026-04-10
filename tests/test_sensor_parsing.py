import pytest

from motor_fault.config import AppConfig
from motor_fault.sensors import parse_sensor_value
from motor_fault.sensors import _fallback_current


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("~0.123", 0.123),
        ("+1.500", 1.5),
        ("-2.250", -2.25),
        ("3.75", 3.75),
    ],
)
def test_parse_sensor_value(raw, expected):
    assert parse_sensor_value(raw) == pytest.approx(expected)


def test_parse_sensor_value_returns_none_for_empty_string():
    assert parse_sensor_value("") is None


def test_fallback_current_uses_config_value():
    config = AppConfig(sensor_read_fallback_enabled=True, sensor_read_fallback_value=0.0)
    assert _fallback_current(config) == pytest.approx(0.0)


def test_ordered_sensor_names_only_returns_configured_ports():
    config = AppConfig(
        sensor_ports={
            "I1": "/dev/serial0",
            "I2": "/dev/ttyUSB0",
        }
    )
    assert config.ordered_sensor_names() == ("I1", "I2")
    assert config.missing_sensor_names() == ("I3",)
