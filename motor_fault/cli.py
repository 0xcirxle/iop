"""CLI entry points for local checks and Raspberry Pi deployment."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace

from motor_fault_model import BUFFER_N

from .app import MonitorConfigurationError, MotorFaultMonitor, configure_logging
from .config import AppConfig
from .predictor import MotorFaultPredictor
from .sensors import SensorReadError, build_sensor_reader, parse_sensor_value


def _cmd_predict(args: argparse.Namespace) -> int:
    config = AppConfig()
    predictor = MotorFaultPredictor(
        config.model_path,
        buffer_n=config.rolling_buffer_size,
    )
    current_prediction = None
    for _ in range(args.repeats):
        current_prediction = predictor.update(args.i1, args.i2, args.i3)
    if current_prediction is None:
        raise ValueError("--repeats must be at least 1")
    payload = {
        "currents": {
            "I1": args.i1,
            "I2": args.i2,
            "I3": args.i3,
        },
        "repeats_used": args.repeats,
        "prediction": current_prediction.as_dict(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


def _cmd_test_sensors(args: argparse.Namespace) -> int:
    config = AppConfig()
    reader = build_sensor_reader(config)
    reader.open()
    failures = 0
    try:
        if args.live:
            print("Streaming live serial data. Press Ctrl+C to stop.", file=sys.stderr)
            for name, port in config.sensor_ports.items():
                print(f"listening on {name}: {port}", file=sys.stderr)
            _stream_live_sensor_data(reader, poll_timeout=args.poll_timeout)
            return 0
        for index in range(args.samples):
            try:
                sample = reader.read_currents()
                print(f"sample {index + 1}: {sample.currents}")
            except SensorReadError as exc:
                failures += 1
                print(f"sample {index + 1}: ERROR: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nStopped sensor stream.", file=sys.stderr)
        return 0
    finally:
        reader.close()
    return 0 if failures == 0 else 1


def _stream_live_sensor_data(reader, poll_timeout: float) -> None:
    for connection in reader.connections.values():
        connection.timeout = poll_timeout

    while True:
        for name, connection in reader.connections.items():
            port = reader.config.sensor_ports[name]
            try:
                line = connection.readline().decode("ascii", errors="ignore")
            except Exception as exc:
                _print_live_sensor_line(
                    name=name,
                    port=port,
                    raw="",
                    value=None,
                    error=f"read_error={exc}",
                )
                continue

            raw = line.strip()
            if not raw:
                continue

            try:
                value = parse_sensor_value(line)
            except ValueError as exc:
                _print_live_sensor_line(
                    name=name,
                    port=port,
                    raw=raw,
                    value=None,
                    error=f"parse_error={exc}",
                )
                continue

            _print_live_sensor_line(name=name, port=port, raw=raw, value=value, error=None)


def _print_live_sensor_line(
    *,
    name: str,
    port: str,
    raw: str,
    value: float | None,
    error: str | None,
) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if error is not None:
        print(
            f"{timestamp} {name} {port} raw={raw!r} {error}",
            flush=True,
        )
        return
    print(
        f"{timestamp} {name} {port} raw={raw!r} value={value}",
        flush=True,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    configure_logging(verbose=args.verbose)
    config = AppConfig()
    if args.interval is not None:
        config = replace(config, sample_interval=args.interval)
    try:
        monitor = MotorFaultMonitor(config)
    except MonitorConfigurationError as exc:
        print(exc, file=sys.stderr)
        return 1
    monitor.open()
    try:
        if args.once:
            try:
                print(json.dumps(monitor.run_once(), indent=2, ensure_ascii=True))
            except SensorReadError as exc:
                print(f"Sensor read failed: {exc}", file=sys.stderr)
                return 1
        else:
            monitor.run_forever()
    finally:
        monitor.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Motor fault monitor and sensor tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser(
        "predict",
        help="Replay one RMS triple into the rolling threshold model for debugging",
    )
    predict.add_argument("--i1", required=True, type=float)
    predict.add_argument("--i2", required=True, type=float)
    predict.add_argument("--i3", required=True, type=float)
    predict.add_argument(
        "--repeats",
        default=BUFFER_N,
        type=int,
        help="How many times to feed the same RMS triple into a fresh inferencer",
    )
    predict.set_defaults(func=_cmd_predict)

    test_sensors = subparsers.add_parser(
        "test-sensors",
        help="Read raw values from the configured serial sensors",
    )
    test_sensors.add_argument("--samples", default=3, type=int)
    test_sensors.add_argument(
        "--live",
        action="store_true",
        help="Stream raw serial lines as they arrive",
    )
    test_sensors.add_argument(
        "--poll-timeout",
        default=0.2,
        type=float,
        help="Per-port read timeout while streaming live serial data",
    )
    test_sensors.set_defaults(func=_cmd_test_sensors)

    run = subparsers.add_parser("run", help="Run the monitor")
    run.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    run.add_argument(
        "--interval",
        type=float,
        help="Override the sampling interval in seconds for this run",
    )
    run.add_argument("--verbose", action="store_true")
    run.set_defaults(func=_cmd_run)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
