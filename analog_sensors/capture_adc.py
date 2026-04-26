#!/usr/bin/env python3
"""
capture_adc.py
==============

Single-channel high-speed capture from the Waveshare High-Precision AD/DA
board (ADS1256) on a Raspberry Pi 4.

This script:
  - configures the ADS1256 for the requested DRATE / gain
  - selects one analog channel (AIN0..AIN7, single-ended vs AINCOM)
  - puts the chip in RDATAC (Read Data Continuous) mode
  - reads samples driven by the falling edge of /DRDY
  - applies a hall-effect-sensor scaling (V -> A) loaded from .env
  - prints live status and writes every sample to CSV

Why RDATAC:
  The "set MUX, send RDATA, read 3 bytes" round-trip is the path used by most
  example code, but the channel-switch settling and per-sample command overhead
  caps it well below 10 kHz from Python. RDATAC tells the chip "keep producing
  samples on the same MUX as fast as DRATE allows; just clock 24 bits out every
  time DRDY drops." This is the only practical way to sustain 10 kHz on a
  single channel from Python on a Pi 4.

Run:
    set -a; source .env; set +a
    sudo python3 capture_adc.py

Stop with Ctrl+C at any time. Partially written CSV is still valid.
"""

import csv
import os
import signal
import sys
import time
from dataclasses import dataclass

import RPi.GPIO as GPIO
import spidev


# ---------------------------------------------------------------------------
# ADS1256 register / command constants (datasheet pp.30-35)
# ---------------------------------------------------------------------------
REG_STATUS = 0x00
REG_MUX = 0x01
REG_ADCON = 0x02
REG_DRATE = 0x03

CMD_WAKEUP = 0x00
CMD_RDATA = 0x01
CMD_RDATAC = 0x03
CMD_SDATAC = 0x0F
CMD_RREG = 0x10
CMD_WREG = 0x50
CMD_SELFCAL = 0xF0
CMD_SYNC = 0xFC
CMD_RESET = 0xFE

# Mapping from a target SPS to the DRATE register value.
# (Datasheet table 13.) We pick the smallest DRATE >= target so the chip
# itself does the oversampling/filtering for us.
DRATE_TABLE = {
    30000: 0xF0,
    15000: 0xE0,
    7500: 0xD0,
    3750: 0xC0,
    2000: 0xB0,
    1000: 0xA1,
    500: 0x92,
    100: 0x82,
    60: 0x72,
    50: 0x63,
    30: 0x53,
    25: 0x43,
    15: 0x33,
    10: 0x23,
    5: 0x13,
}

GAIN_TABLE = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6}

# Waveshare board fixed pin mapping (BCM numbering). These pins are hard-wired
# on the HAT; they are not user-configurable.
RST_PIN = 18  # ADS1256 /RESET
CS_PIN = 22  # ADS1256 /CS  (driven manually, not by SPI peripheral)
DRDY_PIN = 17  # ADS1256 /DRDY (input, low = new sample ready)


# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
@dataclass
class Config:
    channel: int
    drate_sps: int
    gain: int
    spi_hz: int
    target_rate_hz: int
    hall_offset_v: float
    hall_sens_v_per_a: float
    capture_samples: int
    csv_path: str
    print_interval_s: float


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def load_config() -> Config:
    cfg = Config(
        channel=_env_int("ADC_CHANNEL", 0),
        drate_sps=_env_int("ADC_DRATE_SPS", 15000),
        gain=_env_int("ADC_GAIN", 1),
        spi_hz=_env_int("SPI_HZ", 1_920_000),
        target_rate_hz=_env_int("TARGET_SAMPLE_RATE_HZ", 10_000),
        hall_offset_v=_env_float("HALL_OFFSET_V", 2.5),
        hall_sens_v_per_a=_env_float("HALL_SENSITIVITY_V_PER_A", 0.185),
        capture_samples=_env_int("CAPTURE_SAMPLES", 100_000),
        csv_path=_env_str("CAPTURE_FILE_NAME", "capture_phase1.csv"),
        print_interval_s=_env_float("PRINT_INTERVAL_S", 1.0),
    )
    if cfg.channel not in range(8):
        raise ValueError(f"ADC_CHANNEL must be 0..7, got {cfg.channel}")
    if cfg.drate_sps not in DRATE_TABLE:
        raise ValueError(
            f"ADC_DRATE_SPS={cfg.drate_sps} not supported. "
            f"Pick one of: {sorted(DRATE_TABLE.keys(), reverse=True)}"
        )
    if cfg.gain not in GAIN_TABLE:
        raise ValueError(f"ADC_GAIN must be one of {list(GAIN_TABLE.keys())}")
    if cfg.drate_sps < cfg.target_rate_hz:
        raise ValueError(
            f"ADC_DRATE_SPS ({cfg.drate_sps}) must be >= "
            f"TARGET_SAMPLE_RATE_HZ ({cfg.target_rate_hz})"
        )
    return cfg


# ---------------------------------------------------------------------------
# ADS1256 driver (just enough for single-channel RDATAC)
# ---------------------------------------------------------------------------
class ADS1256:
    # VREF on the Waveshare board is 2.5 V. Full-scale input is +/-(2*VREF/GAIN).
    VREF = 2.5

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # +/- full-scale voltage at the chosen gain.
        self.fullscale_v = 2.0 * self.VREF / cfg.gain
        # Pre-compute the lsb -> volts factor. ADS1256 output is signed 24-bit.
        self.lsb_to_v = self.fullscale_v / (1 << 23)

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(RST_PIN, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(CS_PIN, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(DRDY_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)  # /dev/spidev0.0
        self.spi.max_speed_hz = cfg.spi_hz
        self.spi.mode = 0b01  # CPOL=0, CPHA=1 (datasheet figure 1)

    # ---- low-level helpers ----
    def _cs_low(self):
        GPIO.output(CS_PIN, GPIO.LOW)

    def _cs_high(self):
        GPIO.output(CS_PIN, GPIO.HIGH)

    def _wait_drdy(self, timeout_s: float = 0.5) -> bool:
        """Block until /DRDY goes low (new sample ready)."""
        deadline = time.monotonic() + timeout_s
        while GPIO.input(DRDY_PIN) == GPIO.HIGH:
            if time.monotonic() > deadline:
                return False
        return True

    def _write_reg(self, reg: int, value: int) -> None:
        self._cs_low()
        # WREG: 0x50 | reg, then (n-1)=0, then the value byte.
        self.spi.writebytes([CMD_WREG | reg, 0x00, value & 0xFF])
        self._cs_high()

    def _send_cmd(self, cmd: int) -> None:
        self._cs_low()
        self.spi.writebytes([cmd])
        self._cs_high()

    # ---- public API ----
    def reset(self) -> None:
        # Hardware reset pulse on /RST.
        GPIO.output(RST_PIN, GPIO.LOW)
        time.sleep(0.01)
        GPIO.output(RST_PIN, GPIO.HIGH)
        time.sleep(0.05)

    def configure_single_channel(self) -> None:
        cfg = self.cfg
        # Wait for first DRDY after reset.
        if not self._wait_drdy(timeout_s=1.0):
            raise RuntimeError(
                "ADS1256: DRDY never went low after reset. "
                "Check that the HAT is seated and SPI is on."
            )

        # MUX: positive input = AINx, negative input = AINCOM (single-ended).
        mux_val = ((cfg.channel & 0x07) << 4) | 0x08
        # ADCON: clock out off, sensor detect off, gain bits in [2:0].
        adcon_val = GAIN_TABLE[cfg.gain]
        # STATUS: enable buffer (bit 1 = BUFEN). MSB-first, auto-cal off.
        status_val = 0x06

        self._write_reg(REG_STATUS, status_val)
        self._write_reg(REG_MUX, mux_val)
        self._write_reg(REG_ADCON, adcon_val)
        self._write_reg(REG_DRATE, DRATE_TABLE[cfg.drate_sps])

        # Self-calibrate at the new settings.
        self._send_cmd(CMD_SELFCAL)
        if not self._wait_drdy(timeout_s=1.0):
            raise RuntimeError("ADS1256: self-cal did not complete.")

    def start_continuous(self) -> None:
        """Enter RDATAC. After this, every DRDY -> 3 SCLK bytes = one sample."""
        self._cs_low()  # CS stays LOW for the whole stream
        self.spi.writebytes([CMD_RDATAC])
        # t6 (datasheet): wait >=50*tCLKIN ~ 6.5 us before first read
        time.sleep(50e-6)

    def stop_continuous(self) -> None:
        self.spi.writebytes([CMD_SDATAC])
        self._cs_high()

    def read_one_continuous(self) -> int:
        """One sample in RDATAC mode. Caller must wait for DRDY first."""
        # xfer2 with 3 dummy bytes -> 24-bit signed result.
        b = self.spi.xfer2([0x00, 0x00, 0x00])
        raw = (b[0] << 16) | (b[1] << 8) | b[2]
        if raw & 0x800000:  # sign-extend 24 -> 32 bits
            raw -= 1 << 24
        return raw

    def to_volts(self, raw: int) -> float:
        return raw * self.lsb_to_v

    def close(self) -> None:
        try:
            self.stop_continuous()
        except Exception:
            pass
        try:
            self.spi.close()
        except Exception:
            pass
        GPIO.cleanup([RST_PIN, CS_PIN, DRDY_PIN])


# ---------------------------------------------------------------------------
# Capture loop
# ---------------------------------------------------------------------------
_stop = False


def _on_sigint(signum, frame):
    del signum, frame
    global _stop
    _stop = True


def run(cfg: Config) -> None:
    print("=" * 60)
    print("IOP ADC capture")
    print("=" * 60)
    print(f"  channel        : AIN{cfg.channel} (single-ended vs AINCOM)")
    print(f"  ADC DRATE      : {cfg.drate_sps} SPS")
    print(f"  gain           : {cfg.gain}x  -> +/- {2 * ADS1256.VREF / cfg.gain:.3f} V")
    print(f"  SPI clock      : {cfg.spi_hz / 1e6:.2f} MHz")
    print(f"  target app rate: {cfg.target_rate_hz} Hz")
    print(f"  hall offset    : {cfg.hall_offset_v} V")
    print(f"  hall sens      : {cfg.hall_sens_v_per_a} V/A")
    print(f"  capture        : {cfg.capture_samples or 'unlimited'} samples")
    print(f"  csv            : {cfg.csv_path}")
    print("=" * 60)

    adc = ADS1256(cfg)
    adc.reset()
    adc.configure_single_channel()
    adc.start_continuous()

    # Pre-bind hot-path locals for speed.
    drdy_input = GPIO.input
    drdy_pin = DRDY_PIN
    drdy_low = GPIO.LOW
    monotonic = time.monotonic
    read_one = adc.read_one_continuous
    lsb_to_v = adc.lsb_to_v
    offset_v = cfg.hall_offset_v
    sens = cfg.hall_sens_v_per_a
    target_n = cfg.capture_samples if cfg.capture_samples > 0 else None
    print_dt = cfg.print_interval_s

    signal.signal(signal.SIGINT, _on_sigint)

    csv_f = open(cfg.csv_path, "w", newline="", buffering=1024 * 1024)
    writer = csv.writer(csv_f)
    writer.writerow(["sample_index", "t_seconds", "raw", "voltage_v", "current_a"])

    t_start = monotonic()
    t_next_print = t_start + print_dt
    n = 0
    n_at_last_print = 0
    drops = 0  # times we saw DRDY still high when we expected a sample

    try:
        while not _stop and (target_n is None or n < target_n):
            # Spin-wait for /DRDY. At 10 kHz that's 100 us between samples,
            # which is well inside what a Pi 4 can do.
            t_wait_start = monotonic()
            while drdy_input(drdy_pin) != drdy_low:
                if monotonic() - t_wait_start > 0.01:
                    drops += 1
                    break
            else:
                pass

            raw = read_one()
            t_rel = monotonic() - t_start
            volts = raw * lsb_to_v
            current = (volts - offset_v) / sens
            writer.writerow(
                (n, f"{t_rel:.6f}", raw, f"{volts:.6f}", f"{current:.6f}")
            )
            n += 1

            # Periodic console status (do not print every sample - too slow).
            now = monotonic()
            if now >= t_next_print:
                window = now - (t_next_print - print_dt)
                rate = (n - n_at_last_print) / window if window > 0 else 0
                print(
                    f"  n={n:>8}  t={t_rel:7.3f}s  "
                    f"V={volts:+.4f}  I={current:+.4f} A  "
                    f"rate~{rate:7.1f} Hz  drops={drops}"
                )
                n_at_last_print = n
                t_next_print = now + print_dt
    finally:
        elapsed = monotonic() - t_start
        adc.close()
        csv_f.close()
        avg_rate = n / elapsed if elapsed > 0 else 0
        target = cfg.target_rate_hz
        ratio = avg_rate / target if target else 0
        print()
        print("-" * 60)
        print(f"  samples written  : {n}")
        print(f"  elapsed          : {elapsed:.3f} s")
        print(
            f"  average rate     : {avg_rate:.1f} Hz "
            f"({ratio * 100:.1f}% of {target} Hz target)"
        )
        print(f"  drdy timeouts    : {drops}")
        print(f"  csv file         : {cfg.csv_path}")
        print("-" * 60)
        if avg_rate < 0.95 * target:
            print("  WARNING: average rate is below 95% of the target.")
            print("           Try lowering TARGET_SAMPLE_RATE_HZ or run as root.")


def main() -> int:
    try:
        cfg = load_config()
    except Exception as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    try:
        run(cfg)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
