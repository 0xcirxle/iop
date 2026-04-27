#!/usr/bin/env python3
"""
capture_adc.py  (pigpio version)
================================

Single-channel high-speed capture from the Waveshare High-Precision AD/DA
board (ADS1256) on a Raspberry Pi 4, using pigpio.

This is a drop-in replacement for the spidev/RPi.GPIO version, used when
the kernel's SPI pinctrl is broken (the "not valid maps for state default"
case). pigpio configures the pin alternate functions directly via the GPIO
peripheral registers, bypassing the kernel framework entirely.

Trade-off vs spidev:
  - spidev path can sustain ~23 kHz reads from Python on a Pi 4
  - pigpio path tops out around ~5-8 kHz because each SPI transfer goes
    over a Unix socket to the pigpio daemon
  - For 10 kHz target on a single channel, this is borderline. We default
    ADC_DRATE_SPS=7500 in this version.

Prereq:
  sudo apt install -y pigpio python3-pigpio
  sudo systemctl enable --now pigpiod.service

Run:
  set -a; source .env; set +a
  python3 capture_adc.py        # no sudo needed
"""

import csv
import os
import signal
import sys
import time
from dataclasses import dataclass

import pigpio


# ---------------------------------------------------------------------------
# Optional: load .env automatically
# ---------------------------------------------------------------------------
def _load_dotenv():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()


# ---------------------------------------------------------------------------
# ADS1256 constants
# ---------------------------------------------------------------------------
REG_STATUS = 0x00
REG_MUX    = 0x01
REG_ADCON  = 0x02
REG_DRATE  = 0x03

CMD_WAKEUP  = 0x00
CMD_RDATA   = 0x01
CMD_RDATAC  = 0x03
CMD_SDATAC  = 0x0F
CMD_RREG    = 0x10
CMD_WREG    = 0x50
CMD_SELFCAL = 0xF0
CMD_SYNC    = 0xFC
CMD_RESET   = 0xFE

DRATE_TABLE = {
    30000: 0xF0, 15000: 0xE0, 7500: 0xD0, 3750: 0xC0,
    2000: 0xB0, 1000: 0xA1, 500: 0x92, 100: 0x82,
    60: 0x72, 50: 0x63, 30: 0x53, 25: 0x43,
    15: 0x33, 10: 0x23, 5: 0x13,
}
GAIN_TABLE = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6}

# Waveshare HAT fixed pin mapping (BCM)
RST_PIN  = 18
CS_PIN   = 22
DRDY_PIN = 17

# pigpio SPI flags. Bits [1:0] = mode (CPOL, CPHA). Mode 1 means CPOL=0, CPHA=1.
# We DON'T set the "use auxiliary SPI" bit (bit 8) because we want SPI0.
# We DO set bit 5 ("aux SPI 3-wire") to 0 (default).
# spi_open(channel, baud, flags). channel 0 = CE0 (GPIO8). The HAT ignores
# this CS line and uses GPIO22 instead, which we drive manually.
PIGPIO_SPI_FLAGS = 0b01   # mode 1


# ---------------------------------------------------------------------------
# Config
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


def _i(name, default): return int(os.environ.get(name, str(default)))
def _f(name, default): return float(os.environ.get(name, str(default)))
def _s(name, default): return os.environ.get(name, default)


def load_config() -> Config:
    cfg = Config(
        channel           = _i("ADC_CHANNEL", 0),
        drate_sps         = _i("ADC_DRATE_SPS", 7500),
        gain              = _i("ADC_GAIN", 1),
        spi_hz            = _i("SPI_HZ", 1_000_000),
        target_rate_hz    = _i("TARGET_SAMPLE_RATE_HZ", 5_000),
        hall_offset_v     = _f("HALL_OFFSET_V", 0.1037),
        hall_sens_v_per_a = _f("HALL_SENSITIVITY_V_PER_A", 0.3948),
        capture_samples   = _i("CAPTURE_SAMPLES", 30_000),
        csv_path          = _s("CAPTURE_FILE_NAME", "capture_phase1.csv"),
        print_interval_s  = _f("PRINT_INTERVAL_S", 1.0),
    )
    if cfg.channel not in range(8):
        raise ValueError(f"ADC_CHANNEL must be 0..7, got {cfg.channel}")
    if cfg.drate_sps not in DRATE_TABLE:
        raise ValueError(f"ADC_DRATE_SPS={cfg.drate_sps} not supported. "
                         f"Pick one of: {sorted(DRATE_TABLE.keys(), reverse=True)}")
    if cfg.gain not in GAIN_TABLE:
        raise ValueError(f"ADC_GAIN must be one of {list(GAIN_TABLE.keys())}")
    return cfg


# ---------------------------------------------------------------------------
# ADS1256 driver via pigpio
# ---------------------------------------------------------------------------
class ADS1256:
    VREF = 2.5

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.fullscale_v = 2.0 * self.VREF / cfg.gain
        self.lsb_to_v = self.fullscale_v / (1 << 23)

        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError(
                "Could not connect to pigpiod. Run: sudo systemctl start pigpiod"
            )

        # Configure GPIO pins.
        self.pi.set_mode(RST_PIN, pigpio.OUTPUT)
        self.pi.set_mode(CS_PIN,  pigpio.OUTPUT)
        self.pi.set_mode(DRDY_PIN, pigpio.INPUT)
        self.pi.set_pull_up_down(DRDY_PIN, pigpio.PUD_UP)
        self.pi.write(RST_PIN, 1)
        self.pi.write(CS_PIN,  1)

        # Open SPI0 channel 0 (we ignore its hardware CS, drive CS_PIN manually).
        self.spi_handle = self.pi.spi_open(0, cfg.spi_hz, PIGPIO_SPI_FLAGS)

    def _cs_low(self):  self.pi.write(CS_PIN, 0)
    def _cs_high(self): self.pi.write(CS_PIN, 1)

    def _wait_drdy(self, timeout_s=1.0):
        deadline = time.monotonic() + timeout_s
        while self.pi.read(DRDY_PIN) == 1:
            if time.monotonic() > deadline:
                return False
        return True

    def _write_reg(self, reg, value):
        self._cs_low()
        self.pi.spi_write(self.spi_handle,
                          bytes([CMD_WREG | reg, 0x00, value & 0xFF]))
        self._cs_high()

    def _send_cmd(self, cmd):
        self._cs_low()
        self.pi.spi_write(self.spi_handle, bytes([cmd]))
        self._cs_high()

    def reset(self):
        self.pi.write(RST_PIN, 0); time.sleep(0.01)
        self.pi.write(RST_PIN, 1); time.sleep(0.05)

    def read_status_register(self):
        """Read STATUS register. Returns the byte. Used to verify chip is alive."""
        self._cs_low()
        self.pi.spi_write(self.spi_handle, bytes([CMD_RREG | REG_STATUS, 0x00]))
        time.sleep(0.001)
        count, data = self.pi.spi_read(self.spi_handle, 1)
        self._cs_high()
        return data[0] if count == 1 else 0

    def configure_single_channel(self):
        cfg = self.cfg
        if not self._wait_drdy(1.0):
            raise RuntimeError("ADS1256: DRDY never went low after reset.")

        mux_val   = ((cfg.channel & 0x07) << 4) | 0x08
        adcon_val = GAIN_TABLE[cfg.gain]
        status_val = 0x06   # BUFEN=1, MSB-first, auto-cal off

        self._write_reg(REG_STATUS, status_val)
        self._write_reg(REG_MUX,    mux_val)
        self._write_reg(REG_ADCON,  adcon_val)
        self._write_reg(REG_DRATE,  DRATE_TABLE[cfg.drate_sps])

        self._send_cmd(CMD_SELFCAL)
        if not self._wait_drdy(1.0):
            raise RuntimeError("ADS1256: self-cal did not complete.")

    def start_continuous(self):
        self._cs_low()
        self.pi.spi_write(self.spi_handle, bytes([CMD_RDATAC]))
        time.sleep(50e-6)

    def stop_continuous(self):
        try:
            self.pi.spi_write(self.spi_handle, bytes([CMD_SDATAC]))
        except Exception:
            pass
        self._cs_high()

    def read_one_continuous(self):
        count, b = self.pi.spi_read(self.spi_handle, 3)
        if count != 3:
            return 0
        raw = (b[0] << 16) | (b[1] << 8) | b[2]
        if raw & 0x800000:
            raw -= 1 << 24
        return raw

    def close(self):
        try: self.stop_continuous()
        except Exception: pass
        try: self.pi.spi_close(self.spi_handle)
        except Exception: pass
        try: self.pi.stop()
        except Exception: pass


# ---------------------------------------------------------------------------
# Capture loop
# ---------------------------------------------------------------------------
_stop = False
def _on_sigint(signum, frame):
    global _stop
    _stop = True


def run(cfg: Config):
    print("=" * 60)
    print("IOP ADC capture (pigpio)")
    print("=" * 60)
    print(f"  channel        : AIN{cfg.channel} (single-ended vs AINCOM)")
    print(f"  ADC DRATE      : {cfg.drate_sps} SPS")
    print(f"  gain           : {cfg.gain}x  -> +/- {2*ADS1256.VREF/cfg.gain:.3f} V")
    print(f"  SPI clock      : {cfg.spi_hz/1e6:.2f} MHz")
    print(f"  target app rate: {cfg.target_rate_hz} Hz")
    print(f"  hall offset    : {cfg.hall_offset_v} V")
    print(f"  hall sens      : {cfg.hall_sens_v_per_a} V/A")
    print(f"  capture        : {cfg.capture_samples or 'unlimited'} samples")
    print(f"  csv            : {cfg.csv_path}")
    print("=" * 60)

    adc = ADS1256(cfg)
    adc.reset()

    # Sanity check: read STATUS register before doing anything else.
    status = adc.read_status_register()
    print(f"  STATUS register: 0x{status:02X}  "
          f"(chip ID nibble: 0x{status >> 4:X}, expect 0x3)")
    if (status >> 4) != 0x3:
        adc.close()
        raise RuntimeError(
            f"ADS1256 not responding correctly (STATUS=0x{status:02X}). "
            f"Expected upper nibble 0x3. Check HAT seating and AINCOM jumper."
        )
    print("  chip detected OK")
    print("=" * 60)

    adc.configure_single_channel()
    adc.start_continuous()

    monotonic = time.monotonic
    read_one  = adc.read_one_continuous
    wait_drdy = adc._wait_drdy
    lsb_to_v  = adc.lsb_to_v
    offset_v  = cfg.hall_offset_v
    sens      = cfg.hall_sens_v_per_a
    target_n  = cfg.capture_samples if cfg.capture_samples > 0 else None
    print_dt  = cfg.print_interval_s

    signal.signal(signal.SIGINT, _on_sigint)

    csv_f = open(cfg.csv_path, "w", newline="", buffering=1024 * 1024)
    writer = csv.writer(csv_f)
    writer.writerow(["sample_index", "t_seconds", "raw", "voltage_v", "current_a"])

    t_start = monotonic()
    t_next_print = t_start + print_dt
    n = 0
    n_at_last_print = 0
    drops = 0

    try:
        while not _stop and (target_n is None or n < target_n):
            if not wait_drdy(timeout_s=0.05):
                drops += 1
                continue
            raw = read_one()
            t_rel = monotonic() - t_start
            volts = raw * lsb_to_v
            current = (volts - offset_v) / sens
            writer.writerow((n, f"{t_rel:.6f}", raw, f"{volts:.6f}",
                             f"{current:.6f}"))
            n += 1

            now = monotonic()
            if now >= t_next_print:
                window = now - (t_next_print - print_dt)
                rate = (n - n_at_last_print) / window if window > 0 else 0
                print(f"  n={n:>8}  t={t_rel:7.3f}s  "
                      f"V={volts:+.4f}  I={current:+.4f} A  "
                      f"rate~{rate:7.1f} Hz  drops={drops}")
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
        print(f"  average rate     : {avg_rate:.1f} Hz "
              f"({ratio*100:.1f}% of {target} Hz target)")
        print(f"  drdy timeouts    : {drops}")
        print(f"  csv file         : {cfg.csv_path}")
        print("-" * 60)


def main():
    try:
        cfg = load_config()
    except Exception as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    try:
        run(cfg)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
