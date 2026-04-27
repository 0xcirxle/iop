#!/usr/bin/env python3
"""
capture_adc_v2.py
=================

Single-channel high-speed capture from the Waveshare High-Precision AD/DA
board (ADS1256), using the same SPI/register-write pattern as the working
Waveshare demo.

Why this version exists:
  v1 (capture_adc.py) used RDATAC continuous mode for max speed (~17 kHz on
  a Pi 4), and on most setups it works. On this particular hardware/kernel
  combination, RDATAC returned all-zero samples even though the chip was
  alive (verified via the demo working correctly on the same Pi).

  This version mirrors the demo's per-sample pattern:
    - set MUX to target channel
    - issue SYNC + WAKEUP (start fresh conversion, release it)
    - wait for DRDY
    - send RDATA, read 3 bytes
  This is slower (~2-3 kHz on a single channel from Python) but is exactly
  what we know works on your hardware.

Run:
    set -a; source .env; set +a
    sudo -E env "PATH=$PATH" python3 capture_adc_v2.py
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
# ADS1256 constants - same as the demo
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
    15: 0x33, 10: 0x20, 5: 0x13,
}
GAIN_TABLE = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6}

# Waveshare HAT pins (BCM)
RST_PIN  = 18
CS_PIN   = 22
DRDY_PIN = 17


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
        drate_sps         = _i("ADC_DRATE_SPS", 30000),
        gain              = _i("ADC_GAIN", 1),
        spi_hz            = _i("SPI_HZ", 1_000_000),
        target_rate_hz    = _i("TARGET_SAMPLE_RATE_HZ", 2_000),
        hall_offset_v     = _f("HALL_OFFSET_V", 0.0046),
        hall_sens_v_per_a = _f("HALL_SENSITIVITY_V_PER_A", 1.0),
        capture_samples   = _i("CAPTURE_SAMPLES", 10_000),
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
# ADS1256 driver -- exactly mirrors the demo's pattern
# ---------------------------------------------------------------------------
class ADS1256:
    VREF = 2.5

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.fullscale_v = 2.0 * self.VREF / cfg.gain
        # 23-bit signed full-scale -> volts conversion
        self.lsb_to_v = self.fullscale_v / 0x7FFFFF

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(RST_PIN,  GPIO.OUT)
        GPIO.setup(CS_PIN,   GPIO.OUT)
        GPIO.setup(DRDY_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        self.spi = spidev.SpiDev(0, 0)
        self.spi.max_speed_hz = cfg.spi_hz
        self.spi.mode = 0b01

    # --- low level helpers, mirroring the demo ---
    def _cs_low(self):  GPIO.output(CS_PIN, GPIO.LOW)
    def _cs_high(self): GPIO.output(CS_PIN, GPIO.HIGH)

    def _wait_drdy(self):
        # demo's pattern: poll, with a hard upper bound
        for _ in range(400000):
            if GPIO.input(DRDY_PIN) == 0:
                return True
        return False

    def _write_cmd(self, cmd):
        self._cs_low()
        self.spi.writebytes([cmd])
        self._cs_high()

    def _write_reg(self, reg, data):
        self._cs_low()
        self.spi.writebytes([CMD_WREG | reg, 0x00, data])
        self._cs_high()

    def _read_reg(self, reg):
        self._cs_low()
        self.spi.writebytes([CMD_RREG | reg, 0x00])
        result = self.spi.readbytes(1)
        self._cs_high()
        return result[0]

    # --- public methods, mirroring the demo ---
    def reset(self):
        # exact demo sequence
        GPIO.output(RST_PIN, GPIO.HIGH)
        time.sleep(0.2)
        GPIO.output(RST_PIN, GPIO.LOW)
        time.sleep(0.2)
        GPIO.output(RST_PIN, GPIO.HIGH)

    def read_chip_id(self):
        # demo waits for DRDY then reads STATUS register, returns upper nibble
        self._wait_drdy()
        status = self._read_reg(REG_STATUS)
        return status >> 4

    def config_adc(self):
        # EXACT demo register sequence (ConfigADC function)
        self._wait_drdy()
        gain  = GAIN_TABLE[self.cfg.gain]
        drate = DRATE_TABLE[self.cfg.drate_sps]

        buf = [0, 0, 0, 0, 0, 0, 0, 0]
        # demo's STATUS value: BUFEN=1, ACAL=0, MSB-first
        buf[0] = (0 << 3) | (1 << 2) | (0 << 1)   # = 0x04
        buf[1] = 0x08                              # MUX: AIN0 vs AINCOM (will be reoverridden per read)
        buf[2] = (0 << 5) | (0 << 3) | (gain << 0)
        buf[3] = drate

        # Burst write: WREG starting at reg 0, count = 4 registers
        self._cs_low()
        self.spi.writebytes([CMD_WREG | 0, 0x03])  # write 4 regs (count = n-1 = 3)
        self.spi.writebytes(buf)
        self._cs_high()
        time.sleep(0.001)

    def set_channel(self, channel):
        # Single-ended: positive=AINx, negative=AINCOM (=0x08)
        # demo: WriteReg(REG_MUX, (Channel<<4) | (1<<3))
        self._write_reg(REG_MUX, (channel << 4) | 0x08)

    def read_one(self, channel):
        """Mirror the demo's GetChannalValue exactly: SetChannel, SYNC, WAKEUP, RDATA."""
        self.set_channel(channel)
        self._write_cmd(CMD_SYNC)
        self._write_cmd(CMD_WAKEUP)
        # Now wait for DRDY then read 3 bytes
        if not self._wait_drdy():
            return None
        self._cs_low()
        self.spi.writebytes([CMD_RDATA])
        buf = self.spi.readbytes(3)
        self._cs_high()
        raw = ((buf[0] << 16) & 0xFF0000) | ((buf[1] << 8) & 0xFF00) | (buf[2] & 0xFF)
        if raw & 0x800000:
            # demo's sign extension is buggy ("read &= 0xF000000") but the math
            # below is the correct two's complement extension for 24->32 bits
            raw -= 1 << 24
        return raw

    def to_volts(self, raw):
        return raw * self.lsb_to_v

    def close(self):
        try: self.spi.close()
        except Exception: pass
        try: GPIO.cleanup([RST_PIN, CS_PIN, DRDY_PIN])
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
    print("IOP ADC capture (demo-compatible)")
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

    chip_id = adc.read_chip_id()
    if chip_id == 3:
        print("  chip ID read OK (=3)")
    else:
        adc.close()
        raise RuntimeError(f"Chip ID read failed. Expected 3, got {chip_id}. "
                           f"Verify VCC/VREF jumpers and AINCOM jumper.")

    adc.config_adc()
    print("  ADC configured")
    print("=" * 60)

    monotonic = time.monotonic
    read_one  = adc.read_one
    lsb_to_v  = adc.lsb_to_v
    offset_v  = cfg.hall_offset_v
    sens      = cfg.hall_sens_v_per_a
    target_n  = cfg.capture_samples if cfg.capture_samples > 0 else None
    print_dt  = cfg.print_interval_s
    channel   = cfg.channel

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
            raw = read_one(channel)
            if raw is None:
                drops += 1
                continue
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
