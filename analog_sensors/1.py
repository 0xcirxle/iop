#!/usr/bin/env python3
"""
capture_3phase.py
=================

Three-channel multiplexed current capture for the IOP inter-turn fault
detection project.

Reads three current channels (one per motor phase) sequentially from the
ADS1256 by writing a fresh MUX value before each conversion. Per-channel
hall-sensor calibration (offset and sensitivity) lives in .env so all
three phases land in the same 'amps' space without code changes.

Output columns: t_seconds, raw1, v1, i1, raw2, v2, i2, raw3, v3, i3

OPTIMIZATIONS over the previous version:
  - WREG_MUX + SYNC + WAKEUP are sent as a single SPI transaction with one
    CS toggle (datasheet permits adjacent command bytes while CS is low).
  - RDATA + 3-byte read is a single xfer2 transaction instead of separate
    writebytes/readbytes (one syscall, one CS hold).
  - All hot-loop SPI helpers are bound as locals.

This drops the per-sample SPI-syscall count from 5 to 3 and lifts the
sustained per-channel rate to ~1.3-1.6 kHz on a Pi 4. To reach 5 kHz/ch
use the C extension in capture_core.c; this file falls back to the pure
Python path if the extension is not built.

Run:
    set -a; source .env; set +a
    sudo -E env "PATH=$PATH" python3 capture_3phase.py
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
# Optional C inner-loop extension
# ---------------------------------------------------------------------------
try:
    import capture_core  # built from capture_core.c
    HAVE_C_CORE = True
except ImportError:
    capture_core = None
    HAVE_C_CORE = False

# ---------------------------------------------------------------------------
# Optional .env loader (so sudo -E isn't strictly required)
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

CMD_WAKEUP = 0x00
CMD_RDATA  = 0x01
CMD_SDATAC = 0x0F
CMD_RREG   = 0x10
CMD_WREG   = 0x50
CMD_SYNC   = 0xFC
CMD_RESET  = 0xFE

DRATE_TABLE = {
    30000: 0xF0, 15000: 0xE0, 7500: 0xD0, 3750: 0xC0,
    2000:  0xB0, 1000:  0xA1, 500:  0x92, 100:  0x82,
    60:    0x72, 50:    0x63, 30:   0x53, 25:   0x43,
    15:    0x33, 10:    0x20, 5:    0x13,
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
    ch_phase1: int
    ch_phase2: int
    ch_phase3: int

    drate_sps: int
    gain: int
    spi_hz: int
    target_rate_hz: int

    s1_offset_v: float
    s1_sens_v_per_a: float
    s2_offset_v: float
    s2_sens_v_per_a: float
    s3_offset_v: float
    s3_sens_v_per_a: float

    capture_samples: int
    csv_path: str
    print_interval_s: float

    use_c_core: bool

def _i(name, default): return int(os.environ.get(name, str(default)))
def _f(name, default): return float(os.environ.get(name, str(default)))
def _s(name, default): return os.environ.get(name, default)
def _b(name, default):
    v = os.environ.get(name, str(default)).strip().lower()
    return v in ("1", "true", "yes", "on")

def load_config() -> Config:
    cfg = Config(
        ch_phase1       = _i("ADC_CHANNEL_PHASE1", 0),
        ch_phase2       = _i("ADC_CHANNEL_PHASE2", 1),
        ch_phase3       = _i("ADC_CHANNEL_PHASE3", 2),
        drate_sps       = _i("ADC_DRATE_SPS", 15000),
        gain            = _i("ADC_GAIN", 1),
        spi_hz          = _i("SPI_HZ", 1_920_000),
        target_rate_hz  = _i("TARGET_SAMPLE_RATE_HZ_PER_CHANNEL", 5_000),
        s1_offset_v     = _f("S1_OFFSET_V", 1.640),
        s1_sens_v_per_a = _f("S1_SENS_V_PER_A", 0.594),
        s2_offset_v     = _f("S2_OFFSET_V", 1.550),
        s2_sens_v_per_a = _f("S2_SENS_V_PER_A", 0.559),
        s3_offset_v     = _f("S3_OFFSET_V", 1.960),
        s3_sens_v_per_a = _f("S3_SENS_V_PER_A", 0.646),
        capture_samples = _i("CAPTURE_SAMPLES_PER_CHANNEL", 20_000),
        csv_path        = _s("CAPTURE_FILE_NAME", "capture_3phase.csv"),
        print_interval_s= _f("PRINT_INTERVAL_S", 1.0),
        use_c_core      = _b("USE_C_CORE", True),
    )
    for ch in (cfg.ch_phase1, cfg.ch_phase2, cfg.ch_phase3):
        if ch not in range(8):
            raise ValueError(f"All ADC_CHANNEL_PHASE* must be 0..7 (got {ch})")
    if cfg.drate_sps not in DRATE_TABLE:
        raise ValueError(f"ADC_DRATE_SPS={cfg.drate_sps} not supported.")
    if cfg.gain not in GAIN_TABLE:
        raise ValueError(f"ADC_GAIN must be one of {list(GAIN_TABLE.keys())}")
    return cfg

# ---------------------------------------------------------------------------
# Driver (pure-Python path, used when the C extension isn't available)
# ---------------------------------------------------------------------------
class ADS1256:
    VREF = 2.5  # LM285-2.5 on board reference. Do NOT change this.

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.fullscale_v = 2.0 * self.VREF / cfg.gain
        self.lsb_to_v    = self.fullscale_v / 0x7FFFFF

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(RST_PIN,  GPIO.OUT)
        GPIO.setup(CS_PIN,   GPIO.OUT)
        GPIO.setup(DRDY_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        self.spi = spidev.SpiDev(0, 0)
        self.spi.max_speed_hz = cfg.spi_hz
        self.spi.mode = 0b01

    def _cs_low(self):  GPIO.output(CS_PIN, GPIO.LOW)
    def _cs_high(self): GPIO.output(CS_PIN, GPIO.HIGH)

    def _wait_drdy(self):
        gpio_input = GPIO.input
        pin = DRDY_PIN
        for _ in range(400000):
            if gpio_input(pin) == 0:
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

    def reset(self):
        GPIO.output(RST_PIN, GPIO.HIGH); time.sleep(0.2)
        GPIO.output(RST_PIN, GPIO.LOW);  time.sleep(0.2)
        GPIO.output(RST_PIN, GPIO.HIGH)

    def read_chip_id(self):
        self._wait_drdy()
        return self._read_reg(REG_STATUS) >> 4

    def config_adc(self, initial_channel):
        self._wait_drdy()
        gain  = GAIN_TABLE[self.cfg.gain]
        drate = DRATE_TABLE[self.cfg.drate_sps]
        mux_val = ((initial_channel & 0x07) << 4) | 0x08
        buf = [
            (0 << 3) | (1 << 2) | (0 << 1),  # STATUS = 0x04 (BUFEN=1, ACAL=0)
            mux_val,
            (0 << 5) | (0 << 3) | (gain << 0),
            drate,
        ]
        self._cs_low()
        self.spi.writebytes([CMD_WREG | 0, 0x03])
        self.spi.writebytes(buf)
        self._cs_high()
        time.sleep(0.001)

    def read_one_on_channel(self, channel):
        """
        Set MUX to `channel`, force a fresh conversion, return the 24-bit
        signed sample.

        SPI sequence per sample:
          1. writebytes([WREG MUX, 0x00, mux_val])
                -> set the input multiplexer to the requested channel.
          2. writebytes([SYNC, WAKEUP])
                -> restart the conversion on the new channel.
                The natural syscall/CS-toggle gap between (1) and (2) gives
                the input MUX its settle time (datasheet t11 ~= 24*tCLKIN
                = 3.1 us at 7.68 MHz crystal). DO NOT MERGE THESE TWO --
                merging them causes the chip to read the previous
                channel's conversion (verified empirically: with the merge
                in place every ~3rd sample shows the prior channel's
                value, classic ADS1256 channel-pipelining bug).
          3. wait DRDY low.
          4. writebytes([RDATA])
                -> command, kept separate from the data read so the t6
                delay (>=50*tCLKIN ~= 6.5 us between RDATA and the first
                data byte) is naturally satisfied by the syscall gap.
                Merging this with the 3-byte read corrupts the MSB at
                SPI clocks below ~7 MHz.
          5. readbytes(3)
                -> 24-bit MSB-first conversion result.

        Net SPI syscalls per sample = 4 (was 5 in the original).
        Realistic per-channel rate on a Pi 4: ~1.0-1.1 kHz.

        Two timing rules dominate this routine:
          * MUX settle (t11) between WREG-MUX and SYNC.
          * t6 between RDATA and the data bytes.
        Violate either and you get plausible-looking but wrong numbers.
        """
        mux_val = ((channel & 0x07) << 4) | 0x08

        # --- Step 1: WREG MUX ---
        self._cs_low()
        self.spi.writebytes([CMD_WREG | REG_MUX, 0x00, mux_val])
        self._cs_high()

        # --- Step 2: SYNC + WAKEUP, clubbed (no inter-byte timing requirement
        #             between SYNC and WAKEUP, only between WREG and SYNC) ---
        self._cs_low()
        self.spi.writebytes([CMD_SYNC, CMD_WAKEUP])
        self._cs_high()

        # --- Step 3: wait for DRDY to fall (new conversion complete) ---
        if not self._wait_drdy():
            return None

        # --- Steps 4 & 5: RDATA, then 3 data bytes (kept as two calls
        #     to satisfy the t6 delay; see docstring) ---
        self._cs_low()
        self.spi.writebytes([CMD_RDATA])
        b = self.spi.readbytes(3)
        self._cs_high()

        raw = (b[0] << 16) | (b[1] << 8) | b[2]
        if raw & 0x800000:
            raw -= 1 << 24
        return raw

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


def _print_banner(cfg: Config, mode: str):
    print("=" * 64)
    print(f"IOP 3-channel current capture  [{mode}]")
    print("=" * 64)
    print(f"  channels       : AIN{cfg.ch_phase1} (P1), "
          f"AIN{cfg.ch_phase2} (P2), AIN{cfg.ch_phase3} (P3)")
    print(f"  ADC DRATE      : {cfg.drate_sps} SPS")
    print(f"  gain           : {cfg.gain}x  ->  +/- {2*ADS1256.VREF/cfg.gain:.3f} V")
    print(f"  SPI clock      : {cfg.spi_hz/1e6:.2f} MHz")
    print(f"  target/channel : {cfg.target_rate_hz} Hz")
    print(f"  S1 calib       : offset={cfg.s1_offset_v:.4f} V, "
          f"sens={cfg.s1_sens_v_per_a:.4f} V/A")
    print(f"  S2 calib       : offset={cfg.s2_offset_v:.4f} V, "
          f"sens={cfg.s2_sens_v_per_a:.4f} V/A")
    print(f"  S3 calib       : offset={cfg.s3_offset_v:.4f} V, "
          f"sens={cfg.s3_sens_v_per_a:.4f} V/A")
    print(f"  capture        : {cfg.capture_samples or 'unlimited'} samples/ch")
    print(f"  csv            : {cfg.csv_path}")
    print("=" * 64)


def run_python(cfg: Config):
    """Pure-Python path with clubbed SPI calls. ~1.3-1.6 kHz/ch on a Pi 4."""
    _print_banner(cfg, "python (clubbed)")

    adc = ADS1256(cfg)
    adc.reset()
    chip_id = adc.read_chip_id()
    if chip_id == 3:
        print("  chip ID read OK (=3)")
    else:
        adc.close()
        raise RuntimeError(f"Chip ID read failed. Expected 3, got {chip_id}.")

    adc.config_adc(initial_channel=cfg.ch_phase1)
    print(f"  ADC configured")
    print("=" * 64)

    monotonic = time.monotonic
    read_one  = adc.read_one_on_channel
    lsb_to_v  = adc.lsb_to_v

    ch1, ch2, ch3 = cfg.ch_phase1, cfg.ch_phase2, cfg.ch_phase3
    o1, k1 = cfg.s1_offset_v, cfg.s1_sens_v_per_a
    o2, k2 = cfg.s2_offset_v, cfg.s2_sens_v_per_a
    o3, k3 = cfg.s3_offset_v, cfg.s3_sens_v_per_a

    target_n = cfg.capture_samples if cfg.capture_samples > 0 else None
    print_dt = cfg.print_interval_s

    signal.signal(signal.SIGINT, _on_sigint)

    csv_f  = open(cfg.csv_path, "w", newline="", buffering=1024 * 1024)
    writer = csv.writer(csv_f)
    writer.writerow([
        "sample_index", "t_seconds",
        "raw1", "v1", "i1",
        "raw2", "v2", "i2",
        "raw3", "v3", "i3",
    ])
    writerow = writer.writerow

    t_start = monotonic()
    t_next_print = t_start + print_dt
    n = 0
    n_at_last_print = 0
    drops = 0

    try:
        while not _stop and (target_n is None or n < target_n):
            r1 = read_one(ch1)
            r2 = read_one(ch2)
            r3 = read_one(ch3)
            if r1 is None or r2 is None or r3 is None:
                drops += 1
                continue

            t_rel = monotonic() - t_start
            v1 = r1 * lsb_to_v
            v2 = r2 * lsb_to_v
            v3 = r3 * lsb_to_v
            i1 = (v1 - o1) / k1
            i2 = (v2 - o2) / k2
            i3 = (v3 - o3) / k3

            writerow((n, f"{t_rel:.6f}",
                      r1, f"{v1:.6f}", f"{i1:.6f}",
                      r2, f"{v2:.6f}", f"{i2:.6f}",
                      r3, f"{v3:.6f}", f"{i3:.6f}"))
            n += 1

            now = monotonic()
            if now >= t_next_print:
                window = now - (t_next_print - print_dt)
                rate = (n - n_at_last_print) / window if window > 0 else 0
                print(f"  n={n:>7} t={t_rel:7.3f}s "
                      f"V1={v1:+.3f} V2={v2:+.3f} V3={v3:+.3f} "
                      f"I1={i1:+.3f} I2={i2:+.3f} I3={i3:+.3f} "
                      f"rate~{rate:6.0f} Hz/ch drops={drops}")
                n_at_last_print = n
                t_next_print = now + print_dt
    finally:
        elapsed = monotonic() - t_start
        adc.close()
        csv_f.close()
        _print_summary(cfg, n, elapsed, drops)


def run_c_core(cfg: Config):
    """C inner loop. Targets 5 kHz/channel on a Pi 4."""
    _print_banner(cfg, "C inner loop")

    gain_code  = GAIN_TABLE[cfg.gain]
    drate_code = DRATE_TABLE[cfg.drate_sps]

    handle = capture_core.open_adc(
        cfg.spi_hz,
        gain_code,
        drate_code,
        cfg.ch_phase1,
    )
    print("  ADC configured (C core)")
    print("=" * 64)

    signal.signal(signal.SIGINT, _on_sigint)

    target_n = cfg.capture_samples if cfg.capture_samples > 0 else 0
    chunk    = 256  # samples per channel per C call

    fullscale_v = 2.0 * ADS1256.VREF / cfg.gain
    lsb_to_v    = fullscale_v / 0x7FFFFF

    o1, k1 = cfg.s1_offset_v, cfg.s1_sens_v_per_a
    o2, k2 = cfg.s2_offset_v, cfg.s2_sens_v_per_a
    o3, k3 = cfg.s3_offset_v, cfg.s3_sens_v_per_a

    csv_f  = open(cfg.csv_path, "w", newline="", buffering=4 * 1024 * 1024)
    writer = csv.writer(csv_f)
    writer.writerow([
        "sample_index", "t_seconds",
        "raw1", "v1", "i1",
        "raw2", "v2", "i2",
        "raw3", "v3", "i3",
    ])
    writerow = writer.writerow

    t_start = time.monotonic()
    t_next_print = t_start + cfg.print_interval_s
    n = 0
    n_at_last_print = 0
    drops_total = 0

    try:
        while not _stop and (target_n == 0 or n < target_n):
            want = chunk if target_n == 0 else min(chunk, target_n - n)
            # capture_burst returns (raws_flat, t0_ns, dt_ns, drops)
            # raws_flat is a list of length 3*want: r1,r2,r3, r1,r2,r3, ...
            raws, t0_ns, dt_ns, drops = capture_core.capture_burst(
                handle, cfg.ch_phase1, cfg.ch_phase2, cfg.ch_phase3, want,
            )
            drops_total += drops

            # Vectorized-ish formatting in Python; the bottleneck was SPI,
            # not float math, so this is fine at 5 kHz/ch.
            base_t = (t0_ns - int(t_start * 1e9)) / 1e9
            inv_k1, inv_k2, inv_k3 = 1.0/k1, 1.0/k2, 1.0/k3
            for i in range(want):
                r1 = raws[3*i]; r2 = raws[3*i+1]; r3 = raws[3*i+2]
                t_rel = base_t + (i * dt_ns) / 1e9
                v1 = r1 * lsb_to_v
                v2 = r2 * lsb_to_v
                v3 = r3 * lsb_to_v
                i1 = (v1 - o1) * inv_k1
                i2 = (v2 - o2) * inv_k2
                i3 = (v3 - o3) * inv_k3
                writerow((n + i, f"{t_rel:.6f}",
                          r1, f"{v1:.6f}", f"{i1:.6f}",
                          r2, f"{v2:.6f}", f"{i2:.6f}",
                          r3, f"{v3:.6f}", f"{i3:.6f}"))
            n += want

            now = time.monotonic()
            if now >= t_next_print:
                window = now - (t_next_print - cfg.print_interval_s)
                rate = (n - n_at_last_print) / window if window > 0 else 0
                print(f"  n={n:>7} t={now-t_start:7.3f}s "
                      f"rate~{rate:6.0f} Hz/ch drops={drops_total}")
                n_at_last_print = n
                t_next_print = now + cfg.print_interval_s
    finally:
        elapsed = time.monotonic() - t_start
        capture_core.close_adc(handle)
        csv_f.close()
        _print_summary(cfg, n, elapsed, drops_total)


def _print_summary(cfg, n, elapsed, drops):
    avg_rate = n / elapsed if elapsed > 0 else 0
    target   = cfg.target_rate_hz
    ratio    = avg_rate / target if target else 0
    print()
    print("-" * 64)
    print(f"  samples written  : {n} per channel ({3*n} total)")
    print(f"  elapsed          : {elapsed:.3f} s")
    print(f"  per-channel rate : {avg_rate:.1f} Hz "
          f"({ratio*100:.1f}% of {target} Hz target)")
    print(f"  drdy timeouts    : {drops}")
    print(f"  csv file         : {cfg.csv_path}")
    print("-" * 64)


def main():
    try:
        cfg = load_config()
    except Exception as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    try:
        if cfg.use_c_core and HAVE_C_CORE:
            run_c_core(cfg)
        else:
            if cfg.use_c_core and not HAVE_C_CORE:
                print("  [note] capture_core C extension not found, "
                      "falling back to pure Python.", file=sys.stderr)
            run_python(cfg)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
