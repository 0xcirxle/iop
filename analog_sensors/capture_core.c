/*
 * capture_core.c
 * ==============
 *
 * Native inner loop for the IOP 3-phase capture pipeline.
 * Target hardware: Waveshare High-Precision AD/DA HAT (ADS1256).
 * Marker: pipelined_v1
 *
 * Implementation follows ADS1256 datasheet Figure 19 ("Cycling the
 * ADS1256 Input Multiplexer", page 21) and matches the pattern used
 * by PiPyADC (ul-gh) and the Curious Scientist Arduino driver. Both
 * are known-working references for this exact chip.
 *
 * Key design points (these are NOT optimizations -- they are
 * correctness requirements; getting any one wrong corrupts data):
 *
 *  1. CS goes low ONCE for the entire burst and stays low across the
 *     whole multi-channel multi-sample cycle. Never toggle CS between
 *     commands. (Datasheet p.34: "CS must stay low during the entire
 *     command sequence.")
 *
 *  2. Pipelined MUX writes: set MUX to the NEXT channel before reading
 *     the CURRENT channel's data. Because of the SINC5 filter, the
 *     conversion register holds the value of whatever channel was
 *     selected during the most recent conversion period -- not whatever
 *     channel the MUX is set to RIGHT NOW.
 *
 *  3. Delays in the right places, with values from the datasheet:
 *       - 4us between SYNC and WAKEUP (datasheet 24*tCLKIN ~ 3.1us)
 *       - 7us between RDATA cmd and reading data bytes (50*tCLKIN ~ 6.5us)
 *     Anything else is unnecessary -- DRDY is the canonical timing
 *     signal, the chip auto-delays DRDY until the filter is settled.
 *
 *  4. The first sample of the burst is a "priming" sample whose data
 *     is discarded; it sets the MUX to channel 0 and starts the pipe.
 *     From sample 2 onwards, each iteration reads channel N's data
 *     while setting MUX to channel N+1.
 *
 * SPI: /dev/spidev0.0 via SPI_IOC_MESSAGE ioctl, with kernel
 *      delay_usecs for the t11/t6 windows (precise, lower jitter
 *      than userspace busy-waits, especially on 32-bit Pi OS).
 * GPIO: direct /dev/gpiomem register access for CS, RST, DRDY.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <linux/spi/spidev.h>

/* ---- ADS1256 constants ------------------------------------------------- */
#define REG_STATUS  0x00
#define REG_MUX     0x01
#define REG_ADCON   0x02
#define REG_DRATE   0x03

#define CMD_WAKEUP  0x00
#define CMD_RDATA   0x01
#define CMD_SDATAC  0x0F
#define CMD_RREG    0x10
#define CMD_WREG    0x50
#define CMD_SYNC    0xFC
#define CMD_RESET   0xFE

/* Waveshare HAT BCM pin mapping */
#define PIN_RST   18
#define PIN_CS    22
#define PIN_DRDY  17

/* GPIO peripheral register offsets (BCM2711 / BCM2837 / BCM2835 layout) */
#define GPIO_BLOCK_LEN   0xB4
#define GPFSEL0_OFF   (0x00 / 4)
#define GPSET0_OFF    (0x1C / 4)
#define GPCLR0_OFF    (0x28 / 4)
#define GPLEV0_OFF    (0x34 / 4)
#define GPPUD_OFF     (0x94 / 4)
#define GPPUDCLK0_OFF (0x98 / 4)

/* Datasheet timing windows. tCLKIN = 1/7.68MHz = 130.2 ns. */
#define T11_SYNC_TO_WAKEUP_US  4   /* 24*tCLKIN ~ 3.1us, +margin */
#define T6_RDATA_TO_DATA_US    7   /* 50*tCLKIN ~ 6.5us, +margin */

/* DRDY poll budget. Worst case at DRATE=15000 SPS, the chip can delay
 * DRDY by up to several conversion cycles after SYNC while the SINC5
 * filter settles. 5 ms is plenty of headroom for any DRATE >= 1 kHz. */
#define DRDY_TIMEOUT_NS  (5L * 1000L * 1000L)

/* ---- Handle ------------------------------------------------------------ */
typedef struct {
    int spi_fd;
    uint32_t spi_hz;
    int gpiomem_fd;
    volatile uint32_t *gpio;
} adc_t;

/* ---- helpers ----------------------------------------------------------- */
static inline int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static inline void ns_sleep(int64_t ns) {
    struct timespec req = { .tv_sec  = ns / 1000000000LL,
                            .tv_nsec = ns % 1000000000LL };
    nanosleep(&req, NULL);
}

static inline void gpio_set_output(volatile uint32_t *gpio, int pin) {
    int reg = pin / 10;
    int shift = (pin % 10) * 3;
    uint32_t v = gpio[GPFSEL0_OFF + reg];
    v &= ~(7u << shift);
    v |=  (1u << shift);
    gpio[GPFSEL0_OFF + reg] = v;
}

static inline void gpio_set_input(volatile uint32_t *gpio, int pin) {
    int reg = pin / 10;
    int shift = (pin % 10) * 3;
    uint32_t v = gpio[GPFSEL0_OFF + reg];
    v &= ~(7u << shift);
    gpio[GPFSEL0_OFF + reg] = v;
}

static inline void gpio_high(volatile uint32_t *gpio, int pin) {
    gpio[GPSET0_OFF] = 1u << pin;
    __sync_synchronize();
}

static inline void gpio_low(volatile uint32_t *gpio, int pin) {
    gpio[GPCLR0_OFF] = 1u << pin;
    __sync_synchronize();
}

static void gpio_set_pullup(volatile uint32_t *gpio, int pin) {
    gpio[GPPUD_OFF] = 2;
    ns_sleep(2000);
    gpio[GPPUDCLK0_OFF] = 1u << pin;
    ns_sleep(2000);
    gpio[GPPUD_OFF] = 0;
    gpio[GPPUDCLK0_OFF] = 0;
}

static inline void cs_low (adc_t *a) { gpio_low (a->gpio, PIN_CS); }
static inline void cs_high(adc_t *a) { gpio_high(a->gpio, PIN_CS); }

/* Wait for DRDY low. The chip auto-delays DRDY until the digital filter
 * has settled, so we don't need any manual edge tracking -- a level
 * poll is sufficient AS LONG AS we reach this function fast enough that
 * we don't miss the prior cycle's high-low transition. In practice that
 * means: don't sleep for tens of milliseconds between samples. */
static int wait_drdy(adc_t *a) {
    int64_t deadline = now_ns() + DRDY_TIMEOUT_NS;
    volatile uint32_t *gpio = a->gpio;
    for (;;) {
        if (((gpio[GPLEV0_OFF] >> PIN_DRDY) & 1u) == 0) return 0;
        if (now_ns() > deadline) return -1;
    }
}

/* SPI write a few bytes. Used outside the burst hot path. */
static int spi_write(adc_t *a, const uint8_t *tx, size_t n) {
    struct spi_ioc_transfer tr = {
        .tx_buf = (unsigned long)tx,
        .rx_buf = 0,
        .len = (uint32_t)n,
        .speed_hz = a->spi_hz,
        .bits_per_word = 8,
        .delay_usecs = 0,
        .cs_change = 0,
    };
    return ioctl(a->spi_fd, SPI_IOC_MESSAGE(1), &tr);
}

/* ---- ADS1256 ops (init/config) ----------------------------------------- */

static int adc_reset(adc_t *a) {
    gpio_high(a->gpio, PIN_RST); ns_sleep(200000000);
    gpio_low (a->gpio, PIN_RST); ns_sleep(200000000);
    gpio_high(a->gpio, PIN_RST);
    ns_sleep(50000000);
    return 0;
}

static int adc_read_reg(adc_t *a, uint8_t reg, uint8_t *out) {
    /* Two-segment with t6 gap. Used only at startup to read chip ID. */
    uint8_t tx0[2] = { (uint8_t)(CMD_RREG | reg), 0x00 };
    uint8_t tx1   = 0x00;
    uint8_t rx1   = 0x00;
    struct spi_ioc_transfer tr[2] = {
        { .tx_buf = (unsigned long)tx0, .rx_buf = 0,
          .len = 2, .speed_hz = a->spi_hz, .bits_per_word = 8,
          .delay_usecs = T6_RDATA_TO_DATA_US, .cs_change = 0 },
        { .tx_buf = (unsigned long)&tx1, .rx_buf = (unsigned long)&rx1,
          .len = 1, .speed_hz = a->spi_hz, .bits_per_word = 8,
          .delay_usecs = 0, .cs_change = 0 },
    };
    cs_low(a);
    int r = ioctl(a->spi_fd, SPI_IOC_MESSAGE(2), tr);
    cs_high(a);
    if (r < 0) return r;
    *out = rx1;
    return 0;
}

static int adc_config(adc_t *a, uint8_t gain_code, uint8_t drate_code,
                      int initial_channel)
{
    if (wait_drdy(a) != 0) return -1;
    uint8_t mux_val = (uint8_t)(((initial_channel & 0x07) << 4) | 0x08);
    uint8_t tx[6] = {
        (uint8_t)(CMD_WREG | 0x00), 0x03,
        0x04,                  /* STATUS: BUFEN=1, ACAL=0, ORDER=MSB */
        mux_val,
        (uint8_t)(gain_code),
        drate_code,
    };
    cs_low(a);
    int r = spi_write(a, tx, 6);
    cs_high(a);
    ns_sleep(1000000);
    return r;
}

/* ---- Pipelined burst read (THE fast path) ------------------------------
 *
 * Reads `n_per_ch` samples from each of three channels (round-robin),
 * implementing the channel-cycling pattern from datasheet Figure 19.
 *
 * CS is held low for the entire burst. The pipeline:
 *
 *   [priming step at start of burst]
 *   wait DRDY
 *   WREG MUX = ch[0]    (set MUX to first channel; chip starts conversion)
 *
 *   [main loop, for i = 0 .. (3*n_per_ch - 1)]
 *       wait DRDY                          // ch[(i) mod 3] conv done
 *       WREG MUX = ch[(i+1) mod 3]         // queue NEXT channel
 *       SYNC, delay 4us, WAKEUP            // restart converter
 *       RDATA, delay 7us, read 3 bytes     // CURRENT channel's data
 *       store sample as channel ch[i mod 3]
 *
 * The first 3 iterations populate channels 1, 2, 3 of sample 0.
 * Iteration 4 produces channel 1 of sample 1. And so on.
 *
 * Returns:
 *    0 on success
 *   -1 on DRDY timeout or ioctl failure (caller checks `drops`)
 *
 * Output buf layout: [s0c1, s0c2, s0c3, s1c1, s1c2, s1c3, ...]
 */
static int adc_burst(adc_t *a,
                     int ch1, int ch2, int ch3,
                     int n_per_ch,
                     int32_t *out_buf,
                     int *out_drops)
{
    int drops = 0;
    int channels[3] = { ch1, ch2, ch3 };
    uint8_t mux_vals[3] = {
        (uint8_t)(((ch1 & 0x07) << 4) | 0x08),
        (uint8_t)(((ch2 & 0x07) << 4) | 0x08),
        (uint8_t)(((ch3 & 0x07) << 4) | 0x08),
    };
    (void)channels;  /* indices only; mux_vals carries the value */

    cs_low(a);

    /* --- Priming step: set MUX to channel 1 and SYNC the converter,
     * so that when we enter the loop, the first DRDY wait is for a
     * conversion of channel 1. */
    {
        if (wait_drdy(a) != 0) { cs_high(a); return -1; }
        uint8_t tx_prime[6] = {
            (uint8_t)(CMD_WREG | REG_MUX), 0x00, mux_vals[0],
            CMD_SYNC,
            /* gap is satisfied by the kernel's per-segment delay below */
            CMD_WAKEUP,
            0x00,  /* unused padding -- replaced below with multi-segment */
        };
        (void)tx_prime;
        /* Send WREG MUX (3 bytes), 4us delay, SYNC + WAKEUP (2 bytes). */
        uint8_t tx_a[3] = {
            (uint8_t)(CMD_WREG | REG_MUX), 0x00, mux_vals[0],
        };
        uint8_t tx_b[2] = { CMD_SYNC, CMD_WAKEUP };
        struct spi_ioc_transfer tr[2] = {
            { .tx_buf = (unsigned long)tx_a, .rx_buf = 0,
              .len = 3, .speed_hz = a->spi_hz, .bits_per_word = 8,
              .delay_usecs = T11_SYNC_TO_WAKEUP_US, .cs_change = 0 },
            { .tx_buf = (unsigned long)tx_b, .rx_buf = 0,
              .len = 2, .speed_hz = a->spi_hz, .bits_per_word = 8,
              .delay_usecs = 0, .cs_change = 0 },
        };
        if (ioctl(a->spi_fd, SPI_IOC_MESSAGE(2), tr) < 0) {
            cs_high(a);
            return -1;
        }
    }

    /* --- Main pipelined loop ---
     *
     * Each iteration reads the channel that was queued one iteration
     * earlier, and queues the next channel for the iteration after this.
     *
     * Total iterations = 3 * n_per_ch + 1 priming-discard read.
     * The +1 is because iteration 0 reads "stale" data corresponding
     * to whatever was in the converter before our burst started. We
     * still need to consume it (otherwise the chip's RDRY pipeline gets
     * confused on subsequent reads).
     */
    int total_iters = 3 * n_per_ch;

    for (int i = 0; i <= total_iters; i++) {
        /* current channel = index of the channel whose data we read here */
        int cur_idx  = (i - 1) % 3;          /* -1 wraps to 2 in iter 0 */
        if (cur_idx < 0) cur_idx += 3;

        /* next channel = the one we queue this iteration */
        int next_idx = i % 3;

        /* Wait for the current conversion to complete. */
        if (wait_drdy(a) != 0) { drops++; goto done; }

        /* Queue next channel + restart converter, AND read current data,
         * all in a single ioctl with two segments and the t6 delay between
         * the RDATA cmd and the data bytes. The full byte sequence:
         *
         *   WREG | MUX, 0x00, mux_vals[next]   (3 bytes, set next channel)
         *   SYNC                                (1 byte, restart converter)
         *   <kernel delay 4us, t11>
         *   WAKEUP                              (1 byte)
         *   RDATA                               (1 byte, request current data)
         *   <kernel delay 7us, t6>
         *   0x00, 0x00, 0x00                    (3 bytes, MOSI clocks while
         *                                        we capture MISO data)
         */
        uint8_t tx_pre[5] = {
            (uint8_t)(CMD_WREG | REG_MUX), 0x00, mux_vals[next_idx],
            CMD_SYNC,
            /* t11 gap is supplied by kernel between segments */
        };
        /* Two consecutive bytes after the SYNC delay: WAKEUP + RDATA */
        uint8_t tx_mid[2] = { CMD_WAKEUP, CMD_RDATA };
        uint8_t tx_data[3] = { 0x00, 0x00, 0x00 };
        uint8_t rx_data[3] = { 0, 0, 0 };

        struct spi_ioc_transfer tr[3] = {
            /* segment 0: WREG MUX + SYNC, then t11 delay */
            { .tx_buf = (unsigned long)tx_pre, .rx_buf = 0,
              .len = 4, .speed_hz = a->spi_hz, .bits_per_word = 8,
              .delay_usecs = T11_SYNC_TO_WAKEUP_US, .cs_change = 0 },
            /* segment 1: WAKEUP + RDATA, then t6 delay */
            { .tx_buf = (unsigned long)tx_mid, .rx_buf = 0,
              .len = 2, .speed_hz = a->spi_hz, .bits_per_word = 8,
              .delay_usecs = T6_RDATA_TO_DATA_US, .cs_change = 0 },
            /* segment 2: 3 data bytes */
            { .tx_buf = (unsigned long)tx_data, .rx_buf = (unsigned long)rx_data,
              .len = 3, .speed_hz = a->spi_hz, .bits_per_word = 8,
              .delay_usecs = 0, .cs_change = 0 },
        };
        if (ioctl(a->spi_fd, SPI_IOC_MESSAGE(3), tr) < 0) {
            drops++;
            goto done;
        }

        /* Decode the 24-bit signed result. */
        int32_t raw = ((int32_t)rx_data[0] << 16) |
                      ((int32_t)rx_data[1] <<  8) |
                      ((int32_t)rx_data[2]);
        if (raw & 0x800000) raw |= ~0xFFFFFF;

        /* Store -- skip iteration 0 (it's the priming read with stale data). */
        if (i > 0) {
            int sample_idx = (i - 1) / 3;     /* which sample number    */
            int chan       = (i - 1) % 3;     /* which channel slot     */
            out_buf[3 * sample_idx + chan] = raw;
        }
    }

done:
    cs_high(a);
    if (out_drops) *out_drops = drops;
    return drops > 0 ? -1 : 0;
}

/* ---- /dev/gpiomem setup ------------------------------------------------ */
static int setup_gpio(adc_t *a) {
    a->gpiomem_fd = open("/dev/gpiomem", O_RDWR | O_SYNC);
    if (a->gpiomem_fd < 0) {
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, "/dev/gpiomem");
        return -1;
    }
    void *map = mmap(NULL, GPIO_BLOCK_LEN, PROT_READ | PROT_WRITE,
                     MAP_SHARED, a->gpiomem_fd, 0);
    if (map == MAP_FAILED) {
        close(a->gpiomem_fd);
        a->gpiomem_fd = -1;
        PyErr_SetFromErrno(PyExc_OSError);
        return -1;
    }
    a->gpio = (volatile uint32_t *)map;

    gpio_high(a->gpio, PIN_CS);
    gpio_high(a->gpio, PIN_RST);
    gpio_set_output(a->gpio, PIN_CS);
    gpio_set_output(a->gpio, PIN_RST);
    gpio_set_input (a->gpio, PIN_DRDY);
    gpio_set_pullup(a->gpio, PIN_DRDY);
    return 0;
}

static void teardown_gpio(adc_t *a) {
    if (a->gpio) {
        munmap((void *)a->gpio, GPIO_BLOCK_LEN);
        a->gpio = NULL;
    }
    if (a->gpiomem_fd >= 0) {
        close(a->gpiomem_fd);
        a->gpiomem_fd = -1;
    }
}

/* ---- spidev setup ------------------------------------------------------ */
static int setup_spi(adc_t *a, uint32_t hz) {
    a->spi_fd = open("/dev/spidev0.0", O_RDWR);
    if (a->spi_fd < 0) {
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, "/dev/spidev0.0");
        return -1;
    }
    uint8_t mode = SPI_MODE_1;
    uint8_t bits = 8;
    a->spi_hz = hz;
    if (ioctl(a->spi_fd, SPI_IOC_WR_MODE, &mode) < 0 ||
        ioctl(a->spi_fd, SPI_IOC_WR_BITS_PER_WORD, &bits) < 0 ||
        ioctl(a->spi_fd, SPI_IOC_WR_MAX_SPEED_HZ, &hz) < 0)
    {
        PyErr_SetFromErrno(PyExc_OSError);
        close(a->spi_fd);
        a->spi_fd = -1;
        return -1;
    }
    return 0;
}

/* ---- Python bindings --------------------------------------------------- */
static void adc_destructor(PyObject *capsule) {
    adc_t *a = (adc_t *)PyCapsule_GetPointer(capsule, "iop.adc_t");
    if (!a) return;
    if (a->spi_fd >= 0) close(a->spi_fd);
    teardown_gpio(a);
    free(a);
}

static PyObject *py_open_adc(PyObject *self, PyObject *args) {
    int spi_hz, gain_code, drate_code, initial_channel;
    if (!PyArg_ParseTuple(args, "iiii", &spi_hz, &gain_code, &drate_code,
                          &initial_channel))
        return NULL;

    adc_t *a = calloc(1, sizeof(*a));
    if (!a) return PyErr_NoMemory();
    a->spi_fd = -1;
    a->gpiomem_fd = -1;

    if (setup_spi(a, (uint32_t)spi_hz) < 0) { free(a); return NULL; }
    if (setup_gpio(a) < 0) { close(a->spi_fd); free(a); return NULL; }

    adc_reset(a);

    if (wait_drdy(a) < 0) {
        teardown_gpio(a); close(a->spi_fd); free(a);
        PyErr_SetString(PyExc_RuntimeError, "DRDY never asserted after reset");
        return NULL;
    }
    uint8_t status = 0;
    if (adc_read_reg(a, REG_STATUS, &status) < 0) {
        teardown_gpio(a); close(a->spi_fd); free(a);
        PyErr_SetString(PyExc_RuntimeError, "STATUS read failed");
        return NULL;
    }
    if ((status >> 4) != 3) {
        teardown_gpio(a); close(a->spi_fd); free(a);
        PyErr_Format(PyExc_RuntimeError,
                     "Bad ADS1256 chip ID: expected 3, got %d (status=0x%02X)",
                     status >> 4, status);
        return NULL;
    }

    if (adc_config(a, (uint8_t)gain_code, (uint8_t)drate_code,
                   initial_channel) < 0)
    {
        teardown_gpio(a); close(a->spi_fd); free(a);
        PyErr_SetString(PyExc_RuntimeError, "adc_config failed");
        return NULL;
    }

    return PyCapsule_New(a, "iop.adc_t", adc_destructor);
}

static PyObject *py_close_adc(PyObject *self, PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    if (PyCapsule_CheckExact(capsule)) {
        adc_t *a = (adc_t *)PyCapsule_GetPointer(capsule, "iop.adc_t");
        if (a) {
            if (a->spi_fd >= 0) { close(a->spi_fd); a->spi_fd = -1; }
            teardown_gpio(a);
        }
        if (PyErr_Occurred()) PyErr_Clear();
    }
    Py_RETURN_NONE;
}

/*
 * capture_burst(handle, ch1, ch2, ch3, n_per_ch)
 *   -> (raws_list, t0_ns, dt_ns_avg, drops)
 *
 * Reads n_per_ch samples from each of the three channels using the
 * pipelined cycling pattern. Releases the GIL across the burst.
 *
 * Returns a flat Python list of 3*n_per_ch ints (r1,r2,r3, r1,r2,r3, ...).
 */
static PyObject *py_capture_burst(PyObject *self, PyObject *args) {
    PyObject *capsule;
    int ch1, ch2, ch3, n_per_ch;
    if (!PyArg_ParseTuple(args, "Oiiii", &capsule, &ch1, &ch2, &ch3, &n_per_ch))
        return NULL;
    if (n_per_ch <= 0) {
        PyErr_SetString(PyExc_ValueError, "n_per_ch must be > 0");
        return NULL;
    }
    adc_t *a = (adc_t *)PyCapsule_GetPointer(capsule, "iop.adc_t");
    if (!a || a->spi_fd < 0) {
        PyErr_SetString(PyExc_RuntimeError, "ADC handle is closed");
        return NULL;
    }

    int32_t *buf = malloc(sizeof(int32_t) * 3 * (size_t)n_per_ch);
    if (!buf) return PyErr_NoMemory();

    int drops = 0;
    int64_t t0 = 0, t_last = 0;
    int rc = 0;

    Py_BEGIN_ALLOW_THREADS
    t0 = now_ns();
    rc = adc_burst(a, ch1, ch2, ch3, n_per_ch, buf, &drops);
    t_last = now_ns();
    Py_END_ALLOW_THREADS

    (void)rc; /* drops carries the error info already */

    int64_t total_ns = t_last - t0;
    int64_t dt_avg   = (n_per_ch > 0) ? (total_ns / n_per_ch) : 0;

    PyObject *list = PyList_New((Py_ssize_t)(3 * n_per_ch));
    if (!list) { free(buf); return NULL; }
    for (int i = 0; i < 3 * n_per_ch; i++) {
        PyObject *v = PyLong_FromLong((long)buf[i]);
        if (!v) { Py_DECREF(list); free(buf); return NULL; }
        PyList_SET_ITEM(list, i, v);
    }
    free(buf);

    /* "N" steals our reference into the tuple. */
    return Py_BuildValue("(NLLi)", list,
                         (long long)t0, (long long)dt_avg, drops);
}

static PyMethodDef methods[] = {
    { "open_adc",      py_open_adc,      METH_VARARGS,
      "open_adc(spi_hz, gain_code, drate_code, initial_channel) -> handle" },
    { "close_adc",     py_close_adc,     METH_VARARGS,
      "close_adc(handle) -> None" },
    { "capture_burst", py_capture_burst, METH_VARARGS,
      "capture_burst(handle, ch1, ch2, ch3, n_per_ch) "
      "-> (raws_list, t0_ns, dt_ns_avg, drops)" },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "capture_core",
    "Native ADS1256 + GPIO inner loop (pipelined) for the IOP capture pipeline.",
    -1,
    methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_capture_core(void) {
    return PyModule_Create(&moduledef);
}
