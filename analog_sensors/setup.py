"""
setup.py for capture_core
=========================

Builds the C extension that runs the SPI/GPIO inner loop. Uses
/dev/gpiomem direct register access (no libgpiod), so the only build
dependencies are a C compiler and the Python development headers.

Usage:
    pip install setuptools wheel
    sudo apt-get install -y build-essential python3-dev
    python3 setup.py build_ext --inplace
"""
from setuptools import setup, Extension

ext = Extension(
    name="capture_core",
    sources=["capture_core.c"],
    extra_compile_args=[
        "-O3",
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
        "-fno-strict-aliasing",
    ],
)

setup(
    name="capture_core",
    version="0.2.0",
    description="Native ADS1256 + GPIO inner loop for the IOP capture pipeline.",
    ext_modules=[ext],
)
