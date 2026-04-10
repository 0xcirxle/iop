#!/usr/bin/env python3
"""Convenience entry point for quickly validating sensor reads."""

import sys

from motor_fault.cli import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["test-sensors", "--live"])
    raise SystemExit(main())
