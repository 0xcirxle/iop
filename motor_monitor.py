#!/usr/bin/env python3
"""Convenience entry point for the Raspberry Pi monitor."""

import sys

from motor_fault.cli import main


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1].startswith("-"):
        sys.argv.insert(1, "run")
    raise SystemExit(main())
