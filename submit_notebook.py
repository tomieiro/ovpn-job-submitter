#!/usr/bin/env python3
"""Source entry point used by PyInstaller."""

from dgx_slurm.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
