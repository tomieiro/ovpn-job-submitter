#!/usr/bin/env python3
"""Source entry point used by PyInstaller for the windowed build."""

from dgx_slurm.gui import main


if __name__ == "__main__":
    raise SystemExit(main())
