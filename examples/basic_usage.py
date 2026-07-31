#!/usr/bin/env python
"""Canonical usage against a real DGX/SLURM cluster.

Before running this, edit client.py's DEFAULT_SSH_HOST / DEFAULT_REMOTE_BASE_DIR
(src/dgx_slurm/client.py) to match your actual cluster, and point --ovpn at a
real .ovpn file. You will be prompted for your cluster password via getpass.

Run:
    python examples/basic_usage.py /path/to/client.ovpn cluster-user experiment.ipynb
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dgx_slurm import DGXClient, JobState, Resources


async def main() -> None:
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} <ovpn> <username> <notebook>")
        raise SystemExit(2)

    ovpn, username, notebook = sys.argv[1], sys.argv[2], sys.argv[3]

    client = DGXClient(ovpn=ovpn, username=username)
    try:
        job = client.submit(
            notebook,
            resources=Resources(
                gpus=1,
                cpus=4,
                memory="0",
                time_limit="00:20:00",
                partition="devwork",
            ),
        )
        print(f"Submitted job {job.id}")

        result = await job.wait()
    finally:
        client.close()

    print()
    print("state:            ", result.state.value)
    print("exit_code:        ", result.exit_code)
    print("executed_notebook:", result.executed_notebook)
    print("output_files:     ", result.output_files)

    if result.state is not JobState.COMPLETED:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
