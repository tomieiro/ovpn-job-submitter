#!/usr/bin/env python
"""Submitting a notebook together with extra project files (data, source
code) via the `include=` parameter. Everything under each included path is
copied into payload/ on the remote bundle, preserving relative structure.

Run:
    python examples/submit_with_includes.py /path/to/client.ovpn cluster-user
"""

from __future__ import annotations

import asyncio
import sys

from dgx_slurm import DGXClient, Resources


async def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <ovpn> <username>")
        raise SystemExit(2)

    ovpn, username = sys.argv[1], sys.argv[2]

    client = DGXClient(ovpn=ovpn, username=username)
    try:
        job = client.submit(
            "experiment.ipynb",
            resources=Resources(gpus=1, cpus=8, partition="devwork"),
            include=[
                "data/",
                "src/",
                "config.json",
            ],
        )
        print(f"Submitted job {job.id}")
        result = await job.wait()
    finally:
        client.close()

    print("state:", result.state.value)


if __name__ == "__main__":
    asyncio.run(main())
