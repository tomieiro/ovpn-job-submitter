#!/usr/bin/env python
"""Reattaching to a job submitted earlier (e.g. from a previous kernel
session), by its SLURM job ID. Requires that the job was originally
submitted through the same LocalJobStore path (the default is
~/.dgx-slurm/jobs.json).

Run:
    python examples/attach_to_job.py /path/to/client.ovpn cluster-user 48192
"""

from __future__ import annotations

import asyncio
import sys

from dgx_slurm import DGXClient


async def main() -> None:
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} <ovpn> <username> <job_id>")
        raise SystemExit(2)

    ovpn, username, job_id = sys.argv[1], sys.argv[2], sys.argv[3]

    client = DGXClient(ovpn=ovpn, username=username)
    try:
        job = client.attach(job_id)
        print(f"Reattached to job {job.id}, current status: {job.status().state.value}")
        result = await job.wait()
    finally:
        client.close()

    print("state:", result.state.value)


if __name__ == "__main__":
    asyncio.run(main())
