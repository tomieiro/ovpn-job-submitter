"""Opt-in smoke test against a real DGX/SLURM cluster.

Not run by default. Requires:
  - DGX_SLURM_SMOKE_OVPN: path to a real .ovpn file
  - DGX_SLURM_SMOKE_USERNAME: cluster username
  - network access to the real cluster's OpenVPN endpoint

Run with: pytest -m cluster tests/test_cluster_smoke.py
"""

import os
from pathlib import Path

import pytest

from dgx_slurm import DGXClient, JobState, Resources

pytestmark = pytest.mark.cluster


@pytest.mark.asyncio
async def test_gpu_notebook():
    ovpn = os.environ.get("DGX_SLURM_SMOKE_OVPN")
    username = os.environ.get("DGX_SLURM_SMOKE_USERNAME")
    if not ovpn or not username:
        pytest.skip(
            "DGX_SLURM_SMOKE_OVPN and DGX_SLURM_SMOKE_USERNAME must be set "
            "to run the real-cluster smoke test"
        )

    notebook = Path(__file__).parent / "fixtures" / "gpu_smoke.ipynb"

    client = DGXClient(ovpn=ovpn, username=username)
    try:
        job = client.submit(
            notebook,
            resources=Resources(
                gpus=1, cpus=4, memory="0", time_limit="00:20:00", partition="devwork"
            ),
        )
        result = await job.wait()
    finally:
        client.close()

    assert result.state is JobState.COMPLETED
    assert result.exit_code == 0
    assert "CUDA available: True" in result.stdout
    assert result.executed_notebook is not None
    assert result.executed_notebook.exists()
