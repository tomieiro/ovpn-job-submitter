#!/usr/bin/env python
"""Runs the full DGXClient/DGXJob flow end-to-end on your own machine,
with no VPN, SSH, SLURM, or Docker involved.

It replaces the network layers (VPNConnection, SSHTransport, SlurmScheduler)
with tiny local stand-ins backed by a plain temp directory, but the notebook
itself is executed for real, by the exact same runner
(dgx_slurm/templates/execute_notebook.py) that ships inside the Docker image
that would run on the real DGX. You will see the same live
"[Job <id>] STATE" / cell-by-cell output that job.wait() prints against a
real cluster.

Requirements:
    - the dgx_slurm package installed in the active environment
    - a Jupyter "python3" kernel available, e.g.:
        python -m ipykernel install --user --name python3

Run:
    python examples/local_dry_run.py
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dgx_slurm import DGXClient, JobState, Resources
from dgx_slurm.models import JobStatus
from dgx_slurm.ssh import RemoteCommandResult, RemoteFileChunk
from dgx_slurm.storage import LocalJobStore


class LocalFilesystemTransport:
    """Stands in for SSHTransport: the "remote" is just a local directory."""

    def __init__(self, remote_root: Path) -> None:
        self._remote_root = remote_root

    def connect(self) -> None:
        pass

    def execute(self, command: str) -> RemoteCommandResult:
        if command.startswith("mkdir -p"):
            target = command.split("mkdir -p", 1)[1].strip().replace("'", "")
            (self._remote_root / target).mkdir(parents=True, exist_ok=True)
            return RemoteCommandResult(command=command, exit_code=0, stdout="", stderr="")
        raise RuntimeError(f"unsupported command in local dry run: {command}")

    def upload_directory(self, local: Path, remote: str) -> None:
        import shutil

        shutil.copytree(local, self._remote_root / remote, dirs_exist_ok=True)

    def download_directory(self, remote: str, local: Path) -> None:
        import shutil

        shutil.copytree(self._remote_root / remote, local, dirs_exist_ok=True)

    def read_from(self, remote_file: str, offset: int) -> RemoteFileChunk:
        path = self._remote_root / remote_file
        if not path.exists():
            return RemoteFileChunk(text="", new_offset=offset)
        data = path.read_bytes()[offset:]
        return RemoteFileChunk(text=data.decode(errors="replace"), new_offset=offset + len(data))

    def close(self) -> None:
        pass


class NoopVPN:
    """Stands in for VPNConnection: there is nowhere to connect to."""

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass


class LocalProcessScheduler:
    """Stands in for SlurmScheduler: runs the notebook runner as a plain
    subprocess instead of `docker run` inside a SLURM allocation."""

    _EXECUTE_NOTEBOOK = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "dgx_slurm"
        / "templates"
        / "execute_notebook.py"
    )

    def __init__(self, remote_root: Path) -> None:
        self._remote_root = remote_root
        self._processes: dict[str, subprocess.Popen] = {}
        self._counter = 0

    def submit(self, remote_job_dir: str, script_name: str = "runImage.slurm") -> str:
        self._counter += 1
        job_id = str(10000 + self._counter)
        job_name = Path(remote_job_dir).name
        job_dir = self._remote_root / remote_job_dir

        logs_dir = job_dir / "logs"
        outputs_dir = job_dir / "outputs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        out_log = logs_dir / f"{job_name}-{job_id}.out"
        err_log = logs_dir / f"{job_name}-{job_id}.err"
        out_log.write_text(f"=== node: {os.uname().nodename} ===\n=== job: {job_id} ===\n")

        env = os.environ.copy()
        env["DGX_NOTEBOOK_WORKDIR"] = str(job_dir)

        out_handle = out_log.open("ab")
        err_handle = err_log.open("ab")
        process = subprocess.Popen(
            [
                sys.executable,
                str(self._EXECUTE_NOTEBOOK),
                str(job_dir / "payload" / "notebook.ipynb"),
                str(outputs_dir / "notebook.executed.ipynb"),
            ],
            cwd=str(job_dir),
            env=env,
            stdout=out_handle,
            stderr=err_handle,
        )
        self._processes[job_id] = process
        return job_id

    def status(self, job_id: str) -> JobStatus:
        process = self._processes[job_id]
        exit_code = process.poll()
        if exit_code is None:
            state = JobState.RUNNING
        elif exit_code == 0:
            state = JobState.COMPLETED
        else:
            state = JobState.FAILED
        return JobStatus(job_id=job_id, state=state, exit_code=exit_code, raw_state=state.value)

    def cancel(self, job_id: str) -> None:
        self._processes[job_id].terminate()


async def main() -> None:
    examples_dir = Path(__file__).parent
    notebook = examples_dir / "experiment.ipynb"
    remote_root = Path(tempfile.mkdtemp(prefix="dgx-slurm-dryrun-remote-"))
    bundle_workdir = Path(tempfile.mkdtemp(prefix="dgx-slurm-dryrun-bundles-"))

    client = DGXClient(
        ovpn="/dev/null",  # unused: vpn/transport are injected below
        username="local-demo",
        ssh_host="local.invalid",
        ssh_port=22,
        vpn=NoopVPN(),
        transport=LocalFilesystemTransport(remote_root),
        scheduler=LocalProcessScheduler(remote_root),
        job_store=LocalJobStore(examples_dir / ".dgx-results" / "jobs.json"),
        workdir_root=bundle_workdir,
        project_root=examples_dir,
    )

    print(f"Submitting {notebook.name}...")
    job = client.submit(
        notebook,
        resources=Resources(gpus=0, cpus=1, memory="0", time_limit="00:05:00", partition="devwork"),
    )
    print(f"Submitted as job {job.id}\n")

    result = await job.wait(
        poll_interval=0.2,
        download_outputs=True,
        destination=examples_dir / ".dgx-results" / job.id,
    )

    print()
    print(f"state:             {result.state.value}")
    print(f"exit_code:         {result.exit_code}")
    print(f"executed_notebook: {result.executed_notebook}")
    print(f"output_files:      {result.output_files}")

    client.close()


if __name__ == "__main__":
    asyncio.run(main())
