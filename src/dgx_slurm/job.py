"""DGXJob: a handle to a single submitted SLURM job."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Awaitable, Callable

from .models import JobResult, JobStatus
from .streaming import LogStreamer

DEFAULT_RESULTS_DIR = Path(".dgx-results")


class DGXJob:
    """Represents one submitted job; monitors it and retrieves its results."""

    def __init__(
        self,
        *,
        job_id: str,
        job_name: str,
        remote_job_dir: str,
        scheduler,
        transport,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        print_fn: Callable[[str], None] = print,
    ) -> None:
        self._job_id = job_id
        self._job_name = job_name
        self._remote_job_dir = remote_job_dir
        self._scheduler = scheduler
        self._transport = transport
        self._sleep = sleep
        self._print_fn = print_fn

    @property
    def id(self) -> str:
        return self._job_id

    def status(self) -> JobStatus:
        return self._scheduler.status(self._job_id)

    async def wait(
        self,
        *,
        stream: bool = True,
        poll_interval: float = 2.0,
        download_outputs: bool = True,
        destination: Path | str | None = None,
    ) -> JobResult:
        streamer = LogStreamer(
            self._transport,
            stdout_path=self._remote_stdout_path(),
            stderr_path=self._remote_stderr_path(),
            print_fn=self._print_fn,
        )

        last_state = None
        status = None
        while True:
            status = await asyncio.to_thread(self._scheduler.status, self._job_id)

            if status.state != last_state:
                if stream:
                    self._print_fn(f"[Job {self._job_id}] {status.state.value}")
                last_state = status.state

            streamer.poll(echo=stream)

            if status.state.is_terminal():
                break

            await self._sleep(poll_interval)

        # Catch any trailing output written between the last poll and job exit.
        streamer.poll(echo=stream)

        executed_notebook = None
        output_files: tuple[Path, ...] = ()
        if download_outputs:
            dest = Path(destination) if destination is not None else DEFAULT_RESULTS_DIR / self._job_id
            downloaded = self.download_outputs(dest)
            for file_path in downloaded:
                if file_path.name == "notebook.executed.ipynb":
                    executed_notebook = file_path
            output_files = tuple(f for f in downloaded if f.name != "notebook.executed.ipynb")

        return JobResult(
            job_id=self._job_id,
            state=status.state,
            exit_code=status.exit_code,
            stdout=streamer.stdout,
            stderr=streamer.stderr,
            executed_notebook=executed_notebook,
            output_files=output_files,
        )

    def cancel(self) -> None:
        self._scheduler.cancel(self._job_id)

    def download_outputs(self, destination: Path | str) -> tuple[Path, ...]:
        destination = Path(destination)
        self._transport.download_directory(f"{self._remote_job_dir}/outputs", destination)
        return tuple(sorted(p for p in destination.rglob("*") if p.is_file()))

    def _remote_stdout_path(self) -> str:
        return f"{self._remote_job_dir}/logs/{self._job_name}-{self._job_id}.out"

    def _remote_stderr_path(self) -> str:
        return f"{self._remote_job_dir}/logs/{self._job_name}-{self._job_id}.err"
