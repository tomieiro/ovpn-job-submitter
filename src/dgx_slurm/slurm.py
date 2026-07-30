"""SLURM job control (sbatch, squeue, sacct, scancel) over an SSH transport."""

from __future__ import annotations

import re
import shlex

from .errors import SchedulerError
from .models import JobState, JobStatus
from .ssh import SSHTransport


class SlurmScheduler:
    """Submits and monitors jobs on the remote SLURM cluster."""

    def __init__(self, transport: SSHTransport) -> None:
        self._transport = transport

    def submit(self, remote_job_dir: str, script_name: str = "runImage.slurm") -> str:
        command = (
            f"cd {shlex.quote(remote_job_dir)} && "
            f"sbatch --parsable {shlex.quote(script_name)}"
        )
        result = self._transport.execute(command)
        if result.exit_code != 0:
            raise SchedulerError(
                f"sbatch failed (exit {result.exit_code}): {result.stderr.strip()}"
            )
        job_id = result.stdout.strip().splitlines()[-1].split(";")[0].strip() if result.stdout.strip() else ""
        if not job_id.isdigit():
            raise SchedulerError(f"unexpected sbatch output: {result.stdout!r}")
        return job_id

    def status(self, job_id: str) -> JobStatus:
        squeue_state = self._squeue_state(job_id)
        if squeue_state:
            return JobStatus(
                job_id=job_id,
                state=JobState.from_slurm(squeue_state),
                exit_code=None,
                raw_state=squeue_state,
            )

        raw_state, exit_code = self._sacct_state(job_id)
        if not raw_state:
            raw_state, exit_code = self._scontrol_state(job_id)
        if not raw_state:
            return JobStatus(job_id=job_id, state=JobState.UNKNOWN, exit_code=None, raw_state="")
        return JobStatus(
            job_id=job_id,
            state=JobState.from_slurm(raw_state),
            exit_code=exit_code,
            raw_state=raw_state,
        )

    def cancel(self, job_id: str) -> None:
        command = f"scancel {shlex.quote(job_id)}"
        result = self._transport.execute(command)
        if result.exit_code != 0:
            raise SchedulerError(
                f"scancel failed for job {job_id} (exit {result.exit_code}): "
                f"{result.stderr.strip()}"
            )

    def _squeue_state(self, job_id: str) -> str:
        command = f"squeue -j {shlex.quote(job_id)} -h -o %T"
        result = self._transport.execute(command)
        return result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""

    def _sacct_state(self, job_id: str) -> tuple[str, int | None]:
        command = f"sacct -j {shlex.quote(job_id)} -n -P -o JobID,State,ExitCode"
        result = self._transport.execute(command)
        for line in result.stdout.strip().splitlines():
            fields = line.split("|")
            if len(fields) != 3:
                continue
            row_job_id, state, exit_code_field = fields
            if row_job_id.strip() != job_id:
                continue
            exit_code = None
            if exit_code_field.strip():
                code_part = exit_code_field.split(":")[0].strip()
                if code_part.lstrip("-").isdigit():
                    exit_code = int(code_part)
            return state.strip(), exit_code
        return "", None

    def _scontrol_state(self, job_id: str) -> tuple[str, int | None]:
        """Fallback for clusters whose accounting database omits recent jobs."""
        command = f"scontrol show job -o {shlex.quote(job_id)}"
        result = self._transport.execute(command)
        if result.exit_code != 0 or not result.stdout.strip():
            return "", None

        state_match = re.search(r"(?:^|\s)JobState=(\S+)", result.stdout)
        exit_match = re.search(r"(?:^|\s)ExitCode=(-?\d+):\d+", result.stdout)
        if state_match is None:
            return "", None

        exit_code = int(exit_match.group(1)) if exit_match else None
        return state_match.group(1), exit_code
