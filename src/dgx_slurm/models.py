"""Public data models: Resources, JobState, JobStatus, JobResult."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .errors import ConfigurationError

_MEMORY_RE = re.compile(r"^\d+[KMGT]?$")
_TIME_LIMIT_RE = re.compile(r"^(\d+-)?\d{1,2}:\d{2}:\d{2}$")
_PARTITION_RE = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class Resources:
    """SLURM resource request for a single job submission."""

    gpus: int = 1
    cpus: int = 1
    memory: str = "0"
    time_limit: str = "00:20:00"
    partition: str = "devwork"
    nodes: int = 1
    tasks: int = 1

    def __post_init__(self) -> None:
        if self.gpus < 0:
            raise ConfigurationError(f"gpus must be >= 0, got {self.gpus}")
        if self.cpus < 1:
            raise ConfigurationError(f"cpus must be >= 1, got {self.cpus}")
        if self.nodes < 1:
            raise ConfigurationError(f"nodes must be >= 1, got {self.nodes}")
        if self.tasks < 1:
            raise ConfigurationError(f"tasks must be >= 1, got {self.tasks}")
        if not _MEMORY_RE.match(self.memory):
            raise ConfigurationError(
                f"memory must match <int>[K|M|G|T], got {self.memory!r}"
            )
        if not _TIME_LIMIT_RE.match(self.time_limit):
            raise ConfigurationError(
                f"time_limit must match [D-]HH:MM:SS, got {self.time_limit!r}"
            )
        if not _PARTITION_RE.match(self.partition):
            raise ConfigurationError(
                f"partition must contain only letters, digits, '.', '_', '-', "
                f"got {self.partition!r}"
            )


class JobState(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_slurm(cls, raw: str) -> "JobState":
        """Map a raw squeue/sacct state string to a JobState.

        sacct states can carry suffixes such as "CANCELLED by 1000", so we
        match on the leading token.
        """
        token = raw.strip().split()[0] if raw.strip() else ""
        try:
            return cls(token)
        except ValueError:
            return cls.UNKNOWN

    def is_terminal(self) -> bool:
        return self in _TERMINAL_STATES


_TERMINAL_STATES = frozenset(
    {
        JobState.COMPLETED,
        JobState.FAILED,
        JobState.CANCELLED,
        JobState.TIMEOUT,
        JobState.OUT_OF_MEMORY,
    }
)


@dataclass(frozen=True)
class JobStatus:
    """A single point-in-time snapshot of a job's SLURM state."""

    job_id: str
    state: JobState
    exit_code: int | None
    raw_state: str


@dataclass(frozen=True)
class JobResult:
    """The final outcome of a submitted job, returned by DGXJob.wait()."""

    job_id: str
    state: JobState
    exit_code: int | None
    stdout: str
    stderr: str
    executed_notebook: Path | None
    output_files: tuple[Path, ...] = field(default_factory=tuple)
