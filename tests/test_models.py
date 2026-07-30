import pytest

from dgx_slurm.errors import ConfigurationError
from dgx_slurm.models import JobResult, JobState, JobStatus, Resources


def test_resources_accepts_valid_values():
    r = Resources(gpus=2, cpus=8, memory="32G", time_limit="01:30:00", partition="devwork")
    assert r.gpus == 2
    assert r.cpus == 8
    assert r.memory == "32G"
    assert r.partition == "devwork"


def test_resources_defaults():
    r = Resources()
    assert r.gpus == 1
    assert r.cpus == 1
    assert r.memory == "0"
    assert r.time_limit == "00:20:00"
    assert r.partition == "devwork"
    assert r.nodes == 1
    assert r.tasks == 1


def test_resources_rejects_negative_gpu():
    with pytest.raises(ConfigurationError):
        Resources(gpus=-1)


def test_resources_rejects_cpu_less_than_one():
    with pytest.raises(ConfigurationError):
        Resources(cpus=0)


def test_resources_accepts_memory_zero():
    r = Resources(memory="0")
    assert r.memory == "0"


def test_resources_rejects_invalid_memory_format():
    with pytest.raises(ConfigurationError):
        Resources(memory="lots")


def test_resources_rejects_invalid_time_limit_format():
    with pytest.raises(ConfigurationError):
        Resources(time_limit="soon")


def test_resources_rejects_unsafe_partition():
    with pytest.raises(ConfigurationError):
        Resources(partition="devwork; rm -rf /")


def test_resources_rejects_partition_with_spaces():
    with pytest.raises(ConfigurationError):
        Resources(partition="dev work")


def test_resources_rejects_nodes_less_than_one():
    with pytest.raises(ConfigurationError):
        Resources(nodes=0)


def test_resources_rejects_tasks_less_than_one():
    with pytest.raises(ConfigurationError):
        Resources(tasks=0)


def test_resources_is_frozen():
    r = Resources()
    with pytest.raises(Exception):
        r.gpus = 5


def test_job_state_maps_known_states():
    assert JobState.from_slurm("PENDING") is JobState.PENDING
    assert JobState.from_slurm("RUNNING") is JobState.RUNNING
    assert JobState.from_slurm("COMPLETED") is JobState.COMPLETED
    assert JobState.from_slurm("FAILED") is JobState.FAILED
    assert JobState.from_slurm("CANCELLED") is JobState.CANCELLED
    assert JobState.from_slurm("CANCELLED by 1000") is JobState.CANCELLED
    assert JobState.from_slurm("TIMEOUT") is JobState.TIMEOUT
    assert JobState.from_slurm("OUT_OF_MEMORY") is JobState.OUT_OF_MEMORY


def test_job_state_unknown_falls_back():
    assert JobState.from_slurm("SOMETHING_WEIRD") is JobState.UNKNOWN
    assert JobState.from_slurm("") is JobState.UNKNOWN


def test_job_state_is_terminal():
    assert JobState.COMPLETED.is_terminal()
    assert JobState.FAILED.is_terminal()
    assert JobState.CANCELLED.is_terminal()
    assert JobState.TIMEOUT.is_terminal()
    assert JobState.OUT_OF_MEMORY.is_terminal()
    assert not JobState.PENDING.is_terminal()
    assert not JobState.RUNNING.is_terminal()
    assert not JobState.UNKNOWN.is_terminal()


def test_job_status_holds_state_and_exit_code():
    status = JobStatus(job_id="123", state=JobState.RUNNING, exit_code=None, raw_state="RUNNING")
    assert status.job_id == "123"
    assert status.state is JobState.RUNNING
    assert status.exit_code is None


def test_job_result_is_frozen_dataclass():
    result = JobResult(
        job_id="123",
        state=JobState.COMPLETED,
        exit_code=0,
        stdout="hello",
        stderr="",
        executed_notebook=None,
        output_files=(),
    )
    assert result.job_id == "123"
    assert result.exit_code == 0
    with pytest.raises(Exception):
        result.exit_code = 1
