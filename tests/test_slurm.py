import pytest

from dgx_slurm.errors import SchedulerError
from dgx_slurm.models import JobState
from dgx_slurm.slurm import SlurmScheduler
from dgx_slurm.ssh import RemoteCommandResult


class FakeTransport:
    def __init__(self):
        self.commands = []
        self.responses = {}  # substring -> (stdout, stderr, exit_code)
        self.default_response = ("", "", 0)

    def execute(self, command):
        self.commands.append(command)
        for substring, response in self.responses.items():
            if substring in command:
                stdout, stderr, exit_code = response
                return RemoteCommandResult(
                    command=command, exit_code=exit_code, stdout=stdout, stderr=stderr
                )
        stdout, stderr, exit_code = self.default_response
        return RemoteCommandResult(
            command=command, exit_code=exit_code, stdout=stdout, stderr=stderr
        )


@pytest.fixture
def transport():
    return FakeTransport()


@pytest.fixture
def scheduler(transport):
    return SlurmScheduler(transport)


def test_sbatch_parsable_returns_job_id(scheduler, transport):
    transport.responses["sbatch --parsable"] = ("48192\n", "", 0)
    job_id = scheduler.submit("/remote/job-42")
    assert job_id == "48192"
    assert any("sbatch --parsable" in c for c in transport.commands)
    assert any("cd /remote/job-42" in c for c in transport.commands)


def test_sbatch_failure_raises_scheduler_error(scheduler, transport):
    transport.responses["sbatch --parsable"] = (
        "",
        "sbatch: error: invalid partition",
        1,
    )
    with pytest.raises(SchedulerError):
        scheduler.submit("/remote/job-42")


def test_sbatch_unexpected_output_raises_scheduler_error(scheduler, transport):
    transport.responses["sbatch --parsable"] = ("not-a-job-id\n", "", 0)
    with pytest.raises(SchedulerError):
        scheduler.submit("/remote/job-42")


def test_squeue_returns_pending(scheduler, transport):
    transport.responses["squeue"] = ("PENDING\n", "", 0)
    status = scheduler.status("48192")
    assert status.state is JobState.PENDING
    assert status.job_id == "48192"


def test_squeue_returns_running(scheduler, transport):
    transport.responses["squeue"] = ("RUNNING\n", "", 0)
    status = scheduler.status("48192")
    assert status.state is JobState.RUNNING


def test_sacct_returns_completed_when_job_left_queue(scheduler, transport):
    transport.responses["squeue"] = ("", "", 0)
    transport.responses["sacct"] = ("48192|COMPLETED|0:0\n48192.batch|COMPLETED|0:0\n", "", 0)
    status = scheduler.status("48192")
    assert status.state is JobState.COMPLETED
    assert status.exit_code == 0


def test_sacct_returns_failed(scheduler, transport):
    transport.responses["squeue"] = ("", "", 0)
    transport.responses["sacct"] = ("48192|FAILED|1:0\n", "", 0)
    status = scheduler.status("48192")
    assert status.state is JobState.FAILED


def test_sacct_returns_exit_code(scheduler, transport):
    transport.responses["squeue"] = ("", "", 0)
    transport.responses["sacct"] = ("48192|FAILED|3:0\n", "", 0)
    status = scheduler.status("48192")
    assert status.exit_code == 3


def test_status_unknown_when_no_squeue_or_sacct_record(scheduler, transport):
    transport.responses["squeue"] = ("", "", 0)
    transport.responses["sacct"] = ("", "", 0)
    status = scheduler.status("48192")
    assert status.state is JobState.UNKNOWN


def test_scancel_receives_correct_job_id(scheduler, transport):
    transport.responses["scancel"] = ("", "", 0)
    scheduler.cancel("48192")
    assert any(c == "scancel 48192" for c in transport.commands)


def test_scancel_failure_raises_scheduler_error(scheduler, transport):
    transport.responses["scancel"] = ("", "scancel: error: Invalid job id", 1)
    with pytest.raises(SchedulerError):
        scheduler.cancel("48192")


def test_unrecognized_squeue_state_maps_to_unknown(scheduler, transport):
    transport.responses["squeue"] = ("GARBAGE STATE\n", "", 0)
    status = scheduler.status("48192")
    assert status.state is JobState.UNKNOWN
