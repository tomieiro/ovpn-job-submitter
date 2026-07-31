import asyncio
from pathlib import Path

import pytest

from dgx_slurm.job import DGXJob
from dgx_slurm.models import JobState, JobStatus
from dgx_slurm.ssh import RemoteFileChunk


class ScriptedScheduler:
    def __init__(self, states):
        self._states = list(states)
        self.cancel_calls = []

    def status(self, job_id):
        state = self._states.pop(0) if len(self._states) > 1 else self._states[0]
        return JobStatus(job_id=job_id, state=state, exit_code=0 if state.is_terminal() else None, raw_state=state.value)

    def cancel(self, job_id):
        self.cancel_calls.append(job_id)


class FakeTransport:
    def __init__(self):
        self.logs = {}
        self.downloaded = []

    def read_from(self, remote_file, offset):
        content = self.logs.get(remote_file, "")
        chunk = content[offset:]
        return RemoteFileChunk(text=chunk, new_offset=offset + len(chunk))

    def download_directory(self, remote, local):
        self.downloaded.append((remote, local))
        local = Path(local)
        local.mkdir(parents=True, exist_ok=True)
        (local / "notebook.executed.ipynb").write_text("{}")
        (local / "result.txt").write_text("done")


async def fake_sleep(_seconds):
    return None


def make_job(states, transport=None, remote_job_dir="/remote/job-48192"):
    transport = transport or FakeTransport()
    scheduler = ScriptedScheduler(states)
    job = DGXJob(
        job_id="48192",
        job_name="dgx-notebook",
        remote_job_dir=remote_job_dir,
        scheduler=scheduler,
        transport=transport,
        sleep=fake_sleep,
        print_fn=lambda *_a, **_k: None,
    )
    return job, scheduler, transport


def test_status_returns_current_job_status():
    job, scheduler, _ = make_job([JobState.RUNNING])
    status = job.status()
    assert status.state is JobState.RUNNING
    assert status.job_id == "48192"


@pytest.mark.asyncio
async def test_wait_completes_on_terminal_state():
    job, _, _ = make_job([JobState.PENDING, JobState.RUNNING, JobState.COMPLETED])
    result = await job.wait(download_outputs=False)
    assert result.state is JobState.COMPLETED
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_wait_prints_state_transitions():
    printed = []
    transport = FakeTransport()
    scheduler = ScriptedScheduler([JobState.PENDING, JobState.RUNNING, JobState.COMPLETED])
    job = DGXJob(
        job_id="48192",
        job_name="dgx-notebook",
        remote_job_dir="/remote/job-48192",
        scheduler=scheduler,
        transport=transport,
        sleep=fake_sleep,
        print_fn=lambda msg: printed.append(msg),
    )
    await job.wait(download_outputs=False)
    joined = "\n".join(printed)
    assert "PENDING" in joined
    assert "RUNNING" in joined
    assert "COMPLETED" in joined


@pytest.mark.asyncio
async def test_wait_streams_stdout_incrementally():
    transport = FakeTransport()
    transport.logs["/remote/job-48192/logs/dgx-notebook-48192.out"] = "first chunk"
    job, scheduler, _ = make_job([JobState.RUNNING, JobState.COMPLETED], transport=transport)
    result = await job.wait(download_outputs=False)
    assert "first chunk" in result.stdout


@pytest.mark.asyncio
async def test_wait_does_not_duplicate_logs():
    transport = FakeTransport()
    transport.logs["/remote/job-48192/logs/dgx-notebook-48192.out"] = "abc"
    job, scheduler, _ = make_job(
        [JobState.RUNNING, JobState.RUNNING, JobState.COMPLETED], transport=transport
    )
    result = await job.wait(download_outputs=False)
    assert result.stdout == "abc"


@pytest.mark.asyncio
async def test_unknown_temporary_state_does_not_end_wait():
    job, _, _ = make_job([JobState.RUNNING, JobState.UNKNOWN, JobState.RUNNING, JobState.COMPLETED])
    result = await job.wait(download_outputs=False)
    assert result.state is JobState.COMPLETED


@pytest.mark.asyncio
async def test_failed_preserves_stderr():
    transport = FakeTransport()
    transport.logs["/remote/job-48192/logs/dgx-notebook-48192.err"] = "traceback here"
    job, _, _ = make_job([JobState.RUNNING, JobState.FAILED], transport=transport)
    result = await job.wait(download_outputs=False)
    assert result.state is JobState.FAILED
    assert "traceback here" in result.stderr


def test_cancel_calls_scheduler_cancel():
    job, scheduler, _ = make_job([JobState.RUNNING])
    job.cancel()
    assert scheduler.cancel_calls == ["48192"]


@pytest.mark.asyncio
async def test_outputs_are_downloaded(tmp_path):
    transport = FakeTransport()
    job, _, _ = make_job([JobState.COMPLETED], transport=transport)
    result = await job.wait(download_outputs=True, destination=tmp_path / "results")
    assert transport.downloaded == [
        ("/remote/job-48192/outputs", tmp_path / "results")
    ]
    assert result.executed_notebook == tmp_path / "results" / "notebook.executed.ipynb"
    assert (tmp_path / "results" / "result.txt") in result.output_files


def test_download_outputs_method(tmp_path):
    job, _, transport = make_job([JobState.COMPLETED])
    files = job.download_outputs(tmp_path / "dl")
    assert (tmp_path / "dl" / "notebook.executed.ipynb") in files or (
        tmp_path / "dl" / "notebook.executed.ipynb"
    ).exists()
