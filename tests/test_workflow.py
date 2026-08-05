import inspect
from pathlib import Path

import nbformat
import pytest

from dgx_slurm.errors import ConfigurationError, NotebookExecutionError
from dgx_slurm.models import JobResult, JobState, Resources
from dgx_slurm.workflow import (
    discover_ovpn,
    discover_username,
    project_includes,
    run_notebook,
    run_notebook_async,
)


def required_job_args(vpn_dir, *, include_project_files=True):
    return {
        "include_project_files": include_project_files,
        "vpn_dir": vpn_dir,
        "ssh_host": "c4aiscm2",
        "ssh_port": 22,
        "partition": "devwork",
        "gpus": 1,
        "cpus": 8,
        "memory": "0",
        "time_limit": "04:00:00",
    }


def test_infrastructure_and_resources_are_required():
    for function in (run_notebook, run_notebook_async):
        signature = inspect.signature(function)
        for name in (
            "include_project_files",
            "vpn_dir",
            "ssh_host",
            "ssh_port",
            "partition",
            "gpus",
            "cpus",
            "memory",
            "time_limit",
        ):
            assert signature.parameters[name].default is inspect.Parameter.empty


def make_layout(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    notebook = project / "experiment.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[nbformat.v4.new_code_cell("print('ok')")]
        ),
        notebook,
    )
    ssh_dir = tmp_path / "SSH"
    ssh_dir.mkdir()
    ovpn = ssh_dir / "c4ai.icmc.usp.br.ovpn"
    ovpn.write_text("client\ncert client-mtomieiro-cert.pem\n")
    return project, notebook, ovpn


def test_discovers_ovpn_in_explicit_vpn_directory(tmp_path):
    _, _, ovpn = make_layout(tmp_path)
    assert discover_ovpn(ovpn.parent) == ovpn


def test_discover_ovpn_requires_unambiguous_config(tmp_path):
    make_layout(tmp_path)
    (tmp_path / "SSH" / "second.ovpn").write_text("client\n")
    with pytest.raises(ConfigurationError, match="multiple"):
        discover_ovpn(tmp_path / "SSH")


def test_discover_ovpn_rejects_missing_directory(tmp_path):
    with pytest.raises(ConfigurationError, match="not found"):
        discover_ovpn(tmp_path / "missing")


def test_discovers_username_from_certificate_directive(tmp_path):
    _, _, ovpn = make_layout(tmp_path)
    assert discover_username(ovpn) == "mtomieiro"


def test_project_includes_only_source_siblings(tmp_path):
    project, notebook, _ = make_layout(tmp_path)
    data = project / "data.nc"
    data.write_text("data")
    (project / ".dgx-results").mkdir()
    old_result = project / "old.executed.ipynb"
    old_result.write_text("{}")
    output = project / "experiment.executed.ipynb"

    assert project_includes(notebook, output=output) == (data,)


class FakeJob:
    id = "48192"

    def __init__(self, state):
        self.state = state
        self.include = None
        self.resources = None

    async def wait(self, *, destination, **_kwargs):
        destination = Path(destination)
        destination.mkdir(parents=True)
        executed = destination / "notebook.executed.ipynb"
        executed.write_text('{"cells": []}')
        return JobResult(
            job_id=self.id,
            state=self.state,
            exit_code=0 if self.state is JobState.COMPLETED else 1,
            stdout="",
            stderr="",
            executed_notebook=executed,
        )


class FakeClient:
    instances = []
    final_state = JobState.COMPLETED

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.job = FakeJob(self.final_state)
        self.instances.append(self)

    def submit(self, _notebook, *, include, resources):
        self.job.include = include
        self.job.resources = resources
        return self.job

    def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_high_level_workflow_owns_connection_bundle_and_result(
    tmp_path, monkeypatch
):
    project, notebook, ovpn = make_layout(tmp_path)
    data = project / "data.nc"
    data.write_text("data")
    monkeypatch.setattr("dgx_slurm.workflow.DGXClient", FakeClient)
    FakeClient.instances.clear()
    FakeClient.final_state = JobState.COMPLETED

    result = await run_notebook_async(
        notebook,
        **required_job_args(ovpn.parent),
    )

    client = FakeClient.instances[-1]
    assert client.kwargs["ovpn"] == ovpn
    assert client.kwargs["username"] == "mtomieiro"
    assert client.kwargs["ssh_host"] == "c4aiscm2"
    assert client.job.include == (data,)
    assert client.job.resources == Resources(
        gpus=1,
        cpus=8,
        memory="0",
        time_limit="04:00:00",
        partition="devwork",
    )
    assert client.closed is True
    assert result.executed_notebook == project / "experiment.executed.ipynb"
    assert result.executed_notebook.exists()


@pytest.mark.asyncio
async def test_high_level_workflow_can_skip_project_files(tmp_path, monkeypatch):
    _, notebook, ovpn = make_layout(tmp_path)
    monkeypatch.setattr("dgx_slurm.workflow.DGXClient", FakeClient)
    FakeClient.instances.clear()
    FakeClient.final_state = JobState.COMPLETED

    await run_notebook_async(
        notebook,
        **required_job_args(ovpn.parent, include_project_files=False),
    )

    assert FakeClient.instances[-1].job.include == ()


@pytest.mark.asyncio
async def test_password_provider_reaches_the_client(tmp_path, monkeypatch):
    """The GUI has no console, so getpass must be replaceable."""
    _, notebook, ovpn = make_layout(tmp_path)
    monkeypatch.setattr("dgx_slurm.workflow.DGXClient", FakeClient)
    FakeClient.instances.clear()
    FakeClient.final_state = JobState.COMPLETED

    await run_notebook_async(
        notebook,
        **required_job_args(ovpn.parent),
        password_provider=lambda: "from-the-window",
    )

    assert FakeClient.instances[-1].kwargs["password_provider"]() == "from-the-window"


@pytest.mark.asyncio
async def test_client_keeps_its_own_password_default_without_provider(
    tmp_path, monkeypatch
):
    _, notebook, ovpn = make_layout(tmp_path)
    monkeypatch.setattr("dgx_slurm.workflow.DGXClient", FakeClient)
    FakeClient.instances.clear()
    FakeClient.final_state = JobState.COMPLETED

    await run_notebook_async(notebook, **required_job_args(ovpn.parent))

    assert "password_provider" not in FakeClient.instances[-1].kwargs


@pytest.mark.asyncio
async def test_high_level_workflow_returns_partial_notebook_then_raises(
    tmp_path, monkeypatch
):
    project, notebook, ovpn = make_layout(tmp_path)
    monkeypatch.setattr("dgx_slurm.workflow.DGXClient", FakeClient)
    FakeClient.instances.clear()
    FakeClient.final_state = JobState.FAILED

    with pytest.raises(NotebookExecutionError, match="FAILED"):
        await run_notebook_async(
            notebook,
            **required_job_args(ovpn.parent),
        )

    assert (project / "experiment.executed.ipynb").exists()
