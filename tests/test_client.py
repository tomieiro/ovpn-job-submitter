from pathlib import Path

import nbformat
import pytest

from dgx_slurm.client import DGXClient
from dgx_slurm.errors import BundleError, SubmissionError
from dgx_slurm.job import DGXJob
from dgx_slurm.models import Resources
from dgx_slurm.storage import LocalJobStore


def make_notebook(path):
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("print('hi')"))
    nbformat.write(nb, str(path))
    return path


class FakeVPN:
    def __init__(self):
        self.connect_calls = 0
        self.disconnect_calls = 0

    def connect(self):
        self.connect_calls += 1

    def disconnect(self):
        self.disconnect_calls += 1


class FakeTransport:
    def __init__(self, calls):
        self.connect_calls = 0
        self.close_calls = 0
        self.executed = []
        self.uploaded = []
        self._calls = calls

    def connect(self):
        self.connect_calls += 1
        self._calls.append("ssh.connect")

    def execute(self, command):
        self.executed.append(command)
        self._calls.append(f"execute:{command}")

        class Result:
            exit_code = 0
            stdout = ""
            stderr = ""

        return Result()

    def upload_directory(self, local, remote):
        self.uploaded.append((local, remote))
        self._calls.append("upload_directory")

    def close(self):
        self.close_calls += 1


class FakeScheduler:
    def __init__(self, calls, job_id="48192"):
        self._job_id = job_id
        self._calls = calls
        self.submitted_dirs = []

    def submit(self, remote_job_dir, script_name="runImage.slurm"):
        self.submitted_dirs.append(remote_job_dir)
        self._calls.append("scheduler.submit")
        return self._job_id

    def status(self, job_id):
        raise NotImplementedError


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture
def client_parts(tmp_path):
    calls = []
    vpn = FakeVPN()
    transport = FakeTransport(calls)
    scheduler = FakeScheduler(calls)
    store = LocalJobStore(tmp_path / "jobs.json")
    return calls, vpn, transport, scheduler, store


def make_client(project, tmp_path, client_parts, ovpn_file=None):
    calls, vpn, transport, scheduler, store = client_parts
    ovpn_file = ovpn_file or (tmp_path / "client.ovpn")
    if not ovpn_file.exists():
        ovpn_file.write_text("client\n")
    return DGXClient(
        ovpn=ovpn_file,
        username="cluster-user",
        vpn=vpn,
        transport=transport,
        scheduler=scheduler,
        job_store=store,
        workdir_root=tmp_path / "bundles",
        project_root=project,
    )


def test_submit_validates_notebook(project, tmp_path, client_parts):
    client = make_client(project, tmp_path, client_parts)
    bad = project / "not_a_notebook.py"
    bad.write_text("print('hi')")
    with pytest.raises(BundleError):
        client.submit(bad, resources=Resources())


def test_submit_ensures_vpn(project, tmp_path, client_parts):
    calls, vpn, *_ = client_parts
    client = make_client(project, tmp_path, client_parts)
    nb = make_notebook(project / "experiment.ipynb")
    client.submit(nb, resources=Resources())
    assert vpn.connect_calls == 1


def test_submit_opens_ssh(project, tmp_path, client_parts):
    calls, vpn, transport, *_ = client_parts
    client = make_client(project, tmp_path, client_parts)
    nb = make_notebook(project / "experiment.ipynb")
    client.submit(nb, resources=Resources())
    assert transport.connect_calls == 1


def test_submit_creates_bundle(project, tmp_path, client_parts):
    client = make_client(project, tmp_path, client_parts)
    nb = make_notebook(project / "experiment.ipynb")
    client.submit(nb, resources=Resources())
    bundle_dirs = list((tmp_path / "bundles").glob("*"))
    assert bundle_dirs
    assert (bundle_dirs[0] / "payload" / "notebook.ipynb").exists()


def test_submit_creates_remote_directory(project, tmp_path, client_parts):
    calls, vpn, transport, *_ = client_parts
    client = make_client(project, tmp_path, client_parts)
    nb = make_notebook(project / "experiment.ipynb")
    client.submit(nb, resources=Resources())
    assert any("mkdir -p" in c for c in transport.executed)


def test_submit_uploads_bundle(project, tmp_path, client_parts):
    calls, vpn, transport, *_ = client_parts
    client = make_client(project, tmp_path, client_parts)
    nb = make_notebook(project / "experiment.ipynb")
    client.submit(nb, resources=Resources())
    assert transport.uploaded


def test_submit_creates_logs_before_sbatch(project, tmp_path, client_parts):
    calls, *_ = client_parts
    client = make_client(project, tmp_path, client_parts)
    nb = make_notebook(project / "experiment.ipynb")
    client.submit(nb, resources=Resources())
    mkdir_index = next(i for i, c in enumerate(calls) if "mkdir -p" in c and "logs" in c)
    submit_index = calls.index("scheduler.submit")
    assert mkdir_index < submit_index


def test_submit_returns_dgx_job(project, tmp_path, client_parts):
    client = make_client(project, tmp_path, client_parts)
    nb = make_notebook(project / "experiment.ipynb")
    job = client.submit(nb, resources=Resources())
    assert isinstance(job, DGXJob)
    assert job.id == "48192"


def test_attach_recovers_existing_job(project, tmp_path, client_parts):
    calls, vpn, transport, scheduler, store = client_parts
    store.save("48192", {"job_name": "dgx-notebook-abc", "remote_job_dir": "dgx-slurm-jobs/dgx-notebook-abc"})
    client = make_client(project, tmp_path, client_parts)
    job = client.attach("48192")
    assert isinstance(job, DGXJob)
    assert job.id == "48192"


def test_attach_unknown_job_raises(project, tmp_path, client_parts):
    client = make_client(project, tmp_path, client_parts)
    with pytest.raises(SubmissionError):
        client.attach("does-not-exist")


def test_failure_before_sbatch_creates_no_job_record(project, tmp_path, client_parts):
    calls, vpn, transport, scheduler, store = client_parts

    def failing_submit(remote_job_dir, script_name="runImage.slurm"):
        raise RuntimeError("network blip")

    scheduler.submit = failing_submit
    client = make_client(project, tmp_path, client_parts)
    nb = make_notebook(project / "experiment.ipynb")
    with pytest.raises(RuntimeError):
        client.submit(nb, resources=Resources())
    assert store.load("48192") is None


def test_close_disconnects_own_resources(project, tmp_path, client_parts):
    calls, vpn, transport, *_ = client_parts
    client = make_client(project, tmp_path, client_parts)
    client.close()
    assert vpn.disconnect_calls == 1
    assert transport.close_calls == 1
