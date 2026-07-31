"""End-to-end flow test using an in-process fake "remote" filesystem instead
of a real DGX cluster: submit -> build bundle -> fake upload -> fake sbatch
-> PENDING -> RUNNING -> logs appear -> COMPLETED -> download outputs.
"""

from pathlib import Path

import nbformat
import pytest

from dgx_slurm.client import DGXClient
from dgx_slurm.models import JobState, Resources
from dgx_slurm.ssh import RemoteCommandResult, RemoteFileChunk
from dgx_slurm.storage import LocalJobStore


def make_notebook(path):
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("print('CUDA available: True')"))
    nbformat.write(nb, str(path))
    return path


class FakeRemoteFilesystemTransport:
    """Treats a local directory as the remote filesystem SSH would reach."""

    def __init__(self, remote_root: Path):
        self._remote_root = Path(remote_root)
        self.connected = False
        self.closed = False

    def connect(self):
        self.connected = True

    def execute(self, command: str) -> RemoteCommandResult:
        if command.startswith("mkdir -p"):
            target = command.split("mkdir -p", 1)[1].strip().strip("'\"")
            (self._remote_root / target).mkdir(parents=True, exist_ok=True)
            return RemoteCommandResult(command=command, exit_code=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command in integration fake: {command}")

    def upload_directory(self, local: Path, remote: str) -> None:
        import shutil

        destination = self._remote_root / remote
        shutil.copytree(local, destination, dirs_exist_ok=True)

    def download_directory(self, remote: str, local: Path) -> None:
        import shutil

        source = self._remote_root / remote
        shutil.copytree(source, local, dirs_exist_ok=True)

    def read_from(self, remote_file: str, offset: int) -> RemoteFileChunk:
        path = self._remote_root / remote_file
        if not path.exists():
            return RemoteFileChunk(text="", new_offset=offset)
        data = path.read_bytes()[offset:]
        return RemoteFileChunk(text=data.decode(), new_offset=offset + len(data))

    def close(self):
        self.closed = True


class FakeVPN:
    def __init__(self):
        self.connect_calls = 0

    def connect(self):
        self.connect_calls += 1

    def disconnect(self):
        pass


class SimulatedSlurmScheduler:
    """Simulates a job that goes PENDING -> RUNNING -> COMPLETED, writing
    logs and outputs into the fake remote filesystem as it progresses,
    the same way the real runImage.slurm script would on the compute node.
    """

    def __init__(self, remote_root: Path):
        self._remote_root = remote_root
        self._job_id = "48192"
        self._job_name = None
        self._remote_job_dir = None
        self._sequence = [JobState.PENDING, JobState.RUNNING, JobState.RUNNING, JobState.COMPLETED]
        self._step = 0

    def submit(self, remote_job_dir: str, script_name: str = "runImage.slurm") -> str:
        self._remote_job_dir = remote_job_dir
        self._job_name = Path(remote_job_dir).name
        return self._job_id

    def status(self, job_id: str):
        from dgx_slurm.models import JobStatus

        state = self._sequence[min(self._step, len(self._sequence) - 1)]
        self._step += 1

        job_dir = self._remote_root / self._remote_job_dir
        out_log = job_dir / "logs" / f"{self._job_name}-{self._job_id}.out"

        if state is JobState.RUNNING:
            with out_log.open("a") as fh:
                fh.write(f"=== node: compute-node ===\n")
                fh.write("[Cell 1/1] START\n")
                fh.write("CUDA available: True\n")
                fh.write("[Cell 1/1] DONE\n")
        elif state is JobState.COMPLETED:
            with out_log.open("a") as fh:
                fh.write("Notebook completed successfully.\n")
                fh.write("Done (image will be removed from the node on exit).\n")
            outputs_dir = job_dir / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            (outputs_dir / "notebook.executed.ipynb").write_text("{}")
            (outputs_dir / "result.txt").write_text("0.9234")

        exit_code = 0 if state is JobState.COMPLETED else None
        return JobStatus(job_id=job_id, state=state, exit_code=exit_code, raw_state=state.value)

    def cancel(self, job_id):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_full_submit_and_wait_flow(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    nb_path = make_notebook(project / "experiment.ipynb")

    remote_root = tmp_path / "fake-remote"
    remote_root.mkdir()

    transport = FakeRemoteFilesystemTransport(remote_root)
    vpn = FakeVPN()
    scheduler = SimulatedSlurmScheduler(remote_root)
    store = LocalJobStore(tmp_path / "jobs.json")

    ovpn_file = tmp_path / "client.ovpn"
    ovpn_file.write_text("client\n")

    client = DGXClient(
        ovpn=ovpn_file,
        username="cluster-user",
        vpn=vpn,
        transport=transport,
        scheduler=scheduler,
        job_store=store,
        workdir_root=tmp_path / "bundles",
        project_root=project,
    )

    job = client.submit(
        nb_path,
        resources=Resources(gpus=1, cpus=4, memory="0", time_limit="00:20:00", partition="devwork"),
    )

    assert vpn.connect_calls == 1
    assert transport.connected is True

    printed = []
    job._print_fn = printed.append

    result = await job.wait(
        stream=True, poll_interval=0.0, download_outputs=True, destination=tmp_path / "results"
    )

    assert result.job_id == "48192"
    assert result.state is JobState.COMPLETED
    assert result.exit_code == 0
    assert "CUDA available: True" in result.stdout
    assert "Notebook completed successfully" in result.stdout
    assert result.executed_notebook is not None
    assert result.executed_notebook.exists()
    assert any(f.name == "result.txt" for f in result.output_files)

    assert store.load("48192") is not None
