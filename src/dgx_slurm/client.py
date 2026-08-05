"""DGXClient: the public entry point for submitting notebooks to the DGX."""

from __future__ import annotations

import getpass
import shlex
import socket
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Sequence

from .bundle import NotebookBundleBuilder
from .errors import SubmissionError
from .job import DGXJob
from .models import Resources
from .slurm import SlurmScheduler
from .ssh import SSHTransport
from .storage import LocalJobStore
from .vpn import VPNConnection

DEFAULT_REMOTE_BASE_DIR = "dgx-slurm-jobs"
DEFAULT_JOB_STORE_PATH = Path.home() / ".dgx-slurm" / "jobs.json"


def _default_is_reachable(host: str, port: int) -> Callable[[], bool]:
    def check() -> bool:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            return False

    return check


class DGXClient:
    """Submits notebooks to a SLURM-managed DGX cluster over OpenVPN/SSH."""

    def __init__(
        self,
        *,
        ovpn: Path | str,
        username: str,
        ssh_host: str,
        ssh_port: int,
        known_hosts_path: Path | str | None = None,
        sudo_openvpn: bool = True,
        vpn: VPNConnection | None = None,
        transport: SSHTransport | None = None,
        bundle_builder: NotebookBundleBuilder | None = None,
        scheduler: SlurmScheduler | None = None,
        job_store: LocalJobStore | None = None,
        password_provider: Callable[[], str] = getpass.getpass,
        host_key_confirmer: Callable[[str, str], bool] | None = None,
        print_fn: Callable[[str], None] = print,
        workdir_root: Path | None = None,
        project_root: Path | None = None,
        job_name_factory: Callable[[], str] = lambda: f"dgx-notebook-{uuid.uuid4().hex[:8]}",
    ) -> None:
        self._ovpn_path = Path(ovpn)
        self._username = username
        self._ssh_host = ssh_host
        self._ssh_port = ssh_port
        self._known_hosts_path = known_hosts_path
        self._sudo_openvpn = sudo_openvpn
        self._vpn = vpn
        self._transport = transport
        self._bundle_builder = bundle_builder or NotebookBundleBuilder()
        self._scheduler = scheduler
        self._job_store = job_store or LocalJobStore(DEFAULT_JOB_STORE_PATH)
        self._password_provider = password_provider
        self._host_key_confirmer = host_key_confirmer
        self._print_fn = print_fn
        self._workdir_root = Path(workdir_root) if workdir_root else Path(tempfile.gettempdir())
        self._project_root = Path(project_root) if project_root else Path.cwd()
        self._job_name_factory = job_name_factory

        self._password: str | None = None
        self._connected = False

    def submit(
        self,
        notebook: Path | str,
        *,
        resources: Resources | None = None,
        include: Sequence[Path | str] = (),
    ) -> DGXJob:
        resources = resources or Resources()
        notebook = Path(notebook)

        self._bundle_builder.validate_notebook(notebook)

        self._ensure_connected()

        job_name = self._job_name_factory()
        remote_job_dir = f"{DEFAULT_REMOTE_BASE_DIR}/{job_name}"

        self._print_fn("Empacotando o notebook e os arquivos incluídos...")
        bundle_root = self._bundle_builder.build(
            notebook=notebook,
            job_name=job_name,
            resources=resources,
            include=include,
            workdir=self._workdir_root / f"dgx-slurm-bundle-{job_name}",
            project_root=self._project_root,
        )

        self._transport.execute(f"mkdir -p {shlex.quote(remote_job_dir)}/logs")
        self._transport.execute(f"mkdir -p {shlex.quote(remote_job_dir)}/outputs")
        self._transport.upload_directory(bundle_root, remote_job_dir)

        self._print_fn("Submetendo ao SLURM...")
        job_id = self._scheduler.submit(remote_job_dir)

        self._job_store.save(
            job_id, {"job_name": job_name, "remote_job_dir": remote_job_dir}
        )

        return DGXJob(
            job_id=job_id,
            job_name=job_name,
            remote_job_dir=remote_job_dir,
            scheduler=self._scheduler,
            transport=self._transport,
        )

    def attach(self, job_id: str) -> DGXJob:
        record = self._job_store.load(job_id)
        if record is None:
            raise SubmissionError(f"no local record found for job {job_id}")

        self._ensure_connected()

        return DGXJob(
            job_id=job_id,
            job_name=record["job_name"],
            remote_job_dir=record["remote_job_dir"],
            scheduler=self._scheduler,
            transport=self._transport,
        )

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()
        if self._vpn is not None:
            self._vpn.disconnect()
        self._connected = False

    def _ensure_connected(self) -> None:
        if self._connected:
            return

        if self._vpn is None:
            self._vpn = VPNConnection(
                ovpn_path=self._ovpn_path,
                username=self._username,
                password=self._get_password(),
                is_reachable=_default_is_reachable(
                    self._ssh_host, self._ssh_port
                ),
                use_sudo=self._sudo_openvpn,
            )
        self._print_fn("Conectando à VPN, se necessário...")
        self._vpn.connect()

        self._print_fn(f"Conectando a {self._ssh_host} por SSH...")
        if self._transport is None:
            self._transport = SSHTransport(
                host=self._ssh_host,
                port=self._ssh_port,
                username=self._username,
                password=self._get_password(),
                known_hosts_path=self._known_hosts_path,
                host_key_confirmer=self._host_key_confirmer,
                print_fn=self._print_fn,
            )
        self._transport.connect()

        if self._scheduler is None:
            self._scheduler = SlurmScheduler(self._transport)

        self._connected = True

    def _get_password(self) -> str:
        if self._password is None:
            self._password = self._password_provider()
        return self._password
