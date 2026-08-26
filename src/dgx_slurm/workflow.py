"""High-level, batteries-included notebook execution workflow."""

from __future__ import annotations

import asyncio
import getpass
import re
import shutil
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .client import DGXClient
from .errors import ConfigurationError, NotebookExecutionError, SubmissionError
from .models import JobResult, JobState, Resources
from .storage import LocalJobStore
from .client import DEFAULT_JOB_STORE_PATH

_IGNORED_PROJECT_NAMES = {
    ".dgx-results",
    ".git",
    ".venv",
    "__pycache__",
}
_CERT_USERNAME_RE = re.compile(r"^client-(?P<username>.+)-cert\.pem$")


def discover_ovpn(vpn_dir: Path | str) -> Path:
    """Find the single .ovpn configuration in an explicit VPN directory."""
    vpn_dir = Path(vpn_dir).expanduser().resolve()
    if not vpn_dir.is_dir():
        raise ConfigurationError(f"VPN directory not found: {vpn_dir}")

    candidates = sorted(vpn_dir.glob("*.ovpn"))
    if not candidates:
        raise ConfigurationError(f"no .ovpn found in VPN directory: {vpn_dir}")
    if len(candidates) > 1:
        raise ConfigurationError(
            f"multiple .ovpn files found in VPN directory: {vpn_dir}"
        )
    return candidates[0].resolve()


def discover_username(ovpn: Path | str) -> str:
    """Infer the cluster username from the certificate named by the config."""
    ovpn = Path(ovpn)
    try:
        lines = ovpn.read_text().splitlines()
    except OSError as exc:
        raise ConfigurationError(f"could not read OpenVPN configuration: {exc}") from exc

    for line in lines:
        fields = line.strip().split()
        if len(fields) != 2 or fields[0] != "cert":
            continue
        match = _CERT_USERNAME_RE.match(Path(fields[1]).name)
        if match:
            return match.group("username")

    return getpass.getuser()


def project_includes(
    notebook: Path | str,
    *,
    output: Path | str,
) -> tuple[Path, ...]:
    """Return safe project siblings, excluding generated/local-only content."""
    notebook = Path(notebook).resolve()
    output = Path(output).resolve()
    return tuple(
        path
        for path in sorted(notebook.parent.iterdir())
        if path != notebook
        and path.resolve() != output
        and path.name not in _IGNORED_PROJECT_NAMES
        and not path.name.endswith(".executed.ipynb")
    )


async def run_notebook_async(
    notebook: Path | str,
    *,
    include_project_files: bool,
    vpn_dir: Path | str,
    ssh_host: str,
    ssh_port: int,
    partition: str,
    gpus: int,
    cpus: int,
    memory: str,
    time_limit: str,
    username: str | None = None,
    output: Path | str | None = None,
    stream: bool = True,
    detach: bool = False,
    password_provider: Callable[[], str] | None = None,
    host_key_confirmer: Callable[[str, str], bool] | None = None,
) -> JobResult | str:
    """Submit, wait, download, and return a fully executed notebook.

    The VPN directory, SSH endpoint, and SLURM allocation are always explicit.
    The username is inferred from the certificate unless supplied.
    """
    notebook = Path(notebook).expanduser().resolve()
    if not notebook.is_file():
        raise ConfigurationError(f"notebook not found: {notebook}")

    ovpn_path = discover_ovpn(vpn_dir)
    cluster_username = username or discover_username(ovpn_path)
    output_path = (
        Path(output).expanduser().resolve()
        if output is not None
        else notebook.with_name(f"{notebook.stem}.executed.ipynb")
    )
    if output_path == notebook:
        raise ConfigurationError("output must not overwrite the source notebook")

    includes = (
        project_includes(notebook, output=output_path)
        if include_project_files
        else ()
    )
    known_hosts = Path.home() / ".ssh" / "known_hosts"
    download_root = notebook.parent / ".dgx-results"

    with tempfile.TemporaryDirectory(prefix="dgx-slurm-bundles-") as workdir:
        client_options = {}
        if password_provider is not None:
            client_options["password_provider"] = password_provider
        if host_key_confirmer is not None:
            client_options["host_key_confirmer"] = host_key_confirmer
        client = DGXClient(
            ovpn=ovpn_path,
            username=cluster_username,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            known_hosts_path=known_hosts if known_hosts.is_file() else None,
            project_root=notebook.parent,
            workdir_root=Path(workdir),
            **client_options,
        )
        try:
            job = client.submit(
                notebook,
                include=includes,
                resources=Resources(
                    gpus=gpus,
                    cpus=cpus,
                    memory=memory,
                    time_limit=time_limit,
                    partition=partition,
                ),
                metadata={
                    "notebook": str(notebook),
                    "output": str(output_path),
                    "vpn_dir": str(Path(vpn_dir).expanduser().resolve()),
                    "ssh_host": ssh_host,
                    "ssh_port": ssh_port,
                    "username": cluster_username,
                    "submitted_at": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                },
            )
            print(f"Job submetido: {job.id}")
            if detach:
                return job.id
            result = await job.wait(
                stream=stream,
                download_outputs=True,
                destination=download_root / job.id,
            )
        finally:
            client.close()

    if result.executed_notebook is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result.executed_notebook, output_path)
        result = replace(result, executed_notebook=output_path)
        print(f"Notebook executado: {output_path}")

    print(f"Estado final: {result.state.value} (exit code: {result.exit_code})")
    if result.state is not JobState.COMPLETED:
        partial = (
            f" Partial notebook: {result.executed_notebook}."
            if result.executed_notebook is not None
            else ""
        )
        raise NotebookExecutionError(
            f"job {result.job_id} ended as {result.state.value} "
            f"(exit code {result.exit_code}).{partial}"
        )
    return result


def _save_logs(result: JobResult, destination: Path) -> tuple[Path, ...]:
    """Write the job's streamed stdout and stderr next to its outputs.

    ``download_outputs`` only fetches ``outputs/``. A job that dies before
    producing any output therefore yields an empty results directory, and the
    log -- which is the only diagnostic there is -- exists solely on the console
    and vanishes when the VPN drops right after collection. For a failed job the
    log IS the artefact, so persist it.
    """
    destination.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for text, name in ((result.stdout, "job.out"), (result.stderr, "job.err")):
        path = destination / name
        path.write_text(text or "", encoding="utf-8")
        saved.append(path)
        size = len(text or "")
        print(f"Log salvo: {path} ({size} bytes)")
        if size == 0:
            print(
                f"  aviso: {name} veio vazio -- o job pode ter morrido antes de "
                "o SLURM materializar o arquivo, ou o caminho remoto de log "
                "difere do esperado"
            )
    return tuple(saved)


async def collect_results_async(
    job_id: str,
    *,
    password_provider: Callable[[], str] | None = None,
    host_key_confirmer: Callable[[str, str], bool] | None = None,
    save_logs: bool = False,
    keep_remote: int = 4,
) -> JobResult:
    """Reconnect once a detached job has finished and download its outputs."""
    if not job_id.isdigit():
        raise ConfigurationError(f"job id must be numeric, got {job_id!r}")

    store = LocalJobStore(DEFAULT_JOB_STORE_PATH)
    record = store.load(job_id)
    if record is None:
        raise SubmissionError(f"no local record found for job {job_id}")
    required = {"vpn_dir", "ssh_host", "ssh_port", "notebook", "output"}
    missing = sorted(required.difference(record))
    if missing:
        raise SubmissionError(
            f"job {job_id} predates asynchronous collection metadata; "
            f"missing: {', '.join(missing)}"
        )

    ovpn_path = discover_ovpn(record["vpn_dir"])
    known_hosts = Path.home() / ".ssh" / "known_hosts"
    client_options = {}
    if password_provider is not None:
        client_options["password_provider"] = password_provider
    if host_key_confirmer is not None:
        client_options["host_key_confirmer"] = host_key_confirmer
    client = DGXClient(
        ovpn=ovpn_path,
        username=discover_username(ovpn_path),
        ssh_host=record["ssh_host"],
        ssh_port=int(record["ssh_port"]),
        known_hosts_path=known_hosts if known_hosts.is_file() else None,
        **client_options,
    )
    try:
        job = client.attach(job_id)
        status = job.status()
        if status.state in {JobState.PENDING, JobState.RUNNING}:
            raise SubmissionError(
                f"job {job_id} is {status.state.value}; try collecting again later"
            )
        notebook = Path(record["notebook"])
        output_path = Path(record["output"])
        destination = notebook.parent / ".dgx-results" / job_id
        if status.state is JobState.UNKNOWN:
            print(
                f"Job {job_id} is no longer available in SLURM accounting; "
                "collecting its remote outputs directly."
            )
            result = job.collect_available(
                status, destination=destination, stream=True
            )
        else:
            result = await job.wait(
                stream=True,
                download_outputs=True,
                destination=destination,
            )
        if keep_remote > 0:
            removidos = client.prune_remote_jobs(
                keep=keep_remote, protect=record.get("job_name")
            )
            if removidos:
                print(
                    f"Removidos {len(removidos)} diretorios antigos no cluster "
                    f"(mantidos os {keep_remote} mais recentes):"
                )
                for nome in removidos:
                    print(f"  {nome}")
    finally:
        client.close()

    if save_logs:
        _save_logs(result, destination)

    if result.executed_notebook is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result.executed_notebook, output_path)
        result = replace(result, executed_notebook=output_path)
        print(f"Notebook executado: {output_path}")
    print(f"Estado final: {result.state.value} (exit code: {result.exit_code})")
    return result


def run_notebook(
    notebook: Path | str,
    *,
    include_project_files: bool,
    vpn_dir: Path | str,
    ssh_host: str,
    ssh_port: int,
    partition: str,
    gpus: int,
    cpus: int,
    memory: str,
    time_limit: str,
    **kwargs,
) -> JobResult:
    """Synchronous convenience wrapper around :func:`run_notebook_async`."""
    return asyncio.run(
        run_notebook_async(
            notebook,
            include_project_files=include_project_files,
            vpn_dir=vpn_dir,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            partition=partition,
            gpus=gpus,
            cpus=cpus,
            memory=memory,
            time_limit=time_limit,
            **kwargs,
        )
    )


def collect_results(job_id: str, **kwargs) -> JobResult:
    """Synchronous wrapper around :func:`collect_results_async`."""
    return asyncio.run(collect_results_async(job_id, **kwargs))
