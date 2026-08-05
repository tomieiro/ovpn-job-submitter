"""High-level, batteries-included notebook execution workflow."""

from __future__ import annotations

import asyncio
import getpass
import re
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Callable

from .client import DGXClient
from .errors import ConfigurationError, NotebookExecutionError
from .models import JobResult, JobState, Resources

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
    password_provider: Callable[[], str] | None = None,
) -> JobResult:
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
        client_options = (
            {"password_provider": password_provider}
            if password_provider is not None
            else {}
        )
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
            )
            print(f"Job submetido: {job.id}")
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
