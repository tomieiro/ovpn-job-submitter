"""SSH/SFTP transport to the DGX login node, built on Paramiko.

SSHTransport only ever executes commands the library itself constructs
(sbatch, squeue, sacct, mkdir, scancel, ...). It never forwards
notebook-supplied text to the remote shell.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import paramiko

from .errors import SSHError


@dataclass(frozen=True)
class RemoteCommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class RemoteFileChunk:
    text: str
    new_offset: int


ClientFactory = Callable[[], "paramiko.SSHClient"]


class SSHTransport:
    """A reusable SSH + SFTP connection to a single remote host."""

    def __init__(
        self,
        *,
        host: str,
        username: str,
        port: int = 22,
        password: str | None = None,
        key_filename: str | None = None,
        known_hosts_path: Path | str | None = None,
        client_factory: ClientFactory = paramiko.SSHClient,
        connect_timeout: float = 30.0,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._key_filename = key_filename
        self._known_hosts_path = str(known_hosts_path) if known_hosts_path else None
        self._client_factory = client_factory
        self._connect_timeout = connect_timeout

        self._client: paramiko.SSHClient | None = None
        self._sftp = None

    def connect(self) -> None:
        client = self._client_factory()
        client.load_system_host_keys()
        if self._known_hosts_path:
            client.load_host_keys(self._known_hosts_path)
        client.set_missing_host_key_policy(paramiko.RejectPolicy())

        try:
            client.connect(
                hostname=self._host,
                port=self._port,
                username=self._username,
                password=self._password,
                key_filename=self._key_filename,
                timeout=self._connect_timeout,
            )
        except paramiko.SSHException as exc:
            raise SSHError(f"SSH connection to {self._host} failed: {exc}") from exc
        except OSError as exc:
            raise SSHError(f"SSH connection to {self._host} failed: {exc}") from exc

        self._client = client
        self._sftp = client.open_sftp()

    def execute(self, command: str) -> RemoteCommandResult:
        self._require_connected()
        _stdin, stdout, stderr = self._client.exec_command(command)
        exit_code = stdout.channel.recv_exit_status()
        return RemoteCommandResult(
            command=command,
            exit_code=exit_code,
            stdout=stdout.read().decode("utf-8", errors="replace"),
            stderr=stderr.read().decode("utf-8", errors="replace"),
        )

    def upload_directory(self, local: Path, remote: str) -> None:
        self._require_connected()
        local = Path(local)
        self._mkdir_remote(remote)
        for root, dirnames, filenames in os.walk(local):
            root_path = Path(root)
            relative_root = root_path.relative_to(local)
            remote_root = self._remote_join(remote, relative_root)
            for dirname in dirnames:
                self._mkdir_remote(self._remote_join(remote_root, dirname))
            for filename in filenames:
                local_file = root_path / filename
                remote_file = f"{remote_root}/{filename}" if remote_root else filename
                self._sftp.put(str(local_file), remote_file)

    def download_directory(self, remote: str, local: Path) -> None:
        self._require_connected()
        local = Path(local)
        local.mkdir(parents=True, exist_ok=True)
        for entry in self._sftp.listdir_attr(remote):
            remote_path = f"{remote}/{entry.filename}"
            local_path = local / entry.filename
            if stat.S_ISDIR(entry.st_mode):
                self.download_directory(remote_path, local_path)
            else:
                self._sftp.get(remote_path, str(local_path))

    def read_from(self, remote_file: str, offset: int) -> RemoteFileChunk:
        self._require_connected()
        try:
            with self._sftp.open(remote_file, "rb") as handle:
                handle.seek(offset)
                data = handle.read()
        except (FileNotFoundError, OSError):
            return RemoteFileChunk(text="", new_offset=offset)
        text = data.decode("utf-8", errors="replace")
        return RemoteFileChunk(text=text, new_offset=offset + len(data))

    def close(self) -> None:
        if self._sftp is not None:
            self._sftp.close()
            self._sftp = None
        if self._client is not None:
            self._client.close()
            self._client = None

    def _mkdir_remote(self, remote_path: str) -> None:
        try:
            self._sftp.mkdir(remote_path)
        except OSError:
            pass  # already exists

    @staticmethod
    def _remote_join(base: str, relative: Path | str) -> str:
        relative_str = str(relative)
        if relative_str in ("", "."):
            return base
        return f"{base}/{relative_str}"

    def _require_connected(self) -> None:
        if self._client is None or self._sftp is None:
            raise SSHError("SSHTransport.connect() must be called before use")
