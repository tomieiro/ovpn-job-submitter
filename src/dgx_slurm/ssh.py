"""SSH/SFTP transport to the DGX login node, built on Paramiko.

SSHTransport only ever executes commands the library itself constructs
(sbatch, squeue, sacct, mkdir, scancel, ...). It never forwards
notebook-supplied text to the remote shell.
"""

from __future__ import annotations

import base64
import hashlib
import os
import socket
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import paramiko

from .errors import SSHError

DEFAULT_KNOWN_HOSTS = Path.home() / ".ssh" / "known_hosts"


def format_size(num_bytes: float) -> str:
    """Sizes readable enough to tell a stalled transfer from a slow one."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024 or unit == "GB":
            precision = 0 if unit == "B" else 1
            return f"{num_bytes:.{precision}f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def format_duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


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
HostKeyConfirmer = Callable[[str, str], bool]


def format_fingerprint(key: paramiko.PKey) -> str:
    """Render a host key the way OpenSSH shows it when asking for trust."""
    digest = hashlib.sha256(key.asbytes()).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


def fetch_host_key(
    host: str,
    port: int,
    *,
    timeout: float = 30.0,
    transport_factory: Callable[..., paramiko.Transport] = paramiko.Transport,
) -> paramiko.PKey:
    """Read the key the server presents, without authenticating to it."""
    try:
        sock = socket.create_connection((host, port), timeout)
    except OSError as exc:
        raise SSHError(f"could not reach {host}:{port} to read its key: {exc}") from exc

    transport = transport_factory(sock)
    try:
        transport.start_client(timeout=timeout)
        key = transport.get_remote_server_key()
    except paramiko.SSHException as exc:
        raise SSHError(f"could not read the host key of {host}: {exc}") from exc
    finally:
        transport.close()
    return key


def remember_host_key(
    known_hosts_path: Path | str, host: str, port: int, key: paramiko.PKey
) -> None:
    """Append one accepted key, leaving every existing entry untouched."""
    path = Path(known_hosts_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        path.parent.chmod(0o700)

    entry = host if port == 22 else f"[{host}]:{port}"
    line = f"{entry} {key.get_name()} {key.get_base64()}\n"
    existing = path.read_text() if path.is_file() else ""
    separator = "" if not existing or existing.endswith("\n") else "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{separator}{line}")


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
        host_key_confirmer: HostKeyConfirmer | None = None,
        key_fetcher: Callable[[str, int], paramiko.PKey] = fetch_host_key,
        print_fn: Callable[[str], None] = print,
        progress_interval: float = 5.0,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._key_filename = key_filename
        self._known_hosts_path = str(known_hosts_path) if known_hosts_path else None
        self._client_factory = client_factory
        self._connect_timeout = connect_timeout
        self._host_key_confirmer = host_key_confirmer
        self._key_fetcher = key_fetcher
        self._print_fn = print_fn
        self._progress_interval = progress_interval
        self._now = now

        self._client: paramiko.SSHClient | None = None
        self._sftp = None

    def connect(self) -> None:
        try:
            client = self._open_client()
        except paramiko.SSHException as exc:
            if not self._may_learn_host_key(exc):
                raise self._as_ssh_error(exc) from exc
            self._learn_host_key()
            try:
                client = self._open_client()
            except (paramiko.SSHException, OSError) as retry_exc:
                raise self._as_ssh_error(retry_exc) from retry_exc
        except OSError as exc:
            raise self._as_ssh_error(exc) from exc

        self._client = client
        self._sftp = client.open_sftp()

    def _open_client(self) -> paramiko.SSHClient:
        client = self._client_factory()
        client.load_system_host_keys()
        if self._known_hosts_path:
            client.load_host_keys(self._known_hosts_path)
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        client.connect(
            hostname=self._host,
            port=self._port,
            username=self._username,
            password=self._password,
            key_filename=self._key_filename,
            timeout=self._connect_timeout,
        )
        return client

    def _as_ssh_error(self, exc: Exception) -> SSHError:
        if isinstance(exc, paramiko.BadHostKeyException):
            return SSHError(
                f"SSH connection to {self._host} failed: the key presented by the "
                "server does not match the one already in known_hosts. Nothing was "
                "sent; confirm the new identification with the cluster before "
                "replacing the saved entry."
            )
        return SSHError(
            f"SSH connection to {self._host} failed: {exc}"
            f"{self._known_hosts_hint(exc)}"
        )

    def _may_learn_host_key(self, exc: paramiko.SSHException) -> bool:
        """A key that is merely unknown can be accepted; a changed one cannot."""
        return (
            self._host_key_confirmer is not None
            and not isinstance(exc, paramiko.BadHostKeyException)
            and "known_hosts" in str(exc)
        )

    def _learn_host_key(self) -> None:
        key = self._key_fetcher(self._host, self._port)
        if not self._host_key_confirmer(self._host, format_fingerprint(key)):
            raise SSHError(
                f"the identification of {self._host} was not accepted, so nothing "
                "was connected or sent."
            )
        target = Path(self._known_hosts_path or DEFAULT_KNOWN_HOSTS)
        remember_host_key(target, self._host, self._port, key)
        self._known_hosts_path = str(target)

    def _known_hosts_hint(self, exc: Exception) -> str:
        """Explain how to trust the server when its key is unknown."""
        if "known_hosts" not in str(exc):
            return ""
        port_flag = "" if self._port == 22 else f"-p {self._port} "
        return (
            f". Connect once with `ssh {port_flag}{self._username}@{self._host}` "
            "while the VPN is up, check the fingerprint and accept the key so "
            "it is saved to known_hosts."
        )

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

        uploads: list[tuple[Path, str]] = []
        for root, dirnames, filenames in os.walk(local):
            root_path = Path(root)
            relative_root = root_path.relative_to(local)
            remote_root = self._remote_join(remote, relative_root)
            for dirname in dirnames:
                self._mkdir_remote(self._remote_join(remote_root, dirname))
            for filename in filenames:
                local_file = root_path / filename
                remote_file = f"{remote_root}/{filename}" if remote_root else filename
                uploads.append((local_file, remote_file))

        total = sum(local_file.stat().st_size for local_file, _ in uploads)
        self._print_fn(
            f"Enviando {len(uploads)} arquivo(s) ({format_size(total)})..."
        )
        started = self._now()
        sent = 0
        for local_file, remote_file in uploads:
            size = local_file.stat().st_size
            self._print_fn(f"  {local_file.name} ({format_size(size)})")
            self._sftp.put(
                str(local_file),
                remote_file,
                callback=self._upload_reporter(sent, total, started),
            )
            sent += size
        elapsed = self._now() - started
        self._print_fn(
            f"Envio concluído: {format_size(total)} em {format_duration(elapsed)}."
        )

    def _upload_reporter(
        self, already_sent: int, total: int, started: float
    ) -> Callable[[int, int], None]:
        """Report the whole transfer, not each file, and not too often."""
        last_report = self._now()

        def report(transferred: int, _file_total: int) -> None:
            nonlocal last_report
            now = self._now()
            if now - last_report < self._progress_interval:
                return
            last_report = now
            sent = already_sent + transferred
            elapsed = now - started
            speed = f", {format_size(sent / elapsed)}/s" if elapsed > 0 else ""
            percent = 100 * sent // total if total else 100
            self._print_fn(
                f"    {percent}% — {format_size(sent)} de {format_size(total)}{speed}"
            )

        return report

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
