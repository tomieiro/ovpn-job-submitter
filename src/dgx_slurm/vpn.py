"""OpenVPN session management.

The library hands the .ovpn file to OpenVPN unmodified and never parses
its directives (CA, cert, key, remote, routes, ...). Credentials live only
in memory and in a short-lived 0600 temp file that OpenVPN reads once at
startup.
"""

from __future__ import annotations

import shutil
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from .errors import VPNError

ProcessLauncher = Callable[..., "subprocess.Popen"]


class VPNConnection:
    """Starts and supervises an OpenVPN process for a single .ovpn file."""

    def __init__(
        self,
        *,
        ovpn_path: Path,
        username: str,
        password: str,
        is_reachable: Callable[[], bool],
        process_launcher: ProcessLauncher = subprocess.Popen,
        openvpn_binary: str = "openvpn",
        connect_timeout: float = 60.0,
        poll_interval: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ovpn_path = Path(ovpn_path)
        self._username = username
        self._password = password
        self._is_reachable = is_reachable
        self._process_launcher = process_launcher
        self._openvpn_binary = openvpn_binary
        self._connect_timeout = connect_timeout
        self._poll_interval = poll_interval
        self._sleep = sleep
        self._now = now

        self._process: subprocess.Popen | None = None
        self._started_by_us = False
        self._auth_dir: Path | None = None

    @property
    def started_by_us(self) -> bool:
        return self._started_by_us

    def connect(self) -> None:
        if self._is_reachable():
            self._started_by_us = False
            return

        self._auth_dir = Path(tempfile.mkdtemp(prefix="dgx-slurm-"))
        self._auth_dir.chmod(0o700)
        auth_file = self._auth_dir / "auth"
        auth_file.write_text(f"{self._username}\n{self._password}\n")
        auth_file.chmod(0o600)

        try:
            command = [
                self._openvpn_binary,
                "--config",
                str(self._ovpn_path),
                "--auth-user-pass",
                str(auth_file),
                "--auth-nocache",
            ]
            self._process = self._process_launcher(
                command, cwd=str(self._ovpn_path.parent)
            )
            self._wait_until_ready()
            self._started_by_us = True
        finally:
            self._remove_credentials()

    def disconnect(self) -> None:
        if self._process is not None and self._started_by_us:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10)
        self._process = None
        self._started_by_us = False
        self._remove_credentials()

    def _wait_until_ready(self) -> None:
        deadline = self._now() + self._connect_timeout
        while True:
            exit_code = self._process.poll()
            if exit_code is not None:
                raise VPNError(
                    f"openvpn exited prematurely with code {exit_code} "
                    f"before the cluster became reachable"
                )
            if self._is_reachable():
                return
            if self._now() >= deadline:
                raise VPNError(
                    f"timed out after {self._connect_timeout}s waiting for "
                    f"the cluster to become reachable through the VPN"
                )
            self._sleep(self._poll_interval)

    def _remove_credentials(self) -> None:
        if self._auth_dir is not None and self._auth_dir.exists():
            shutil.rmtree(self._auth_dir, ignore_errors=True)
