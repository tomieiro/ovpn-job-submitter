"""OpenVPN session management.

The library hands the .ovpn file to OpenVPN unmodified and never parses
its directives (CA, cert, key, remote, routes, ...). Credentials live only
in memory and in a short-lived 0600 temp file that OpenVPN reads once at
startup.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Mapping

from .errors import VPNError

ProcessLauncher = Callable[..., "subprocess.Popen"]

OPENVPN_WINDOWS_DOWNLOAD = "https://openvpn.net/community/"
OPENVPN_LINUX_DOWNLOAD = (
    "https://community.openvpn.net/Pages/OpenVPN%20software%20repos"
)
OPENVPN_MACOS_DOWNLOAD = "https://formulae.brew.sh/formula/openvpn"


def discover_openvpn_binary(
    system_name: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return the OpenVPN executable for Linux, Windows, or macOS."""
    system_name = system_name or platform.system()
    environ = os.environ if environ is None else environ

    executable_names = (
        ("openvpn.exe", "openvpn") if system_name == "Windows" else ("openvpn",)
    )
    for executable_name in executable_names:
        executable = shutil.which(executable_name)
        if executable:
            return executable

    candidates: list[Path] = []
    if system_name == "Windows":
        for variable in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
            base = environ.get(variable)
            if base:
                candidates.append(Path(base) / "OpenVPN" / "bin" / "openvpn.exe")
    elif system_name == "Darwin":
        candidates.extend(
            (
                Path("/opt/homebrew/sbin/openvpn"),
                Path("/usr/local/sbin/openvpn"),
            )
        )
    elif system_name != "Linux":
        raise VPNError(
            f"unsupported operating system for automatic VPN startup: {system_name}"
        )

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    if system_name == "Windows":
        install = (
            "Install OpenVPN Community, then reopen the terminal: "
            f"{OPENVPN_WINDOWS_DOWNLOAD}"
        )
    elif system_name == "Darwin":
        install = (
            "Install it with `brew install openvpn`: "
            f"{OPENVPN_MACOS_DOWNLOAD}"
        )
    else:
        install = (
            "Install the OpenVPN package for your Linux distribution: "
            f"{OPENVPN_LINUX_DOWNLOAD}"
        )
    raise VPNError(f"OpenVPN executable not found. {install}")


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
        openvpn_binary: str | None = None,
        use_sudo: bool = False,
        system_name: str | None = None,
        verbosity: int = 1,
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
        self._use_sudo = use_sudo
        self._system_name = system_name or platform.system()
        self._verbosity = verbosity
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

        openvpn_binary = self._openvpn_binary or discover_openvpn_binary(
            self._system_name
        )
        self._auth_dir = Path(tempfile.mkdtemp(prefix="dgx-slurm-"))
        self._auth_dir.chmod(0o700)
        auth_file = self._auth_dir / "auth"
        auth_file.write_text(f"{self._username}\n{self._password}\n")
        auth_file.chmod(0o600)

        try:
            command = [
                openvpn_binary,
                "--config",
                str(self._ovpn_path),
                "--auth-user-pass",
                str(auth_file),
                "--auth-nocache",
                "--verb",
                str(self._verbosity),
            ]
            get_effective_uid = getattr(os, "geteuid", lambda: 1)
            if (
                self._use_sudo
                and self._system_name in {"Linux", "Darwin"}
                and get_effective_uid() != 0
            ):
                command = ["sudo", "--", *command]
            try:
                self._process = self._process_launcher(
                    command, cwd=str(self._ovpn_path.parent)
                )
            except OSError as exc:
                raise VPNError(
                    f"could not start OpenVPN executable: {exc}"
                ) from exc
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
                hint = (
                    " On Windows, run the terminal as Administrator or connect "
                    "with OpenVPN GUI before running the script."
                    if self._system_name == "Windows"
                    else ""
                )
                raise VPNError(
                    f"openvpn exited prematurely with code {exit_code} "
                    f"before the cluster became reachable.{hint}"
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
