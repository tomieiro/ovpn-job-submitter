import os
import stat
from pathlib import Path

import pytest

from dgx_slurm.errors import VPNError
from dgx_slurm import vpn
from dgx_slurm.vpn import VPNConnection, discover_openvpn_binary


class FakeProcess:
    def __init__(self, args, cwd=None, exit_code=None):
        self.args = args
        self.cwd = cwd
        self.terminated = False
        self.killed = False
        self._returncode = exit_code
        self.wait_called = False

    def poll(self):
        return self._returncode

    def terminate(self):
        self.terminated = True
        self._returncode = 0

    def kill(self):
        self.killed = True
        self._returncode = -9

    def wait(self, timeout=None):
        self.wait_called = True
        return self._returncode


@pytest.fixture
def ovpn_file(tmp_path):
    vpn_dir = tmp_path / "vpn-config"
    vpn_dir.mkdir()
    ovpn = vpn_dir / "client.ovpn"
    ovpn.write_text("client\nremote example.com 1194\n")
    return ovpn


def make_connection(
    ovpn_file,
    reachable_sequence,
    launched,
    exit_code=None,
    connect_timeout=5.0,
    is_elevated=lambda: True,
):
    reachable_iter = iter(reachable_sequence)

    def is_reachable():
        try:
            return next(reachable_iter)
        except StopIteration:
            return reachable_sequence[-1]

    def process_launcher(cmd, cwd=None):
        proc = FakeProcess(cmd, cwd=cwd, exit_code=exit_code)
        launched.append(proc)
        return proc

    return VPNConnection(
        ovpn_path=ovpn_file,
        username="cluster-user",
        password="s3cret",
        is_reachable=is_reachable,
        process_launcher=process_launcher,
        openvpn_binary="openvpn",
        connect_timeout=connect_timeout,
        poll_interval=0.0,
        sleep=lambda _: None,
        is_elevated=is_elevated,
    )


def test_executes_openvpn_with_config(ovpn_file):
    launched = []
    conn = make_connection(ovpn_file, [False, True], launched)
    conn.connect()
    assert launched
    cmd = launched[0].args
    assert cmd[0] == "openvpn"
    assert "--config" in cmd
    assert str(ovpn_file) in cmd
    assert cmd[-2:] == ["--verb", "1"]


def test_can_launch_only_openvpn_through_sudo(ovpn_file, monkeypatch):
    launched = []
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    conn = make_connection(ovpn_file, [False, True], launched)
    conn._use_sudo = True
    conn.connect()
    assert launched[0].args[:3] == ["sudo", "--", "openvpn"]


def test_uses_ovpn_directory_as_cwd(ovpn_file):
    launched = []
    conn = make_connection(ovpn_file, [False, True], launched)
    conn.connect()
    assert launched[0].cwd == str(ovpn_file.parent)


def test_creates_auth_file_with_mode_0600(ovpn_file):
    launched = []
    captured_auth_file = {}

    def process_launcher(cmd, cwd=None):
        idx = cmd.index("--auth-user-pass")
        auth_file = Path(cmd[idx + 1])
        captured_auth_file["path"] = auth_file
        captured_auth_file["mode"] = stat.S_IMODE(auth_file.stat().st_mode)
        captured_auth_file["contents"] = auth_file.read_text()
        proc = FakeProcess(cmd, cwd=cwd, exit_code=None)
        launched.append(proc)
        return proc

    conn = VPNConnection(
        ovpn_path=ovpn_file,
        username="cluster-user",
        password="s3cret",
        is_reachable=iter([False, True]).__next__,
        process_launcher=process_launcher,
        openvpn_binary="openvpn",
        connect_timeout=5.0,
        poll_interval=0.0,
        sleep=lambda _: None,
        is_elevated=lambda: True,
    )
    conn.connect()
    assert captured_auth_file["mode"] == 0o600
    assert captured_auth_file["contents"] == "cluster-user\ns3cret\n"


def test_creates_auth_dir_with_mode_0700(ovpn_file):
    captured = {}

    def process_launcher(cmd, cwd=None):
        idx = cmd.index("--auth-user-pass")
        auth_dir = Path(cmd[idx + 1]).parent
        captured["mode"] = stat.S_IMODE(auth_dir.stat().st_mode)
        return FakeProcess(cmd, cwd=cwd, exit_code=None)

    conn = VPNConnection(
        ovpn_path=ovpn_file,
        username="cluster-user",
        password="s3cret",
        is_reachable=iter([False, True]).__next__,
        process_launcher=process_launcher,
        openvpn_binary="openvpn",
        connect_timeout=5.0,
        poll_interval=0.0,
        sleep=lambda _: None,
        is_elevated=lambda: True,
    )
    conn.connect()
    assert captured["mode"] == 0o700


def test_removes_credentials_after_success(ovpn_file):
    launched = []
    conn = make_connection(ovpn_file, [False, True], launched)
    conn.connect()
    assert conn._auth_dir is not None
    assert not conn._auth_dir.exists()


def test_removes_credentials_after_failure(ovpn_file):
    launched = []
    # process exits immediately (premature exit) -> connect() must raise
    conn = make_connection(ovpn_file, [False], launched, exit_code=1)
    with pytest.raises(VPNError):
        conn.connect()
    assert conn._auth_dir is not None
    assert not conn._auth_dir.exists()


def test_does_not_start_vpn_when_already_reachable(ovpn_file):
    launched = []
    conn = make_connection(ovpn_file, [True], launched)
    conn.connect()
    assert launched == []
    assert conn.started_by_us is False


def test_does_not_terminate_external_vpn(ovpn_file):
    launched = []
    conn = make_connection(ovpn_file, [True], launched)
    conn.connect()
    conn.disconnect()
    assert launched == []  # nothing was ever started, nothing to terminate


def test_disconnect_terminates_only_process_we_started(ovpn_file):
    launched = []
    conn = make_connection(ovpn_file, [False, True], launched)
    conn.connect()
    conn.disconnect()
    assert launched[0].terminated is True


def test_does_not_expose_password_in_errors(ovpn_file):
    launched = []
    conn = make_connection(ovpn_file, [False], launched, exit_code=1)
    with pytest.raises(VPNError) as exc_info:
        conn.connect()
    assert "s3cret" not in str(exc_info.value)


def test_timeout_raises_vpn_error_without_password(ovpn_file):
    launched = []
    conn = make_connection(
        ovpn_file, [False, False, False], launched, exit_code=None, connect_timeout=0.05
    )
    with pytest.raises(VPNError) as exc_info:
        conn.connect()
    assert "s3cret" not in str(exc_info.value)


def test_discovers_linux_openvpn_from_path(monkeypatch):
    monkeypatch.setattr(vpn.shutil, "which", lambda name: "/usr/sbin/openvpn")
    assert discover_openvpn_binary("Linux") == "/usr/sbin/openvpn"


def test_discovers_macos_homebrew_openvpn(monkeypatch):
    monkeypatch.setattr(vpn.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        vpn.Path,
        "is_file",
        lambda path: str(path) == "/opt/homebrew/sbin/openvpn",
    )
    assert discover_openvpn_binary("Darwin") == "/opt/homebrew/sbin/openvpn"


def test_discovers_windows_openvpn_in_program_files(tmp_path, monkeypatch):
    monkeypatch.setattr(vpn.shutil, "which", lambda name: None)
    executable = tmp_path / "OpenVPN" / "bin" / "openvpn.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    assert discover_openvpn_binary(
        "Windows", environ={"ProgramFiles": str(tmp_path)}
    ) == str(executable)


@pytest.mark.parametrize(
    ("system_name", "expected"),
    [
        ("Linux", "OpenVPN package"),
        ("Darwin", "brew install openvpn"),
        ("Windows", "openvpn.net/community"),
    ],
)
def test_missing_openvpn_has_platform_install_help(
    system_name, expected, monkeypatch
):
    monkeypatch.setattr(vpn.shutil, "which", lambda name: None)
    monkeypatch.setattr(vpn.Path, "is_file", lambda path: False)
    with pytest.raises(VPNError) as exc_info:
        discover_openvpn_binary(system_name, environ={})
    assert expected in str(exc_info.value)


def test_windows_never_uses_sudo(ovpn_file):
    launched = []
    conn = make_connection(ovpn_file, [False, True], launched)
    conn._use_sudo = True
    conn._system_name = "Windows"
    conn.connect()
    assert launched[0].args[0] == "openvpn"


def test_windows_permission_failure_has_actionable_message(ovpn_file):
    launched = []
    conn = make_connection(ovpn_file, [False], launched, exit_code=1)
    conn._system_name = "Windows"
    with pytest.raises(VPNError, match="Administrator"):
        conn.connect()


def test_windows_without_elevation_fails_before_starting_openvpn(ovpn_file):
    launched = []
    conn = make_connection(
        ovpn_file, [False, True], launched, is_elevated=lambda: False
    )
    conn._system_name = "Windows"
    with pytest.raises(VPNError, match="Run as administrator"):
        conn.connect()
    assert launched == []


def test_windows_elevation_error_mentions_netsh_failure(ovpn_file):
    conn = make_connection(ovpn_file, [False], [], is_elevated=lambda: False)
    conn._system_name = "Windows"
    with pytest.raises(VPNError, match="NETSH"):
        conn.connect()


def test_windows_skips_elevation_check_when_already_reachable(ovpn_file):
    launched = []
    conn = make_connection(ovpn_file, [True], launched, is_elevated=lambda: False)
    conn._system_name = "Windows"
    conn.connect()
    assert launched == []
    assert conn.started_by_us is False


def test_other_systems_do_not_require_elevation(ovpn_file):
    launched = []
    conn = make_connection(
        ovpn_file, [False, True], launched, is_elevated=lambda: False
    )
    conn._system_name = "Linux"
    conn.connect()
    assert launched[0].args[0] == "openvpn"
