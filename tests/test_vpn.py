import os
import stat
from pathlib import Path

import pytest

from dgx_slurm.errors import VPNError
from dgx_slurm.vpn import VPNConnection


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


def make_connection(ovpn_file, reachable_sequence, launched, exit_code=None, connect_timeout=5.0):
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
        connect_timeout=connect_timeout,
        poll_interval=0.0,
        sleep=lambda _: None,
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
        connect_timeout=5.0,
        poll_interval=0.0,
        sleep=lambda _: None,
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
        connect_timeout=5.0,
        poll_interval=0.0,
        sleep=lambda _: None,
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
