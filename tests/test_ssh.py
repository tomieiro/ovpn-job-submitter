import base64
import hashlib
import io
from pathlib import Path

import paramiko
import pytest

from dgx_slurm.errors import SSHError
from dgx_slurm.ssh import SSHTransport, format_fingerprint, remember_host_key


class FakeHostKey:
    def __init__(self, blob=b"host-key-blob"):
        self._blob = blob

    def asbytes(self):
        return self._blob

    def get_name(self):
        return "ssh-ed25519"

    def get_base64(self):
        return base64.b64encode(self._blob).decode("ascii")


class FakeChannelFile(io.BytesIO):
    def __init__(self, data, exit_status=0):
        super().__init__(data)
        self.channel = type("C", (), {"recv_exit_status": lambda self: exit_status})()


class FakeSFTPClient:
    def __init__(self):
        self.files = {}  # remote path -> bytes
        self.dirs_created = []
        self.uploaded = {}
        self.closed = False

    def mkdir(self, path):
        self.dirs_created.append(path)

    def put(self, local_path, remote_path, callback=None):
        self.uploaded[remote_path] = Path(local_path).read_bytes()
        self.files[remote_path] = self.uploaded[remote_path]
        if callback is not None:
            size = len(self.uploaded[remote_path])
            callback(size, size)

    def get(self, remote_path, local_path):
        Path(local_path).write_bytes(self.files[remote_path])

    def listdir_attr(self, path):
        import stat as statmod

        entries = []
        prefix = path.rstrip("/") + "/"
        seen = set()
        for remote_path in self.files:
            if remote_path.startswith(prefix):
                rest = remote_path[len(prefix):]
                name = rest.split("/")[0]
                if name not in seen:
                    seen.add(name)
                    attr = type(
                        "A",
                        (),
                        {
                            "filename": name,
                            "st_mode": statmod.S_IFDIR if "/" in rest else statmod.S_IFREG,
                        },
                    )()
                    entries.append(attr)
        return entries

    def open(self, path, mode="r"):
        if path not in self.files:
            raise FileNotFoundError(path)
        return io.BytesIO(self.files[path])

    def stat(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return type("St", (), {"st_size": len(self.files[path])})()

    def close(self):
        self.closed = True


class FakeSSHClient:
    instances = []

    def __init__(self):
        self.loaded_system_host_keys = False
        self.loaded_host_keys_path = None
        self.missing_host_key_policy = None
        self.connect_kwargs = None
        self.commands = []
        self.command_results = {}
        self.sftp = FakeSFTPClient()
        self.closed = False
        self.raise_on_connect = None
        FakeSSHClient.instances.append(self)

    def load_system_host_keys(self):
        self.loaded_system_host_keys = True

    def load_host_keys(self, path):
        self.loaded_host_keys_path = path

    def set_missing_host_key_policy(self, policy):
        self.missing_host_key_policy = policy

    def connect(self, **kwargs):
        if self.raise_on_connect:
            raise self.raise_on_connect
        self.connect_kwargs = kwargs

    def open_sftp(self):
        return self.sftp

    def exec_command(self, command):
        self.commands.append(command)
        stdout_data, stderr_data, exit_status = self.command_results.get(
            command, (b"", b"", 0)
        )
        stdin = io.BytesIO()
        stdout = FakeChannelFile(stdout_data, exit_status)
        stderr = FakeChannelFile(stderr_data, exit_status)
        return stdin, stdout, stderr

    def close(self):
        self.closed = True


@pytest.fixture
def client_factory():
    FakeSSHClient.instances = []
    return FakeSSHClient


def make_transport(client_factory, **overrides):
    kwargs = dict(
        host="dgx.cluster.internal",
        port=22,
        username="cluster-user",
        password="s3cret",
        client_factory=client_factory,
    )
    kwargs.update(overrides)
    return SSHTransport(**kwargs)


def test_authenticates_with_username_and_password(client_factory):
    transport = make_transport(client_factory)
    transport.connect()
    client = FakeSSHClient.instances[0]
    assert client.connect_kwargs["username"] == "cluster-user"
    assert client.connect_kwargs["password"] == "s3cret"
    assert client.connect_kwargs["hostname"] == "dgx.cluster.internal"


def test_loads_known_hosts(client_factory, tmp_path):
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("")
    transport = make_transport(client_factory, known_hosts_path=known_hosts)
    transport.connect()
    client = FakeSSHClient.instances[0]
    assert client.loaded_system_host_keys is True
    assert client.loaded_host_keys_path == str(known_hosts)


def test_rejects_unknown_host_by_default(client_factory):
    transport = make_transport(client_factory)
    transport.connect()
    client = FakeSSHClient.instances[0]
    assert isinstance(client.missing_host_key_policy, paramiko.RejectPolicy)


def test_wraps_connect_failure_as_ssh_error(client_factory):
    FakeSSHClient.instances = []

    class RaisingClient(FakeSSHClient):
        def __init__(self):
            super().__init__()
            self.raise_on_connect = paramiko.AuthenticationException("bad auth")

    transport = make_transport(client_factory=RaisingClient)
    with pytest.raises(SSHError):
        transport.connect()


def test_unknown_host_key_error_explains_how_to_trust_the_server():
    class RejectingClient(FakeSSHClient):
        def __init__(self):
            super().__init__()
            self.raise_on_connect = paramiko.SSHException(
                "Server 'dgx.cluster.internal' not found in known_hosts"
            )

    transport = make_transport(client_factory=RejectingClient)
    with pytest.raises(SSHError) as exc_info:
        transport.connect()
    assert "ssh cluster-user@dgx.cluster.internal" in str(exc_info.value)


def test_unknown_host_key_hint_includes_custom_port():
    class RejectingClient(FakeSSHClient):
        def __init__(self):
            super().__init__()
            self.raise_on_connect = paramiko.SSHException(
                "not found in known_hosts"
            )

    transport = make_transport(client_factory=RejectingClient, port=2222)
    with pytest.raises(SSHError) as exc_info:
        transport.connect()
    assert "ssh -p 2222 cluster-user@dgx.cluster.internal" in str(exc_info.value)


def make_first_attempt_rejecting_factory(exception):
    """A client that refuses the unknown host once, then behaves normally."""
    state = {"attempts": 0}

    class Client(FakeSSHClient):
        def __init__(self):
            super().__init__()
            state["attempts"] += 1
            if state["attempts"] == 1:
                self.raise_on_connect = exception

    return Client


def test_accepted_host_key_is_saved_and_the_connection_retried(tmp_path):
    FakeSSHClient.instances = []
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("other.host ssh-rsa AAAA\n")
    key = FakeHostKey()
    asked = []
    factory = make_first_attempt_rejecting_factory(
        paramiko.SSHException("Server 'dgx.cluster.internal' not found in known_hosts")
    )

    transport = make_transport(
        client_factory=factory,
        known_hosts_path=known_hosts,
        key_fetcher=lambda host, port: key,
        host_key_confirmer=lambda host, fingerprint: asked.append(
            (host, fingerprint)
        )
        or True,
    )
    transport.connect()

    assert asked == [("dgx.cluster.internal", format_fingerprint(key))]
    saved = known_hosts.read_text()
    assert "other.host ssh-rsa AAAA\n" in saved
    assert f"dgx.cluster.internal ssh-ed25519 {key.get_base64()}\n" in saved
    assert FakeSSHClient.instances[-1].connect_kwargs is not None


def test_refused_host_key_connects_nothing(tmp_path):
    FakeSSHClient.instances = []
    known_hosts = tmp_path / "known_hosts"
    factory = make_first_attempt_rejecting_factory(
        paramiko.SSHException("Server 'dgx.cluster.internal' not found in known_hosts")
    )

    transport = make_transport(
        client_factory=factory,
        known_hosts_path=known_hosts,
        key_fetcher=lambda host, port: FakeHostKey(),
        host_key_confirmer=lambda host, fingerprint: False,
    )
    with pytest.raises(SSHError, match="not accepted"):
        transport.connect()

    assert not known_hosts.exists()


def test_changed_host_key_is_never_offered_for_acceptance(tmp_path):
    FakeSSHClient.instances = []
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("dgx.cluster.internal ssh-ed25519 AAAA\n")
    key = FakeHostKey()

    def unexpected_confirmer(_host, _fingerprint):
        raise AssertionError("a changed key must not be offered for acceptance")

    class Client(FakeSSHClient):
        def __init__(self):
            super().__init__()
            self.raise_on_connect = paramiko.BadHostKeyException(
                "dgx.cluster.internal", key, key
            )

    transport = make_transport(
        client_factory=Client,
        known_hosts_path=known_hosts,
        key_fetcher=lambda host, port: key,
        host_key_confirmer=unexpected_confirmer,
    )
    with pytest.raises(SSHError, match="does not match"):
        transport.connect()

    assert known_hosts.read_text() == "dgx.cluster.internal ssh-ed25519 AAAA\n"


def test_saved_entry_carries_the_port_when_it_is_not_22(tmp_path):
    known_hosts = tmp_path / "known_hosts"
    key = FakeHostKey()
    remember_host_key(known_hosts, "dgx.cluster.internal", 2222, key)
    assert known_hosts.read_text() == (
        f"[dgx.cluster.internal]:2222 ssh-ed25519 {key.get_base64()}\n"
    )


def test_saved_entry_never_glues_itself_to_an_unterminated_line(tmp_path):
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("other.host ssh-rsa AAAA")
    remember_host_key(known_hosts, "dgx.cluster.internal", 22, FakeHostKey())
    assert known_hosts.read_text().splitlines()[0] == "other.host ssh-rsa AAAA"
    assert len(known_hosts.read_text().splitlines()) == 2


def test_fingerprint_matches_the_openssh_format():
    key = FakeHostKey()
    expected = base64.b64encode(hashlib.sha256(key.asbytes()).digest())
    assert format_fingerprint(key) == "SHA256:" + expected.decode().rstrip("=")


def test_other_ssh_errors_do_not_get_known_hosts_hint():
    class RaisingClient(FakeSSHClient):
        def __init__(self):
            super().__init__()
            self.raise_on_connect = paramiko.AuthenticationException("bad auth")

    transport = make_transport(client_factory=RaisingClient)
    with pytest.raises(SSHError) as exc_info:
        transport.connect()
    assert "known_hosts" not in str(exc_info.value)


def test_creates_remote_directory(client_factory):
    transport = make_transport(client_factory)
    transport.connect()
    transport.execute("mkdir -p /remote/job-dir")
    client = FakeSSHClient.instances[0]
    assert client.commands == ["mkdir -p /remote/job-dir"]


def test_uploads_bundle_directory(client_factory, tmp_path):
    local = tmp_path / "bundle"
    (local / "payload").mkdir(parents=True)
    (local / "payload" / "notebook.ipynb").write_text("{}")
    (local / "Dockerfile").write_text("FROM x")

    transport = make_transport(client_factory)
    transport.connect()
    transport.upload_directory(local, "/remote/job-42")

    client = FakeSSHClient.instances[0]
    assert client.sftp.files["/remote/job-42/Dockerfile"] == b"FROM x"
    assert client.sftp.files["/remote/job-42/payload/notebook.ipynb"] == b"{}"


def test_upload_reports_size_names_and_completion(client_factory, tmp_path):
    local = tmp_path / "bundle"
    local.mkdir()
    (local / "data.nc").write_bytes(b"x" * 2048)
    (local / "Dockerfile").write_text("FROM x")
    printed = []

    transport = make_transport(client_factory, print_fn=printed.append)
    transport.connect()
    transport.upload_directory(local, "/remote/job-42")

    assert printed[0] == "Enviando 2 arquivo(s) (2.0 KB)..."
    assert "  data.nc (2.0 KB)" in printed
    assert printed[-1].startswith("Envio concluído: 2.0 KB em ")


def test_upload_progress_is_reported_at_most_once_per_interval(
    client_factory, tmp_path
):
    """A 500 MB file must not flood the log with a line per chunk."""
    local = tmp_path / "bundle"
    local.mkdir()
    (local / "big.db").write_bytes(b"x" * 4096)
    printed = []
    clock = iter([0.0, 0.0, 0.0, 1.0, 6.0, 7.0] + [100.0] * 20)

    transport = make_transport(
        client_factory,
        print_fn=printed.append,
        progress_interval=5.0,
        now=lambda: next(clock),
    )
    transport.connect()

    def chunked_put(local_path, remote_path, callback=None):
        for step in (1024, 2048, 3072, 4096):
            callback(step, 4096)

    transport._sftp.put = chunked_put
    transport.upload_directory(local, "/remote/job-42")

    progress_lines = [line for line in printed if line.startswith("    ")]
    assert len(progress_lines) == 1
    assert "%" in progress_lines[0]


def test_executes_command_and_returns_result(client_factory):
    transport = make_transport(client_factory)
    transport.connect()
    client = FakeSSHClient.instances[0]
    client.command_results["echo hi"] = (b"hi\n", b"", 0)
    result = transport.execute("echo hi")
    assert result.exit_code == 0
    assert result.stdout == "hi\n"
    assert result.command == "echo hi"


def test_reads_remote_file_from_offset(client_factory):
    transport = make_transport(client_factory)
    transport.connect()
    client = FakeSSHClient.instances[0]
    client.sftp.files["/remote/job-42/logs/x-1.out"] = b"PENDING\nRUNNING\n"
    chunk = transport.read_from("/remote/job-42/logs/x-1.out", offset=8)
    assert chunk.text == "RUNNING\n"
    assert chunk.new_offset == 16


def test_read_from_missing_file_returns_empty_chunk(client_factory):
    transport = make_transport(client_factory)
    transport.connect()
    chunk = transport.read_from("/remote/job-42/logs/missing.out", offset=0)
    assert chunk.text == ""
    assert chunk.new_offset == 0


def test_downloads_outputs_directory(client_factory, tmp_path):
    transport = make_transport(client_factory)
    transport.connect()
    client = FakeSSHClient.instances[0]
    client.sftp.files["/remote/job-42/outputs/notebook.executed.ipynb"] = b"{}"
    client.sftp.files["/remote/job-42/outputs/result.txt"] = b"done"

    destination = tmp_path / "downloaded"
    transport.download_directory("/remote/job-42/outputs", destination)

    assert (destination / "notebook.executed.ipynb").read_bytes() == b"{}"
    assert (destination / "result.txt").read_bytes() == b"done"


def test_close_closes_sftp_and_client_explicitly(client_factory):
    transport = make_transport(client_factory)
    transport.connect()
    client = FakeSSHClient.instances[0]
    transport.close()
    assert client.sftp.closed is True
    assert client.closed is True
