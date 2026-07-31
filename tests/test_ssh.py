import io
from pathlib import Path

import paramiko
import pytest

from dgx_slurm.errors import SSHError
from dgx_slurm.ssh import SSHTransport


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

    def put(self, local_path, remote_path):
        self.uploaded[remote_path] = Path(local_path).read_bytes()
        self.files[remote_path] = self.uploaded[remote_path]

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
