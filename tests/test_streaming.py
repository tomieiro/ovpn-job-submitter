from dgx_slurm.ssh import RemoteFileChunk
from dgx_slurm.streaming import LogStreamer


class FakeTransport:
    def __init__(self):
        self.files = {}  # path -> full bytes/str content so far

    def read_from(self, remote_file, offset):
        content = self.files.get(remote_file, "")
        chunk = content[offset:]
        return RemoteFileChunk(text=chunk, new_offset=offset + len(chunk))


def test_poll_returns_only_new_stdout(tmp_path=None):
    transport = FakeTransport()
    transport.files["out.log"] = "hello "
    transport.files["err.log"] = ""
    streamer = LogStreamer(transport, stdout_path="out.log", stderr_path="err.log")

    streamer.poll(echo=False)
    assert streamer.stdout == "hello "

    transport.files["out.log"] = "hello world"
    streamer.poll(echo=False)
    assert streamer.stdout == "hello world"


def test_does_not_duplicate_logs_across_polls():
    transport = FakeTransport()
    transport.files["out.log"] = "a"
    streamer = LogStreamer(transport, stdout_path="out.log", stderr_path="err.log")
    streamer.poll(echo=False)
    streamer.poll(echo=False)
    streamer.poll(echo=False)
    assert streamer.stdout == "a"


def test_echoes_new_chunks_via_print_fn():
    transport = FakeTransport()
    transport.files["out.log"] = "line1\n"
    printed = []
    streamer = LogStreamer(
        transport, stdout_path="out.log", stderr_path="err.log", print_fn=printed.append
    )
    streamer.poll(echo=True)
    assert printed == ["line1\n"]


def test_tracks_stdout_and_stderr_independently():
    transport = FakeTransport()
    transport.files["out.log"] = "stdout data"
    transport.files["err.log"] = "stderr data"
    streamer = LogStreamer(transport, stdout_path="out.log", stderr_path="err.log")
    streamer.poll(echo=False)
    assert streamer.stdout == "stdout data"
    assert streamer.stderr == "stderr data"
