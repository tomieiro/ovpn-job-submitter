"""Incremental tail-following of remote stdout/stderr log files."""

from __future__ import annotations

from typing import Callable


class LogStreamer:
    """Polls two remote log files from their last-read offset, echoing new
    output at most once and accumulating the full text seen so far."""

    def __init__(
        self,
        transport,
        *,
        stdout_path: str,
        stderr_path: str,
        print_fn: Callable[[str], None] = print,
    ) -> None:
        self._transport = transport
        self._stdout_path = stdout_path
        self._stderr_path = stderr_path
        self._print_fn = print_fn

        self._stdout_offset = 0
        self._stderr_offset = 0
        self._stdout_chunks: list[str] = []
        self._stderr_chunks: list[str] = []

    def poll(self, *, echo: bool = True) -> None:
        self._poll_one(is_stdout=True, echo=echo)
        self._poll_one(is_stdout=False, echo=echo)

    def _poll_one(self, *, is_stdout: bool, echo: bool) -> None:
        path = self._stdout_path if is_stdout else self._stderr_path
        offset = self._stdout_offset if is_stdout else self._stderr_offset

        chunk = self._transport.read_from(path, offset)

        if is_stdout:
            self._stdout_offset = chunk.new_offset
        else:
            self._stderr_offset = chunk.new_offset

        if not chunk.text:
            return

        (self._stdout_chunks if is_stdout else self._stderr_chunks).append(chunk.text)
        if echo:
            self._print_fn(chunk.text)

    @property
    def stdout(self) -> str:
        return "".join(self._stdout_chunks)

    @property
    def stderr(self) -> str:
        return "".join(self._stderr_chunks)
