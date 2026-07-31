"""Local persistence of minimal job metadata, used by DGXClient.attach()."""

from __future__ import annotations

import json
from pathlib import Path


class LocalJobStore:
    """A flat JSON file mapping job_id -> {job_name, remote_job_dir}.

    Never stores credentials, .ovpn contents, or other secrets.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    def save(self, job_id: str, record: dict) -> None:
        data = self._read_all()
        data[job_id] = record
        self._write_all(data)

    def load(self, job_id: str) -> dict | None:
        return self._read_all().get(job_id)

    def _read_all(self) -> dict:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text())

    def _write_all(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2))
