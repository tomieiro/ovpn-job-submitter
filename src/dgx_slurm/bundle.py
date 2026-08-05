"""Builds the local job-bundle/ directory uploaded to the DGX cluster."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Sequence

import nbformat

from .errors import BundleError
from .models import Resources

FORBIDDEN_INCLUDE_SUFFIXES = {".ovpn", ".pem", ".key", ".p12", ".pfx"}
IGNORED_DIR_NAMES = {
    ".ipynb_checkpoints",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
}


class NotebookBundleBuilder:
    """Validates inputs and assembles the on-disk bundle for a submission."""

    def __init__(self, templates_dir: Path | None = None) -> None:
        self._templates_dir = templates_dir or Path(
            str(importlib_resources.files("dgx_slurm") / "templates")
        )

    def validate_notebook(self, notebook: Path) -> None:
        self._validate_notebook(Path(notebook))

    def build(
        self,
        *,
        notebook: Path,
        job_name: str,
        resources: Resources,
        workdir: Path,
        project_root: Path,
        include: Sequence[Path] = (),
    ) -> Path:
        notebook = Path(notebook)
        project_root = Path(project_root).resolve()

        self._validate_notebook(notebook)

        bundle_root = Path(workdir)
        payload_dir = bundle_root / "payload"
        runner_dir = bundle_root / "runner"
        for directory in (payload_dir, runner_dir, bundle_root / "outputs", bundle_root / "logs"):
            directory.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(notebook, payload_dir / "notebook.ipynb")

        for include_path in include:
            self._copy_include(Path(include_path), project_root, payload_dir)

        self._copy_unix_text(
            self._templates_dir / "execute_notebook.py",
            runner_dir / "execute_notebook.py",
        )
        self._copy_unix_text(
            self._templates_dir / "runner-requirements.txt",
            runner_dir / "requirements.txt",
        )
        self._copy_unix_text(
            self._templates_dir / "Dockerfile", bundle_root / "Dockerfile"
        )
        self._copy_unix_text(
            self._templates_dir / "dockerignore", bundle_root / ".dockerignore"
        )

        self._write_unix_text(
            bundle_root / "runImage.slurm",
            self._render_slurm_script(job_name, resources),
        )

        manifest = {
            "job_name": job_name,
            "notebook": notebook.name,
            "resources": asdict(resources),
            "include": [str(Path(p).relative_to(project_root)) for p in include],
        }
        self._write_unix_text(
            bundle_root / "manifest.json", json.dumps(manifest, indent=2)
        )

        return bundle_root

    @staticmethod
    def _write_unix_text(destination: Path, text: str) -> None:
        """The bundle runs on Linux: sbatch rejects a script with CRLF, and
        Path.write_text would produce exactly that on Windows."""
        destination.write_text(text, encoding="utf-8", newline="\n")

    @classmethod
    def _copy_unix_text(cls, source: Path, destination: Path) -> None:
        """Copy a template as text, so a CRLF checkout cannot travel with it."""
        cls._write_unix_text(destination, Path(source).read_text(encoding="utf-8"))

    def _validate_notebook(self, notebook: Path) -> None:
        if notebook.suffix != ".ipynb":
            raise BundleError(f"notebook must have a .ipynb extension, got {notebook.name}")
        if not notebook.is_file():
            raise BundleError(f"notebook not found: {notebook}")
        try:
            nbformat.read(str(notebook), as_version=4)
        except Exception as exc:  # nbformat raises varied errors on bad/malformed JSON
            raise BundleError(f"notebook is not valid: {exc}") from exc

    def _copy_include(self, source: Path, project_root: Path, payload_dir: Path) -> None:
        if not source.exists():
            raise BundleError(f"include path not found: {source}")

        try:
            resolved = source.resolve(strict=True)
        except OSError as exc:
            raise BundleError(f"could not resolve include path {source}: {exc}") from exc

        if project_root not in resolved.parents and resolved != project_root:
            raise BundleError(
                f"include path escapes project root: {source} -> {resolved}"
            )

        try:
            relative = resolved.relative_to(project_root)
        except ValueError as exc:
            raise BundleError(f"include path escapes project root: {source}") from exc

        if source.suffix in FORBIDDEN_INCLUDE_SUFFIXES:
            raise BundleError(f"include path has a forbidden extension: {source}")

        destination = payload_dir / relative

        if resolved.is_dir():
            self._copy_tree(resolved, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resolved, destination)

    def _copy_tree(self, source_dir: Path, destination_dir: Path) -> None:
        destination_dir.mkdir(parents=True, exist_ok=True)
        for entry in sorted(source_dir.iterdir()):
            if entry.name in IGNORED_DIR_NAMES:
                continue
            if entry.suffix in FORBIDDEN_INCLUDE_SUFFIXES:
                continue
            target = destination_dir / entry.name
            if entry.is_dir() and not entry.is_symlink():
                self._copy_tree(entry, target)
            elif entry.is_file():
                shutil.copyfile(entry, target)

    def _render_slurm_script(self, job_name: str, resources: Resources) -> str:
        template = (self._templates_dir / "runImage.slurm").read_text()
        replacements = {
            "__JOB_NAME__": job_name,
            "__TIME_LIMIT__": resources.time_limit,
            "__NODES__": str(resources.nodes),
            "__TASKS__": str(resources.tasks),
            "__CPUS__": str(resources.cpus),
            "__GPUS__": str(resources.gpus),
            "__MEMORY__": resources.memory,
            "__PARTITION__": resources.partition,
        }
        for placeholder, value in replacements.items():
            template = template.replace(placeholder, value)
        return template
