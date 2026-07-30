#!/usr/bin/env python
"""Executes a notebook cell-by-cell inside the job container.

Usage: execute_notebook.py <input.ipynb> <output.ipynb>

Prints progress and cell output to stdout/stderr so the submitting
library can stream it back to the local Jupyter cell. Always writes the
(possibly partial) executed notebook to the output path, then exits 0 on
success or non-zero on the first cell error.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def _print_flush(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def _emit_cell_outputs(cell) -> None:
    for output in cell.get("outputs", []):
        output_type = output.get("output_type")
        if output_type == "stream":
            stream = sys.stderr if output.get("name") == "stderr" else sys.stdout
            _print_flush(output.get("text", ""), end="", file=stream)
        elif output_type == "error":
            traceback_lines = output.get("traceback", [])
            _print_flush("\n".join(traceback_lines), file=sys.stderr)
        elif output_type in ("execute_result", "display_data"):
            text_plain = output.get("data", {}).get("text/plain")
            if text_plain:
                _print_flush(text_plain)


def _install_runtime_kernelspec(root: Path) -> str:
    """Point notebook execution at the runner's exact Python environment."""
    kernel_name = "dgx-runtime"
    kernel_dir = root / "kernels" / kernel_name
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "DGX runtime",
                "language": "python",
            }
        )
    )
    return kernel_name


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        _print_flush("usage: execute_notebook.py <input.ipynb> <output.ipynb>", file=sys.stderr)
        return 2

    input_path, output_path = argv[1], argv[2]
    workdir = os.environ.get("DGX_NOTEBOOK_WORKDIR", "/workspace")

    notebook = nbformat.read(input_path, as_version=4)
    code_cells = [c for c in notebook.cells if c.get("cell_type") == "code"]
    total = len(code_cells)

    progress = {"index": 0}

    def on_cell_start(cell, cell_index=None, **_kwargs):
        if cell.get("cell_type") != "code":
            return
        progress["index"] += 1
        _print_flush(f"\n[Cell {progress['index']}/{total}] START")

    def on_cell_executed(cell, cell_index=None, execute_reply=None, **_kwargs):
        if cell.get("cell_type") != "code":
            return
        _emit_cell_outputs(cell)
        _print_flush(f"[Cell {progress['index']}/{total}] DONE")

    exit_code = 0
    with tempfile.TemporaryDirectory(prefix="dgx-kernels-") as kernels_root:
        kernels_path = Path(kernels_root)
        kernel_name = _install_runtime_kernelspec(kernels_path)
        previous_jupyter_path = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = str(kernels_path)
        try:
            client = NotebookClient(
                notebook,
                kernel_name=kernel_name,
                timeout=None,
                allow_errors=False,
                resources={"metadata": {"path": workdir}},
                on_cell_start=on_cell_start,
                on_cell_executed=on_cell_executed,
            )
            client.execute()
        except CellExecutionError as exc:
            _print_flush(str(exc), file=sys.stderr)
            exit_code = 1
        finally:
            if previous_jupyter_path is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous_jupyter_path
            nbformat.write(notebook, output_path)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
