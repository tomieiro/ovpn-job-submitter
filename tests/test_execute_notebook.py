import subprocess
import sys
from importlib import resources as importlib_resources

import nbformat
import pytest

EXECUTOR = str(importlib_resources.files("dgx_slurm") / "templates" / "execute_notebook.py")


def make_notebook(cells_source):
    nb = nbformat.v4.new_notebook()
    for source in cells_source:
        nb.cells.append(nbformat.v4.new_code_cell(source))
    return nb


def run_executor(tmp_path, nb, workdir=None, extra_env=None):
    input_path = tmp_path / "notebook.ipynb"
    output_path = tmp_path / "notebook.executed.ipynb"
    nbformat.write(nb, str(input_path))

    env = {"PATH": "/usr/bin:/bin"}
    import os

    env.update(os.environ)
    if workdir is not None:
        env["DGX_NOTEBOOK_WORKDIR"] = str(workdir)
    if extra_env:
        env.update(extra_env)

    proc = subprocess.run(
        [sys.executable, EXECUTOR, str(input_path), str(output_path)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc, output_path


def test_executes_cells_in_order(tmp_path):
    nb = make_notebook(["x = 1", "x += 1", "print(x)"])
    proc, output_path = run_executor(tmp_path, nb, workdir=tmp_path)
    assert proc.returncode == 0
    assert "2" in proc.stdout


def test_preserves_stdout(tmp_path):
    nb = make_notebook(["print('hello from cell')"])
    proc, _ = run_executor(tmp_path, nb, workdir=tmp_path)
    assert "hello from cell" in proc.stdout


def test_preserves_stderr(tmp_path):
    nb = make_notebook(["import sys; sys.stderr.write('warn message\\n')"])
    proc, _ = run_executor(tmp_path, nb, workdir=tmp_path)
    assert "warn message" in proc.stderr


def test_saves_executed_notebook(tmp_path):
    nb = make_notebook(["print('ok')"])
    proc, output_path = run_executor(tmp_path, nb, workdir=tmp_path)
    assert output_path.exists()
    executed = nbformat.read(str(output_path), as_version=4)
    assert executed.cells[0].outputs


def test_preserves_rich_outputs(tmp_path):
    nb = make_notebook(["3 + 4"])
    proc, output_path = run_executor(tmp_path, nb, workdir=tmp_path)
    executed = nbformat.read(str(output_path), as_version=4)
    outputs = executed.cells[0].outputs
    assert any(o.get("output_type") == "execute_result" for o in outputs)
    assert "7" in proc.stdout


def test_returns_zero_on_success(tmp_path):
    nb = make_notebook(["1 + 1"])
    proc, _ = run_executor(tmp_path, nb, workdir=tmp_path)
    assert proc.returncode == 0


def test_returns_nonzero_on_error(tmp_path):
    nb = make_notebook(["raise ValueError('boom')"])
    proc, _ = run_executor(tmp_path, nb, workdir=tmp_path)
    assert proc.returncode != 0
    assert "boom" in proc.stdout or "boom" in proc.stderr


def test_stops_after_first_failure(tmp_path):
    nb = make_notebook(["raise ValueError('boom')", "print('should not run')"])
    proc, _ = run_executor(tmp_path, nb, workdir=tmp_path)
    assert "should not run" not in proc.stdout


def test_saves_partial_notebook_on_error(tmp_path):
    nb = make_notebook(["print('first')", "raise ValueError('boom')", "print('third')"])
    proc, output_path = run_executor(tmp_path, nb, workdir=tmp_path)
    assert output_path.exists()
    executed = nbformat.read(str(output_path), as_version=4)
    assert executed.cells[0].outputs  # first cell executed
    assert executed.cells[2].execution_count is None  # third cell never ran


def test_uses_configured_workdir(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "marker.txt").write_text("present")
    nb = make_notebook(["import os; print(os.getcwd()); print(os.listdir('.'))"])
    proc, _ = run_executor(tmp_path, nb, workdir=workdir)
    assert str(workdir) in proc.stdout
    assert "marker.txt" in proc.stdout


def test_default_workdir_is_workspace():
    source = importlib_resources.files("dgx_slurm") / "templates" / "execute_notebook.py"
    content = source.read_text()
    assert 'os.environ.get("DGX_NOTEBOOK_WORKDIR", "/workspace")' in content


def test_supports_pip_install_cell(tmp_path):
    # Build a tiny local wheel-less package and install it via pip from a
    # local path, so the test never touches the network.
    pkg_dir = tmp_path / "tinypkg"
    pkg_dir.mkdir()
    (pkg_dir / "pyproject.toml").write_text(
        "[build-system]\n"
        'requires = ["setuptools>=68"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        "[project]\n"
        'name = "tinypkg"\n'
        'version = "0.0.1"\n'
    )
    (pkg_dir / "tinypkg").mkdir()
    (pkg_dir / "tinypkg" / "__init__.py").write_text('VALUE = "installed"\n')

    workdir = tmp_path / "work"
    workdir.mkdir()
    nb = make_notebook(
        [
            f"%pip install --no-cache-dir --no-index --no-build-isolation {pkg_dir}",
            "import tinypkg; print(tinypkg.VALUE)",
        ]
    )
    proc, _ = run_executor(tmp_path, nb, workdir=workdir)
    assert proc.returncode == 0
    assert "installed" in proc.stdout
