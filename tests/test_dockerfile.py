import nbformat
import pytest

from dgx_slurm.bundle import NotebookBundleBuilder
from dgx_slurm.models import Resources


def make_notebook(path):
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("print('hi')"))
    nbformat.write(nb, str(path))
    return path


@pytest.fixture
def bundle_root(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    nb_path = make_notebook(project / "experiment.ipynb")
    builder = NotebookBundleBuilder()
    return builder.build(
        notebook=nb_path,
        job_name="dgx-notebook",
        resources=Resources(),
        workdir=tmp_path / "bundle",
        project_root=project,
    )


def test_uses_fixed_base_image(bundle_root):
    content = (bundle_root / "Dockerfile").read_text()
    assert content.startswith("FROM nvcr.io/nvidia/pytorch:26.05-py3")


def test_installs_runner_requirements(bundle_root):
    content = (bundle_root / "Dockerfile").read_text()
    assert "runner/requirements.txt" in content
    assert "pip install" in content
    assert (bundle_root / "runner" / "requirements.txt").exists()


def test_copies_notebook_executor(bundle_root):
    content = (bundle_root / "Dockerfile").read_text()
    assert "execute_notebook.py" in content
    assert (bundle_root / "runner" / "execute_notebook.py").exists()


def test_copies_payload(bundle_root):
    content = (bundle_root / "Dockerfile").read_text()
    assert "COPY payload/ /workspace/" in content


def test_sets_pythonunbuffered(bundle_root):
    content = (bundle_root / "Dockerfile").read_text()
    assert "ENV PYTHONUNBUFFERED=1" in content


def test_cmd_runs_execute_notebook(bundle_root):
    content = (bundle_root / "Dockerfile").read_text()
    assert 'CMD ["python", "/opt/notebook-runner/execute_notebook.py"' in content


def test_does_not_contain_user_scientific_dependencies(bundle_root):
    content = (bundle_root / "Dockerfile").read_text()
    pip_install_lines = [
        line for line in content.splitlines() if "pip install" in line or "requirement" in line
    ]
    joined = " ".join(pip_install_lines).lower()
    for forbidden in ("cupy", "numpy", "pandas", "torch"):
        assert forbidden not in joined


def test_dockerfile_is_static_template_not_freeform(bundle_root):
    # The generated Dockerfile must be byte-identical to the packaged
    # template: the library never injects arbitrary user-provided
    # Dockerfile fragments.
    from importlib import resources as importlib_resources

    template = (
        importlib_resources.files("dgx_slurm") / "templates" / "Dockerfile"
    ).read_text()
    assert (bundle_root / "Dockerfile").read_text() == template
