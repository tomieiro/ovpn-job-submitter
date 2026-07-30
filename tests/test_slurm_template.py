import re

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
def build(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    nb_path = make_notebook(project / "experiment.ipynb")
    builder = NotebookBundleBuilder()

    def _build(resources, job_name="dgx-notebook"):
        return builder.build(
            notebook=nb_path,
            job_name=job_name,
            resources=resources,
            workdir=tmp_path / f"bundle-{job_name}",
            project_root=project,
        )

    return _build


def test_renders_resources_into_sbatch_directives(build):
    bundle_root = build(
        Resources(gpus=2, cpus=8, memory="64G", time_limit="02:00:00", partition="research")
    )
    content = (bundle_root / "runImage.slurm").read_text()
    assert "#SBATCH --cpus-per-task=8" in content
    assert "#SBATCH --gres=gpu:2" in content
    assert "#SBATCH --mem=64G" in content
    assert "#SBATCH --time=02:00:00" in content
    assert "#SBATCH --partition=research" in content
    assert "__" not in content  # no leftover placeholders


def test_renders_sanitized_job_name(build):
    bundle_root = build(Resources(), job_name="dgx-notebook-48192")
    content = (bundle_root / "runImage.slurm").read_text()
    assert "#SBATCH --job-name=dgx-notebook-48192" in content


def test_uses_logs_x_j_pattern(build):
    bundle_root = build(Resources())
    content = (bundle_root / "runImage.slurm").read_text()
    assert "--output=logs/%x-%j.out" in content
    assert "--error=logs/%x-%j.err" in content


def test_uses_job_id_based_image_tag(build):
    bundle_root = build(Resources())
    content = (bundle_root / "runImage.slurm").read_text()
    assert 'IMAGE_TAG="${IMAGE_TAG:-dgx-notebook-${SLURM_JOB_ID}:latest}"' in content


def test_runs_docker_build(build):
    bundle_root = build(Resources())
    content = (bundle_root / "runImage.slurm").read_text()
    assert re.search(r"docker build\s", content)


def test_runs_docker_run_with_rm(build):
    bundle_root = build(Resources())
    content = (bundle_root / "runImage.slurm").read_text()
    assert "docker run" in content
    assert "--rm" in content


def test_mounts_outputs_directory(build):
    bundle_root = build(Resources())
    content = (bundle_root / "runImage.slurm").read_text()
    assert "type=bind,src=${SCRIPT_DIR}/outputs,dst=/workspace/outputs" in content


def test_installs_trap_exit(build):
    bundle_root = build(Resources())
    content = (bundle_root / "runImage.slurm").read_text()
    assert "trap cleanup EXIT" in content


def test_cleanup_runs_docker_rmi(build):
    bundle_root = build(Resources())
    content = (bundle_root / "runImage.slurm").read_text()
    assert "docker rmi" in content


def test_cleanup_preserves_exit_code(build):
    bundle_root = build(Resources())
    content = (bundle_root / "runImage.slurm").read_text()
    assert "local status=$?" in content
    assert 'exit "${status}"' in content


def test_does_not_run_docker_system_prune(build):
    bundle_root = build(Resources())
    content = (bundle_root / "runImage.slurm").read_text()
    assert "system prune" not in content


def test_two_submissions_render_distinct_job_names(build):
    first = build(Resources(), job_name="dgx-notebook-a")
    second = build(Resources(), job_name="dgx-notebook-b")
    assert "dgx-notebook-a" in (first / "runImage.slurm").read_text()
    assert "dgx-notebook-b" in (second / "runImage.slurm").read_text()
