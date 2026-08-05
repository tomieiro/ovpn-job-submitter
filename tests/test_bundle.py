import json
import os

import nbformat
import pytest

from dgx_slurm.bundle import NotebookBundleBuilder
from dgx_slurm.errors import BundleError
from dgx_slurm.models import Resources


def make_notebook(path):
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("print('hi')"))
    nbformat.write(nb, str(path))
    return path


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture
def builder():
    return NotebookBundleBuilder()


def test_accepts_valid_notebook(builder, project, tmp_path):
    nb_path = make_notebook(project / "experiment.ipynb")
    out = tmp_path / "bundle"
    bundle_root = builder.build(
        notebook=nb_path,
        job_name="dgx-notebook",
        resources=Resources(),
        workdir=out,
        project_root=project,
    )
    assert (bundle_root / "payload" / "notebook.ipynb").exists()


BUNDLE_TEXT_FILES = (
    "runImage.slurm",
    "Dockerfile",
    ".dockerignore",
    "manifest.json",
    "runner/execute_notebook.py",
    "runner/requirements.txt",
)


def test_bundle_text_files_never_carry_windows_line_breaks(
    builder, project, tmp_path
):
    """sbatch refuses a script with CRLF, and write_text emits it on Windows."""
    nb_path = make_notebook(project / "experiment.ipynb")
    bundle_root = builder.build(
        notebook=nb_path,
        job_name="dgx-notebook",
        resources=Resources(),
        workdir=tmp_path / "bundle",
        project_root=project,
    )
    for name in BUNDLE_TEXT_FILES:
        assert b"\r" not in (bundle_root / name).read_bytes(), name


def test_crlf_templates_are_normalised_on_the_way_into_the_bundle(
    project, tmp_path
):
    """A Windows checkout of the repository stores the templates as CRLF."""
    templates = tmp_path / "templates"
    templates.mkdir()
    source = NotebookBundleBuilder()._templates_dir
    for template in source.iterdir():
        if template.is_file():
            crlf = template.read_text(encoding="utf-8").replace("\n", "\r\n")
            (templates / template.name).write_bytes(crlf.encode("utf-8"))

    nb_path = make_notebook(project / "experiment.ipynb")
    bundle_root = NotebookBundleBuilder(templates_dir=templates).build(
        notebook=nb_path,
        job_name="dgx-notebook",
        resources=Resources(),
        workdir=tmp_path / "bundle",
        project_root=project,
    )
    for name in BUNDLE_TEXT_FILES:
        assert b"\r" not in (bundle_root / name).read_bytes(), name


def test_rejects_nonexistent_notebook(builder, project, tmp_path):
    with pytest.raises(BundleError):
        builder.build(
            notebook=project / "missing.ipynb",
            job_name="dgx-notebook",
            resources=Resources(),
            workdir=tmp_path / "bundle",
            project_root=project,
        )


def test_rejects_invalid_json(builder, project, tmp_path):
    bad = project / "bad.ipynb"
    bad.write_text("{not valid json")
    with pytest.raises(BundleError):
        builder.build(
            notebook=bad,
            job_name="dgx-notebook",
            resources=Resources(),
            workdir=tmp_path / "bundle",
            project_root=project,
        )


def test_rejects_non_ipynb_extension(builder, project, tmp_path):
    script = project / "script.py"
    script.write_text("print('hi')")
    with pytest.raises(BundleError):
        builder.build(
            notebook=script,
            job_name="dgx-notebook",
            resources=Resources(),
            workdir=tmp_path / "bundle",
            project_root=project,
        )


def test_copies_notebook_as_payload_notebook_ipynb(builder, project, tmp_path):
    nb_path = make_notebook(project / "my_experiment.ipynb")
    bundle_root = builder.build(
        notebook=nb_path,
        job_name="dgx-notebook",
        resources=Resources(),
        workdir=tmp_path / "bundle",
        project_root=project,
    )
    payload_nb = bundle_root / "payload" / "notebook.ipynb"
    assert payload_nb.exists()
    nbformat.read(str(payload_nb), as_version=4)


def test_copies_safe_includes(builder, project, tmp_path):
    nb_path = make_notebook(project / "experiment.ipynb")
    data_dir = project / "data"
    data_dir.mkdir()
    (data_dir / "input.csv").write_text("a,b\n1,2\n")
    bundle_root = builder.build(
        notebook=nb_path,
        job_name="dgx-notebook",
        resources=Resources(),
        include=[data_dir],
        workdir=tmp_path / "bundle",
        project_root=project,
    )
    assert (bundle_root / "payload" / "data" / "input.csv").read_text() == "a,b\n1,2\n"


def test_rejects_path_traversal(builder, project, tmp_path):
    nb_path = make_notebook(project / "experiment.ipynb")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    traversal_path = project / ".." / "outside.txt"
    with pytest.raises(BundleError):
        builder.build(
            notebook=nb_path,
            job_name="dgx-notebook",
            resources=Resources(),
            include=[traversal_path],
            workdir=tmp_path / "bundle",
            project_root=project,
        )


def test_rejects_symlink_escaping_project(builder, project, tmp_path):
    nb_path = make_notebook(project / "experiment.ipynb")
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "secret.txt").write_text("secret")
    link = project / "linked"
    os.symlink(outside_dir, link)
    with pytest.raises(BundleError):
        builder.build(
            notebook=nb_path,
            job_name="dgx-notebook",
            resources=Resources(),
            include=[link],
            workdir=tmp_path / "bundle",
            project_root=project,
        )


@pytest.mark.parametrize(
    "filename", ["client.ovpn", "server.pem", "id.key", "cert.p12", "cert.pfx"]
)
def test_rejects_vpn_and_credential_files(builder, project, tmp_path, filename):
    nb_path = make_notebook(project / "experiment.ipynb")
    forbidden = project / filename
    forbidden.write_text("secret material")
    with pytest.raises(BundleError):
        builder.build(
            notebook=nb_path,
            job_name="dgx-notebook",
            resources=Resources(),
            include=[forbidden],
            workdir=tmp_path / "bundle",
            project_root=project,
        )


def test_does_not_include_ipynb_checkpoints(builder, project, tmp_path):
    nb_path = make_notebook(project / "experiment.ipynb")
    checkpoints = project / ".ipynb_checkpoints"
    checkpoints.mkdir()
    (checkpoints / "experiment-checkpoint.ipynb").write_text("{}")
    src_dir = project / "src"
    src_dir.mkdir()
    (src_dir / "helper.py").write_text("x = 1")
    (src_dir / ".ipynb_checkpoints").mkdir()
    (src_dir / ".ipynb_checkpoints" / "helper-checkpoint.py").write_text("x = 1")
    bundle_root = builder.build(
        notebook=nb_path,
        job_name="dgx-notebook",
        resources=Resources(),
        include=[src_dir],
        workdir=tmp_path / "bundle",
        project_root=project,
    )
    assert not (bundle_root / "payload" / "src" / ".ipynb_checkpoints").exists()


def test_generates_outputs_and_logs_dirs(builder, project, tmp_path):
    nb_path = make_notebook(project / "experiment.ipynb")
    bundle_root = builder.build(
        notebook=nb_path,
        job_name="dgx-notebook",
        resources=Resources(),
        workdir=tmp_path / "bundle",
        project_root=project,
    )
    assert (bundle_root / "outputs").is_dir()
    assert (bundle_root / "logs").is_dir()


def test_manifest_has_no_secrets(builder, project, tmp_path):
    nb_path = make_notebook(project / "experiment.ipynb")
    bundle_root = builder.build(
        notebook=nb_path,
        job_name="dgx-notebook",
        resources=Resources(gpus=2, cpus=4, partition="devwork"),
        workdir=tmp_path / "bundle",
        project_root=project,
    )
    manifest = json.loads((bundle_root / "manifest.json").read_text())
    assert manifest["job_name"] == "dgx-notebook"
    assert manifest["resources"]["gpus"] == 2
    dumped = json.dumps(manifest).lower()
    for forbidden in ("password", "secret", "private_key", "ovpn_contents"):
        assert forbidden not in dumped
