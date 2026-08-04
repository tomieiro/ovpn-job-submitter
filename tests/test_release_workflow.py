from pathlib import Path


WORKFLOW = Path(".github/workflows/release.yml")
HOOKS_DIR = Path("pyinstaller-hooks")
TEMPLATES_DIR = Path("src/dgx_slurm/templates")


def test_frozen_build_bundles_every_template_including_python_ones():
    """--collect-data drops .py data files, so the runner template went missing."""
    content = WORKFLOW.read_text()
    build_start = content.index("- name: Build standalone executable")
    build_step = content[build_start:content.index("- name: Smoke-test Linux executable")]

    assert f"--additional-hooks-dir {HOOKS_DIR.name}" in build_step
    assert "--collect-data dgx_slurm" not in build_step

    hook = (HOOKS_DIR / "hook-dgx_slurm.py").read_text()
    assert "include_py_files=True" in hook
    assert "templates/*" in hook

    assert any(path.suffix == ".py" for path in TEMPLATES_DIR.iterdir())


def test_macos_build_forces_known_universal_cryptography_wheel():
    content = WORKFLOW.read_text()
    install_start = content.index(
        "- name: Install compatible macOS cryptography wheel"
    )
    application_install = content.index("- name: Install application and packager")
    build_start = content.index("- name: Build standalone executable")
    install_step = content[install_start:application_install]

    assert "if: matrix.platform == 'macos'" in install_step
    assert "--only-binary=:all:" in install_step
    assert "cryptography==48.0.1" in install_step
    assert install_start < application_install < build_start


def test_every_platform_smoke_tests_the_frozen_executable():
    content = WORKFLOW.read_text()
    assert "./dist/ovpn-job-submitter --help" in content
    assert '& ".\\dist\\ovpn-job-submitter.exe" --help' in content


def test_release_command_has_explicit_repository_context():
    content = WORKFLOW.read_text()
    release_start = content.index("- name: Publish GitHub Release")
    release_step = content[release_start:]

    assert "GH_TOKEN: ${{ github.token }}" in release_step
    assert "GH_REPO: ${{ github.repository }}" in release_step
    assert 'gh release create "${GITHUB_REF_NAME}"' in release_step
