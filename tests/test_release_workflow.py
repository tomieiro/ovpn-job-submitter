from pathlib import Path


WORKFLOW = Path(".github/workflows/release.yml")


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
