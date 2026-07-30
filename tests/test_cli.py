from dgx_slurm.cli import (
    DEFAULT_CPUS,
    DEFAULT_GPUS,
    DEFAULT_MEMORY,
    DEFAULT_PARTITION,
    DEFAULT_SSH_HOST,
    DEFAULT_SSH_PORT,
    DEFAULT_TIME_LIMIT,
    main,
)
from dgx_slurm.errors import ConfigurationError


def test_cli_passes_required_paths_and_defaults():
    calls = []

    def runner(notebook, **kwargs):
        calls.append((notebook, kwargs))

    assert main(["project/test.ipynb", "SSH"], runner=runner) == 0
    assert calls == [
        (
            "project/test.ipynb",
            {
                "include_project_files": False,
                "vpn_dir": "SSH",
                "ssh_host": DEFAULT_SSH_HOST,
                "ssh_port": DEFAULT_SSH_PORT,
                "partition": DEFAULT_PARTITION,
                "gpus": DEFAULT_GPUS,
                "cpus": DEFAULT_CPUS,
                "memory": DEFAULT_MEMORY,
                "time_limit": DEFAULT_TIME_LIMIT,
            },
        )
    ]


def test_cli_include_files_and_optional_overrides():
    calls = []

    def runner(notebook, **kwargs):
        calls.append((notebook, kwargs))

    exit_code = main(
        [
            "experiment.ipynb",
            "vpn",
            "--include-files",
            "--ssh-host",
            "login.internal",
            "--ssh-port",
            "2222",
            "--partition",
            "research",
            "--gpus",
            "2",
            "--cpus",
            "16",
            "--memory",
            "128G",
            "--time-limit",
            "02:30:00",
        ],
        runner=runner,
    )

    assert exit_code == 0
    _, options = calls[0]
    assert options == {
        "include_project_files": True,
        "vpn_dir": "vpn",
        "ssh_host": "login.internal",
        "ssh_port": 2222,
        "partition": "research",
        "gpus": 2,
        "cpus": 16,
        "memory": "128G",
        "time_limit": "02:30:00",
    }


def test_cli_prints_library_errors_without_traceback(capsys):
    def runner(_notebook, **_kwargs):
        raise ConfigurationError("notebook inválido")

    assert main(["invalid.ipynb", "vpn"], runner=runner) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Erro: notebook inválido\n"
