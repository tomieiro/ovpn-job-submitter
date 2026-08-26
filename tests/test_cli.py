from dgx_slurm.cli import (
    DEFAULT_CPUS,
    DEFAULT_GPUS,
    DEFAULT_MEMORY,
    DEFAULT_PARTITION,
    DEFAULT_SSH_HOST,
    DEFAULT_SSH_PORT,
    DEFAULT_TIME_LIMIT,
    confirm_host_key,
    main,
)
from dgx_slurm.errors import ConfigurationError


def test_cli_passes_required_paths_and_defaults():
    calls = []

    def runner(notebook, **kwargs):
        calls.append((notebook, kwargs))

    assert main(["project/test.ipynb", "SSH"], runner=runner) == 0
    notebook, options = calls[0]
    assert notebook == "project/test.ipynb"
    assert options.pop("host_key_confirmer") is confirm_host_key
    assert options == {
        "include_project_files": False,
        "vpn_dir": "SSH",
        "ssh_host": DEFAULT_SSH_HOST,
        "ssh_port": DEFAULT_SSH_PORT,
        "partition": DEFAULT_PARTITION,
        "gpus": DEFAULT_GPUS,
        "cpus": DEFAULT_CPUS,
        "memory": DEFAULT_MEMORY,
        "time_limit": DEFAULT_TIME_LIMIT,
        "detach": False,
    }


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
    options.pop("host_key_confirmer")
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
        "detach": False,
    }


def test_cli_async_detaches_after_submission():
    calls = []

    def runner(notebook, **kwargs):
        calls.append((notebook, kwargs))

    assert main(["experiment.ipynb", "vpn", "--async"], runner=runner) == 0
    assert calls[0][1]["detach"] is True


def test_cli_collects_by_id_without_submission_paths():
    calls = []

    def collector(job_id, **kwargs):
        calls.append((job_id, kwargs))

    assert main(["--collect-results", "48192"], collector=collector) == 0
    assert calls[0][0] == "48192"
    assert calls[0][1]["host_key_confirmer"] is confirm_host_key
    assert calls[0][1]["save_logs"] is False


def test_cli_collect_even_error_requests_log_persistence():
    calls = []

    def collector(job_id, **kwargs):
        calls.append((job_id, kwargs))

    assert main(
        ["--collect-results", "48192", "--collect-even-error"],
        collector=collector,
    ) == 0
    assert calls[0][1]["save_logs"] is True


def test_save_logs_writes_stdout_and_stderr_even_when_empty(tmp_path):
    from dgx_slurm.models import JobResult, JobState
    from dgx_slurm.workflow import _save_logs

    # Um job que morre antes de escrever em outputs/ e o caso que motiva a
    # flag: outputs vazio, e o log e a unica evidencia.
    result = JobResult(
        job_id="48192", state=JobState.FAILED, exit_code=1,
        stdout="docker build falhou\n", stderr="", executed_notebook=None,
    )
    saved = _save_logs(result, tmp_path)

    assert {p.name for p in saved} == {"job.out", "job.err"}
    assert (tmp_path / "job.out").read_text() == "docker build falhou\n"
    assert (tmp_path / "job.err").read_text() == ""


def test_cli_host_key_prompt_accepts_only_an_explicit_yes(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", type("TTY", (), {"isatty": lambda self: True})())
    monkeypatch.setattr("builtins.input", lambda _prompt: "s")
    assert confirm_host_key("c4aiscm2", "SHA256:abc") is True

    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    assert confirm_host_key("c4aiscm2", "SHA256:abc") is False
    assert "SHA256:abc" in capsys.readouterr().out


def test_cli_host_key_prompt_refuses_without_a_terminal(monkeypatch):
    monkeypatch.setattr("sys.stdin", type("Pipe", (), {"isatty": lambda self: False})())
    assert confirm_host_key("c4aiscm2", "SHA256:abc") is False


def test_cli_prints_library_errors_without_traceback(capsys):
    def runner(_notebook, **_kwargs):
        raise ConfigurationError("notebook inválido")

    assert main(["invalid.ipynb", "vpn"], runner=runner) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Erro: notebook inválido\n"


def test_cli_without_arguments_reports_a_configuration_error(capsys):
    """The guard clauses raise ConfigurationError, so it must be imported."""
    assert main([]) == 1
    assert "Erro:" in capsys.readouterr().err


def test_cli_rejects_positional_arguments_with_collect_results(capsys):
    def collector(*_args, **_kwargs):
        raise AssertionError("collection must not start")

    assert main(
        ["notebook.ipynb", "vpn", "--collect-results", "48192"], collector=collector
    ) == 1
    assert "Erro:" in capsys.readouterr().err
