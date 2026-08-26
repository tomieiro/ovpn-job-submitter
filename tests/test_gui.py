import gc
import sys
import time
import tomllib
from pathlib import Path

import pytest

tk = pytest.importorskip("tkinter")

from dgx_slurm import gui
from dgx_slurm.errors import ConfigurationError
from dgx_slurm.gui import QueueWriter, captured_output, validate_selection


@pytest.fixture
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # headless runners have no display
        pytest.skip(f"no display available: {exc}")
    root.withdraw()
    yield root
    # Drop the app (still held by its pending `after` callback) before the
    # interpreter goes away, or its Tk variables complain while being freed.
    for callback_id in root.tk.call("after", "info"):
        root.after_cancel(str(callback_id))
    gc.collect()
    root.destroy()


@pytest.fixture
def layout(tmp_path):
    notebook = tmp_path / "experiment.ipynb"
    notebook.write_text('{"cells": []}')
    vpn_dir = tmp_path / "SSH"
    vpn_dir.mkdir()
    (vpn_dir / "c4ai.icmc.usp.br.ovpn").write_text("client\n")
    return notebook, vpn_dir


def test_accepts_a_notebook_and_a_vpn_directory(layout):
    notebook, vpn_dir = layout
    assert validate_selection(str(notebook), str(vpn_dir)) == (notebook, vpn_dir)


def test_requires_a_notebook(layout):
    _, vpn_dir = layout
    with pytest.raises(ConfigurationError, match="Escolha o notebook"):
        validate_selection("   ", str(vpn_dir))


def test_requires_a_vpn_directory(layout):
    notebook, _ = layout
    with pytest.raises(ConfigurationError, match="Escolha a pasta"):
        validate_selection(str(notebook), "")


def test_rejects_missing_notebook(tmp_path, layout):
    _, vpn_dir = layout
    with pytest.raises(ConfigurationError, match="não encontrado"):
        validate_selection(str(tmp_path / "absent.ipynb"), str(vpn_dir))


def test_rejects_file_that_is_not_a_notebook(tmp_path, layout):
    _, vpn_dir = layout
    other = tmp_path / "data.nc"
    other.write_text("data")
    with pytest.raises(ConfigurationError, match=r"\.ipynb"):
        validate_selection(str(other), str(vpn_dir))


def test_rejects_missing_vpn_directory(tmp_path, layout):
    notebook, _ = layout
    with pytest.raises(ConfigurationError, match="Pasta da VPN"):
        validate_selection(str(notebook), str(tmp_path / "absent"))


def test_rejects_vpn_directory_without_ovpn(tmp_path, layout):
    notebook, _ = layout
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ConfigurationError, match="Nenhum arquivo .ovpn"):
        validate_selection(str(notebook), str(empty))


def test_queue_writer_forwards_text():
    written = []
    writer = QueueWriter(written.append)
    assert writer.write("linha\n") == len("linha\n")
    writer.write("")
    writer.flush()
    assert written == ["linha\n"]
    assert writer.isatty() is False


def test_captured_output_routes_prints_and_restores_streams():
    written = []
    saved_stdout, saved_stderr = sys.stdout, sys.stderr

    with captured_output(QueueWriter(written.append)):
        print("progresso")
        print("falha", file=sys.stderr)

    assert "progresso\n" in "".join(written)
    assert "falha\n" in "".join(written)
    assert sys.stdout is saved_stdout
    assert sys.stderr is saved_stderr


def test_captured_output_restores_streams_after_failure():
    saved_stdout = sys.stdout
    with pytest.raises(RuntimeError):
        with captured_output(QueueWriter(lambda _text: None)):
            raise RuntimeError("boom")
    assert sys.stdout is saved_stdout


def pump(root, until, timeout=10.0):
    """Run the Tk event loop until a condition holds, as mainloop would."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        root.update()
        if until():
            return True
        time.sleep(0.02)
    return False


def test_window_runs_the_job_and_reports_the_executed_notebook(
    tk_root, layout, monkeypatch
):
    notebook, vpn_dir = layout
    calls = {}
    shown = []

    def fake_runner(selected_notebook, **kwargs):
        calls["notebook"] = selected_notebook
        calls.update(kwargs)
        print("Job submetido: 48192")
        return None

    monkeypatch.setattr(gui.simpledialog, "askstring", lambda *a, **k: "senha")
    monkeypatch.setattr(gui.messagebox, "showinfo", lambda *args: shown.append(args))

    app = gui.SubmitterApp(tk_root, runner=fake_runner, is_elevated=lambda: True)
    app.submit_panel.notebook.set(str(notebook))
    app.submit_panel.vpn_dir.set(str(vpn_dir))
    app.submit_panel.include_files.set(True)
    app.submit_panel.start_job()

    assert pump(tk_root, lambda: bool(shown))
    assert calls["notebook"] == notebook
    assert calls["vpn_dir"] == vpn_dir
    assert calls["include_project_files"] is True
    assert calls["password_provider"]() == "senha"
    assert "Job submetido: 48192" in app._log.get("1.0", "end")
    assert str(notebook.with_name("experiment.executed.ipynb")) in shown[-1][1]


def test_window_reports_failures_without_closing(tk_root, layout, monkeypatch):
    notebook, vpn_dir = layout
    shown = []

    def failing_runner(*_args, **_kwargs):
        raise gui.DGXError("VPN caiu")

    monkeypatch.setattr(gui.simpledialog, "askstring", lambda *a, **k: "senha")
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args: shown.append(args))

    app = gui.SubmitterApp(tk_root, runner=failing_runner, is_elevated=lambda: True)
    app.submit_panel.notebook.set(str(notebook))
    app.submit_panel.vpn_dir.set(str(vpn_dir))
    app.submit_panel.start_job()

    assert pump(tk_root, lambda: bool(shown))
    assert "VPN caiu" in shown[-1][1]
    assert app.submit_panel.run_button.instate(["!disabled"])


def test_host_key_question_crosses_from_the_job_thread_to_the_window(
    tk_root, monkeypatch
):
    import threading

    asked = []
    monkeypatch.setattr(
        gui.messagebox,
        "askyesno",
        lambda _title, message: asked.append(message) or True,
    )
    app = gui.SubmitterApp(tk_root, is_elevated=lambda: True)
    answers = []

    worker = threading.Thread(
        target=lambda: answers.append(
            app.confirm_host_key("c4aiscm2", "SHA256:abc")
        ),
        daemon=True,
    )
    worker.start()

    assert pump(tk_root, lambda: bool(answers))
    assert answers == [True]
    assert "SHA256:abc" in asked[0]


def test_invalid_selection_never_starts_a_job(tk_root, monkeypatch):
    shown = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args: shown.append(args))

    def unexpected_runner(*_args, **_kwargs):
        raise AssertionError("job must not start")

    app = gui.SubmitterApp(tk_root, runner=unexpected_runner, is_elevated=lambda: True)
    app.submit_panel.start_job()

    assert app._worker is None
    assert "Escolha o notebook" in shown[-1][1]


def test_gui_entry_point_is_declared_as_a_windowed_script():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    scripts = pyproject["project"]["gui-scripts"]
    assert scripts["ovpn-job-submitter-gui"] == "dgx_slurm.gui:main"


def test_windows_release_builds_the_gui_executable():
    workflow = Path(".github/workflows/release.yml").read_text()
    build_start = workflow.index("- name: Build Windows GUI executable")
    build_step = workflow[build_start:workflow.index("- name: Upload executable")]

    assert "--windowed" in build_step
    assert "--additional-hooks-dir pyinstaller-hooks" in build_step
    assert "submit_notebook_gui.py" in build_step
    assert "ovpn-job-submitter-windows-x86_64-gui.exe" in workflow


@pytest.fixture
def store(tmp_path):
    from dgx_slurm.storage import LocalJobStore

    return LocalJobStore(tmp_path / "jobs.json")


def saved_job(notebook, vpn_dir, *, submitted_at):
    return {
        "job_name": "dgx-notebook-abc",
        "notebook": str(notebook),
        "output": str(notebook.with_name(f"{notebook.stem}.executed.ipynb")),
        "vpn_dir": str(vpn_dir),
        "ssh_host": "c4aiscm2",
        "ssh_port": 22,
        "submitted_at": submitted_at,
    }


def test_sorted_jobs_puts_the_newest_first():
    records = {
        "48100": {"submitted_at": "2026-08-25T17:40:00+00:00"},
        "48213": {"submitted_at": "2026-08-26T09:12:00+00:00"},
    }
    assert [job_id for job_id, _ in gui.sorted_jobs(records)] == ["48213", "48100"]


def test_sorted_jobs_keeps_records_that_predate_the_timestamp():
    """Jobs submitted by an older build have no date; they must still show."""
    records = {"48100": {}, "48213": {"submitted_at": "2026-08-26T09:12:00+00:00"}}
    assert [job_id for job_id, _ in gui.sorted_jobs(records)] == ["48213", "48100"]


def test_format_submitted_falls_back_when_there_is_no_usable_date():
    assert gui.format_submitted(None) == "—"
    assert gui.format_submitted("") == "—"
    assert gui.format_submitted("ontem") == "—"
    assert gui.format_submitted("2026-08-26T09:12:00+00:00") != "—"


def test_detach_button_submits_without_waiting(tk_root, layout, monkeypatch, store):
    notebook, vpn_dir = layout
    calls = {}
    shown = []

    def fake_runner(selected_notebook, **kwargs):
        calls["notebook"] = selected_notebook
        calls.update(kwargs)
        return "48213"

    monkeypatch.setattr(gui.simpledialog, "askstring", lambda *a, **k: "senha")
    monkeypatch.setattr(gui.messagebox, "showinfo", lambda *args: shown.append(args))

    app = gui.SubmitterApp(
        tk_root, runner=fake_runner, store=store, is_elevated=lambda: True
    )
    app.submit_panel.notebook.set(str(notebook))
    app.submit_panel.vpn_dir.set(str(vpn_dir))
    app.submit_panel.start_job(detach=True)

    assert pump(tk_root, lambda: bool(shown))
    assert calls["detach"] is True
    message = shown[-1][1]
    assert "48213" in message
    assert "Coletar resultados" in message
    assert "executado salvo" not in message


def test_collect_tab_lists_the_saved_jobs(tk_root, layout, store):
    notebook, vpn_dir = layout
    store.save("48100", saved_job(notebook, vpn_dir, submitted_at="2026-08-25T17:40:00+00:00"))
    store.save("48213", saved_job(notebook, vpn_dir, submitted_at="2026-08-26T09:12:00+00:00"))

    app = gui.SubmitterApp(tk_root, store=store, is_elevated=lambda: True)
    tree = app.collect_panel.tree

    assert list(tree.get_children()) == ["48213", "48100"]
    assert tree.set("48213", "notebook") == "experiment.ipynb"


def test_collect_sends_the_selected_job_and_keeps_four_remote_dirs(
    tk_root, layout, monkeypatch, store
):
    notebook, vpn_dir = layout
    store.save("48213", saved_job(notebook, vpn_dir, submitted_at="2026-08-26T09:12:00+00:00"))
    calls = {}
    shown = []

    def fake_collector(job_id, **kwargs):
        calls["job_id"] = job_id
        calls.update(kwargs)
        return gui.JobResult(
            job_id=job_id,
            state=gui.JobState.COMPLETED,
            exit_code=0,
            stdout="",
            stderr="",
            executed_notebook=notebook.with_name("experiment.executed.ipynb"),
        )

    monkeypatch.setattr(gui.simpledialog, "askstring", lambda *a, **k: "senha")
    monkeypatch.setattr(gui.messagebox, "showinfo", lambda *args: shown.append(args))

    app = gui.SubmitterApp(
        tk_root, collector=fake_collector, store=store, is_elevated=lambda: True
    )
    app.collect_panel.tree.selection_set("48213")
    app.collect_panel.collect_selected()

    assert pump(tk_root, lambda: bool(shown))
    assert calls["job_id"] == "48213"
    assert calls["keep_remote"] == 4
    assert calls["save_logs"] is True
    assert calls["password_provider"]() == "senha"
    assert "experiment.executed.ipynb" in shown[-1][1]


def test_collect_reports_a_job_that_produced_no_notebook(
    tk_root, layout, monkeypatch, store
):
    """A failed job still collects: its logs are the only artefact there is."""
    notebook, vpn_dir = layout
    store.save("48213", saved_job(notebook, vpn_dir, submitted_at="2026-08-26T09:12:00+00:00"))
    shown = []

    def fake_collector(job_id, **_kwargs):
        return gui.JobResult(
            job_id=job_id,
            state=gui.JobState.FAILED,
            exit_code=1,
            stdout="",
            stderr="mount: /home nao montado",
            executed_notebook=None,
        )

    monkeypatch.setattr(gui.simpledialog, "askstring", lambda *a, **k: "senha")
    monkeypatch.setattr(gui.messagebox, "showinfo", lambda *args: shown.append(args))

    app = gui.SubmitterApp(
        tk_root, collector=fake_collector, store=store, is_elevated=lambda: True
    )
    app.collect_panel.tree.selection_set("48213")
    app.collect_panel.collect_selected()

    assert pump(tk_root, lambda: bool(shown))
    assert "failed" in shown[-1][1].lower()
    assert "log" in shown[-1][1].lower()


def test_collect_without_a_selection_never_connects(tk_root, monkeypatch, store):
    shown = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args: shown.append(args))

    def unexpected_collector(*_args, **_kwargs):
        raise AssertionError("collection must not start")

    app = gui.SubmitterApp(
        tk_root, collector=unexpected_collector, store=store, is_elevated=lambda: True
    )
    app.collect_panel.collect_selected()

    assert app._worker is None
    assert "Escolha um job" in shown[-1][1]


def test_collect_refuses_a_job_saved_before_the_metadata_existed(
    tk_root, monkeypatch, store
):
    store.save("48100", {"job_name": "dgx-notebook-abc"})
    shown = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args: shown.append(args))

    def unexpected_collector(*_args, **_kwargs):
        raise AssertionError("collection must not start")

    app = gui.SubmitterApp(
        tk_root, collector=unexpected_collector, store=store, is_elevated=lambda: True
    )
    app.collect_panel.tree.selection_set("48100")
    app.collect_panel.collect_selected()

    assert app._worker is None
    assert "versão antiga" in shown[-1][1]


def test_detached_submission_refreshes_the_collect_list(
    tk_root, layout, monkeypatch, store
):
    """The job the user just sent must appear without reopening the program."""
    notebook, vpn_dir = layout
    shown = []

    def fake_runner(_notebook, **_kwargs):
        store.save(
            "48213",
            saved_job(notebook, vpn_dir, submitted_at="2026-08-26T09:12:00+00:00"),
        )
        return "48213"

    monkeypatch.setattr(gui.simpledialog, "askstring", lambda *a, **k: "senha")
    monkeypatch.setattr(gui.messagebox, "showinfo", lambda *args: shown.append(args))

    app = gui.SubmitterApp(
        tk_root, runner=fake_runner, store=store, is_elevated=lambda: True
    )
    assert list(app.collect_panel.tree.get_children()) == []

    app.submit_panel.notebook.set(str(notebook))
    app.submit_panel.vpn_dir.set(str(vpn_dir))
    app.submit_panel.start_job(detach=True)

    assert pump(tk_root, lambda: bool(shown))
    assert list(app.collect_panel.tree.get_children()) == ["48213"]


def test_both_screens_are_disabled_while_a_job_runs(tk_root, layout, monkeypatch, store):
    notebook, vpn_dir = layout
    released = []
    shown = []

    def slow_runner(*_args, **_kwargs):
        while not released:
            time.sleep(0.01)
        return None

    monkeypatch.setattr(gui.simpledialog, "askstring", lambda *a, **k: "senha")
    monkeypatch.setattr(gui.messagebox, "showinfo", lambda *args: shown.append(args))

    app = gui.SubmitterApp(
        tk_root, runner=slow_runner, store=store, is_elevated=lambda: True
    )
    app.submit_panel.notebook.set(str(notebook))
    app.submit_panel.vpn_dir.set(str(vpn_dir))
    app.submit_panel.start_job()

    assert pump(tk_root, lambda: app.submit_panel.run_button.instate(["disabled"]))
    assert app.submit_panel.detach_button.instate(["disabled"])
    assert app.collect_panel.collect_button.instate(["disabled"])

    released.append(True)
    assert pump(tk_root, lambda: bool(shown))
    assert app.collect_panel.collect_button.instate(["!disabled"])
