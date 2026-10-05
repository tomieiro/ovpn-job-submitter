"""Desktop front-end for submitting a notebook and collecting it later.

Tkinter ships with CPython, so the window costs no extra dependency and is
bundled by PyInstaller as-is. The window owns a single worker thread and a
single log pane; each screen is a panel that hands it one unit of work and a
message to show when that work succeeds.
"""

from __future__ import annotations

import contextlib
import queue
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from .cli import (
    DEFAULT_GPUS,
    DEFAULT_MEMORY,
    DEFAULT_PARTITION,
    DEFAULT_SSH_HOST,
    DEFAULT_SSH_PORT,
    DEFAULT_TIME_LIMIT,
)
from .client import DEFAULT_JOB_STORE_PATH
from .errors import ConfigurationError, DGXError
from .models import JobResult, JobState
from .storage import LocalJobStore
from .vpn import windows_process_is_elevated
from .workflow import collect_results, discover_ovpn, discover_username, run_notebook

WINDOW_TITLE = "ovpn-job-submitter"
MISSING = "—"

# Mirrors the CLI's --keep-remote default. Fixed rather than exposed: the window
# exists for people who should not have to reason about cluster housekeeping.
KEEP_REMOTE = 4

ELEVATION_WARNING = (
    "Sem privilégios de administrador o OpenVPN não consegue configurar o "
    "adaptador da VPN. Reabra como administrador."
)
DETACH_HINT = (
    '"Enviar sem aguardar" devolve o programa assim que o job entra na fila. '
    "Você pode fechar tudo e baixar o resultado depois, na aba "
    '"Coletar resultados".'
)
COLLECT_HINT = (
    "Jobs enviados a partir deste computador. Escolha um que já tenha "
    "terminado no cluster para baixar o notebook executado e os logs."
)


def validate_selection(notebook: str, vpn_dir: str) -> tuple[Path, Path]:
    """Check both selections before any connection attempt is made."""
    if not notebook.strip():
        raise ConfigurationError("Escolha o notebook (.ipynb).")
    if not vpn_dir.strip():
        raise ConfigurationError("Escolha a pasta com o .ovpn e os certificados.")

    notebook_path = Path(notebook).expanduser()
    if not notebook_path.is_file():
        raise ConfigurationError(f"Notebook não encontrado: {notebook_path}")
    if notebook_path.suffix.lower() != ".ipynb":
        raise ConfigurationError("O arquivo escolhido não é um notebook .ipynb.")

    vpn_path = Path(vpn_dir).expanduser()
    if not vpn_path.is_dir():
        raise ConfigurationError(f"Pasta da VPN não encontrada: {vpn_path}")
    if not sorted(vpn_path.glob("*.ovpn")):
        raise ConfigurationError(f"Nenhum arquivo .ovpn na pasta: {vpn_path}")

    return notebook_path, vpn_path


def sorted_jobs(records: dict[str, dict]) -> list[tuple[str, dict]]:
    """Saved jobs, newest first.

    Jobs submitted before ``submitted_at`` was recorded have no date; they sort
    last by job id rather than disappearing, since their outputs may still be
    waiting on the cluster.
    """

    def key(item: tuple[str, dict]) -> tuple[str, int]:
        job_id, record = item
        return (
            record.get("submitted_at") or "",
            int(job_id) if job_id.isdigit() else 0,
        )

    return sorted(records.items(), key=key, reverse=True)


def format_submitted(value: str | None) -> str:
    """Render a stored ISO timestamp in local time, or a dash when unusable."""
    if not value:
        return MISSING
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return MISSING
    if moment.tzinfo is not None:
        moment = moment.astimezone()
    return moment.strftime("%d/%m %H:%M")


class QueueWriter:
    """Text stream that forwards everything printed to the GUI thread."""

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit

    def write(self, text: str) -> int:
        if text:
            self._emit(text)
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


@contextlib.contextmanager
def captured_output(stream: QueueWriter):
    """Send stdout and stderr to the log pane for the duration of the job."""
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = stream
    try:
        yield
    finally:
        sys.stdout, sys.stderr = saved_stdout, saved_stderr


class SubmitPanel(ttk.Frame):
    """First screen: pick a notebook, pick the VPN folder, send it."""

    def __init__(self, master: tk.Misc, *, shell: "SubmitterApp", runner) -> None:
        super().__init__(master, padding=12)
        self._shell = shell
        self._runner = runner

        self.notebook = tk.StringVar()
        self.vpn_dir = tk.StringVar()
        self.include_files = tk.BooleanVar(value=False)
        self.gpus = tk.StringVar(value=str(DEFAULT_GPUS))
        self.cpus = tk.StringVar(value=str(DEFAULT_GPUS * 4))
        self.time_limit = tk.StringVar(value=DEFAULT_TIME_LIMIT)
        self.gpus.trace_add("write", self._update_default_cpus)

        self._build()

    def _build(self) -> None:
        self.columnconfigure(1, weight=1)

        ttk.Label(self, text="Notebook (.ipynb)").grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        ttk.Entry(self, textvariable=self.notebook).grid(
            row=0, column=1, sticky="ew", padx=8, pady=(0, 4)
        )
        ttk.Button(self, text="Escolher...", command=self._choose_notebook).grid(
            row=0, column=2, pady=(0, 4)
        )

        ttk.Label(self, text="Pasta da VPN (.ovpn + certificados)").grid(
            row=1, column=0, sticky="w", pady=4
        )
        ttk.Entry(self, textvariable=self.vpn_dir).grid(
            row=1, column=1, sticky="ew", padx=8, pady=4
        )
        ttk.Button(self, text="Escolher...", command=self._choose_vpn_dir).grid(
            row=1, column=2, pady=4
        )

        ttk.Checkbutton(
            self,
            text="Enviar também os outros arquivos da pasta do notebook",
            variable=self.include_files,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 4))

        resources = ttk.Frame(self)
        resources.grid(row=3, column=0, columnspan=3, sticky="ew", pady=4)
        for column in range(3):
            resources.columnconfigure(column, weight=1)
        self._resource_field(resources, "Horas (HH:MM:SS)", self.time_limit, 0)
        self._resource_field(resources, "GPUs", self.gpus, 1)
        self._resource_field(resources, "CPUs", self.cpus, 2)

        buttons = ttk.Frame(self)
        buttons.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        self.run_button = ttk.Button(
            buttons, text="Executar e aguardar", command=self.start_job
        )
        self.run_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.detach_button = ttk.Button(
            buttons,
            text="Enviar sem aguardar",
            command=lambda: self.start_job(detach=True),
        )
        self.detach_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        ttk.Label(self, text=DETACH_HINT, wraplength=560).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )

    def _resource_field(
        self, master: ttk.Frame, label: str, variable: tk.StringVar, column: int
    ) -> None:
        field = ttk.Frame(master)
        field.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 8, 8))
        ttk.Label(field, text=label).pack(anchor="w")
        ttk.Entry(field, textvariable=variable, width=14).pack(fill="x")

    def _update_default_cpus(self, *_args) -> None:
        try:
            self.cpus.set(str(int(self.gpus.get()) * 4))
        except ValueError:
            pass

    def _choose_notebook(self) -> None:
        selected = filedialog.askopenfilename(
            title="Escolha o notebook",
            filetypes=[("Notebook Jupyter", "*.ipynb"), ("Todos os arquivos", "*.*")],
        )
        if selected:
            self.notebook.set(selected)

    def _choose_vpn_dir(self) -> None:
        selected = filedialog.askdirectory(title="Escolha a pasta da VPN")
        if selected:
            self.vpn_dir.set(selected)

    def start_job(self, *, detach: bool = False) -> None:
        if self._shell.busy():
            return

        try:
            notebook, vpn_dir = validate_selection(
                self.notebook.get(), self.vpn_dir.get()
            )
            username = discover_username(discover_ovpn(vpn_dir))
        except DGXError as exc:
            messagebox.showerror(WINDOW_TITLE, str(exc))
            return

        password = self._shell.ask_password(f"Senha de {username}@{DEFAULT_SSH_HOST}:")
        if not password:
            return

        include_files = self.include_files.get()
        try:
            gpus = int(self.gpus.get())
            cpus = int(self.cpus.get())
        except ValueError:
            messagebox.showerror(WINDOW_TITLE, "GPUs e CPUs devem ser números inteiros.")
            return
        if gpus < 1 or cpus < 1:
            messagebox.showerror(WINDOW_TITLE, "GPUs e CPUs devem ser maiores que zero.")
            return
        time_limit = self.time_limit.get().strip()
        executed = notebook.with_name(f"{notebook.stem}.executed.ipynb")

        def work() -> str:
            outcome = self._runner(
                notebook,
                include_project_files=include_files,
                vpn_dir=vpn_dir,
                ssh_host=DEFAULT_SSH_HOST,
                ssh_port=DEFAULT_SSH_PORT,
                partition=DEFAULT_PARTITION,
                gpus=gpus,
                cpus=cpus,
                memory=DEFAULT_MEMORY,
                time_limit=time_limit,
                detach=detach,
                password_provider=lambda: password,
                host_key_confirmer=self._shell.confirm_host_key,
            )
            if detach:
                return (
                    f"Job {outcome} enviado.\n\n"
                    "Pode fechar o programa. Quando o job terminar no cluster, "
                    'abra a aba "Coletar resultados" e baixe o notebook.'
                )
            return f"Notebook executado salvo em:\n{executed}"

        self._shell.run_in_worker(work)

    def set_busy(self, busy: bool) -> None:
        state = ["disabled"] if busy else ["!disabled"]
        self.run_button.state(state)
        self.detach_button.state(state)


class CollectPanel(ttk.Frame):
    """Second screen: pick a job sent earlier and download what it left behind.

    Logs are always saved. ``download_outputs`` fetches only ``outputs/``, so a
    job that died before writing anything yields an empty directory -- and for a
    failed job the log is the only artefact there is.
    """

    COLUMNS = ("job", "notebook", "submitted")
    HEADINGS = {"job": "Job", "notebook": "Notebook", "submitted": "Enviado em"}

    def __init__(
        self,
        master: tk.Misc,
        *,
        shell: "SubmitterApp",
        collector,
        store: LocalJobStore | None = None,
    ) -> None:
        super().__init__(master, padding=12)
        self._shell = shell
        self._collector = collector
        self._store = store or LocalJobStore(DEFAULT_JOB_STORE_PATH)

        self._build()
        self.refresh()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)

        ttk.Label(self, text=COLLECT_HINT, wraplength=560).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        self.tree = ttk.Treeview(
            self,
            columns=self.COLUMNS,
            show="headings",
            height=6,
            selectmode="browse",
        )
        for column in self.COLUMNS:
            self.tree.heading(column, text=self.HEADINGS[column])
        self.tree.column("job", width=90, anchor="w", stretch=False)
        self.tree.column("notebook", width=320, anchor="w")
        self.tree.column("submitted", width=120, anchor="w", stretch=False)
        self.tree.grid(row=1, column=0, sticky="nsew")
        self.rowconfigure(1, weight=1)

        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)

        buttons = ttk.Frame(self)
        buttons.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        self.collect_button = ttk.Button(
            buttons, text="Coletar selecionado", command=self.collect_selected
        )
        self.collect_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.refresh_button = ttk.Button(
            buttons, text="Atualizar lista", command=self.refresh
        )
        self.refresh_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

    def refresh(self) -> None:
        """Rebuild the list from the store, keeping the current selection."""
        selected = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        for job_id, record in sorted_jobs(self._store.list_all()):
            notebook = record.get("notebook")
            self.tree.insert(
                "",
                "end",
                iid=job_id,
                values=(
                    job_id,
                    Path(notebook).name if notebook else MISSING,
                    format_submitted(record.get("submitted_at")),
                ),
            )
        for job_id in selected:
            if self.tree.exists(job_id):
                self.tree.selection_set(job_id)

    def collect_selected(self) -> None:
        if self._shell.busy():
            return

        selection = self.tree.selection()
        if not selection:
            messagebox.showerror(WINDOW_TITLE, "Escolha um job na lista.")
            return

        job_id = selection[0]
        record = self._store.list_all().get(job_id, {})
        if "vpn_dir" not in record:
            messagebox.showerror(
                WINDOW_TITLE,
                f"O job {job_id} foi enviado por uma versão antiga do programa "
                "e não guardou a pasta da VPN, então não há como reconectar "
                "para coletá-lo.",
            )
            return

        try:
            username = discover_username(discover_ovpn(record["vpn_dir"]))
        except DGXError as exc:
            messagebox.showerror(WINDOW_TITLE, str(exc))
            return

        host = record.get("ssh_host", DEFAULT_SSH_HOST)
        password = self._shell.ask_password(f"Senha de {username}@{host}:")
        if not password:
            return

        def work() -> str:
            result = self._collector(
                job_id,
                password_provider=lambda: password,
                host_key_confirmer=self._shell.confirm_host_key,
                save_logs=True,
                keep_remote=KEEP_REMOTE,
            )
            if result.executed_notebook is not None:
                return f"Notebook executado salvo em:\n{result.executed_notebook}"
            return (
                f"O job {job_id} terminou como {result.state.value} e não "
                "produziu notebook executado.\n\nOs logs baixados estão "
                "listados no painel abaixo."
            )

        self._shell.run_in_worker(work)

    def set_busy(self, busy: bool) -> None:
        state = ["disabled"] if busy else ["!disabled"]
        self.collect_button.state(state)
        self.refresh_button.state(state)


class SubmitterApp:
    """The window: a tab per screen, one worker thread, one shared log pane."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        runner: Callable[..., object] = run_notebook,
        collector: Callable[..., JobResult] = collect_results,
        store: LocalJobStore | None = None,
        is_elevated: Callable[[], bool] = windows_process_is_elevated,
        system_name: str | None = None,
    ) -> None:
        self._root = root
        self._system_name = system_name or sys.platform
        self._messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._status = tk.StringVar(value="Pronto.")

        root.title(WINDOW_TITLE)
        root.minsize(720, 560)
        self._build_widgets(runner, collector, store, is_elevated)
        self._root.after(100, self._drain_messages)

    def _build_widgets(self, runner, collector, store, is_elevated) -> None:
        frame = ttk.Frame(self._root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        self._root.columnconfigure(0, weight=1)
        self._root.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        tabs = ttk.Notebook(frame)
        tabs.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.submit_panel = SubmitPanel(tabs, shell=self, runner=runner)
        self.collect_panel = CollectPanel(
            tabs, shell=self, collector=collector, store=store
        )
        tabs.add(self.submit_panel, text="Enviar")
        tabs.add(self.collect_panel, text="Coletar resultados")

        self._log = tk.Text(frame, height=14, wrap="word", state="disabled")
        self._log.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        frame.rowconfigure(1, weight=1)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self._log.yview)
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(12, 0))
        self._log.configure(yscrollcommand=scrollbar.set)

        ttk.Label(frame, textvariable=self._status).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        if self._system_name.startswith("win") and not is_elevated():
            self._show_elevation_warning(frame)

    def _show_elevation_warning(self, frame: ttk.Frame) -> None:
        warning = ttk.Frame(frame)
        warning.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        warning.columnconfigure(0, weight=1)
        label = ttk.Label(
            warning, text=ELEVATION_WARNING, wraplength=520, foreground="#b00020"
        )
        label.grid(row=0, column=0, sticky="w")
        ttk.Button(
            warning, text="Reabrir como administrador", command=self._relaunch_elevated
        ).grid(row=0, column=1, padx=(8, 0))

    def _relaunch_elevated(self) -> None:
        import ctypes

        parameters = "" if getattr(sys, "frozen", False) else "-m dgx_slurm.gui"
        started = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, parameters, None, 1
        )
        if started > 32:
            self._root.destroy()

    # -- services the panels rely on -------------------------------------

    def busy(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def ask_password(self, prompt: str) -> str | None:
        """Ask on the GUI thread, before the worker starts. Never persisted."""
        return simpledialog.askstring(
            WINDOW_TITLE, prompt, show="*", parent=self._root
        )

    def confirm_host_key(self, host: str, fingerprint: str) -> bool:
        """Ask on the GUI thread, from the job thread, and wait for the answer."""
        answer: queue.Queue[bool] = queue.Queue(maxsize=1)
        self._messages.put(("confirm", (host, fingerprint, answer)))
        return answer.get()

    def run_in_worker(self, work: Callable[[], str]) -> None:
        """Run one unit of work off the GUI thread.

        ``work`` returns the message to show once it succeeds, so each panel
        decides how to describe its own outcome.
        """
        if self.busy():
            return
        self._set_busy(True)
        self._status.set("Executando...")
        self._worker = threading.Thread(target=self._work, args=(work,), daemon=True)
        self._worker.start()

    def _work(self, work: Callable[[], str]) -> None:
        writer = QueueWriter(lambda text: self._messages.put(("log", text)))
        try:
            with captured_output(writer):
                message = work()
        except DGXError as exc:
            self._messages.put(("error", str(exc)))
        except Exception as exc:  # noqa: BLE001 - the window must survive any failure
            self._messages.put(("error", f"{type(exc).__name__}: {exc}"))
        else:
            self._messages.put(("done", message))

    # -- GUI thread ------------------------------------------------------

    def _ask_host_key(self, host: str, fingerprint: str) -> bool:
        return bool(
            messagebox.askyesno(
                WINDOW_TITLE,
                f"Primeira conexão com {host}.\n\n"
                f"Identificação do servidor:\n{fingerprint}\n\n"
                "Confere com a identificação divulgada pelo cluster? "
                "Se sim, ela será salva em known_hosts e não será perguntada "
                "de novo.",
            )
        )

    def _drain_messages(self) -> None:
        try:
            while True:
                kind, payload = self._messages.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "confirm":
                    host, fingerprint, answer = payload
                    answer.put(self._ask_host_key(host, fingerprint))
                elif kind == "done":
                    self._finish("Concluído.")
                    messagebox.showinfo(WINDOW_TITLE, str(payload))
                else:
                    self._finish("Falhou.")
                    messagebox.showerror(WINDOW_TITLE, str(payload))
        except queue.Empty:
            pass
        self._root.after(100, self._drain_messages)

    def _append_log(self, text: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", text)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _finish(self, status: str) -> None:
        self._status.set(status)
        self._set_busy(False)
        # A detached submission only becomes collectable once it is stored, so
        # refresh here rather than making the user reopen the program.
        self.collect_panel.refresh()

    def _set_busy(self, busy: bool) -> None:
        for panel in (self.submit_panel, self.collect_panel):
            panel.set_busy(busy)


def main(argv: list[str] | None = None) -> int:
    """Open the window; arguments are accepted only for entry-point symmetry."""
    root = tk.Tk()
    SubmitterApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
