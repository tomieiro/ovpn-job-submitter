"""Minimal desktop front-end for submitting a notebook without a terminal.

Tkinter ships with CPython, so the window costs no extra dependency and is
bundled by PyInstaller as-is. The submission itself runs on a worker thread
and everything the workflow prints is mirrored into the log pane.
"""

from __future__ import annotations

import contextlib
import queue
import sys
import threading
from pathlib import Path
from typing import Callable

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from .cli import (
    DEFAULT_CPUS,
    DEFAULT_GPUS,
    DEFAULT_MEMORY,
    DEFAULT_PARTITION,
    DEFAULT_SSH_HOST,
    DEFAULT_SSH_PORT,
    DEFAULT_TIME_LIMIT,
)
from .errors import ConfigurationError, DGXError
from .vpn import windows_process_is_elevated
from .workflow import discover_ovpn, discover_username, run_notebook

WINDOW_TITLE = "ovpn-job-submitter"
ELEVATION_WARNING = (
    "Sem privilégios de administrador o OpenVPN não consegue configurar o "
    "adaptador da VPN. Reabra como administrador."
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


class SubmitterApp:
    """The single window: two pickers, one checkbox, and a log pane."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        runner: Callable[..., object] = run_notebook,
        is_elevated: Callable[[], bool] = windows_process_is_elevated,
        system_name: str | None = None,
    ) -> None:
        self._root = root
        self._runner = runner
        self._system_name = system_name or sys.platform
        self._messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self._worker: threading.Thread | None = None

        self._notebook = tk.StringVar()
        self._vpn_dir = tk.StringVar()
        self._include_files = tk.BooleanVar(value=False)
        self._status = tk.StringVar(value="Pronto.")

        root.title(WINDOW_TITLE)
        root.minsize(640, 440)
        self._build_widgets(is_elevated)
        self._root.after(100, self._drain_messages)

    def _build_widgets(self, is_elevated: Callable[[], bool]) -> None:
        frame = ttk.Frame(self._root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        self._root.columnconfigure(0, weight=1)
        self._root.rowconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Notebook (.ipynb)").grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        ttk.Entry(frame, textvariable=self._notebook).grid(
            row=0, column=1, sticky="ew", padx=8, pady=(0, 4)
        )
        ttk.Button(frame, text="Escolher...", command=self._choose_notebook).grid(
            row=0, column=2, pady=(0, 4)
        )

        ttk.Label(frame, text="Pasta da VPN (.ovpn + certificados)").grid(
            row=1, column=0, sticky="w", pady=4
        )
        ttk.Entry(frame, textvariable=self._vpn_dir).grid(
            row=1, column=1, sticky="ew", padx=8, pady=4
        )
        ttk.Button(frame, text="Escolher...", command=self._choose_vpn_dir).grid(
            row=1, column=2, pady=4
        )

        ttk.Checkbutton(
            frame,
            text="Enviar também os outros arquivos da pasta do notebook",
            variable=self._include_files,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 4))

        self._submit_button = ttk.Button(
            frame, text="Executar no cluster", command=self._start_job
        )
        self._submit_button.grid(row=3, column=0, columnspan=3, sticky="ew", pady=8)

        self._log = tk.Text(frame, height=16, wrap="word", state="disabled")
        self._log.grid(row=4, column=0, columnspan=3, sticky="nsew")
        frame.rowconfigure(4, weight=1)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self._log.yview)
        scrollbar.grid(row=4, column=3, sticky="ns")
        self._log.configure(yscrollcommand=scrollbar.set)

        ttk.Label(frame, textvariable=self._status).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )

        if self._system_name.startswith("win") and not is_elevated():
            self._show_elevation_warning(frame)

    def _show_elevation_warning(self, frame: ttk.Frame) -> None:
        warning = ttk.Frame(frame)
        warning.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 0))
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

    def _choose_notebook(self) -> None:
        selected = filedialog.askopenfilename(
            title="Escolha o notebook",
            filetypes=[("Notebook Jupyter", "*.ipynb"), ("Todos os arquivos", "*.*")],
        )
        if selected:
            self._notebook.set(selected)

    def _choose_vpn_dir(self) -> None:
        selected = filedialog.askdirectory(title="Escolha a pasta da VPN")
        if selected:
            self._vpn_dir.set(selected)

    def _start_job(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return

        try:
            notebook, vpn_dir = validate_selection(
                self._notebook.get(), self._vpn_dir.get()
            )
            username = discover_username(discover_ovpn(vpn_dir))
        except DGXError as exc:
            messagebox.showerror(WINDOW_TITLE, str(exc))
            return

        password = simpledialog.askstring(
            WINDOW_TITLE,
            f"Senha de {username}@{DEFAULT_SSH_HOST}:",
            show="*",
            parent=self._root,
        )
        if not password:
            return

        self._submit_button.state(["disabled"])
        self._status.set("Executando...")
        self._worker = threading.Thread(
            target=self._run_job,
            args=(notebook, vpn_dir, self._include_files.get(), password),
            daemon=True,
        )
        self._worker.start()

    def _run_job(
        self, notebook: Path, vpn_dir: Path, include_files: bool, password: str
    ) -> None:
        writer = QueueWriter(lambda text: self._messages.put(("log", text)))
        try:
            with captured_output(writer):
                self._runner(
                    notebook,
                    include_project_files=include_files,
                    vpn_dir=vpn_dir,
                    ssh_host=DEFAULT_SSH_HOST,
                    ssh_port=DEFAULT_SSH_PORT,
                    partition=DEFAULT_PARTITION,
                    gpus=DEFAULT_GPUS,
                    cpus=DEFAULT_CPUS,
                    memory=DEFAULT_MEMORY,
                    time_limit=DEFAULT_TIME_LIMIT,
                    password_provider=lambda: password,
                    host_key_confirmer=self._confirm_host_key,
                )
        except DGXError as exc:
            self._messages.put(("error", str(exc)))
        except Exception as exc:  # noqa: BLE001 - the window must survive any failure
            self._messages.put(("error", f"{type(exc).__name__}: {exc}"))
        else:
            self._messages.put(("done", notebook.with_name(f"{notebook.stem}.executed.ipynb")))

    def _confirm_host_key(self, host: str, fingerprint: str) -> bool:
        """Ask on the GUI thread, from the job thread, and wait for the answer."""
        answer: queue.Queue[bool] = queue.Queue(maxsize=1)
        self._messages.put(("confirm", (host, fingerprint, answer)))
        return answer.get()

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
                    messagebox.showinfo(
                        WINDOW_TITLE, f"Notebook executado salvo em:\n{payload}"
                    )
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
        self._submit_button.state(["!disabled"])


def main(argv: list[str] | None = None) -> int:
    """Open the window; arguments are accepted only for entry-point symmetry."""
    root = tk.Tk()
    SubmitterApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
