"""Command-line entry point for notebook submission."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from .errors import DGXError
from .models import JobResult
from .workflow import run_notebook

DEFAULT_SSH_HOST = "c4aiscm2"
DEFAULT_SSH_PORT = 22
DEFAULT_PARTITION = "devwork"
DEFAULT_GPUS = 1
DEFAULT_CPUS = 8
DEFAULT_MEMORY = "0"
DEFAULT_TIME_LIMIT = "04:00:00"

NotebookRunner = Callable[..., JobResult]


def confirm_host_key(host: str, fingerprint: str) -> bool:
    """Ask once, in the terminal, before trusting an unknown server key."""
    if sys.stdin is None or not sys.stdin.isatty():
        return False
    print(f"Primeira conexão com {host}.")
    print(f"Identificação do servidor: {fingerprint}")
    answer = input(
        "Confere com a identificação divulgada pelo cluster? [s/N] "
    )
    return answer.strip().lower() in {"s", "sim", "y", "yes"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ovpn-job-submitter",
        description=(
            "Executa um notebook no cluster C4AI e baixa o .ipynb executado."
        ),
    )
    parser.add_argument("notebook", help="caminho do arquivo .ipynb")
    parser.add_argument(
        "vpn_dir",
        help="pasta que contém o .ovpn e seus certificados",
    )
    parser.add_argument(
        "--include-files",
        action="store_true",
        help="envia também os outros arquivos da pasta do notebook",
    )
    parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST)
    parser.add_argument("--ssh-port", type=int, default=DEFAULT_SSH_PORT)
    parser.add_argument("--partition", default=DEFAULT_PARTITION)
    parser.add_argument("--gpus", type=int, default=DEFAULT_GPUS)
    parser.add_argument("--cpus", type=int, default=DEFAULT_CPUS)
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--time-limit", default=DEFAULT_TIME_LIMIT)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: NotebookRunner = run_notebook,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        runner(
            args.notebook,
            include_project_files=args.include_files,
            vpn_dir=args.vpn_dir,
            ssh_host=args.ssh_host,
            ssh_port=args.ssh_port,
            partition=args.partition,
            gpus=args.gpus,
            cpus=args.cpus,
            memory=args.memory,
            time_limit=args.time_limit,
            host_key_confirmer=confirm_host_key,
        )
    except DGXError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nOperação interrompida.", file=sys.stderr)
        return 130
    return 0
