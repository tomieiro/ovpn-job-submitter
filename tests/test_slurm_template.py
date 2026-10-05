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
    assert "#SBATCH --mem=" not in content
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
    assert "--progress=plain" in content


def test_runs_docker_run_with_rm(build):
    bundle_root = build(Resources())
    content = (bundle_root / "runImage.slurm").read_text()
    assert "docker run" in content
    assert "--rm" in content


def test_gives_the_container_room_for_dataloader_workers(build):
    """The default 64 MB /dev/shm kills PyTorch workers with a bus error."""
    bundle_root = build(Resources())
    content = (bundle_root / "runImage.slurm").read_text()
    assert "--shm-size=1g" in content
    assert "--ulimit memlock=-1" in content
    assert "--ulimit stack=67108864" in content


def test_does_not_share_the_host_ipc_namespace(build):
    """Nodes are shared, so grow /dev/shm instead of opening the host's."""
    bundle_root = build(Resources())
    assert "--ipc=host" not in (bundle_root / "runImage.slurm").read_text()


def test_mounts_outputs_directory(build):
    bundle_root = build(Resources())
    content = (bundle_root / "runImage.slurm").read_text()
    assert "type=bind,src=${SCRIPT_DIR}/outputs,dst=/workspace/outputs" in content


def test_mounts_shared_home_datasets_read_only(build):
    bundle_root = build(Resources())
    content = (bundle_root / "runImage.slurm").read_text()
    assert 'SHARED_DATASETS="${DGX_SHARED_DATASETS:-${HOME}/datasets}"' in content
    assert "type=bind,src=${SHARED_DATASETS},dst=/datasets,readonly" in content
    assert '"${DATASET_MOUNT[@]}"' in content


def test_installs_trap_exit(build):
    bundle_root = build(Resources())
    content = (bundle_root / "runImage.slurm").read_text()
    assert "trap cleanup EXIT TERM INT HUP" in content


def test_cleanup_runs_docker_rmi(build):
    bundle_root = build(Resources())
    content = (bundle_root / "runImage.slurm").read_text()
    assert "docker rmi" in content


def test_cleanup_force_removes_the_named_container(build):
    bundle_root = build(Resources())
    content = (bundle_root / "runImage.slurm").read_text()
    assert 'CONTAINER_NAME="${CONTAINER_NAME:-dgx-notebook-${SLURM_JOB_ID}}"' in content
    assert 'docker rm -f "${CONTAINER_NAME}"' in content
    assert '--name "${CONTAINER_NAME}"' in content
    assert '--label "dgx.slurm_job=${SLURM_JOB_ID}"' in content


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


def test_template_restricts_docker_to_the_slurm_gpu_allocation(build):
    """`--gpus all` entrega todas as GPUs do no, ignorando a alocacao.

    Observado em producao: um job que pediu 4 GPUs enxergava as 8 do no, e
    poderia usar placas alocadas a outro usuario. O template deve repassar ao
    docker a lista que o SLURM expoe em CUDA_VISIBLE_DEVICES.
    """
    content = (build(Resources(gpus=4)) / "runImage.slurm").read_text()

    assert "--gpus all \\" not in content, "o --gpus all incondicional voltou"
    assert "CUDA_VISIBLE_DEVICES" in content
    assert '"${GPU_ARGS[@]}"' in content
    # array em vez de eval: o caminho do job entra em --mount e um eval
    # quebraria com espacos
    assert "eval docker run" not in content


@pytest.mark.parametrize(
    "alocadas, esperado",
    [
        ("0", '"device=0"'),
        ("0,1,2,3", '"device=0,1,2,3"'),
        ("2,5", '"device=2,5"'),
    ],
)
def test_gpu_args_survive_docker_csv_splitting(build, alocadas, esperado):
    """O valor de --gpus precisa chegar ao docker com aspas literais.

    O docker faz split CSV no valor de --gpus. Sem as aspas internas,
    `device=0,1,2,3` vira os campos `device=0`, `1`, `2` e `3`; o `1` e lido
    como *count* e o daemon recusa com "cannot set both Count and DeviceIDs on
    device request". Foi o que matou o job 11882.

    Com uma GPU nao ha virgula e nada quebra, entao um teste de string sobre o
    template nao pega o defeito -- este executa o bloco e inspeciona o argv.
    """
    import subprocess
    import textwrap

    content = (build(Resources(gpus=4)) / "runImage.slurm").read_text()
    bloco = content[content.index("LOCAL_GPU_IDS="):content.index("echo \"=== notebook execution")]

    script = textwrap.dedent(
        """
        set -u
        nvidia-smi() {{ return 1; }}
        CUDA_VISIBLE_DEVICES="{alocadas}"
        SLURM_JOB_GPUS=""
        SLURM_STEP_GPUS=""
        {bloco}
        for a in "${{GPU_ARGS[@]}}"; do printf '%s\\n' "$a"; done
        """
    ).format(alocadas=alocadas, bloco=bloco)

    saida = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    ).stdout.splitlines()

    argv = [linha for linha in saida if not linha.startswith("===")]
    assert argv == ["--gpus", esperado], (
        f"argv inesperado: {argv!r}; docker recusaria a forma sem aspas"
    )


def test_build_retries_and_falls_back_when_the_registry_is_unreachable(build, tmp_path):
    """Uma queda de DNS no no nao pode custar a submissao inteira.

    O job 11883 passou a noite na fila e morreu no docker build porque o no nao
    resolveu nvcr.io. Nenhuma hora de GPU foi gasta -- e justamente por isso o
    prejuizo passou despercebido ate a coleta da manha seguinte.

    O BuildKit faz um HEAD no registry para resolver o manifesto do FROM mesmo
    com a imagem base ja no no, e ela esta la: cleanup() so remove a imagem do
    job. O builder legado usa a copia local. Testa-se com um docker falso, para
    exercitar o fluxo real do script em vez de casar strings.
    """
    import os
    import subprocess

    content = (build(Resources(gpus=4)) / "runImage.slurm").read_text()
    inicio = content.index("build_image() {")
    fim = content.index("\nbuild_image\n", inicio) + len("\nbuild_image\n")
    bloco = content[inicio:fim]

    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM nvcr.io/nvidia/pytorch:26.05-py3\n")

    binario = tmp_path / "bin"
    binario.mkdir()
    (binario / "docker").write_text(
        "#!/usr/bin/env bash\n"
        'contador="$TMPDIR_TESTE/n"\n'
        'n=$(( $(cat "$contador" 2>/dev/null || echo 0) + 1 ))\n'
        'echo "$n" > "$contador"\n'
        '[[ "${DOCKER_BUILDKIT:-}" == "0" ]] && exit "${LEGADO_FALHA:-0}"\n'
        'if (( n <= ${FALHAS:-0} )); then exit 1; fi\n'
        "exit 0\n"
    )
    (binario / "docker").chmod(0o755)

    def roda(falhas, legado_falha=0):
        script = tmp_path / "s.sh"
        script.write_text(
            "#!/bin/bash\nset -euo pipefail\nsleep() { :; }\n"
            + bloco
            + '\necho MARCADOR_POS_BUILD\n'
        )
        script.chmod(0o755)
        (tmp_path / "n").unlink(missing_ok=True)
        env = dict(
            os.environ,
            PATH=f"{binario}:{os.environ['PATH']}",
            DOCKERFILE=str(dockerfile),
            SCRIPT_DIR=str(tmp_path),
            IMAGE_TAG="teste:latest",
            TMPDIR_TESTE=str(tmp_path),
            FALHAS=str(falhas),
            LEGADO_FALHA=str(legado_falha),
        )
        return subprocess.run(
            [str(script)], capture_output=True, text=True, env=env
        )

    # duas quedas transitorias: a terceira tentativa do BuildKit resolve
    r = roda(falhas=2)
    assert r.returncode == 0, r.stderr
    assert "MARCADOR_POS_BUILD" in r.stdout

    # registry inalcancavel o tempo todo: cai no builder legado e segue
    r = roda(falhas=99)
    assert r.returncode == 0, r.stderr
    assert "builder legado" in r.stdout
    assert "MARCADOR_POS_BUILD" in r.stdout

    # nem o legado constroi: precisa ABORTAR, nunca seguir para o docker run
    r = roda(falhas=99, legado_falha=1)
    assert r.returncode != 0, "build falho nao abortou o script"
    assert "MARCADOR_POS_BUILD" not in r.stdout, (
        "seguiu para o docker run com a imagem inexistente"
    )


def test_template_removes_the_uploaded_payload_when_the_job_ends(build):
    """O payload e copia do que o usuario ja tem; logs e outputs precisam ficar.

    Um bundle com dados passa de centenas de MB e o submitter cria um diretorio
    por submissao, entao sem limpeza o home do cluster enche sozinho. Logs e
    outputs nao podem ir junto: uma coleta com --async acontece depois.
    """
    content = (build(Resources()) / "runImage.slurm").read_text()

    assert 'rm -rf "${SCRIPT_DIR}/payload" "${SCRIPT_DIR}/runner"' in content
    limpeza = content[content.index("cleanup()"):content.index("trap cleanup EXIT")]
    assert "payload" in limpeza, "a limpeza deve rodar no trap, como a da imagem"
    for preservado in ("/logs", "/outputs"):
        assert f'rm -rf "${{SCRIPT_DIR}}{preservado}"' not in content


def test_gpu_args_translate_cgroup_indices_to_host_uuids(build):
    """Indices do cgroup nao valem para o daemon do docker.

    Com ConstrainDevices, o job 12484 recebeu IDX 2-3 mas via
    CUDA_VISIBLE_DEVICES=0,1; `device=0,1` pos o container nas placas 0-1 do
    host, em cima do job 12483. Dentro da alocacao o nvidia-smi traduz os
    indices locais para os UUIDs das placas alocadas.
    """
    import subprocess
    import textwrap

    content = (build(Resources(gpus=2)) / "runImage.slurm").read_text()
    bloco = content[content.index("LOCAL_GPU_IDS="):content.index("echo \"=== notebook execution")]
    script = textwrap.dedent(
        """
        set -euo pipefail
        nvidia-smi() {{
            [ "$2" = "0,1" ] || return 9
            printf 'GPU-aaa\\nGPU-bbb\\n'
        }}
        CUDA_VISIBLE_DEVICES="0,1"
        SLURM_JOB_GPUS="2,3"
        {bloco}
        for a in "${{GPU_ARGS[@]}}"; do printf '%s\\n' "$a"; done
        """
    ).format(bloco=bloco)
    saida = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    argv = [linha for linha in saida if not linha.startswith("===")]
    assert argv == ["--gpus", '"device=GPU-aaa,GPU-bbb"']


def test_gpu_args_fall_back_to_global_slurm_indices(build):
    """Se o nvidia-smi nao responder, usa os indices globais do SLURM."""
    import subprocess
    import textwrap

    content = (build(Resources(gpus=2)) / "runImage.slurm").read_text()
    bloco = content[content.index("LOCAL_GPU_IDS="):content.index("echo \"=== notebook execution")]
    script = textwrap.dedent(
        """
        set -euo pipefail
        nvidia-smi() {{ printf 'GPU-only-one\\n'; }}
        CUDA_VISIBLE_DEVICES="0,1"
        SLURM_JOB_GPUS="2,3"
        {bloco}
        for a in "${{GPU_ARGS[@]}}"; do printf '%s\\n' "$a"; done
        """
    ).format(bloco=bloco)
    saida = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    argv = [linha for linha in saida if not linha.startswith("===")]
    assert argv == ["--gpus", '"device=2,3"']


def test_scancel_removes_the_container_before_the_kill_wait(build, tmp_path):
    """TERM precisa disparar o cleanup enquanto o container ainda roda.

    Com `docker run` em primeiro plano o bash adiava o trap ate o cliente
    docker terminar; o SIGKILL do SLURM chegava antes e o container do job
    12484 sobreviveu ao scancel, ocupando GPUs fora da alocacao.
    """
    import os
    import signal
    import subprocess
    import textwrap
    import time

    content = (build(Resources(gpus=2)) / "runImage.slurm").read_text()
    limpeza = content[content.index("cleanup()"):content.index("mkdir -p")]
    execucao = content[content.index("docker run \\"):]
    log = tmp_path / "docker.log"
    fake = tmp_path / "bin" / "docker"
    fake.parent.mkdir()
    fake.write_text(
        "#!/bin/bash\n"
        f'echo "$*" >> {log}\n'
        '[ "$1" = run ] && exec sleep 60\n'
        "exit 0\n"
    )
    fake.chmod(0o755)
    script = textwrap.dedent(
        """
        set -euo pipefail
        SLURM_JOB_ID=7
        SCRIPT_DIR={tmp}
        IMAGE_TAG=img
        CONTAINER_NAME=dgx-notebook-7
        GPU_ARGS=(--gpus all)
        DATASET_MOUNT=()
        """
    ).format(tmp=tmp_path) + limpeza + execucao
    process = subprocess.Popen(
        ["bash", "-c", script],
        env={**os.environ, "PATH": f"{fake.parent}:{os.environ['PATH']}"},
    )
    for _ in range(50):
        if log.exists() and "run" in log.read_text():
            break
        time.sleep(0.1)
    started = time.monotonic()
    process.send_signal(signal.SIGTERM)
    process.wait(timeout=10)
    assert time.monotonic() - started < 5
    assert "rm -f dgx-notebook-7" in log.read_text()
