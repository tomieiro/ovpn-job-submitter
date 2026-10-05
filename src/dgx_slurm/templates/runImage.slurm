#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=__TIME_LIMIT__
#SBATCH --nodes=__NODES__
#SBATCH --ntasks=__TASKS__
#SBATCH --cpus-per-task=__CPUS__
#SBATCH --gres=gpu:__GPUS__
#SBATCH --partition=__PARTITION__

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DOCKERFILE="${DOCKERFILE:-${SCRIPT_DIR}/Dockerfile}"
IMAGE_TAG="${IMAGE_TAG:-dgx-notebook-${SLURM_JOB_ID}:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-dgx-notebook-${SLURM_JOB_ID}}"

cleanup() {
    local status=$?

    trap - EXIT TERM INT HUP

    # `docker run` is a client of the daemon. If SLURM kills the batch shell,
    # disconnecting that client does not stop the container; explicitly remove
    # the job-named container so GPU ranks cannot survive their allocation.
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    docker rmi "${IMAGE_TAG}" >/dev/null 2>&1 || true

    # O payload e uma copia do que o usuario ja tem na propria maquina, e um
    # bundle com dados pode passar de centenas de MB. Some-se um diretorio por
    # submissao e o home do cluster enche sozinho. Logs e outputs FICAM: uma
    # coleta com --async acontece depois, as vezes muito depois.
    rm -rf "${SCRIPT_DIR}/payload" "${SCRIPT_DIR}/runner" >/dev/null 2>&1 || true

    exit "${status}"
}

trap cleanup EXIT TERM INT HUP

mkdir -p "${SCRIPT_DIR}/outputs"

# The user's home is shared by the compute nodes. Expose its dataset catalog
# read-only so large immutable corpora never need to enter each Docker image.
SHARED_DATASETS="${DGX_SHARED_DATASETS:-${HOME}/datasets}"
DATASET_MOUNT=()
if [ -d "${SHARED_DATASETS}" ]; then
    DATASET_MOUNT=(--mount "type=bind,src=${SHARED_DATASETS},dst=/datasets,readonly")
    echo "=== shared datasets: ${SHARED_DATASETS} -> /datasets (ro) ==="
else
    echo "=== shared datasets absent: ${SHARED_DATASETS} ==="
fi

echo "=== node: $(hostname) ==="
echo "=== job: ${SLURM_JOB_ID} ==="
echo "=== docker build: ${IMAGE_TAG} ==="

# O job 11883 morreu aqui depois de uma noite na fila: o no nao resolveu
# nvcr.io ("lookup nvcr.io on 127.0.0.53:53: i/o timeout"). Nenhuma hora de GPU
# foi gasta, mas a submissao inteira se perdeu por uma queda de DNS transitoria.
#
# Duas camadas de defesa. Primeiro repetir, que resolve a queda passageira.
# Depois o builder legado: o BuildKit faz um HEAD no registry para resolver o
# manifesto do FROM mesmo quando a imagem base ja esta no no -- e ela esta,
# porque o cleanup() so remove a imagem do job. Sem rede, o legado constroi a
# partir da copia local; o BuildKit nao.
if docker image inspect "$(awk '/^FROM /{print $2; exit}' "${DOCKERFILE}")" \
        >/dev/null 2>&1; then
    echo "=== imagem base ja presente no no ==="
fi

build_image() {
    local espera
    for espera in 15 60 120; do
        if docker build \
                --progress=plain \
                -t "${IMAGE_TAG}" \
                -f "${DOCKERFILE}" \
                "${SCRIPT_DIR}"; then
            return 0
        fi
        echo "=== build falhou; nova tentativa em ${espera}s ==="
        sleep "${espera}"
    done

    echo "=== tentando o builder legado, sem consultar o registry ==="
    DOCKER_BUILDKIT=0 docker build \
        -t "${IMAGE_TAG}" \
        -f "${DOCKERFILE}" \
        "${SCRIPT_DIR}"
}

build_image

# O SLURM ja restringe as GPUs do job, expondo os indices alocados em
# CUDA_VISIBLE_DEVICES (ou SLURM_JOB_GPUS, conforme a versao). `--gpus all`
# ignora isso e entrega TODAS as GPUs do no ao container: um job que pede 4
# enxerga 8 e pode, por engano, usar placas alocadas a outro usuario. Repassar
# a lista mantem o container dentro da alocacao.
#
# Usa array em vez de eval: o caminho do job entra em --mount e um eval
# quebraria se ele contivesse espacos.
#
# Mas CUDA_VISIBLE_DEVICES e relativo ao cgroup do job: com ConstrainDevices o
# SLURM mostra as placas alocadas como 0..N-1. O docker fala com o daemon, fora
# do cgroup, e le "device=0,1" como as GPUs 0 e 1 DO HOST. O job 12484 recebeu
# IDX 2-3, rodou nas placas 0-1 em cima do job 12483, e o 12483 morreu de OOM.
# Dentro da alocacao o nvidia-smi enxerga exatamente as placas do job, entao os
# indices locais viram UUIDs, que o docker resolve sem ambiguidade. Sem
# nvidia-smi, SLURM_JOB_GPUS traz indices globais.
LOCAL_GPU_IDS="${CUDA_VISIBLE_DEVICES:-}"
GPU_IDS=""
if [[ -n "${LOCAL_GPU_IDS}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    GPU_IDS="$(nvidia-smi -i "${LOCAL_GPU_IDS}" --query-gpu=uuid \
        --format=csv,noheader 2>/dev/null | tr -d ' ' | paste -sd, - || true)"
    if [[ "$(tr ',' '\n' <<<"${GPU_IDS}" | grep -c . || true)" \
            != "$(tr ',' '\n' <<<"${LOCAL_GPU_IDS}" | grep -c . || true)" ]]; then
        GPU_IDS=""
    fi
fi
if [[ -z "${GPU_IDS}" ]]; then
    GPU_IDS="${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-${LOCAL_GPU_IDS}}}"
fi
if [[ -n "${GPU_IDS}" ]]; then
    # As aspas dentro do valor sao literais e obrigatorias: o docker faz split
    # CSV em --gpus, entao "device=0,1,2,3" viraria os campos `device=0`, `1`,
    # `2` e `3`, e o `1` seria lido como *count* -- "cannot set both Count and
    # DeviceIDs on device request". Com uma GPU nao ha virgula e o erro nao
    # aparece, que foi por que passou despercebido ate o job 11882.
    GPU_ARGS=(--gpus "\"device=${GPU_IDS}\"")
    echo "=== gpus alocadas pelo SLURM: ${GPU_IDS} ==="
else
    GPU_ARGS=(--gpus all)
    echo "=== aviso: SLURM nao informou GPUs; usando todas do no ==="
fi

echo "=== notebook execution ==="

docker run \
    --rm \
    --name "${CONTAINER_NAME}" \
    --label "dgx.slurm_job=${SLURM_JOB_ID}" \
    --label "dgx.user=${USER}" \
    "${GPU_ARGS[@]}" \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --mount \
    "type=bind,src=${SCRIPT_DIR}/outputs,dst=/workspace/outputs" \
    "${DATASET_MOUNT[@]}" \
    "${IMAGE_TAG}" &

# O bash so executa um trap quando o comando em primeiro plano termina. Com
# `docker run` em primeiro plano, o scancel mandava TERM, o trap esperava o
# cliente docker, o SIGKILL do KillWait chegava antes e o container do job
# 12484 seguiu rodando sem alocacao. `wait` e interrompido pelo sinal, entao o
# cleanup remove o container imediatamente.
wait "$!"

echo
echo "Done (image will be removed from the node on exit)."
