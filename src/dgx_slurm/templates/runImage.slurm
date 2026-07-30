#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=__TIME_LIMIT__
#SBATCH --nodes=__NODES__
#SBATCH --ntasks=__TASKS__
#SBATCH --cpus-per-task=__CPUS__
#SBATCH --gres=gpu:__GPUS__
#SBATCH --mem=__MEMORY__
#SBATCH --partition=__PARTITION__

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DOCKERFILE="${DOCKERFILE:-${SCRIPT_DIR}/Dockerfile}"
IMAGE_TAG="${IMAGE_TAG:-dgx-notebook-${SLURM_JOB_ID}:latest}"

cleanup() {
    local status=$?

    trap - EXIT

    docker rmi "${IMAGE_TAG}" >/dev/null 2>&1 || true

    exit "${status}"
}

trap cleanup EXIT

mkdir -p "${SCRIPT_DIR}/outputs"

echo "=== node: $(hostname) ==="
echo "=== job: ${SLURM_JOB_ID} ==="
echo "=== docker build: ${IMAGE_TAG} ==="

docker build \
    -t "${IMAGE_TAG}" \
    -f "${DOCKERFILE}" \
    "${SCRIPT_DIR}"

echo "=== notebook execution ==="

docker run \
    --rm \
    --gpus all \
    --mount \
    "type=bind,src=${SCRIPT_DIR}/outputs,dst=/workspace/outputs" \
    "${IMAGE_TAG}"

echo
echo "Done (image will be removed from the node on exit)."
