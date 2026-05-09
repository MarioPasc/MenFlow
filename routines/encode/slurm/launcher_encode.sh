#!/bin/bash
# routines/encode/slurm/launcher_encode.sh
# ----------------------------------------
# Submit an `encode` SLURM job on Picasso. The single positional argument is
# the path to a routine YAML config; all SLURM resources (partition, time,
# memory, conda_env, repo_dir, logs_dir, ...) are read from the `slurm:` block
# of that YAML so this launcher is dataset-agnostic.
#
# Usage:
#   bash routines/encode/slurm/launcher_encode.sh routines/encode/configs/picasso/brats_men.yaml
#   bash routines/encode/slurm/launcher_encode.sh --dry-run routines/encode/configs/picasso/brats_men.yaml
#
# On stdout the launcher prints the SLURM job id once submitted; the
# experiments/ orchestrator captures it for `--dependency=afterok` chaining.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    shift
fi

CONFIG="${1:?usage: launcher_encode.sh [--dry-run] <config.yaml>}"
CONFIG_ABS="$(readlink -f "${CONFIG}")"

cd "${REPO_DIR}"

# --- Activate conda for inline Python YAML reads ---
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate menflow 2>/dev/null || true
PYTHON="$(command -v python)"

read_yaml() {
    "$PYTHON" -c "
import yaml
with open('${CONFIG_ABS}') as f:
    cfg = yaml.safe_load(f)
slurm = cfg.get('slurm', {})
print(slurm.get('$1', '$2'))
"
}

JOB_NAME=$(read_yaml job_name menflow_encode)
PARTITION=$(read_yaml partition dgx)
CONSTRAINT=$(read_yaml constraint dgx)
TIME_LIMIT=$(read_yaml time "0-04:00:00")
CPUS=$(read_yaml cpus_per_task 16)
MEM=$(read_yaml mem 64G)
GRES=$(read_yaml gres "gpu:1")
CONDA_ENV=$(read_yaml conda_env menflow)
SLURM_REPO_DIR=$(read_yaml repo_dir "${REPO_DIR}")
LOGS_DIR=$(read_yaml logs_dir "${REPO_DIR}/logs/encode")

echo "========================================="
echo "MenFlow encode — SLURM launcher"
echo "========================================="
echo "Config:     ${CONFIG_ABS}"
echo "Job name:   ${JOB_NAME}"
echo "Partition:  ${PARTITION}"
echo "Constraint: ${CONSTRAINT}"
echo "Time:       ${TIME_LIMIT}"
echo "CPUs:       ${CPUS}"
echo "Memory:     ${MEM}"
echo "GRES:       ${GRES}"
echo "Conda env:  ${CONDA_ENV}"
echo "Repo dir:   ${SLURM_REPO_DIR}"
echo "Logs dir:   ${LOGS_DIR}"

if ! ${DRY_RUN}; then
    mkdir -p "${LOGS_DIR}"
fi

SBATCH_CMD="sbatch \
    --job-name=${JOB_NAME} \
    --partition=${PARTITION} \
    --constraint=${CONSTRAINT} \
    --time=${TIME_LIMIT} \
    --cpus-per-task=${CPUS} \
    --mem=${MEM} \
    --gres=${GRES} \
    --output=${LOGS_DIR}/${JOB_NAME}_%j.out \
    --error=${LOGS_DIR}/${JOB_NAME}_%j.err \
    --export=ALL,CONFIG_PATH=${CONFIG_ABS},CONDA_ENV=${CONDA_ENV},REPO_DIR=${SLURM_REPO_DIR} \
    routines/encode/slurm/worker_encode.sh"

if ${DRY_RUN}; then
    echo ""
    echo "[DRY-RUN] ${SBATCH_CMD}"
    exit 0
fi

SBATCH_OUTPUT=$(eval "${SBATCH_CMD}" 2>&1)
echo ""
echo "${SBATCH_OUTPUT}"

JOB_ID=$(echo "${SBATCH_OUTPUT}" | grep -oP 'job\s+\K[0-9]+' | head -1 || true)
if [[ -z "${JOB_ID}" ]]; then
    JOB_ID=$(echo "${SBATCH_OUTPUT}" | grep -oP '[0-9]+' | head -1 || true)
fi
echo "JOB_ID=${JOB_ID}"
