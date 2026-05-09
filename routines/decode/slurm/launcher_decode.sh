#!/bin/bash
# routines/decode/slurm/launcher_decode.sh
# ----------------------------------------
# Submit a `decode` SLURM job on Picasso. Single positional argument: path to
# routine YAML config. SLURM resources are read from the config's `slurm:`
# block. Optionally accepts `--dependency=<spec>` so the experiment
# orchestrator can chain decode after encode (afterok:<encode_job_id>).
#
# Usage:
#   bash routines/decode/slurm/launcher_decode.sh routines/decode/configs/picasso/brats_men.yaml
#   bash routines/decode/slurm/launcher_decode.sh --dependency=afterok:12345 routines/decode/configs/picasso/brats_men.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DRY_RUN=false
DEPENDENCY=""
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --dependency=*) DEPENDENCY="${1#--dependency=}"; shift ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

CONFIG="${1:?usage: launcher_decode.sh [--dry-run] [--dependency=<spec>] <config.yaml>}"
CONFIG_ABS="$(readlink -f "${CONFIG}")"

cd "${REPO_DIR}"

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

JOB_NAME=$(read_yaml job_name menflow_decode)
PARTITION=$(read_yaml partition dgx)
CONSTRAINT=$(read_yaml constraint dgx)
TIME_LIMIT=$(read_yaml time "0-04:00:00")
CPUS=$(read_yaml cpus_per_task 16)
MEM=$(read_yaml mem 64G)
GRES=$(read_yaml gres "gpu:1")
CONDA_ENV=$(read_yaml conda_env menflow)
SLURM_REPO_DIR=$(read_yaml repo_dir "${REPO_DIR}")
LOGS_DIR=$(read_yaml logs_dir "${REPO_DIR}/logs/decode")

echo "========================================="
echo "MenFlow decode — SLURM launcher"
echo "========================================="
echo "Config:     ${CONFIG_ABS}"
echo "Job name:   ${JOB_NAME}"
echo "Partition:  ${PARTITION}"
echo "Time:       ${TIME_LIMIT}"
echo "Mem/CPU:    ${MEM} / ${CPUS}"
echo "GRES:       ${GRES}"
echo "Conda env:  ${CONDA_ENV}"
echo "Repo dir:   ${SLURM_REPO_DIR}"
echo "Logs dir:   ${LOGS_DIR}"
[[ -n "${DEPENDENCY}" ]] && echo "Dependency: ${DEPENDENCY}"

if ! ${DRY_RUN}; then
    mkdir -p "${LOGS_DIR}"
fi

DEP_FLAG=""
if [[ -n "${DEPENDENCY}" ]]; then
    DEP_FLAG="--dependency=${DEPENDENCY}"
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
    ${DEP_FLAG} \
    --export=ALL,CONFIG_PATH=${CONFIG_ABS},CONDA_ENV=${CONDA_ENV},REPO_DIR=${SLURM_REPO_DIR} \
    routines/decode/slurm/worker_decode.sh"

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
