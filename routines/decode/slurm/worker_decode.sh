#!/bin/bash
# routines/decode/slurm/worker_decode.sh
# --------------------------------------
# Per-job worker for the decode routine.

#SBATCH --job-name=menflow_decode
#SBATCH --output=menflow_decode_%j.out
#SBATCH --error=menflow_decode_%j.err

set -euo pipefail

echo "=== Decode job ${SLURM_JOB_ID} on $(hostname) at $(date) ==="
echo "Config:    ${CONFIG_PATH}"
echo "Conda env: ${CONDA_ENV:-menflow}"
echo "Repo dir:  ${REPO_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-menflow}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"

nvidia-smi || true

python -m routines.decode.cli "${CONFIG_PATH}"

echo "=== Decode complete at $(date) ==="
