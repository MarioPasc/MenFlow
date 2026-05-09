#!/bin/bash
# routines/encode/slurm/worker_encode.sh
# --------------------------------------
# Per-job worker for the encode routine. Receives CONFIG_PATH, CONDA_ENV, and
# REPO_DIR via sbatch --export from launcher_encode.sh.

#SBATCH --job-name=menflow_encode
#SBATCH --output=menflow_encode_%j.out
#SBATCH --error=menflow_encode_%j.err

set -euo pipefail

echo "=== Encode job ${SLURM_JOB_ID} on $(hostname) at $(date) ==="
echo "Config:    ${CONFIG_PATH}"
echo "Conda env: ${CONDA_ENV:-menflow}"
echo "Repo dir:  ${REPO_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-menflow}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"

nvidia-smi || true

python -m routines.encode.cli "${CONFIG_PATH}"

echo "=== Encode complete at $(date) ==="
