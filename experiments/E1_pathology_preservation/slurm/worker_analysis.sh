#!/bin/bash
# experiments/E1_pathology_preservation/slurm/worker_analysis.sh
# ----------------------------------------------------------------------------
# Per-job worker for the E1 analysis step.

#SBATCH --job-name=menflow_e1_analysis
#SBATCH --output=menflow_e1_analysis_%j.out
#SBATCH --error=menflow_e1_analysis_%j.err

set -euo pipefail

echo "=== E1 analysis ${SLURM_JOB_ID} on $(hostname) at $(date) ==="
echo "Config:    ${CONFIG_PATH}"
echo "Conda env: ${CONDA_ENV:-menflow}"
echo "Repo dir:  ${REPO_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-menflow}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"

python -m experiments.E1_pathology_preservation.cli "${CONFIG_PATH}"

echo "=== Analysis complete at $(date) ==="
