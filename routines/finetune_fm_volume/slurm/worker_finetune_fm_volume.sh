#!/usr/bin/env bash
#SBATCH --ntasks=1
# Dynamic SBATCH flags (partition, qos, constraint, gres, cpus-per-task, mem, time,
# output, error, job-name) are all passed by the launcher via sbatch flags.
# This worker reads its parameters from env vars exported by the launcher.

set -euo pipefail

START_TIME=$(date +%s)

# --------------------------------------------------------------------------
# Validate required env vars
# --------------------------------------------------------------------------
: "${BASE_CONFIG:?BASE_CONFIG not set}"
: "${FOLD:?FOLD not set}"
: "${PER_FOLD_CONFIG_DIR:?PER_FOLD_CONFIG_DIR not set}"
: "${RUN_NAME:?RUN_NAME not set}"

# --------------------------------------------------------------------------
# SIF path (overridable)
# --------------------------------------------------------------------------
SIF="${SIF:-${HOME}/sif/menflow_ngc.sif}"
REPO_HOST="${MENFLOW_REPO:-${HOME}/research/code/MenFlow}"
FSCRATCH="/mnt/home/users/tic_163_uma/mpascual/fscratch"

# --------------------------------------------------------------------------
# Job header
# --------------------------------------------------------------------------
echo "=========================================="
echo "Job:          ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "SIF:          ${SIF}"
echo "Base config:  ${BASE_CONFIG}"
echo "Fold:         ${FOLD}"
echo "Run name:     ${RUN_NAME}"
echo "Repo (host):  ${REPO_HOST}"
echo "=========================================="

# --------------------------------------------------------------------------
# GPU info
# --------------------------------------------------------------------------
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    || echo "[warn] nvidia-smi not available"
echo ""

# --------------------------------------------------------------------------
# Materialise per-fold YAML on the host (outside the container; PyYAML must
# be available in the host conda env or via the system python3+pyyaml).
# --------------------------------------------------------------------------
mkdir -p "${PER_FOLD_CONFIG_DIR}"
PER_FOLD_CONFIG="${PER_FOLD_CONFIG_DIR}/${RUN_NAME}.yaml"

python3 - <<PYEOF
import yaml, copy

with open("${BASE_CONFIG}") as fh:
    cfg = yaml.safe_load(fh)

cfg["fold"]     = int("${FOLD}")
cfg["run_name"] = "${RUN_NAME}"

with open("${PER_FOLD_CONFIG}", "w") as fh:
    yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)

print(f"[fold-config] written to ${PER_FOLD_CONFIG}")
PYEOF

echo "Per-fold config: ${PER_FOLD_CONFIG}"
echo ""

# --------------------------------------------------------------------------
# Singularity invocation
# --------------------------------------------------------------------------
# Bind-mounts:
#   /fscratch      → cluster scratch (datasets, checkpoints)
#   $HOME:$HOME    → configs, logs, run artefacts live in user space
#   /scratch       → local node scratch for temp files
#   REPO → /opt/MenFlow  (so pip install --no-deps -e /opt/MenFlow works)
#
# --cleanenv avoids leaking host CUDA / LD_LIBRARY_PATH conflicts.
# --nv passes through GPU driver and CUDA libraries.

singularity exec \
    --nv \
    --cleanenv \
    --writable-tmpfs \
    --bind "${FSCRATCH}:/fscratch" \
    --bind "${HOME}:${HOME}" \
    --bind "/mnt/scratch:/scratch" \
    --bind "${REPO_HOST}:/opt/MenFlow" \
    --env "PYTHONUNBUFFERED=1" \
    --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}" \
    --env "MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}" \
    --env "WANDB_MODE=offline" \
    "${SIF}" \
    bash -c "
set -euo pipefail

echo '[container] installing MenFlow in dev mode (no-deps)...'
pip install --no-deps -e /opt/MenFlow --quiet

echo '[container] python: \$(which python)'
echo '[container] menflow version: \$(python -c \"import menflow; print(menflow.__version__)\" 2>/dev/null || echo n/a)'
echo ''
echo '[container] launching fold=${FOLD}  run=${RUN_NAME}'
echo ''

python -m routines.finetune_fm_volume.cli '${PER_FOLD_CONFIG}'
"

# --------------------------------------------------------------------------
# Timing footer
# --------------------------------------------------------------------------
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished:  $(date)"
echo "Duration:  $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
