#!/usr/bin/env bash
# Submit one finetune_fm_volume job per fold to Picasso.
# Reads kfold and slurm: block from the base YAML; fires K independent jobs.
#
# Usage:
#   bash launcher_finetune_fm_volume.sh <base_config.yaml>           # submit
#   bash launcher_finetune_fm_volume.sh <base_config.yaml> --dry-run # preview
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------------------------------------------------------
# Args
# --------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <base_config.yaml> [--dry-run]" >&2
    exit 1
fi
BASE_CONFIG="$(realpath "$1")"
DRY_RUN=false
[[ "${2:-}" == "--dry-run" ]] && DRY_RUN=true

[[ -f "${BASE_CONFIG}" ]] || { echo "[FATAL] Config not found: ${BASE_CONFIG}" >&2; exit 1; }

# --------------------------------------------------------------------------
# Bootstrap conda so PyYAML is available on the login node
# --------------------------------------------------------------------------
_bootstrap_conda() {
    local env_name
    env_name=$(grep -E "^[[:space:]]*conda_env:" "${BASE_CONFIG}" 2>/dev/null \
        | head -1 | awk '{print $2}' | tr -d '"' | tr -d "'")
    env_name="${env_name:-menflow}"
    if command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${env_name}" 2>/dev/null || true
    fi
}
_bootstrap_conda
PYTHON="${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}"
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"

# --------------------------------------------------------------------------
# Read fields from YAML
# --------------------------------------------------------------------------
read_yaml() {
    # Usage: read_yaml <dot.key.path> <default>
    "$PYTHON" -c "
import yaml, sys
with open('${BASE_CONFIG}') as f:
    cfg = yaml.safe_load(f)
v = cfg
for k in '${1}'.split('.'):
    v = v.get(k, None) if isinstance(v, dict) else None
    if v is None:
        print('${2}')
        sys.exit(0)
print(v)
"
}

KFOLD=$(read_yaml kfold 5)
BASE_RUN_NAME=$(read_yaml run_name "run")
OUTPUT_DIR=$(read_yaml output_dir "outputs")

PARTITION=$(read_yaml slurm.partition dgx2q)
QOS=$(read_yaml slurm.qos dgx2q)
CONSTRAINT=$(read_yaml slurm.constraint dgx)
GRES=$(read_yaml slurm.gres gpu:1)
CPUS=$(read_yaml slurm.cpus_per_task 8)
MEM=$(read_yaml slurm.mem 64G)
TIME_LIMIT=$(read_yaml slurm.time 48:00:00)

# Per-fold scratch dir for materialised configs (inside the log space)
PER_FOLD_CONFIG_DIR="${HOME}/execs/menflow/finetune_fm_volume/fold_configs"
LOGS_BASE="${HOME}/execs/menflow/finetune_fm_volume"

mkdir -p "${PER_FOLD_CONFIG_DIR}"
mkdir -p "${LOGS_BASE}"

echo "=============================================="
echo "Base config:   ${BASE_CONFIG}"
echo "kfold:         ${KFOLD}"
echo "Base run name: ${BASE_RUN_NAME}"
echo "Partition:     ${PARTITION}  QoS: ${QOS}"
echo "Constraint:    ${CONSTRAINT}  gres: ${GRES}"
echo "CPUs: ${CPUS}  Mem: ${MEM}  Time: ${TIME_LIMIT}"
echo "Logs:          ${LOGS_BASE}/<RUN_NAME>_<JOBID>.{out,err}"
echo "Fold configs:  ${PER_FOLD_CONFIG_DIR}"
echo "=============================================="

# --------------------------------------------------------------------------
# Submit one job per fold
# --------------------------------------------------------------------------
SUBMITTED_IDS=()

for FOLD in $(seq 0 $((KFOLD - 1))); do
    RUN_NAME="${BASE_RUN_NAME}_fold_${FOLD}"
    JOB_NAME="menflow-fm-${RUN_NAME}"

    OUT_LOG="${LOGS_BASE}/${RUN_NAME}_%j.out"
    ERR_LOG="${LOGS_BASE}/${RUN_NAME}_%j.err"

    SBATCH_CMD="sbatch --parsable \
        --job-name=${JOB_NAME} \
        --partition=${PARTITION} \
        --qos=${QOS} \
        --constraint=${CONSTRAINT} \
        --gres=${GRES} \
        --cpus-per-task=${CPUS} \
        --mem=${MEM} \
        --time=${TIME_LIMIT} \
        --output=${OUT_LOG} \
        --error=${ERR_LOG} \
        --export=ALL,BASE_CONFIG=${BASE_CONFIG},FOLD=${FOLD},PER_FOLD_CONFIG_DIR=${PER_FOLD_CONFIG_DIR},RUN_NAME=${RUN_NAME} \
        ${SCRIPT_DIR}/worker_finetune_fm_volume.sh"

    if ${DRY_RUN}; then
        echo "[DRY-RUN] fold=${FOLD}  run=${RUN_NAME}"
        echo "  ${SBATCH_CMD}"
    else
        JOB_ID=$(eval "${SBATCH_CMD}")
        SUBMITTED_IDS+=("${JOB_ID}")
        echo "  fold=${FOLD}  run=${RUN_NAME}  jobid=${JOB_ID}  log=${LOGS_BASE}/${RUN_NAME}_${JOB_ID}.out"
    fi
done

if ! ${DRY_RUN}; then
    echo ""
    echo "Submitted ${#SUBMITTED_IDS[@]} jobs: ${SUBMITTED_IDS[*]}"
    echo "Monitor:  squeue -u \$USER"
fi
