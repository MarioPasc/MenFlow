#!/usr/bin/env bash
# Submit one finetune_fm_volume job per fold to Picasso.
# Reads kfold and slurm: block from the base YAML; fires K independent jobs.
#
# Usage:
#   bash launcher_finetune_fm_volume.sh <base_config.yaml>           # submit
#   bash launcher_finetune_fm_volume.sh <base_config.yaml> --dry-run # preview
#
# Pure-shell parser. Does NOT require PyYAML on the login node (the launcher
# only reads top-level scalars and the `slurm:` sub-block, both of which the
# YAML files for this routine guarantee to be simple `key: value` pairs).

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
# Tiny YAML reader (no external deps)
#   yaml_top  KEY  DEFAULT          → "value" of a top-level scalar.
#   yaml_sub  PARENT  KEY  DEFAULT  → scalar inside a one-level-deep block.
# Both strip surrounding quotes and trailing comments.
# --------------------------------------------------------------------------
_strip() {
    # Strip leading whitespace, trailing whitespace, surrounding quotes, and
    # trailing `# comment`.
    sed -e 's/[[:space:]]*#.*$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
        -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/"
}

yaml_top() {
    local key="$1"; local default="${2:-}"
    local val
    val=$(grep -E "^${key}:[[:space:]]" "${BASE_CONFIG}" | head -1 \
        | sed -E "s/^${key}:[[:space:]]*//" | _strip || true)
    if [[ -z "${val}" ]]; then
        echo "${default}"
    else
        echo "${val}"
    fi
}

yaml_sub() {
    local parent="$1"; local key="$2"; local default="${3:-}"
    local val
    # Extract the block following `parent:` until the next non-indented line,
    # then grep for `  key:` inside that block.
    val=$(awk -v p="${parent}" '
        BEGIN { in_block = 0 }
        /^[^[:space:]#]/ {
            in_block = ($0 ~ "^"p":[[:space:]]*$") ? 1 : 0
            next
        }
        in_block { print }
    ' "${BASE_CONFIG}" \
        | grep -E "^[[:space:]]+${key}:[[:space:]]" | head -1 \
        | sed -E "s/^[[:space:]]+${key}:[[:space:]]*//" | _strip || true)
    if [[ -z "${val}" ]]; then
        echo "${default}"
    else
        echo "${val}"
    fi
}

# --------------------------------------------------------------------------
# Read fields from YAML
# --------------------------------------------------------------------------
KFOLD=$(yaml_top kfold 5)
BASE_RUN_NAME=$(yaml_top run_name "run")
OUTPUT_DIR=$(yaml_top output_dir "${HOME}/runs")

# Picasso conventions (per the picasso-sbatch skill): A100 nodes are selected
# via --constraint=dgx; --partition is NOT used. Do not re-introduce a
# partition flag — recent Picasso versions reject named partitions like
# `dgx2q` outright.
CONSTRAINT=$(yaml_sub slurm constraint "dgx")
GRES=$(yaml_sub slurm gres "gpu:1")
CPUS=$(yaml_sub slurm cpus_per_task "8")
MEM=$(yaml_sub slurm mem "64G")
TIME_LIMIT=$(yaml_sub slurm time "48:00:00")

# Sanity-check kfold
if ! [[ "${KFOLD}" =~ ^[0-9]+$ ]] || (( KFOLD < 1 )); then
    echo "[FATAL] could not parse 'kfold' (got: '${KFOLD}'). Check ${BASE_CONFIG}." >&2
    exit 2
fi

# Per-fold scratch dir for materialised configs (inside the log space)
PER_FOLD_CONFIG_DIR="${HOME}/execs/menflow/finetune_fm_volume/fold_configs"
LOGS_BASE="${HOME}/execs/menflow/finetune_fm_volume"

mkdir -p "${PER_FOLD_CONFIG_DIR}"
mkdir -p "${LOGS_BASE}"

echo "=============================================="
echo "Base config:   ${BASE_CONFIG}"
echo "kfold:         ${KFOLD}"
echo "Base run name: ${BASE_RUN_NAME}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Constraint:    ${CONSTRAINT}  gres: ${GRES}"
echo "CPUs: ${CPUS}  Mem: ${MEM}  Time: ${TIME_LIMIT}"
echo "Logs:          ${LOGS_BASE}/<RUN_NAME>_<JOBID>.{out,err}"
echo "Fold configs:  ${PER_FOLD_CONFIG_DIR}"
echo "Dry-run:       ${DRY_RUN}"
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

    SBATCH_ARGS=(
        --parsable
        --job-name="${JOB_NAME}"
        --constraint="${CONSTRAINT}"
        --gres="${GRES}"
        --ntasks=1
        --cpus-per-task="${CPUS}"
        --mem="${MEM}"
        --time="${TIME_LIMIT}"
        --output="${OUT_LOG}"
        --error="${ERR_LOG}"
        --export=ALL,BASE_CONFIG="${BASE_CONFIG}",FOLD="${FOLD}",PER_FOLD_CONFIG_DIR="${PER_FOLD_CONFIG_DIR}",RUN_NAME="${RUN_NAME}"
        "${SCRIPT_DIR}/worker_finetune_fm_volume.sh"
    )

    if ${DRY_RUN}; then
        echo "[DRY-RUN] fold=${FOLD}  run=${RUN_NAME}"
        printf '  sbatch'
        for arg in "${SBATCH_ARGS[@]}"; do printf ' %q' "${arg}"; done
        printf '\n'
    else
        JOB_ID=$(sbatch "${SBATCH_ARGS[@]}")
        SUBMITTED_IDS+=("${JOB_ID}")
        echo "  fold=${FOLD}  run=${RUN_NAME}  jobid=${JOB_ID}  log=${LOGS_BASE}/${RUN_NAME}_${JOB_ID}.out"
    fi
done

if ! ${DRY_RUN}; then
    echo ""
    echo "Submitted ${#SUBMITTED_IDS[@]} jobs: ${SUBMITTED_IDS[*]}"
    echo "Monitor:  squeue -u \$USER"
fi
