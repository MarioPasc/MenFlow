#!/usr/bin/env bash
# routines/finetune_fm_volume/slurm/loginexa/run_loginexa.sh
# --------------------------------------------------------------------------
# Smoke-run the volume-conditional FM finetune on Picasso's `loginexa` node.
#
# loginexa is NOT a SLURM queue. It is an interactive login front-end with
# 4 × Tesla V100-DGXS-32GB and a 30-minute per-process CPU+GPU budget. We
# therefore (a) do not call `sbatch`, (b) activate the venv via
# `source .../menflow/bin/activate` (the loginexa conda is not initialised),
# (c) run a single fold with the downsampled `configs/loginexa.yaml`.
#
# Usage (on loginexa):
#     bash routines/finetune_fm_volume/slurm/loginexa/run_loginexa.sh
#     bash routines/finetune_fm_volume/slurm/loginexa/run_loginexa.sh \
#         /path/to/another_config.yaml
#     bash routines/finetune_fm_volume/slurm/loginexa/run_loginexa.sh \
#         --gpu 2                              # pin to GPU index 2
#     bash routines/finetune_fm_volume/slurm/loginexa/run_loginexa.sh \
#         --fold 1 --gpu 0                     # override fold + GPU
#     bash routines/finetune_fm_volume/slurm/loginexa/run_loginexa.sh --dry-run
#
# The runner enforces a soft wall-clock cap of TIME_BUDGET seconds (default
# 1500 s = 25 min, leaving 5 min of slack below loginexa's 30-min ceiling)
# by wrapping the python call in `timeout`. If the engine has not finished
# by then it receives SIGTERM and exits — partial logs / `last.pt` survive.

set -euo pipefail

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUTINE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_DIR="$(cd "${ROUTINE_DIR}/../.." && pwd)"

# `source <env>/bin/activate` per the loginexa convention (no `conda activate`).
CONDA_ACTIVATE="${CONDA_ACTIVATE:-/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/menflow/bin/activate}"
CONFIG="${ROUTINE_DIR}/configs/loginexa.yaml"
FOLD=""              # if empty, the value in the YAML is used (default 0)
GPU=""               # if empty, CUDA picks all visible GPUs; the YAML uses 1
TIME_BUDGET="${TIME_BUDGET:-1500}"   # seconds; 25 min < loginexa 30 min cap
DRY_RUN=false

# --------------------------------------------------------------------------
# Argument parser
# --------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)         GPU="$2"; shift 2;;
        --fold)        FOLD="$2"; shift 2;;
        --time-budget) TIME_BUDGET="$2"; shift 2;;
        --dry-run)     DRY_RUN=true; shift;;
        -h|--help)
            sed -n '2,28p' "$0"
            exit 0
            ;;
        --*)
            echo "[error] unknown flag: $1" >&2
            exit 2
            ;;
        *)
            CONFIG="$1"; shift;;
    esac
done

if [[ ! -f "${CONFIG}" ]]; then
    echo "[error] config not found: ${CONFIG}" >&2
    exit 2
fi
CONFIG="$(readlink -f "${CONFIG}")"

if [[ ! -f "${CONDA_ACTIVATE}" ]] && ! ${DRY_RUN}; then
    echo "[error] conda activate script not found: ${CONDA_ACTIVATE}" >&2
    echo "        override with CONDA_ACTIVATE=/path/to/env/bin/activate" >&2
    exit 2
fi

# --------------------------------------------------------------------------
# GPU selection
# --------------------------------------------------------------------------
# loginexa has 4 V100-DGXS-32GB. Pin to a single device unless the user
# explicitly unsets CUDA_VISIBLE_DEVICES afterwards. Defaults to 0.
if [[ -n "${GPU}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU}"
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=0
fi

# --------------------------------------------------------------------------
# Per-fold YAML materialisation (only if --fold was passed)
# --------------------------------------------------------------------------
PER_FOLD_CONFIG="${CONFIG}"
if [[ -n "${FOLD}" ]]; then
    PER_FOLD_CONFIG_DIR="${PER_FOLD_CONFIG_DIR:-${HOME}/execs/menflow/finetune_fm_volume/loginexa_fold_configs}"
    mkdir -p "${PER_FOLD_CONFIG_DIR}"
    RUN_NAME_OVERRIDE="loginexa_v100_smoke_fold_${FOLD}"
    PER_FOLD_CONFIG="${PER_FOLD_CONFIG_DIR}/${RUN_NAME_OVERRIDE}.yaml"
fi

# --------------------------------------------------------------------------
# Job header
# --------------------------------------------------------------------------
echo "=========================================="
echo "MenFlow FM finetune — loginexa smoke"
echo "------------------------------------------"
echo "Host:           $(hostname)"
echo "Start:          $(date)"
echo "Repo:           ${REPO_DIR}"
echo "Conda activate: ${CONDA_ACTIVATE}"
echo "Base config:    ${CONFIG}"
echo "Per-fold yaml:  ${PER_FOLD_CONFIG}"
echo "Override fold:  ${FOLD:-<from yaml>}"
echo "CUDA devices:   ${CUDA_VISIBLE_DEVICES}"
echo "Time budget:    ${TIME_BUDGET}s"
echo "Dry-run:        ${DRY_RUN}"
echo "=========================================="

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
fi
echo ""

# --------------------------------------------------------------------------
# Activate venv (no `conda activate` — loginexa convention)
# --------------------------------------------------------------------------
if ${DRY_RUN}; then
    HOST_PY="$(command -v python || command -v python3)"
    echo "[dry-run] would source ${CONDA_ACTIVATE}"
else
    # shellcheck disable=SC1090
    source "${CONDA_ACTIVATE}"
    HOST_PY="$(command -v python)"
fi
echo "Python: ${HOST_PY} ($("${HOST_PY}" --version 2>&1))"
echo ""

# --------------------------------------------------------------------------
# Materialise the per-fold YAML if --fold was passed (PyYAML now available).
# --------------------------------------------------------------------------
if [[ -n "${FOLD}" ]] && ! ${DRY_RUN}; then
    "${HOST_PY}" - <<PYEOF
import yaml
with open("${CONFIG}") as fh:
    cfg = yaml.safe_load(fh) or {}
cfg["fold"]     = int("${FOLD}")
cfg["run_name"] = "${RUN_NAME_OVERRIDE}"
with open("${PER_FOLD_CONFIG}", "w") as fh:
    yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)
print(f"[fold-config] wrote ${PER_FOLD_CONFIG}")
PYEOF
fi

# --------------------------------------------------------------------------
# Launch
# --------------------------------------------------------------------------
cd "${REPO_DIR}"

CMD=(
    timeout --signal=TERM --kill-after=30 "${TIME_BUDGET}"
    "${HOST_PY}" -m routines.finetune_fm_volume.cli "${PER_FOLD_CONFIG}"
)

if ${DRY_RUN}; then
    echo "[DRY-RUN] ${CMD[*]}"
    exit 0
fi

START_TIME=$(date +%s)
set +e
"${CMD[@]}"
EXIT_CODE=$?
set -e
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "------------------------------------------"
echo "Finished:  $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "Duration:  $((ELAPSED / 60))m $((ELAPSED % 60))s"
if [[ ${EXIT_CODE} -eq 124 ]]; then
    echo "[warn] time budget (${TIME_BUDGET}s) reached — engine SIGTERMed before completion."
    echo "       partial artefacts under output_dir/<run_name>/."
fi
exit "${EXIT_CODE}"
