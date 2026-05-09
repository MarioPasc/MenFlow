#!/bin/bash
# experiments/maisi_autoencoder_patology_preservation/slurm/launcher_full.sh
# --------------------------------------------------------------------------
# Submit the full E1 pipeline: encode -> decode -> analysis, with afterok
# dependencies between consecutive steps.
#
# Usage:
#   bash experiments/maisi_autoencoder_patology_preservation/slurm/launcher_full.sh \
#        <encoder_config.yaml> <decoder_config.yaml> <analysis_config.yaml>
#   bash experiments/maisi_autoencoder_patology_preservation/slurm/launcher_full.sh --dry-run \
#        <encoder_config.yaml> <decoder_config.yaml> <analysis_config.yaml>
#
# Invariant enforced before submission:
#     decoder_config.latent_h5 == encoder_config.output_h5
# decoder_config.source_h5 must also be readable on Picasso (used to copy IDs/
# segmentations into the recon H5). The launcher refuses to submit otherwise.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    shift
fi

ENCODE_CFG="${1:?usage: launcher_full.sh [--dry-run] <encode.yaml> <decode.yaml> <analysis.yaml>}"
DECODE_CFG="${2:?need decode config}"
ANALYSIS_CFG="${3:?need analysis config}"

ENCODE_CFG="$(readlink -f "${ENCODE_CFG}")"
DECODE_CFG="$(readlink -f "${DECODE_CFG}")"
ANALYSIS_CFG="$(readlink -f "${ANALYSIS_CFG}")"

cd "${REPO_DIR}"

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate menflow 2>/dev/null || true
PYTHON="$(command -v python)"

# Validate encode.output_h5 == decode.latent_h5
"$PYTHON" - <<PY
import sys, yaml
with open("${ENCODE_CFG}") as f: enc = yaml.safe_load(f)
with open("${DECODE_CFG}") as f: dec = yaml.safe_load(f)
enc_out = enc.get("output_h5")
dec_lat = dec.get("latent_h5")
if not enc_out or not dec_lat:
    print(f"missing output_h5 / latent_h5 in configs", file=sys.stderr)
    sys.exit(2)
if enc_out != dec_lat:
    print(
        f"encode.output_h5 ({enc_out}) must equal decode.latent_h5 ({dec_lat})",
        file=sys.stderr,
    )
    sys.exit(2)
PY

# Validate that analysis.recon_h5 equals decode.output_h5
"$PYTHON" - <<PY
import sys, yaml
with open("${DECODE_CFG}") as f: dec = yaml.safe_load(f)
with open("${ANALYSIS_CFG}") as f: ana = yaml.safe_load(f)
dec_out = dec.get("output_h5")
ana_in = ana.get("recon_h5")
if dec_out != ana_in:
    print(
        f"decode.output_h5 ({dec_out}) must equal analysis.recon_h5 ({ana_in})",
        file=sys.stderr,
    )
    sys.exit(2)
PY

echo "========================================="
echo "MenFlow E1 — full pipeline launcher"
echo "========================================="
echo "Encode cfg:   ${ENCODE_CFG}"
echo "Decode cfg:   ${DECODE_CFG}"
echo "Analysis cfg: ${ANALYSIS_CFG}"
echo ""

# ---------------------------------------------------------------------------
# 1) Submit encode
# ---------------------------------------------------------------------------
DRY_FLAG=""
${DRY_RUN} && DRY_FLAG="--dry-run"

echo "--- [1/3] Submitting encode ---"
ENCODE_OUT=$(bash routines/encode/slurm/launcher_encode.sh ${DRY_FLAG} "${ENCODE_CFG}")
echo "${ENCODE_OUT}"
ENCODE_JOB_ID=$(echo "${ENCODE_OUT}" | grep -oP 'JOB_ID=\K[0-9]+' | head -1 || true)
if [[ -z "${ENCODE_JOB_ID}" && "${DRY_RUN}" != "true" ]]; then
    echo "[ERROR] Could not extract encode job id" >&2
    exit 1
fi
echo "encode_job=${ENCODE_JOB_ID:-DRYRUN}"
echo ""

# ---------------------------------------------------------------------------
# 2) Submit decode (depends on encode)
# ---------------------------------------------------------------------------
echo "--- [2/3] Submitting decode (afterok:${ENCODE_JOB_ID:-DRYRUN}) ---"
DECODE_DEP="afterok:${ENCODE_JOB_ID:-1}"
DECODE_OUT=$(bash routines/decode/slurm/launcher_decode.sh \
    ${DRY_FLAG} \
    --dependency="${DECODE_DEP}" \
    "${DECODE_CFG}")
echo "${DECODE_OUT}"
DECODE_JOB_ID=$(echo "${DECODE_OUT}" | grep -oP 'JOB_ID=\K[0-9]+' | head -1 || true)
if [[ -z "${DECODE_JOB_ID}" && "${DRY_RUN}" != "true" ]]; then
    echo "[ERROR] Could not extract decode job id" >&2
    exit 1
fi
echo "decode_job=${DECODE_JOB_ID:-DRYRUN}"
echo ""

# ---------------------------------------------------------------------------
# 3) Submit analysis (depends on decode)
# ---------------------------------------------------------------------------
echo "--- [3/3] Submitting analysis (afterok:${DECODE_JOB_ID:-DRYRUN}) ---"

read_yaml() {
    "$PYTHON" -c "
import yaml
with open('${ANALYSIS_CFG}') as f:
    cfg = yaml.safe_load(f)
slurm = cfg.get('slurm', {})
print(slurm.get('$1', '$2'))
"
}

JOB_NAME=$(read_yaml job_name menflow_e1_analysis)
PARTITION=$(read_yaml partition cal)
CONSTRAINT=$(read_yaml constraint cal)
TIME_LIMIT=$(read_yaml time "0-02:00:00")
CPUS=$(read_yaml cpus_per_task 8)
MEM=$(read_yaml mem 32G)
GRES=$(read_yaml gres "")
CONDA_ENV=$(read_yaml conda_env menflow)
SLURM_REPO_DIR=$(read_yaml repo_dir "${REPO_DIR}")
LOGS_DIR=$(read_yaml logs_dir "${REPO_DIR}/logs/e1")

if ! ${DRY_RUN}; then
    mkdir -p "${LOGS_DIR}"
fi

GRES_FLAG=""
[[ -n "${GRES}" ]] && GRES_FLAG="--gres=${GRES}"

ANALYSIS_DEP="afterok:${DECODE_JOB_ID:-1}"
ANALYSIS_CMD="sbatch \
    --job-name=${JOB_NAME} \
    --partition=${PARTITION} \
    --constraint=${CONSTRAINT} \
    --time=${TIME_LIMIT} \
    --cpus-per-task=${CPUS} \
    --mem=${MEM} \
    ${GRES_FLAG} \
    --output=${LOGS_DIR}/${JOB_NAME}_%j.out \
    --error=${LOGS_DIR}/${JOB_NAME}_%j.err \
    --dependency=${ANALYSIS_DEP} \
    --export=ALL,CONFIG_PATH=${ANALYSIS_CFG},CONDA_ENV=${CONDA_ENV},REPO_DIR=${SLURM_REPO_DIR} \
    experiments/maisi_autoencoder_patology_preservation/slurm/worker_analysis.sh"

if ${DRY_RUN}; then
    echo "[DRY-RUN] ${ANALYSIS_CMD}"
    echo ""
    echo "Done (dry-run). 3 jobs would be submitted."
    exit 0
fi

ANALYSIS_OUT=$(eval "${ANALYSIS_CMD}" 2>&1)
echo "${ANALYSIS_OUT}"
ANALYSIS_JOB_ID=$(echo "${ANALYSIS_OUT}" | grep -oP 'job\s+\K[0-9]+' | head -1 || true)
echo "analysis_job=${ANALYSIS_JOB_ID}"

echo ""
echo "Pipeline submitted:"
echo "  encode:   ${ENCODE_JOB_ID}"
echo "  decode:   ${DECODE_JOB_ID}  (afterok:${ENCODE_JOB_ID})"
echo "  analysis: ${ANALYSIS_JOB_ID}  (afterok:${DECODE_JOB_ID})"
