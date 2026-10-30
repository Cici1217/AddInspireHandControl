#!/usr/bin/env bash
# Select the account at submission: sbatch --account=YOUR_ACCOUNT cloud.slurm check
#SBATCH --job-name=groot_basket_sim25
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=groot_basket_%j.out

# Based on the existing box-lift Slurm/Apptainer workflow. Default: checks only.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script through Slurm.}"
MODE=${1:-check}
case "$MODE" in check|train) ;; *) echo 'Use check or train' >&2; exit 2 ;; esac
ROOT=${GROOT_CLOUD_ROOT:?Set GROOT_CLOUD_ROOT to your absolute cloud working directory.}
[[ "$ROOT" == /* ]] || { echo 'GROOT_CLOUD_ROOT must be absolute' >&2; exit 2; }
BUNDLE=${BASKET_BUNDLE:-${SLURM_SUBMIT_DIR:?Submit from the extracted bundle directory.}}
CHECKPOINT=${BASKET_CHECKPOINT:-nvidia/GR00T-N1.7-3B}
GROOT_CODE=${BASKET_GROOT_ROOT:-$ROOT/Isaac-GR00T}
export APPTAINER_CACHEDIR=$ROOT/apptainer_cache
export APPTAINER_TMPDIR=/tmp/apptainer-basket-$SLURM_JOB_ID
export APPTAINERENV_HF_HOME=$ROOT/huggingface_cache
export APPTAINERENV_HF_HUB_CACHE=$ROOT/huggingface_cache/hub
mkdir -p "$APPTAINER_TMPDIR"
if [[ -n "${HF_TOKEN:-}" ]]; then export APPTAINERENV_HF_TOKEN="$HF_TOKEN"; fi

IMAGE=${GROOT_CONTAINER_IMAGE:-$ROOT/pytorch_25.08-py3.sif}
apptainer exec --nv --bind "$ROOT:$ROOT" --bind "$BUNDLE:$BUNDLE" --bind "$GROOT_CODE:$GROOT_CODE" "$IMAGE" \
    bash -s -- "$ROOT" "$BUNDLE" "$CHECKPOINT" "$MODE" "$SLURM_JOB_ID" "$GROOT_CODE" <<'INNER'
set -euo pipefail
ROOT=$1
BUNDLE=$2
CHECKPOINT=$3
MODE=$4
BASKET_JOB_ID=$5
REPO=$6
VENV=$ROOT/gr00t_venv
FFMPEG=$ROOT/ffmpeg_env
export PATH="$FFMPEG/bin:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/torch/lib:$FFMPEG/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:$BUNDLE/runtime_deps${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
cd "$BUNDLE"

# Resolve the pinned base model from the existing cloud HF cache.
if [[ "$CHECKPOINT" == nvidia/GR00T-N1.7-3B ]]; then
    CHECKPOINT=$("$VENV/bin/python" - <<'PY'
import json
from huggingface_hub import snapshot_download
manifest = json.load(open('manifest.json'))
print(snapshot_download('nvidia/GR00T-N1.7-3B', revision=manifest['base_model']['revision'], local_files_only=True))
PY
    )
fi

# The explicit check mode is separate. Training uses the existing validated data.
if [[ "$MODE" == check ]]; then
    exec "$VENV/bin/python" train.py --groot-root "$REPO" --checkpoint "$CHECKPOINT"
fi

"$VENV/bin/python" -c 'import torch; assert torch.cuda.device_count() == 1, "Training requires exactly one visible GPU"'
exec "$VENV/bin/python" train.py \
    --groot-root "$REPO" --checkpoint "$CHECKPOINT" --num-gpus 1 \
    --output "$ROOT/outputs/checkpoints/finetuned/basket_sim25_${BASKET_JOB_ID}" --execute
INNER
