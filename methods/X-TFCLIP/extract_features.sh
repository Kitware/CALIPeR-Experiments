#!/bin/bash
set -euo pipefail
FORCE=()
case "${1:-}" in
    "") ;;
    --force) FORCE=(--force) ;;
    *) echo "Usage: $0 [--force]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then echo "Usage: $0 [--force]" >&2; exit 2; fi
# Extract X-TFCLIP features for a CALIPeR split and score gallery retrieval.
# Set SPLIT to the crop split directory name. Use --force to rebuild features.
# Set RUN_NATIVE_ABLATION=1 to also score the native single-clip setting. Select a CUDA device and cache ViT-B-16.pt under ~/.cache/clip.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$(dirname "$0")"

PY="${PY:-python3}"                            # override with the python of an environment with X-TFCLIP's dependencies
SPLIT="${SPLIT:-paperGR}"
DATASET="${DATASET:-combined_brc1_brc2}"
METHOD=x-tfclip
METRIC=cosine                                  # cosine | euclidean
MAX_PER_VIDEO=0                                # video features already include the sampled frames
MAX_CLIPS=0                                    # 0 uses all clips, covering every frame once. Set to 1 for one sampled clip.
CAM_ID=0                                       # constant SIE camera index; CALIPeR camids != AG-VPReID cameras

# "source:weight" -- one X-TFCLIP checkpoint per pretraining video dataset
COMBOS=(
    "agvpreid:weights/checkpoint_ep.pth.tar"
)

export PYTHONPATH="${CALIPER_EVAL_HOME:-$REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # a single GPU (or request one from your scheduler)

for combo in "${COMBOS[@]}"; do
    SOURCE="${combo%%:*}"
    WEIGHT="${combo#*:}"

    echo "############################################################"
    echo "# X-TFCLIP  source=$SOURCE  weight=$WEIGHT  max_clips=$MAX_CLIPS  cam_id=$CAM_ID  split=$SPLIT  dataset=$DATASET"
    echo "############################################################"
    FEATURES_DIR=$("$PY" -c 'import sys; from caliper_eval import paths; print(paths.features_dir(*sys.argv[1:4], dataset=sys.argv[4]))' "$SPLIT" "$METHOD" "$SOURCE" "$DATASET")

    "$PY" -m caliper_eval.extraction --method "$METHOD" --source "$SOURCE" \
        --split "$SPLIT" --dataset "$DATASET" --weight "$WEIGHT" "${FORCE[@]}" -- \
        "$PY" extract_caliper_features.py --weight "$WEIGHT" --source "$SOURCE" --split "$SPLIT" \
            --dataset "$DATASET" --max-clips "$MAX_CLIPS" --cam-id "$CAM_ID" --out-dir '{output}'

    echo
    echo "== Retrieval scores (pooling=video; metric=$METRIC; per range + combined; max-per-video=$MAX_PER_VIDEO) =="
    "$PY" -m caliper_eval.eval_from_features --features-dir "$FEATURES_DIR" \
        --pooling video --metric "$METRIC" --max-per-video "$MAX_PER_VIDEO"
    echo

    # Optional ablation: X-TFCLIP's NATIVE protocol -- one 8-frame rrs_test sample for the whole tracklet
    # (their published eval).
    if [ "${RUN_NATIVE_ABLATION:-0}" == "1" ]; then
        # Keep a different extraction protocol outside published-result discovery.
        NATIVE_DIR=$("$PY" -c 'import os, sys; from caliper_eval import paths; print(os.path.join(paths.ROOT, "ablations", *sys.argv[1:], "rrs1"))' "$DATASET" "$SPLIT" "$METHOD" "$SOURCE")
        echo "== Ablation: native single-clip rrs_test -> $NATIVE_DIR =="
        "$PY" -m caliper_eval.extraction --method "$METHOD" --source "$SOURCE" \
            --weight "$WEIGHT" --split "$SPLIT" --dataset "$DATASET" --output-dir "$NATIVE_DIR" "${FORCE[@]}" -- \
            "$PY" extract_caliper_features.py --weight "$WEIGHT" --source "$SOURCE" --split "$SPLIT" \
            --dataset "$DATASET" --max-clips 1 --cam-id "$CAM_ID" --out-dir '{output}'
        "$PY" -m caliper_eval.eval_from_features --features-dir "$NATIVE_DIR" \
                   --pooling video --metric "$METRIC" --max-per-video "$MAX_PER_VIDEO"
        echo
    fi
done
