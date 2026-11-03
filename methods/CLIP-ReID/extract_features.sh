#!/bin/bash
# Extract CLIP-ReID features for a CALIPeR split and score gallery retrieval.
# Set SPLIT to the crop split directory name. Use --force to rebuild features.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$(dirname "${BASH_SOURCE[0]}")"

SPLIT="${SPLIT:-paperGR}"
DATASET="${DATASET:-combined_brc1_brc2}"
METHOD=clip-reid
MAX_PER_VIDEO=0          # frames-per-tracklet cap (0 = use all)
METRIC=cosine            # distance for ranking: cosine | euclidean
POOLING=video            # match unit: video (1 feat/clip) | image (per crop)

# "source:weight" combos -- source label = pretraining dataset, weight = its ViT-CLIP-ReID checkpoint
COMBOS=(
    "market1501:./weights/Market1501_clipreid_ViT-B-16_60.pth"
    "msmt17:./weights/MSMT17_clipreid_ViT-B-16_60.pth"
)

export PYTHONPATH="${CALIPER_EVAL_HOME:-$REPO_ROOT}:${PYTHONPATH:-}"

for combo in "${COMBOS[@]}"; do
    SOURCE="${combo%%:*}"
    WEIGHT="${combo#*:}"
    echo
    echo "########## $METHOD / $SOURCE ##########"
    if [ ! -f "$WEIGHT" ]; then
        echo "  WEIGHT not found: $WEIGHT  -- skipping $SOURCE"
        continue
    fi
    FEATURES_DIR=$(python3 -c "from caliper_eval import paths; print(paths.features_dir('$SPLIT','$METHOD','$SOURCE', dataset='$DATASET'))")

    if [ "$1" == "--force" ] || ! ls "$FEATURES_DIR"/gallery_*.npz >/dev/null 2>&1; then
        echo "== Step 1: extracting CLIP-ReID features (GPU) -> $FEATURES_DIR =="
        python3 extract_caliper_features.py --weight "$WEIGHT" --source "$SOURCE" \
            --split "$SPLIT" --dataset "$DATASET" --out-dir "$FEATURES_DIR"
    else
        echo "== Step 1: skipped (cache exists in $FEATURES_DIR; pass --force to re-extract) =="
    fi

    echo "== Retrieval scores (pooling=$POOLING; metric=$METRIC; per range + combined; max-per-video=$MAX_PER_VIDEO) =="
    python3 -m caliper_eval.eval_from_features --features-dir "$FEATURES_DIR" \
        --pooling $POOLING --metric "$METRIC" --max-per-video "$MAX_PER_VIDEO"
done
