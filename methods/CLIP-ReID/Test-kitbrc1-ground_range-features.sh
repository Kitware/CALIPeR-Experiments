#!/bin/bash
# CLIP-ReID feature extraction + shared eval on the CALIPeR KITBRC1 ground-range benchmark.
# Loops over (source, weight) combos -> one ViT-CLIP-ReID (non-SIE) checkpoint per pretraining source.
# Run in the CLIP-ReID conda env. Pick the GPU with CUDA_VISIBLE_DEVICES. First model build downloads
# OpenAI ViT-B-16.pt to ~/.cache/clip. Checkpoints: repo README Google Drive (Market1501 / MSMT17 ViT-B-16).
#   ./Test-kitbrc1-ground_range-features.sh           # extract (if needed) + eval, both combos
#   ./Test-kitbrc1-ground_range-features.sh --force   # force re-extract for all combos

cd "$(dirname "$0")"

SPLIT=ground_range_sim
METHOD=clip-reid
MAX_PER_VIDEO=0          # frames-per-tracklet cap (0 = use all); set 16 to mirror the BoT image crosscheck
METRIC=cosine            # distance for ranking: cosine | euclidean
POOLING=video            # match unit: video (1 feat/clip) | image (per crop -> pair with MAX_PER_VIDEO=16)

# "source:weight" combos -- source label = pretraining dataset, weight = its ViT-CLIP-ReID checkpoint
COMBOS=(
    "market1501:./weights/Market1501_clipreid_ViT-B-16_60.pth"
    "msmt17:./weights/MSMT17_clipreid_ViT-B-16_60.pth"
)

export PYTHONPATH="${CALIPER_EVAL_HOME:-$(cd "../../" && pwd)}:$PYTHONPATH"

for combo in "${COMBOS[@]}"; do
    SOURCE="${combo%%:*}"
    WEIGHT="${combo#*:}"
    echo
    echo "########## $METHOD / $SOURCE ##########"
    if [ ! -f "$WEIGHT" ]; then
        echo "  WEIGHT not found: $WEIGHT  -- skipping $SOURCE"
        continue
    fi
    FEATURES_DIR=$(python3 -c "from caliper_eval import paths; print(paths.features_dir('$SPLIT','$METHOD','$SOURCE'))")

    if [ "$1" == "--force" ] || ! ls "$FEATURES_DIR"/gallery_*.npz >/dev/null 2>&1; then
        echo "== Step 1: extracting CLIP-ReID features (GPU) -> $FEATURES_DIR =="
        python3 extract_caliper_features.py --weight "$WEIGHT" --source "$SOURCE" --split "$SPLIT" \
            --out-dir "$FEATURES_DIR"
    else
        echo "== Step 1: skipped (cache exists in $FEATURES_DIR; pass --force to re-extract) =="
    fi

    echo "== Step 2: shared eval (pooling=$POOLING; metric=$METRIC; per range + combined; max-per-video=$MAX_PER_VIDEO) =="
    python3 -m caliper_eval.eval_from_features --features-dir "$FEATURES_DIR" \
        --pooling $POOLING --metric "$METRIC" --max-per-video "$MAX_PER_VIDEO"
done
