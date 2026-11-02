#!/bin/bash
# CAL feature extraction + shared eval on the CALIPeR KITBRC1 ground-range benchmark.
#
# CAL = Clothes-based Adversarial Loss (Gu et al., CVPR2022), vendored in the ReFace/GEFF repo. GEFF is used
# ONLY as a vehicle for CAL + its published checkpoints -- no face / gallery-enrichment code is involved, so
# apex, insightface, opencv and pytorch_lightning are all unnecessary.
#
# CAL is an IMAGE model here (one feature per crop, 4096-D after BN, 384x192 input): GEFF disabled CAL's
# video path (VID_DATASET = []) and every published checkpoint is the 2D ResNet50 -- there is NO CCVID/video
# checkpoint despite the README claiming one. So features are per-crop and BOTH poolings are valid, and
# --max-per-video genuinely applies (unlike the video-native sinet / x-tfclip caches).
#
#   ./Test-kitbrc1-ground_range-features.sh              # extract (if needed) + eval
#   ./Test-kitbrc1-ground_range-features.sh --force      # force re-extract

cd "$(dirname "$0")"

PY="${PY:-python3}"                            # override with the path to your ResNet50 ReID env's python
SPLIT="${SPLIT:-ground_range}"                 # ground_range = REAL 500m/1000m queries; ground_range_sim = pyBSM-degraded
METHOD=cal
METRIC=cosine                                  # cosine | euclidean
MAX_PER_VIDEO=0                                # 0 = all frames, matching the other image methods
POOLING="image video"                          # per-crop features -> both poolings are meaningful

# "source:weight" -- source = the dataset the CAL backbone was trained on.
# `last` (5000 ids, real movie imagery) is the pick for CALIPeR transfer; ltcc/prcc are small and indoor,
# vc-clothes is synthetic (GTA5). Add more lines to compare sources.
COMBOS=(
    "last:weights/CAL/last-checkpoint.pth.tar"
)

export PYTHONPATH="${CALIPER_EVAL_HOME:-$(cd "../../" && pwd)}:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for combo in "${COMBOS[@]}"; do
    SOURCE="${combo%%:*}"
    WEIGHT="${combo#*:}"

    echo "############################################################"
    echo "# CAL  source=$SOURCE  weight=$WEIGHT"
    echo "############################################################"
    if [ ! -f "$WEIGHT" ]; then
        echo "  WEIGHT not found: $WEIGHT  -- skipping $SOURCE"; continue
    fi
    FEATURES_DIR=$("$PY" -c "from caliper_eval import paths; print(paths.features_dir('$SPLIT','$METHOD','$SOURCE'))")

    if [ "$1" == "--force" ] || ! ls "$FEATURES_DIR"/gallery_*.npz >/dev/null 2>&1; then
        echo "== Step 1: extracting CAL per-crop features (GPU) -> $FEATURES_DIR =="
        "$PY" extract_caliper_features.py --weight "$WEIGHT" --source "$SOURCE" --split "$SPLIT" \
            --out-dir "$FEATURES_DIR" || { echo "extract FAILED for $SOURCE"; continue; }
    else
        echo "== Step 1: skipped (cache exists in $FEATURES_DIR; pass --force to re-extract) =="
    fi

    echo
    echo "== Step 2: shared eval (pooling=$POOLING; metric=$METRIC; per range + combined; max-per-video=$MAX_PER_VIDEO) =="
    "$PY" -m caliper_eval.eval_from_features --features-dir "$FEATURES_DIR" \
        --pooling $POOLING --metric "$METRIC" --max-per-video "$MAX_PER_VIDEO"
    echo
done
