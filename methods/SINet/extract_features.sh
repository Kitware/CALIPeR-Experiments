#!/bin/bash
# Extract SINet features for a CALIPeR split and score gallery retrieval.
# Set SPLIT to the crop split directory name. Use --force to rebuild features.
# SINet produces one embedding per tracklet. Use video pooling for its scores.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$(dirname "${BASH_SOURCE[0]}")"

SPLIT="${SPLIT:-paperGR}"
DATASET="${DATASET:-combined_brc1_brc2}"
METHOD=sinet
METRIC=cosine                                # cosine | euclidean
MAX_PER_VIDEO=0                              # frames-per-tracklet cap (0 = use all)

# "source:weight" -- one SiNet checkpoint per pretraining video dataset
COMBOS=(
    "mars:weights/sinet_mars.pth.tar"
    "lsvid:weights/sinet_lsvid.pth.tar"
)

export PYTHONPATH="${CALIPER_EVAL_HOME:-$REPO_ROOT}:${PYTHONPATH:-}"

for combo in "${COMBOS[@]}"; do
    SOURCE="${combo%%:*}"
    WEIGHT="${combo#*:}"
    FEATURES_DIR=$(python3 -c "from caliper_eval import paths; print(paths.features_dir('$SPLIT','$METHOD','$SOURCE',dataset='$DATASET'))")

    echo "############################################################"
    echo "# SINet  source=$SOURCE  weight=$WEIGHT  split=$SPLIT  dataset=$DATASET"
    echo "############################################################"

    if [ "$1" == "--force" ] || ! ls "$FEATURES_DIR"/gallery_*.npz >/dev/null 2>&1; then
        echo "== Step 1: extracting SiNet tracklet features (GPU) -> $FEATURES_DIR =="
        python3 extract_caliper_features.py --weight "$WEIGHT" --source "$SOURCE" \
            --split "$SPLIT" --dataset "$DATASET" --out-dir "$FEATURES_DIR" \
            || { echo "extract FAILED for $SOURCE"; continue; }
    else
        echo "== Step 1: skipped (cache exists in $FEATURES_DIR; pass --force to re-extract) =="
    fi

    echo
    echo "== Retrieval scores (pooling=video; metric=$METRIC; per range + combined; max-per-video=$MAX_PER_VIDEO) =="
    python3 -m caliper_eval.eval_from_features --features-dir "$FEATURES_DIR" \
        --pooling video --metric "$METRIC" --max-per-video "$MAX_PER_VIDEO"
    echo
done
