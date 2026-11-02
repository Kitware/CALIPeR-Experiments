#!/bin/bash
# SiNet feature extraction + shared eval on the CALIPeR KITBRC1 ground-range benchmark.
# Run from this dir (SINet/) in an env with this repo's dependencies. SiNet is a VIDEO model: it emits one
# feature per tracklet, broadcast across that clip's crops, so ONLY video pooling is valid (the cache is
# stamped pooling='video' and the eval refuses image pooling). source = pretraining video dataset (mars|lsvid);
# SiNet has no Market/MSMT variant. Pick GPU via CUDA_VISIBLE_DEVICES.

cd "$(dirname "$0")"

SPLIT=ground_range_sim
METHOD=sinet
METRIC=cosine                                # cosine | euclidean
MAX_PER_VIDEO=0                              # frames-per-tracklet cap (0 = use all)

# "source:weight" -- one SiNet checkpoint per pretraining video dataset
COMBOS=(
    "mars:weights/sinet_mars.pth.tar"
    "lsvid:weights/sinet_lsvid.pth.tar"
)

export PYTHONPATH="${CALIPER_EVAL_HOME:-$(cd "../../" && pwd)}:$PYTHONPATH"

for combo in "${COMBOS[@]}"; do
    SOURCE="${combo%%:*}"
    WEIGHT="${combo#*:}"
    FEATURES_DIR=$(python3 -c "from caliper_eval import paths; print(paths.features_dir('$SPLIT','$METHOD','$SOURCE'))")

    echo "############################################################"
    echo "# SiNet  source=$SOURCE  weight=$WEIGHT"
    echo "############################################################"

    if [ "$1" == "--force" ] || ! ls "$FEATURES_DIR"/gallery_*.npz >/dev/null 2>&1; then
        echo "== Step 1: extracting SiNet tracklet features (GPU) -> $FEATURES_DIR =="
        python3 extract_caliper_features.py --weight "$WEIGHT" --source "$SOURCE" \
            --split "$SPLIT" --out-dir "$FEATURES_DIR" || { echo "extract FAILED for $SOURCE"; continue; }
    else
        echo "== Step 1: skipped (cache exists in $FEATURES_DIR; pass --force to re-extract) =="
    fi

    echo
    echo "== Step 2: shared eval (pooling=video; metric=$METRIC; per range + combined; max-per-video=$MAX_PER_VIDEO) =="
    python3 -m caliper_eval.eval_from_features --features-dir "$FEATURES_DIR" \
        --pooling video --metric "$METRIC" --max-per-video "$MAX_PER_VIDEO"
    echo
done
