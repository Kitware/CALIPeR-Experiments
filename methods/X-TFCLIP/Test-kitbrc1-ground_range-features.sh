#!/bin/bash
# X-TFCLIP feature extraction + shared eval on the CALIPeR KITBRC1 ground-range benchmark.
#
# X-TFCLIP is a VIDEO model: it emits ONE 2048-D feature per tracklet, broadcast across that clip's crops,
# so ONLY video pooling is valid (the cache is stamped pooling='video' and the shared eval refuses image
# pooling; --max-per-video is consequently inert for this cache -- the frames-per-tracklet budget is set
# inside the extractor via --max-clips). source = the pretraining video dataset (agvpreid); X-TFCLIP has no
# Market/MSMT variant. Requires a GPU (the model build hardcodes .to('cuda')); pick it with
# CUDA_VISIBLE_DEVICES. CLIP ViT-B-16.pt is read from ~/.cache/clip.
#
#   ./Test-kitbrc1-ground_range-features.sh              # extract (if needed) + eval
#   ./Test-kitbrc1-ground_range-features.sh --force      # force re-extract
#   RUN_NATIVE_ABLATION=1 ./Test-kitbrc1-ground_range-features.sh   # + X-TFCLIP's native 1-clip protocol

cd "$(dirname "$0")"

PY="${PY:-python3}"                            # override with the path to your torch env's python
SPLIT="${SPLIT:-ground_range}"                 # ground_range = REAL 500m/1000m queries; ground_range_sim = pyBSM-degraded
METHOD=x-tfclip
METRIC=cosine                                  # cosine | euclidean
MAX_PER_VIDEO=0                                # inert for video-native caches; kept for log parity
MAX_CLIPS=0                                    # 0 = ALL RRS clips (every frame used once); 1 = native rrs_test
CAM_ID=0                                       # constant SIE camera index; CALIPeR camids != AG-VPReID cameras

# "source:weight" -- one X-TFCLIP checkpoint per pretraining video dataset
COMBOS=(
    "agvpreid:weights/checkpoint_ep.pth.tar"
)

export PYTHONPATH="${CALIPER_EVAL_HOME:-$(cd "../../" && pwd)}:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for combo in "${COMBOS[@]}"; do
    SOURCE="${combo%%:*}"
    WEIGHT="${combo#*:}"

    echo "############################################################"
    echo "# X-TFCLIP  source=$SOURCE  weight=$WEIGHT  max_clips=$MAX_CLIPS  cam_id=$CAM_ID"
    echo "############################################################"
    if [ ! -f "$WEIGHT" ]; then
        echo "  WEIGHT not found: $WEIGHT  -- skipping $SOURCE"; continue
    fi
    FEATURES_DIR=$("$PY" -c "from caliper_eval import paths; print(paths.features_dir('$SPLIT','$METHOD','$SOURCE'))")

    if [ "$1" == "--force" ] || ! ls "$FEATURES_DIR"/gallery_*.npz >/dev/null 2>&1; then
        echo "== Step 1: extracting X-TFCLIP tracklet features (GPU) -> $FEATURES_DIR =="
        "$PY" extract_caliper_features.py --weight "$WEIGHT" --source "$SOURCE" --split "$SPLIT" \
            --max-clips "$MAX_CLIPS" --cam-id "$CAM_ID" --out-dir "$FEATURES_DIR" \
            || { echo "extract FAILED for $SOURCE"; continue; }
    else
        echo "== Step 1: skipped (cache exists in $FEATURES_DIR; pass --force to re-extract) =="
    fi

    echo
    echo "== Step 2: shared eval (pooling=video; metric=$METRIC; per range + combined; max-per-video=$MAX_PER_VIDEO) =="
    "$PY" -m caliper_eval.eval_from_features --features-dir "$FEATURES_DIR" \
        --pooling video --metric "$METRIC" --max-per-video "$MAX_PER_VIDEO"
    echo

    # Optional ablation: X-TFCLIP's NATIVE protocol -- one 8-frame rrs_test sample for the whole tracklet
    # (their published eval). ~2 min; also doubles as the smoke test for the extractor.
    if [ "$RUN_NATIVE_ABLATION" == "1" ]; then
        NATIVE_DIR="${FEATURES_DIR}-rrs1"
        echo "== Ablation: native single-clip rrs_test -> $NATIVE_DIR =="
        "$PY" extract_caliper_features.py --weight "$WEIGHT" --source "$SOURCE" --split "$SPLIT" \
            --max-clips 1 --cam-id "$CAM_ID" --out-dir "$NATIVE_DIR" \
            && "$PY" -m caliper_eval.eval_from_features --features-dir "$NATIVE_DIR" \
                   --pooling video --metric "$METRIC" --max-per-video "$MAX_PER_VIDEO"
        echo
    fi
done
