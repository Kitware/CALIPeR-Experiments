#!/bin/bash
# PersonViT feature extraction + shared eval on the CALIPeR KITBRC1 ground-range benchmark.
# Loops over (source, weight, config) combos. Run in the PersonViT conda env, from transreid_pytorch/.
#   Step 1 (GPU, once): extract PersonViT features -> gallery_<range>.npz / query_<range>.npz
#   Step 2 (numpy):     shared eval (per range + combined) via caliper_eval.
# ReID checkpoints: HuggingFace lakeAGI/PersonViTReID -- e.g.:
#   python -c "from huggingface_hub import list_repo_files; print('\n'.join(list_repo_files('lakeAGI/PersonViTReID')))"
#   huggingface-cli download lakeAGI/PersonViTReID --local-dir ../reid_ckpts
# CONFIG must match the checkpoint's architecture (the checkpoint's log says which; the extractor also prints
# load-coverage and aborts on a mismatch). Pick the GPU with CUDA_VISIBLE_DEVICES.
#   ./Test-kitbrc1-ground_range-features.sh           # extract (if needed) + eval, all combos
#   ./Test-kitbrc1-ground_range-features.sh --force   # force re-extract for all combos

cd "$(dirname "$0")"

SPLIT=ground_range_sim
METHOD=personvit
MAX_PER_VIDEO=0          # frames-per-tracklet cap (0 = use all); set 16 to mirror the BoT image crosscheck
METRIC=cosine            # distance for ranking: cosine | euclidean
POOLING=video            # match unit: video (1 feat/clip) | image (per crop -> pair with MAX_PER_VIDEO=16)

# "source:weight:config" combos. Verified from each checkpoint's tensors + train_log: these are PLAIN ViT
# (STEM_CONV=False, 256x128, NECK_FEAT=before) -- "csk" is a pretraining tag, NOT a conv stem -- so the
# config is vit_base.yml / vit_small.yml (NOT the *_ics variants). Using ViT-B (matches CLIP-ReID's ViT-B/16);
# for ViT-S swap vitb->vits, e0260->e0220, vit_base->vit_small.
CKPT_ROOT=${CKPT_ROOT:-../reid_ckpts}
COMBOS=(
    "market1501:$CKPT_ROOT/market.vitb.lup.256x128.wopt.csk.4-8.ar.375.n8.e0260/transformer_120.pth:configs/market/vit_base.yml"
    "msmt17:$CKPT_ROOT/msmt.vitb.lup.256x128.wopt.csk.4-8.ar.375.n8.e0260/transformer_120.pth:configs/msmt17/vit_base.yml"
)

export PYTHONPATH="${CALIPER_EVAL_HOME:-$(cd "../../../" && pwd)}:$PYTHONPATH"

for combo in "${COMBOS[@]}"; do
    IFS=':' read -r SOURCE WEIGHT CONFIG <<< "$combo"
    echo
    echo "########## $METHOD / $SOURCE  (config: $CONFIG) ##########"
    if [ ! -f "$WEIGHT" ]; then
        echo "  WEIGHT not found: $WEIGHT  -- set it to the downloaded checkpoint; skipping $SOURCE"
        continue
    fi
    FEATURES_DIR=$(python3 -c "from caliper_eval import paths; print(paths.features_dir('$SPLIT','$METHOD','$SOURCE'))")

    if [ "$1" == "--force" ] || ! ls "$FEATURES_DIR"/gallery_*.npz >/dev/null 2>&1; then
        echo "== Step 1: extracting PersonViT features (GPU) -> $FEATURES_DIR =="
        python3 extract_caliper_features.py --config-file "$CONFIG" --weight "$WEIGHT" --source "$SOURCE" \
            --split "$SPLIT" --out-dir "$FEATURES_DIR"
    else
        echo "== Step 1: skipped (cache exists in $FEATURES_DIR; pass --force to re-extract) =="
    fi

    echo "== Step 2: shared eval (pooling=$POOLING; metric=$METRIC; per range + combined; max-per-video=$MAX_PER_VIDEO) =="
    python3 -m caliper_eval.eval_from_features --features-dir "$FEATURES_DIR" \
        --pooling $POOLING --metric "$METRIC" --max-per-video "$MAX_PER_VIDEO"
done
