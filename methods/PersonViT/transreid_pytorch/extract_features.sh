#!/bin/bash
set -euo pipefail
FORCE=()
case "${1:-}" in
    "") ;;
    --force) FORCE=(--force) ;;
    *) echo "Usage: $0 [--force]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then echo "Usage: $0 [--force]" >&2; exit 2; fi
# Extract PersonViT features for a CALIPeR split and score gallery retrieval.
# Set SPLIT to the crop split directory name. Use --force to rebuild features.
# Get matching weights and configs from lakeAGI/PersonViTReID on Hugging Face. CKPT_ROOT selects the checkpoint directory.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$(dirname "${BASH_SOURCE[0]}")"

SPLIT="${SPLIT:-paperGR}"
DATASET="${DATASET:-combined_brc1_brc2}"
METHOD=personvit
MAX_PER_VIDEO=0          # frames-per-tracklet cap (0 = use all)
METRIC=cosine            # distance for ranking: cosine | euclidean
POOLING=video            # match unit: video (1 feat/clip) | image (per crop)

# "source:weight:config" combos. These checkpoints are plain ViT (no conv stem), so use vit_base.yml /
# vit_small.yml, not the *_ics variants ("csk" in the name is a pretraining tag, not a conv stem).
# Using ViT-B (matches CLIP-ReID's ViT-B/16); for ViT-S swap vitb->vits, e0260->e0220, vit_base->vit_small.
CKPT_ROOT=${CKPT_ROOT:-../reid_ckpts}
COMBOS=(
    "market1501:$CKPT_ROOT/market.vitb.lup.256x128.wopt.csk.4-8.ar.375.n8.e0260/transformer_120.pth:configs/market/vit_base.yml"
    "msmt17:$CKPT_ROOT/msmt.vitb.lup.256x128.wopt.csk.4-8.ar.375.n8.e0260/transformer_120.pth:configs/msmt17/vit_base.yml"
)

export PYTHONPATH="${CALIPER_EVAL_HOME:-$REPO_ROOT}:${PYTHONPATH:-}"

for combo in "${COMBOS[@]}"; do
    IFS=':' read -r SOURCE WEIGHT CONFIG <<< "$combo"
    echo
    echo "########## $METHOD / $SOURCE  (config: $CONFIG) ##########"
    FEATURES_DIR=$(python3 -c 'import sys; from caliper_eval import paths; print(paths.features_dir(*sys.argv[1:4], dataset=sys.argv[4]))' "$SPLIT" "$METHOD" "$SOURCE" "$DATASET")

    python3 -m caliper_eval.extraction --method "$METHOD" --source "$SOURCE" \
        --split "$SPLIT" --dataset "$DATASET" --weight "$WEIGHT" --asset "$CONFIG" "${FORCE[@]}" -- \
        python3 extract_caliper_features.py --config-file "$CONFIG" --weight "$WEIGHT" --source "$SOURCE" \
            --split "$SPLIT" --dataset "$DATASET" --out-dir '{output}'

    echo "== Retrieval scores (pooling=$POOLING; metric=$METRIC; per range + combined; max-per-video=$MAX_PER_VIDEO) =="
    python3 -m caliper_eval.eval_from_features --features-dir "$FEATURES_DIR" \
        --pooling $POOLING --metric "$METRIC" --max-per-video "$MAX_PER_VIDEO"
done
