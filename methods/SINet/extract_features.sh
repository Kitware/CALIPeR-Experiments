#!/bin/bash
set -euo pipefail
FORCE=()
case "${1:-}" in
    "") ;;
    --force) FORCE=(--force) ;;
    *) echo "Usage: $0 [--force]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then echo "Usage: $0 [--force]" >&2; exit 2; fi
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
    FEATURES_DIR=$(python3 -c 'import sys; from caliper_eval import paths; print(paths.features_dir(*sys.argv[1:4], dataset=sys.argv[4]))' "$SPLIT" "$METHOD" "$SOURCE" "$DATASET")

    echo "############################################################"
    echo "# SINet  source=$SOURCE  weight=$WEIGHT  split=$SPLIT  dataset=$DATASET"
    echo "############################################################"

    python3 -m caliper_eval.extraction --method "$METHOD" --source "$SOURCE" \
        --split "$SPLIT" --dataset "$DATASET" --weight "$WEIGHT" "${FORCE[@]}" -- \
        python3 extract_caliper_features.py --weight "$WEIGHT" --source "$SOURCE" \
            --split "$SPLIT" --dataset "$DATASET" --out-dir '{output}'

    echo
    echo "== Retrieval scores (pooling=video; metric=$METRIC; per range + combined; max-per-video=$MAX_PER_VIDEO) =="
    python3 -m caliper_eval.eval_from_features --features-dir "$FEATURES_DIR" \
        --pooling video --metric "$METRIC" --max-per-video "$MAX_PER_VIDEO"
    echo
done
