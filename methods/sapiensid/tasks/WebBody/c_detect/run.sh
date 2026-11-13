#!/bin/bash
# Usage: bash run.sh [--delete]. Sources are removed only after verified output.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
: "${WEBBODY_DATA_ROOT:?Set WEBBODY_DATA_ROOT to the absolute data directory}"
delete=false
case "${1:-}" in
    "") ;;
    --delete) delete=true ;;
    *) echo "Usage: $0 [--delete]" >&2; exit 2 ;;
esac
(( $# <= 1 )) || exit 2
num_gpus=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
(( num_gpus > 0 )) || { echo 'No GPUs found' >&2; exit 1; }
base_dir="${WEBBODY_DATA_ROOT}/raw_img_parquets_640"
save_dir="${WEBBODY_DATA_ROOT}/raw_img_parquets_640_yolo_cropped"
shopt -s nullglob
sources=("$base_dir"/*_parquet)
(( ${#sources[@]} > 0 )) || { echo "No parquet directories in $base_dir" >&2; exit 1; }
mkdir -p "$save_dir"
gpu_list=$(seq -s, 0 $((num_gpus - 1)))
stage=''
trap 'if [[ -n "$stage" ]]; then rm -rf -- "$stage"; fi' EXIT
for source in "${sources[@]}"; do
    [[ -d "$source" && ! -L "$source" ]] || { echo "Invalid source: $source" >&2; exit 1; }
    for (( rank=0; rank<num_gpus; rank++ )); do
        target="$save_dir/$(basename "$source")_rank_$rank"
        [[ ! -e "$target" ]] || { echo "Output already exists: $target" >&2; exit 1; }
    done
    stage=$(mktemp -d "$save_dir/.extract-XXXXXXXX")
    CUDA_VISIBLE_DEVICES="$gpu_list" fabric run --strategy=ddp --devices="$num_gpus" \
        --precision=32-true --main-port="${WEBBODY_MAIN_PORT:-29500}" run_yolo_detection.py \
        --parquet_dir "$source" --save_dir "$stage" --confidence_threshold 0.7 --num_gpu "$num_gpus"
    python3 completion.py "$source" "$stage" "$num_gpus"
    for (( rank=0; rank<num_gpus; rank++ )); do
        mv -T -- "$stage/$(basename "$source")_rank_$rank" "$save_dir/$(basename "$source")_rank_$rank"
    done
    rmdir "$stage"
    stage=''
    if [[ "$delete" == true ]]; then rm -rf -- "$source"; fi
done
