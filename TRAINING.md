# Fine-tuning on CALIPeR

BoT, PersonViT, SINet, and CLIP-ReID have two fine-tuning results in the [published table](RESULTS.md).
FT uses real training crops. FT+syn mixes real crops with simulated long-range imagery.
The commands below give both training recipes and the checkpoint epoch to score.

## Data

Paths below are relative to `$CALIPER_DATA_ROOT`. First export the absolute data and output roots
as shown in the [README](README.md#data); keep them exported when changing into method directories.

**Training corpus**: `splits/combined_brc1_brc2_train_corpus/`. The manifest lists the labelled crops used for training.

| file | contents |
|---|---|
| `train_manifest.csv` | one row per crop; the training list |
| `train_manifest_classes.json` | `{canonical_id: label}` |
| `train_chips.csv`, `train_tracklets.csv` | the unfiltered crop / tracklet lists |

Training uses the manifest's **ground-field rows** (`kind == "field"`, `cls == "ground"`):
**230 identities, 2,623 tracklets, 269,084 crops**. Identity comes from the manifest's `label` column,
never from directory names. `caliper_eval/train_corpus.py` (`GroundFieldCorpus`) applies this filter
for every method.

Training uses tracklets from single-subject source videos. To check this filter, group rows by
`source_clip_id`. Each `source_relpath` already refers to a per-subject trim.

**Crops**: `reid-chips/`. `chip_path` in the manifest is relative to this directory. Use
`--chip-root` to point training at a copy elsewhere (e.g. a `/dev/shm` stage made with
`python -m caliper_eval.stage_chips`).

**Synthetic-turbulence crops** (FT+syn only): a 1:1 re-render of the real 400m ground crops through a
pyBSM atmospheric-turbulence simulation at three strengths (`cn2_1e-13`, `cn2_4e-13`, `cn2_1e-12`),
177 subjects, 502 tracklets and 34,251 crops per strength. Each synthetic crop's `src_chip_path` equals
a real crop's `chip_path`, so it replaces that crop exactly. The required manifest is
`splits/combined_brc1_brc2_sim_train_corpus/sim_train_manifest.csv`; the default synthetic crop root is
`reid-chips-sim/`. Pass `--sim-chip-root` to use a copy of those crops elsewhere. The source repository
does not generate or bundle this manifest or the synthetic crops.

**Range labels**: training manifests normalize the physical `380m` station to the `400m` range label.
References to 400m training or synthetic-source imagery below use that normalized label. The
`paperGR` test gallery uses physical 380m imagery.

**Test split**: `paperGR`, built from
`splits/combined_brc1_brc2_test/test_ground-range_combined_paperGR.json` (170 identities, none of them
training identities). Gallery is real 380m; queries are real 500m and 1000m. See the README for the
build command. Do not use the file of the same name under `combined_brc1_brc2/` (no `_test`): its
gallery contains 195 training identities, which makes it invalid for a fine-tuned checkpoint.

## Shared setup

All four methods use the same corpus, sampler and evaluation:

| | |
|---|---|
| corpus | 230 identities / 2,623 tracklets / 269,084 crops, all identities trained (`--holdout 0`) |
| init | each method's own released checkpoint (below); heads re-initialised for 230 classes |
| sampler | P×K identity sampler, K = 4 |
| image methods | each tracklet capped at 32 frames per epoch (`--frames-per-tracklet 32`), redrawn every epoch → **75,216 images/epoch** |
| SINet (video) | samples tracklets, not images → **2,416 tracklets/epoch**; restricted-random temporal sampling inside each tracklet (`seq_len 4`) |
| input | 256×128 |
| evaluation | `paperGR`, `--pooling video --metric cosine --max-per-video 0` |

The sampler balances identity, not range: each identity contributes whatever range mix the corpus
has, and 126 of the 230 identities have no 1000m data.

## Per-method settings

| | BoT | PersonViT | SINet | CLIP-ReID |
|---|---|---|---|---|
| directory | `methods/reid-strong-baseline` | `methods/PersonViT/transreid_pytorch` | `methods/SINet` | `methods/CLIP-ReID` |
| entry point | `tools/train_caliper.py` | `train_caliper.py` | `train_caliper.py` | `train_caliper.py` |
| config | `configs/caliper_ground_softmax_triplet.yml` | `configs/caliper_ground_vit.yml` | CLI flags | `configs/caliper_ground_clipreid.yml` |
| init checkpoint | `market_resnet50_model_120_rank1_945.pth` (Market1501) | `transformer_120.pth` (MSMT17 ViT-B) | `sinet_lsvid.pth.tar` (LS-VID) | `MSMT17_clipreid_ViT-B-16_60.pth` (MSMT17) |
| backbone | ResNet-50 | ViT-B/16 | video CNN | CLIP ViT-B/16 |
| optimiser, base LR | Adam, 3.5e-4 | SGD, 8e-3 | Adam, 3.0e-4 | Adam, 3.5e-4 (stage 2: 5e-6) |
| schedule | step at 40, 70 | cosine, warmup 20 | step at 160, 320, 480 (γ 0.1) | stage 1: 15 epochs; stage 2: step at 30, 50 |
| epochs | 120 | 120 | 600 | 15 + 60 |
| batch | 64 (16 ids × 4) | 64 | 32 (8 ids × 4) | 64 |
| triplet margin | 0.3 | 0.3 | 0.3 | 0.3 |
| seed | 1234 | 1234 | **1** (upstream parser default) | 1234 |

Download each init checkpoint from its upstream repository (linked in [RESULTS.md](RESULTS.md)). They
are not redistributed here.

**CLIP-ReID has two stages.** Stage 1 learns one text prompt per identity with the image encoder
frozen, over every training crop (no frame cap). Stage 2 trains the image encoder against those
prompts. Stage 2 uses the shared identity sampler. Both stages write checkpoints at their configured
intervals; stage-1 checkpoints contain the learned prompts and can be reused with `--stage1-init`.

## Running

Every entry point needs `caliper_eval` importable: run `python3 -m pip install -e '.[train]'` at the
repository root, or put the root on `PYTHONPATH`. Training expects exactly one visible GPU: set
`CUDA_VISIBLE_DEVICES` to a single device.

**FT (real data only):** run the matching command from its indicated method directory. Replace the
quoted placeholder paths with your crop directory, initial checkpoint, and output directory.

```bash
# BoT  (methods/reid-strong-baseline)
python3 tools/train_caliper.py --config_file configs/caliper_ground_softmax_triplet.yml \
    --init-weight weights/market_resnet50_model_120_rank1_945.pth --chip-root "/absolute/path/to/reid-chips" \
    SOLVER.CHECKPOINT_PERIOD 5 SOLVER.EVAL_PERIOD 999 OUTPUT_DIR "/absolute/path/to/output"

# PersonViT  (methods/PersonViT/transreid_pytorch)
python3 train_caliper.py --config_file configs/caliper_ground_vit.yml \
    --init-weight "/absolute/path/to/msmt17-vit-b/transformer_120.pth" --chip-root "/absolute/path/to/reid-chips" \
    SOLVER.CHECKPOINT_PERIOD 5 SOLVER.EVAL_PERIOD 999 OUTPUT_DIR "/absolute/path/to/output"

# SINet  (methods/SINet)
python3 train_caliper.py --arch sinet --init-weight weights/sinet_lsvid.pth.tar --chip-root "/absolute/path/to/reid-chips" \
    --max_epoch 600 --stepsize 160 320 480 --train_batch 32 --workers 3 \
    --eval_step 25 --ckpt_step 25 --save_dir "/absolute/path/to/output"

# CLIP-ReID  (methods/CLIP-ReID)
python3 train_caliper.py --config_file configs/caliper_ground_clipreid.yml \
    --init-weight weights/MSMT17_clipreid_ViT-B-16_60.pth --chip-root "/absolute/path/to/reid-chips" \
    SOLVER.STAGE2.EVAL_PERIOD 999 OUTPUT_DIR "/absolute/path/to/output"
```

`EVAL_PERIOD 999` disables in-training validation, since `--holdout 0` leaves no validation set.

**FT+syn:** use the matching complete command below from the same method directory. For BoT,
PersonViT, and CLIP-ReID, put every `--...` option **before** the first positional configuration
override (`SOLVER.*`, `OUTPUT_DIR`, etc.). Their parsers treat all later tokens as configuration
overrides, so appending synthetic flags or PersonViT's `--stop-after` after those overrides fails.

```bash
# BoT  (methods/reid-strong-baseline)
python3 tools/train_caliper.py --config_file configs/caliper_ground_softmax_triplet.yml \
    --init-weight weights/market_resnet50_model_120_rank1_945.pth --chip-root "/absolute/path/to/reid-chips" \
    --sim-mode mix --sim-p 0.5 --sim-chip-root "/absolute/path/to/reid-chips-sim" \
    SOLVER.MAX_EPOCHS 30 SOLVER.CHECKPOINT_PERIOD 5 SOLVER.EVAL_PERIOD 999 OUTPUT_DIR "/absolute/path/to/output"

# PersonViT  (methods/PersonViT/transreid_pytorch)
python3 train_caliper.py --config_file configs/caliper_ground_vit.yml \
    --init-weight "/absolute/path/to/msmt17-vit-b/transformer_120.pth" --chip-root "/absolute/path/to/reid-chips" \
    --sim-mode mix --sim-p 0.5 --sim-chip-root "/absolute/path/to/reid-chips-sim" --stop-after 45 \
    SOLVER.CHECKPOINT_PERIOD 5 SOLVER.EVAL_PERIOD 999 OUTPUT_DIR "/absolute/path/to/output"

# SINet  (methods/SINet)
python3 train_caliper.py --arch sinet --init-weight weights/sinet_lsvid.pth.tar --chip-root "/absolute/path/to/reid-chips" \
    --sim-mode mix --sim-p 0.5 --sim-chip-root "/absolute/path/to/reid-chips-sim" \
    --max_epoch 600 --stepsize 160 320 480 --train_batch 32 --workers 3 \
    --eval_step 25 --ckpt_step 25 --save_dir "/absolute/path/to/output"

# CLIP-ReID  (methods/CLIP-ReID)
python3 train_caliper.py --config_file configs/caliper_ground_clipreid.yml \
    --init-weight weights/MSMT17_clipreid_ViT-B-16_60.pth --chip-root "/absolute/path/to/reid-chips" \
    --sim-mode mix --sim-p 0.5 --sim-chip-root "/absolute/path/to/reid-chips-sim" --sim-stage1 real \
    SOLVER.STAGE2.MAX_EPOCHS 5 SOLVER.STAGE2.EVAL_PERIOD 999 OUTPUT_DIR "/absolute/path/to/output"
```

The FT+syn runs ended at epoch 30 for BoT, epoch 45 for PersonViT, stage-2 epoch 5 for
CLIP-ReID, and epoch 600 for SINet. For PersonViT, use `--stop-after 45` to keep the
learning-rate schedule from the 120-epoch run. For CLIP-ReID, `--sim-stage1 real`
fits prompts on real crops. Synthetic crops enter training in stage 2.

Each entry point checks the corpus composition (identities, epoch size, synthetic share) before
training and refuses to start if it does not match the expected values for the chosen mode.

### What `--sim-mode mix` does

Each of the 502 real 400m tracklets that has a synthetic version becomes a group: the real tracklet
with weight `1 − p = 0.5`, and its synthetic versions sharing the other 0.5. The sampler draws exactly
one member per group, **redrawn every epoch**, so every subject keeps seeing both real and synthetic
400m data. Epoch size, step count and identities are unchanged from FT:

| | FT | FT+syn |
|---|--:|--:|
| images/epoch (image methods) | 75,216 | 75,216 |
| tracklets/epoch (SINet) | 2,416 | 2,416 |
| synthetic share of epoch | 0% | 9.9% of images / 9.7% of tracklets |
| identities with 1000m-labelled data | 104 | 191 |

`mix` swaps data rather than adding it: about half of the real 400m data per epoch is replaced
(18.9% → ~9.0% of images). The `paperGR` gallery is real 380m imagery (the normalized 400m range), so a
negative FT+syn result could mean either that synthetic data doesn't help or that losing real
gallery-range training data hurts.

Other `--sim-mode` values (`add`, `dup`, `swapall`) and `--sim-severity` are available, but they were
not used for the published results.

## Which checkpoint to evaluate

Each trainer writes `run.json` beside its checkpoints with the settings, code revision, and
training and validation identities. BoT also writes `epoch_map.jsonl`. Match the chosen epoch
to its global iteration to find the BoT checkpoint.

Training writes checkpoints periodically, and one epoch per method is reported. To choose the epoch, train a separate run with the same settings and hold out 30 identities
(`--holdout 30 --holdout-rule thin-relaxed`). Use the epoch with the highest validation mAP
on 1000m queries. The remaining 200 identities serve as the training set for this selection run.

| | BoT | PersonViT | SINet | CLIP-ReID |
|---|--:|--:|--:|--:|
| epoch to score | 10 | 5 | 600 | 5 (stage 2) |

Use the checkpoint from the epoch shown above for both FT and FT+syn.

### Score a trained BoT checkpoint

Build the `paperGR` split using the [evaluation guide](README.md#reproducing-the-evaluation).
In the BoT environment, start in the repository root. Set `RUN` to the `OUTPUT_DIR`
from your real-data training command. BoT names each checkpoint by iteration, so
look up the final iteration of epoch 10 in that run’s `epoch_map.jsonl`.

```bash
RUN="/absolute/path/to/bot-ft-training-output"
STEP=$(python3 - "$RUN/epoch_map.jsonl" <<'PY'
import json
import sys

with open(sys.argv[1]) as records:
    for line in records:
        record = json.loads(line)
        if record["epoch"] == 10:
            print(record["global_step"])
            break
    else:
        sys.exit("This run has no entry for epoch 10.")
PY
)
WEIGHT="$RUN/resnet50_model_$STEP.pt"
FEATURES="$REID_BENCH_ROOT/fine-tuned/bot/ft/paperGR"

(
  cd methods/reid-strong-baseline
  python3 tools/extract_features.py --config_file configs/caliper_ground_softmax_triplet.yml \
      --split paperGR --source market1501 --out-dir "$FEATURES" \
      MODEL.PRETRAIN_CHOICE self TEST.NECK_FEAT after TEST.WEIGHT "$WEIGHT"
)
python3 -m caliper_eval.eval_from_features --features-dir "$FEATURES" --pooling video
```

Read the `500m` and `1000m` video scores in `$FEATURES/eval_mean_cosine.json`.
For FT+syn, use its training output for `RUN` and set `FEATURES` to
`$REID_BENCH_ROOT/fine-tuned/bot/ft-syn/paperGR`. This keeps both trained models’
features separate from the pretrained BoT results in `$REID_BENCH_ROOT/eval/`.

## Known limitations

| | |
|---|---|
| one seed per run | no repeated runs, so no estimate of run-to-run variance |
| no GSD correction | synthetic crops keep their 400m source size (median 381 px), while real 1000m crops are 233 px, so after resizing the synthetic crops are sharper than the long-range imagery they stand in for. The simulation models turbulence, not the change in apparent size. |
| range not balanced | the sampler balances identity only (see Shared setup) |

## Hardware notes

Training on 256×128 crops is often limited by the dataloader, not the GPU: JPEG decoding and resizing
dominate. If you reproduce this on different hardware, check which side is the bottleneck before
adding GPUs. A full 120-epoch BoT or PersonViT run took about 6–7.4 hours on a 4-GPU, 6-core machine
running three jobs at once.
