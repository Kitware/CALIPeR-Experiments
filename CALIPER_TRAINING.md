# CALIPeR ReID finetuning

Finetune a ReID method on CALIPeR ground field video, score on `paperGR`.

## Data

Ground field only — no aerial, gait or enrollment.

**Train** — `./data/splits/combined_brc1_brc2_train_corpus/`, the labelled crop list
(CSVs saying which crop belongs to whom; not images, not kwcoco):
- `train_chips.csv` one row per crop · `train_tracklets.csv` same at tracklet granularity
- `train_classes.json` `{canonical_id: label}` — identity comes from here, never directory names
  (44 people own two raw subject dirs, so `ImageFolder` gives 234 classes for 232 people)
- `train_manifest.csv` — not the training list; read only to find tracklets sharing a source video

Chips: `./data/reid-chips/` (`chip_path` is relative to this).

Filtering, in `caliper_eval/train_corpus.py`:

| | ids | tracklets | chips |
|---|--:|--:|--:|
| all modalities | 232 | 52,149 | 974,178 |
| ↳ `modality=='field' and cls=='ground'` | 230 | 3,177 | 350,360 |
| ↳ minus tracklets sharing a video with another person (`dedup=True`) | 230 | 2,623 | 269,084 |

Per epoch the sampler draws ≤32 frames/tracklet → **73,664 images**, re-drawn each epoch.

⚠ To check the dedup yourself, group by `source_clip_id`, never `source_relpath` — the latter is the
per-subject trim, 1:1 with tracklet, so it always reports zero sharing.

**Training partition:** use `train_manifest.csv` filtered to `kind == "field" and
cls == "ground"`, classes from `train_manifest_classes.json`. That is `dedup=True`, the default, and
verified to reproduce the specified partition (2,623 tracklets, and our labels match his file 230/230).

Deduplication observations: the rule costs 115 of 230 identities a median 38% of their ground
tracklets and buys **no** detectable label-noise protection — with ≥8 reference tracklets the rate of
"looks more like the other person" is 0%, and the apparent 12.2% is an artifact of single-reference
identities. Worth knowing if data volume ever becomes the binding constraint; not a reason to deviate
from the specified split.

**Test** — `combined_brc1_brc2_test/test_ground-range_combined_paperGR.json`
→ built to kwcoco at `./reid-bench/splits/paperGR/`:
gallery 926 clips / 65,884 crops, query_500m 890 / 45,150, query_1000m 712 / 37,292. 170 ids, 0 train.

⚠ The same filename exists in `combined_brc1_brc2/` (no `_test`): 365 gallery ids, **195 of them
training identities**. That is what `reid_bench` scores — fine zero-shot, invalid finetuned.

```bash
python -m caliper_eval.split_from_combined \
    --split-json ./data/splits/combined_brc1_brc2_test/test_ground-range_combined_paperGR.json \
    --out-dir <your>/splits/paperGR --gallery-token 380m --query-token-per-tok
```
One invocation only — two runs renumber subjects per file and destroy every match.

## Shared modules

| | |
|---|---|
| `caliper_eval/train_corpus.py` | `GroundFieldCorpus` — training items. Pure stdlib. |
| `caliper_eval/train_sampler.py` | `ResamplingIdentitySampler` — P×K, caps frames/tracklet in `__iter__` (runs once per epoch, so the draw is fresh each epoch) |

`pid` dense 0..229 · `camid` = range token (`10m`…`1000m` → 0–5) · `clothes_id` = `(id, collection,
set)`, 412 classes, CAL only.

## Method entrypoints

| method | entry point |
|---|---|
| **bot** | `tools/train_caliper.py` |
| **personvit** | `transreid_pytorch/train_caliper.py` |
| **clip-reid** | `train_caliper.py` |
| **sinet** | — |
| **x-tfclip** | `train_caliper.py` |
| cal | — |

Init = largest source corpus: bot→market1501, clip-reid→msmt17, personvit→msmt17, sinet→lsvid,
x-tfclip→agvpreid, cal→last. Load backbone, reinit heads.

Shared machinery in `caliper_eval/`: `train_entry.py` (single-GPU assert, reporting checkpoint loader,
the standard flags), `train_metrics.py` (per-range val + `val_metrics.jsonl`), `train_metadata.py`
(`run.json` incl. the explicit held-out id list), `train_sampler.py`, `train_corpus.py`.
reid-strong-baseline and PersonViT predate `train_entry.py` and still carry their own copies of those
helpers.

### SINet — where it stands

`SINet/caliper_ground.py` is written and verified: 230 ids / 2,623 tracklets / 269,084 frames, pids
0..229, cams 0..5; holdout `thin` gives 2,318 tracklets over 200 ids with val query cams [4,5] against
gallery [3]. It emits **3-tuples** `(img_paths, pid, camid)` — `utils/data_manager.py:179`'s shape,
and a fourth distinct arity across the repos (bot 3-tuple per IMAGE, personvit/clip-reid 4-tuple per
image, sinet 3-tuple per TRACKLET).

Two things still open:

**1. Entry point.** SINet is not importable: `args = parser.parse_args()` runs at module scope
(`main.py:82`) and `train()`/`test()` read that global. So the pattern used for personvit and
clip-reid — import the repo's pieces into a separate `train_caliper.py` — does not work here. Either add
`--holdout`, `--holdout-rule` and `--chip-root` to `main.py`'s parser plus a provenance call (a small
additive upstream edit, same category as the two already made to reid-strong-baseline), or copy
`main()`'s body into a new entry point (~100 lines, follows the no-upstream-edits convention).

**2. Video sampling.** An "epoch" means something different here. Measured:

| | epoch unit | frames/epoch |
|---|---|--:|
| bot / personvit / clip-reid | 73,664 images (`ResamplingIdentitySampler`, <=32/tracklet, redrawn) | 73,664 |
| sinet | 2,623 tracklets x seq_len 4 | 10,492 |
| x-tfclip | 2,623 tracklets x seq_len 8 | 20,984 |

So a SINet epoch sees ~7x less data than a bot epoch, and the overtraining finding (peak by ~e25 of
120) may not transfer — video methods could need MORE epochs, not fewer. Frame *selection* already
agrees across both video methods (`TemporalRestrictedCrop` and `rrs_train` are both restricted random
sampling: chop into seq_len chunks, draw one frame per chunk, redraw each epoch). `seq_len` is not
free for x-tfclip — its `video_embed` is shaped `(8, 768)` and `TMD` is built with `T=SEQ_LEN`, so
changing it invalidates the AG-VPReID checkpoint.

Sampling recommendation: keep each repo's native sampling and published epoch count, and let the
holdout val curve plus retained checkpoints locate the epoch — same as the image methods, keeping the
controlled variable the data rather than the sampler. Rejected alternative: sampling multiple clips
per tracklet per epoch to close the gap, since running more epochs achieves the same with no
deviation. Unverified: what SINet does with tracklets shorter than `seq_len` (x-tfclip pads by
repeating the final frame).

### X-TFCLIP — where it stands

`datasets/caliper_ground.py`, `configs/caliper_ground_xtfclip.yml` and `train_caliper.py` are written and
smoke-tested (4 epochs, holdout and no-holdout). Verified: `loaded 252 | skipped 4 (classifier) |
mismatch 0 | unfilled 0`; epoch = 2,424 tracklets / **134 iterations** at IMS_PER_BATCH 18 (2,178 /
121 under a 30-id holdout); loss 11.8 → 7.0 and `Acc_clip` 0.67 → 0.87 over 4 epochs; per-range
`val_metrics.jsonl` written; checkpoints land on BOTH run types.

Throughput depends on hardware and concurrent workloads.

⚠ **The first ~4 minutes of every run are not training.** `do_train_stage2` builds the CLIP-Memory
bank before epoch 1 by walking the whole corpus once, forward-only, at `sample='dense'`: 269,084
frames at ~960 frames/s, ~5 training epochs' worth. Do not read a quiet first four minutes as a hung
job. Its largest single forward is the 521-frame tracklet as 65 clips x 8 = **520 images at once**;
measured to fit an A6000 comfortably under `no_grad` + autocast.

**Four upstream defects, none fixed by editing upstream** — `do_train_stage2` runs verbatim and the
names it resolves at call time are rebound from the entry point, the technique CLIP-ReID's entry point
already uses. See `X-TFCLIP/train_caliper.py`'s module docstring. Two of the four are silent; one of
those was not in the original survey:

| defect | silent? |
|---|---|
| `train.py:13` imports `do_train_stage1` -> nonexistent `baselines` package | no, import error |
| `processor_clipreid_stage2.py:267` unpacks `compute_v2()`'s 14 returns into 7 | no, crash |
| `R1_mAP_eval.compute()` uses `gf = feats[0:]`, the WHOLE set, as the gallery (`utils/metrics.py:206-208`) | **YES** |
| `save_checkpoint` gated inside the eval block, single overwritten filename (`:281`) | **YES** |
| `clip_memory_features/` keyed on batch index while its loader is `shuffle=True` (`:86-121`) | **YES** |
| `best_cmc` initialised to a float, assigned only under `is_best`, indexed unconditionally at `:287` | **YES**, at the very end |

The last one is the one a no-holdout run hits: all-zero val -> `is_best` never fires -> `TypeError`
after the final epoch, past every checkpoint but with a non-zero exit the launcher reads as
TRAIN FAILED. Handled by making the first empty-val call compare as best.

**Not used here: `disable_eval_if_no_holdout`.** Every other method calls it; X-TFCLIP must not,
because in this repo disabling evaluation also disables saving. The metric shim tolerates an empty val
set instead, which keeps the save path alive on full-230 runs.

### BoT

```bash
PYTHONPATH=/absolute/path/to/CALIPeR-Experiments:. python tools/train_caliper.py \
    --config_file configs/caliper_ground_softmax_triplet.yml \
    --init-weight ./weights/market_resnet50_model_120_rank1_945.pth
```
`--holdout 30` for the pilot · `--chip-root` for a `/dev/shm` stage · `--frames-per-tracklet`.

Good start logs `loaded 323 | classifier skipped 1 | mismatch 0 | unfilled 0`; it refuses to train on
any shortfall. `PRETRAIN_CHOICE='self'` is unusable (parses start_epoch from the filename → 945; strict
load fails on the 751 vs 230 classifier), hence `'none'` + explicit transfer.

`engine/trainer.py` is the one upstream edit: ignite ≥0.3 made `ModelCheckpoint`'s 3rd positional
`score_function`, so `checkpoint_period` was passed as a callable. Now
`Events.EPOCH_COMPLETED(every=checkpoint_period)`.

## Performance

Profile the data loader as well as GPU execution when tuning throughput.

**Staging to `/dev/shm` does not speed anything up** (678 img/s vs 680 from HDD) -- copying the chips
warms the page cache anyway. tmpfs buys guaranteed residency, not speed, and being swap-backed it can
still be paged out.

*Rejected:* a pre-decoded, pre-resized chip cache to remove decode+resize (59% of dataloader cost).
Abandoned because it cannot serve most methods -- bot resizes BILINEAR, x-tfclip is 288x144, and CAL's
`ResizeWithEqualScale` pads to preserve aspect ratio rather than scaling -- leaving only clip-reid's
stage 2 to benefit, and an end-to-end loader comparison there disagreed for reasons never isolated.

**Schedule finding:** the published 120 epochs badly overtrains this corpus **for BoT** -- val peaks
by ~e25 and declines 10-13 mAP points by e120, across all three holdout rules. It does NOT generalise:
PersonViT ends higher than it started on 4 of 6 curves, CLIP-ReID falls only 3-5 points (and peaks at
its first readout), and SINet is still rising at epoch 600. Epoch selection must be validated separately for each method.
