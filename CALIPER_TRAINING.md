# CALIPeR ReID finetuning

Finetune a ReID method on CALIPeR ground field video, score on `paperGR`.

## Data

Ground field only — no aerial, gait or enrollment.

**Train** — `$CALIPER_DATA_ROOT/splits/combined_brc1_brc2_train_corpus/`, the labelled crop list
(CSVs saying which crop belongs to whom; not images, not kwcoco):
- `train_chips.csv` one row per crop · `train_tracklets.csv` same at tracklet granularity
- `train_classes.json` `{canonical_id: label}` — identity comes from here, never directory names
  (44 people own two raw subject dirs, so `ImageFolder` gives 234 classes for 232 people)
- `train_manifest.csv` — not the training list; read only to find tracklets sharing a source video

Chips: `$CALIPER_DATA_ROOT/reid-chips/` (`chip_path` is relative to this).

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
→ built to kwcoco at `$REID_BENCH_ROOT/splits/paperGR/`:
gallery 926 clips / 65,884 crops, query_500m 890 / 45,150, query_1000m 712 / 37,292. 170 ids, 0 train.

⚠ The same filename exists in `combined_brc1_brc2/` (no `_test`): 365 gallery ids, **195 of them
training identities**. That is what `reid_bench` scores — fine zero-shot, invalid finetuned.

```bash
python -m caliper_eval.split_from_combined \
    --split-json $CALIPER_DATA_ROOT/splits/combined_brc1_brc2_test/test_ground-range_combined_paperGR.json \
    --out-dir <your>/splits/paperGR --gallery-token 380m --query-token-per-tok
```
One invocation only — two runs renumber subjects per file and destroy every match.

## Shared modules

| | |
|---|---|
| `caliper_eval/train_corpus.py` | `GroundFieldCorpus` — training items. Pure stdlib. |
| `caliper_eval/train_sampler.py` | `ResamplingIdentitySampler` — P×K, caps frames/tracklet in `__iter__` (runs once per epoch, so the draw is fresh each epoch) |

`pid` dense 0..229 · `camid` = range token (`10m`…`1000m` → 0–5).

## Method entrypoints

| method | entry point |
|---|---|
| **bot** | `tools/train_caliper.py` |
| **personvit** | `transreid_pytorch/train_caliper.py` |
| **clip-reid** | `train_caliper.py` |
| **sinet** | `train_caliper.py` |

Init = largest source corpus: bot→market1501, clip-reid→msmt17, personvit→msmt17, sinet→lsvid.
Load backbone, reinit heads.

Shared machinery in `caliper_eval/`: `train_entry.py` (single-GPU assert, reporting checkpoint loader,
the standard flags), `train_metrics.py` (per-range val + `val_metrics.jsonl`), `train_metadata.py`
(`run.json` incl. the explicit held-out id list), `train_sampler.py`, `train_corpus.py`.
reid-strong-baseline and PersonViT predate `train_entry.py` and still carry their own copies of those
helpers.

### SINet — where it stands

`SINet/caliper_ground.py` is written and verified: 230 ids / 2,623 tracklets / 269,084 frames, pids
0..229, cams 0..5; holdout `thin` gives 2,318 tracklets over 200 ids with val query cams [4,5] against
gallery [3]. It emits **3-tuples** `(img_paths, pid, camid)` — `utils/data_manager.py:179`'s shape
(bot emits 3-tuples per IMAGE, personvit/clip-reid 4-tuples per image, sinet 3-tuples per TRACKLET).

**Video sampling.** An "epoch" means something different here than for the image methods. Measured:

| | epoch unit | frames/epoch |
|---|---|--:|
| bot / personvit / clip-reid | 73,664 images (`ResamplingIdentitySampler`, <=32/tracklet, redrawn) | 73,664 |
| sinet | 2,623 tracklets x seq_len 4 | 10,492 |

So a SINet epoch sees ~7x less data than a bot epoch, and the overtraining finding (peak by ~e25 of
120) may not transfer — video methods could need MORE epochs, not fewer. Frame selection is
restricted random sampling: chop into seq_len chunks, draw one frame per chunk, redraw each epoch.

Kept each repo's native sampling and published epoch count, and let the holdout val curve plus
retained checkpoints locate the epoch — same as the image methods, keeping the controlled variable
the data rather than the sampler.

### BoT

```bash
PYTHONPATH=/path/to/caliper_reid_eval:. python tools/train_caliper.py \
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

**Training here was CPU-starved, not GPU-starved** -- a four-GPU box behind a six-core CPU, fully
dataloader-bound (GPUs idling at 36-93% during training). If reproducing on different hardware,
check which side you're bound on before assuming more/faster GPUs will help.

MEASURED END-TO-END on that box (a GPU-only microbenchmark undercounted real throughput by ~3x --
don't trust a GPU-only benchmark's epoch-time estimate):

| | s/epoch | samples/s |
|---|--:|--:|
| bot (ResNet50), solo, 200 ids | 122 | 545 |
| bot, 3 concurrent | 166-196 | 383-453 |
| personvit (ViT-B/16), 3 concurrent | 185-221 | ~400 |
| clip-reid stage 2, 4 concurrent (one noisy sample) | ~420 | 176 |

Per-op share of the dataloader, single-threaded: Resize bicubic 35% - JPEG decode 24% - ToTensor 23%
- Normalize 10% - Pad+RandomCrop 8%. Full 120-epoch runs took **6-7.4 h**, not the 1.7-2.9 h a
GPU-only estimate predicted.

**Staging chips to a tmpfs (`/dev/shm`) did not speed anything up** (678 img/s vs 680 from HDD) --
copying the chips warms the page cache anyway. tmpfs buys guaranteed residency, not speed, and being
swap-backed it can still be paged out.

*Rejected:* a pre-decoded, pre-resized chip cache to remove decode+resize (59% of dataloader cost).
Abandoned because it cannot serve most methods -- each uses a different resize convention (e.g. BoT
resizes BILINEAR) -- leaving only CLIP-ReID's stage 2 to benefit, and an end-to-end loader comparison
there disagreed for reasons never isolated.

**Schedule finding:** the published 120 epochs badly overtrains this corpus **for BoT** -- val peaks
by ~e25 and declines 10-13 mAP points by e120, across all three holdout rules. It does NOT generalise:
PersonViT ends higher than it started on 4 of 6 curves, CLIP-ReID falls only 3-5 points (and peaks at
its first readout), and SINet is still rising at epoch 600. Overtraining at this schedule is
BoT-specific, not a corpus-wide property.
