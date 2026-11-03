# How the four methods were trained — real (A) and synthetic (B)

Every number in this document is read out of the runs' own `run.json` provenance records and training
logs, not from the plans. Written 2026-08-28.

Methods: **BoT** (reid-strong-baseline), **PersonViT**, **SINet**, **CLIP-ReID**. Out of scope:
X-TFCLIP and CAL, which have zero-shot numbers only.

---

## 0. Shared setup — identical across all four methods and both conditions

| | |
|---|---|
| corpus | `train_manifest.csv`, filtered to its ground-field rows (`kind == "field"`, `cls == "ground"`) |
| size | **230 identities / 2,623 tracklets / 269,084 chips** / 412 clothes classes |
| holdout | **none** — all 230 identities train; `val_ids = 0` |
| identity label | the manifest's own `label` column, never directory names (44 canonical people own two raw path ids) |
| chips | pre-extracted crops staged on tmpfs: `/dev/shm/reid-chips` (2.7 GB real), `/dev/shm/reid-chips-sim` (1.4 GB synthetic) |
| init | each method's own largest source-domain checkpoint (below) — none trained from scratch |
| entry point | a per-repo `train_caliper.py`, sharing `caliper_eval.train_entry` / `train_corpus` / `train_sampler`. **No upstream repo files were edited.** |
| evaluation | paperGR: 170 test identities (disjoint from the 230 trained on), gallery real 400m, queries real 500m + 1000m, `--pooling video --metric cosine --max-per-video 0` |

**Sampling — the one shared non-obvious choice.** All four use a P×K identity sampler with **K = 4**.
The image methods cap each tracklet at **32 frames per epoch, redrawn every epoch** rather than once
at list-build time, so across a long schedule the model sees essentially every chip while each epoch
stays the size of the test unit of account. That yields **75,216 images/epoch**. SINet is video-native
and samples *tracklets*, not images: **2,416 tracklets/epoch**, with no frame cap — it does its own
restricted-random temporal sampling (`seq_len = 4`) inside each tracklet, so a frame cap here would be
redundant.

⚠ **Range is never balanced by the sampler.** It groups only by identity and tracklet; the range mix
each identity contributes is whatever the corpus supplies. Per identity the median largest single
range is 33% of their epoch, 42 of 230 sit above 50%, and **126 of 230 have no 1000m data at all**
while 1000m is 9.7% of the epoch — which is the headroom the synthetic condition is aimed at.

---

## A. Real-data finetuning (`full230`)

The unmodified real corpus. One run per method, and every synthetic run is compared only against its
*own* method's `full230`, never across methods.

| | BoT | PersonViT | SINet | CLIP-ReID |
|---|---|---|---|---|
| init from | `market_resnet50_model_120_rank1_945.pth` | `transformer_120.pth` (MSMT17 ViT-B) | `sinet_lsvid.pth.tar` | `MSMT17_clipreid_ViT-B-16_60.pth` |
| backbone | ResNet-50 | ViT-B/16 | video CNN | CLIP ViT-B/16 |
| optimiser | Adam | SGD | Adam | Adam |
| base LR | 3.5e-4 | 8e-3 | 3.0e-4 | 3.5e-4 |
| schedule | step `(40, 70)` | cosine, warmup 20 | step `[160, 320, 480]`, γ 0.1 | stage 2 step `(30, 50)`, warmup 500 it |
| epochs | 120 | 120 | 600 | stage 1 **15** + stage 2 **60** |
| batch | 64 (16 ids × 4) | 64 | 32 (8 ids × 4) | 64 |
| epoch size | 75,216 img | 75,216 img | 2,416 tracklets | 75,216 img (stage 2) |
| input | 256×128 | 256×128 | 256×128, seq_len 4 | 256×128 |
| triplet margin | 0.3 | 0.3 | 0.3 | 0.3 |
| checkpoints | every 5 ep | every 5 ep | every 25 ep | every 5 ep (stage 2) |
| **seed** | **1234** | **1234** | ⚠ **1** | **1234** |

⚠ **SINet's seed is 1, not 1234.** It reuses upstream's own parser, whose `--seed` default is 1; the
other three take 1234 from `add_caliper_args`. It is applied consistently within SINet — its real and
synthetic runs both use 1 — so SINet's own contrast is unaffected, but a blanket "seed 1234" does
not hold for SINet.

⚠ **CLIP-ReID is the only two-stage method.** Stage 1 fits one 4-token text phrase per identity
against a frozen cache of every training image, walked **once, uncapped and outside any sampler**;
the image encoder is frozen throughout. Stage 2 then trains the encoder against those phrases and is
the only stage that writes checkpoints. So the two stages see different data: stage 1 sees all
269,084 chips uncapped (66.9% of which is close-range 10/160/250m), stage 2 sees the 75,216-image
capped draw (54.8% close-range).

---

## B. Synthetic-augmented finetuning (`sim_mix50`)

### What the synthetic data is

Not new imagery: a **1:1 re-render of the real 400m ground chips** through a pybsm atmospheric
turbulence simulation at three strengths — `cn2_1e-13` (weak), `cn2_4e-13` (medium), `cn2_1e-12`
(strong). **177 subjects / 502 tracklets / 34,251 chips per strength**, with `src_chip_path` equal to
the real manifest's `chip_path` exactly. That exactness is what lets a synthetic chip drop into the
slot its source occupied, preserving identity, camera count, frame count and framing by construction.
The corpus builder asserts the join and refuses to start on a partial match.

### What `mix` does

Each of the 502 real 400m tracklets becomes a **variant group**: the real tracklet at weight
`1 − p = 0.5`, and the enabled synthetic twins sharing the remaining 0.5 evenly. The sampler draws
**exactly one member per group per epoch, redrawn every epoch**.

⚠ **The per-epoch redraw is the point.** A fixed partition would permanently deprive some subjects of
their real 400m — with a median of 2 real-400m tracklets per subject, a 50% fixed split strands 44 of
177 subjects with none at all. Redrawing costs nothing and leaves every subject seeing both.

### What is held constant against A

| | real (A) | synthetic (B) |
|---|--:|--:|
| images/epoch (image methods) | 75,216 | **75,216** |
| tracklets/epoch (SINet) | 2,416 | **2,416** |
| optimizer steps/epoch | 1,175 @ batch 64 | **1,175** |
| identities | 230 | 230 |
| synthetic share of epoch | 0% | **9.9%** of images / 9.7% of tracklets |
| 1000m-labelled subjects | 104 | **191** |

So B is budget-matched to A: same epoch size, same step count, same identities. **Only the pixels of
about half the 400m tracklets differ.** Both facts are asserted before training starts by
`train_entry.log_composition`, which refuses to run on a corpus that does not match the design — a
wrong substitution otherwise trains happily and reports a meaningless number.

⚠ **`mix` is a swap, not an addition.** It does not add synthetic imagery on top of the real corpus;
it replaces ~half the real 400m footage. Measured: real 400m falls from 14,253 images/epoch (18.9%)
to ~6,783 (9.0%). Since paperGR's **gallery is real 400m**, a negative result is ambiguous between
"synthetic does not help" and "halving the gallery-domain training data hurts" — both predict the same
sign. BoT separates these with a matched `add`/`dup` pair; the other three have neither.

### Per-method synthetic runs

Every arm: `--sim-mode mix --sim-p 0.5 --sim-chip-root /dev/shm/reid-chips-sim`, same seed as that
method's `full230`, same schedule, one run per arm.

| | BoT | PersonViT | SINet | CLIP-ReID |
|---|---|---|---|---|
| sampler | `MixingIdentitySampler` | `MixingIdentitySampler` | `MixingTrackletSampler` | `MixingIdentitySampler` (stage 2) |
| trained to | e30 | **e45** via `--stop-after` | e600 | stage 2 **e5** |
| checkpoints | 30 (swept @5) | 9 | 24 | **1** |
| readout (criterion E) | e10 | e5 | e600 | e5 |

⚠ **PersonViT uses `--stop-after 45`, not `MAX_EPOCHS 45`.** Its LR is cosine and the scheduler is
built from `MAX_EPOCHS`, so setting 45 would *compress* the cosine into 45 epochs — a different model,
not `full230`'s first 45. The scheduler is built for 120 and only the loop is shortened. Confirmed end
to end: LR at e45 logged 5.54e-03, matching the retained 120-epoch cosine.

⚠ **SINet needed the one new algorithm.** Its epoch is a list of tracklets and upstream's sampler has
no notion of variants — left alone, `mix` would have served the real tracklet *and all three
strengths at once*, a 1.57× epoch matching no defined mode, with no error anywhere.

⚠ **CLIP-ReID: stage 1 is fixed, not retrained.** Stage 1 bypasses the sampler entirely, so with
synthetic attached it would have silently received every rendering at once — 371,837 images (1.38×),
unevenly across identities (177 of 230 gain some, median +33%, max +300%; 53 gain none). Since
`cls_ctx` is one phrase per identity, that would have been a differential, identity-dependent change
to the target stage 2 trains against. Resolved by loading the baseline's completed phrase bank via
`--stage1-init` and training stage 2 only — **verified bit-identical** to the baseline's
(max abs diff 0.0 over 230×4×512), with the identity ordering checked against the source run's
`run.json` rather than assumed.

### The fixed-strength variant (`sim_mix50_med`)

`--sim-severity cn2_4e-13` restricts each variant group to `[real, medium]` at 0.5/0.5. **Epoch size,
synthetic share and 1000m coverage are unchanged**, so strength is isolated from everything else.

⚠ **The three strengths may not be ordered as named.** Mean |real − sim| pixel deviation over ~130
chips each: weak `1e-13` **42.6**, medium `4e-13` **41.1**, strong `1e-12` **38.8** — the nominal
ordering is inverted and the spread is narrow.

⚠ **`--sim-severity` was silently dropped by two adapters until 2026-08-28.** PersonViT's and
CLIP-ReID's `CaliperGround` adapters end in `**kwargs` and did not list `sim_severities`, so the keyword
was swallowed without error and a "medium-only" run would have trained on all three. Caught on
CLIP-ReID's first launch from the corpus stats line and fixed in both before any medium run was kept.
BoT's and SINet's entry points always forwarded it correctly.

---

## Known limitations of this design

| | |
|---|---|
| **one seed per arm** | no within-method repeat, so no noise floor. A single arm's delta cannot be judged against run-to-run variance. |
| **`mix` only** | `add` / `dup` / `swapall` were run for BoT but dropped for the other three, so only BoT has the matched null control its headline rested on. |
| **no GSD correction** | synthetic chips keep their 400m source size (median 381 px) while real 1000m chips are 233 px, so after resize they reach the network **sharper** than the real long-range imagery they stand in for. The simulator models turbulence, not the apparent-size drop of a longer standoff. |
| **CLIP-ReID is one point** | stage 2 stopped at e5 because both criteria select e5, so its contrast has no curve and the criteria agreeing carries no information. |
| **criterion not pre-registered** | A (`hold30_thin`) and E (`hold30_thin-relaxed`) select different epochs on three of four methods, so both are always reported; picking the flattering one would be selecting on test. |
