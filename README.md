# CALIPeR-Experiments

Official benchmarking code for the WACV 2027 paper **"CALIPeR: Co-Temporal Atmospheric Long-range
Imagery for Person Re-identification."**

## Layout

- **`caliper_eval/`**: shared, method-agnostic components for the CALIPeR ReID benchmark. Every method
  under `methods/` imports this package, so the parts that must be identical across methods (the
  evaluation metric, the feature-cache format, the training corpus and sampler) are written once.
- **`methods/`**: one subdirectory per method. Each is a fork of its upstream repository with CALIPeR
  integration added on top, keeping its own README and LICENSE. [`methods/UPSTREAM.md`](methods/UPSTREAM.md)
  lists the upstream commit each fork is based on and the files we changed:
  `reid-strong-baseline` (BoT), `SINet`, `CLIP-ReID`, `PersonViT` (fine-tuned and zero-shot), and
  `X-TFCLIP`, `GEFF` (CAL), `sapiensid` (zero-shot only).

## Results

See the [published results](RESULTS.md) for zero-shot and fine-tuned retrieval scores. The
[training guide](TRAINING.md) gives the settings and readout epoch for each fine-tuned model.

## Install

```bash
pip install -e .            # evaluation only (numpy)
pip install -e .[train]     # also the training modules (torch)
```

Each method also needs its own upstream dependencies; see the README in its `methods/` subdirectory.

## Data

| variable | default | what it points to |
|---|---|---|
| `CALIPER_DATA_ROOT` | `./data` | the CALIPeR release: `splits/` and `reid-chips/` |
| `REID_BENCH_ROOT` | `./reid-bench` | where this code writes crop-level splits, features and eval results |
| `CALIPER_EVAL_HOME` | the repository root | where method scripts find `caliper_eval` if it isn't installed |

Defaults are relative to the current directory unless stated otherwise. `REID_BENCH_ROOT` has this
layout:

```
splits/<split>/{gallery,query_*}.kwcoco.json        crop-level splits
eval/<dataset>/<split>/<method>/<source>/           feature .npz files and eval .json results
```

## Reproducing the evaluation

**1. Build the crop-level splits** from the video-level split JSONs (CPU only). Run each command once;
building a split's gallery and queries in separate invocations renumbers subjects and breaks
matching.

```bash
S=$CALIPER_DATA_ROOT/splits/combined_brc1_brc2_test
O=$REID_BENCH_ROOT/splits
python -m caliper_eval.split_from_combined --split-json $S/test_ground-range_combined_paperGR.json \
    --out-dir $O/paperGR --gallery-token 380m --query-token-per-tok
python -m caliper_eval.split_from_combined --split-json $S/test_cross-collection_combined.json \
    --out-dir $O/cross-collection_test --gallery-token brc1 --query-token brc2 --mated-only
python -m caliper_eval.split_from_combined --split-json $S/test_ground-aerial_combined.json \
    --out-dir $O/ground-aerial_test --gallery-token ground --query-token aerial
python -m caliper_eval.split_from_combined --split-json $S/test_enroll-field_combined.json \
    --out-dir $O/enroll-field_test --gallery-token enroll --query-token field
python -m caliper_eval.split_from_combined --split-json $S/test_clothing_combined_brc1.json \
    --out-dir $O/clothing-brc1_test --gallery-token set1 --query-token set2
python -m caliper_eval.split_from_combined --split-json $S/test_clothing_combined_brc2.json \
    --out-dir $O/clothing-brc2_test --gallery-token set1 --query-token set2
```

All of these use the `combined_brc1_brc2_test` partition: 170 test identities, none of them used for
training.

| split | column in RESULTS.md | gallery → query |
|---|---|---|
| `paperGR` | Ground-Range 500m / 1000m | real 380m → real 500m, 1000m |
| `ground-aerial_test` | Ground-vs-Elevated | ground cameras → aerial cameras |
| `enroll-field_test` | Enroll-vs-Field | enrollment stills + gait → field video |
| `clothing-brc1_test`, `clothing-brc2_test` | Same-collect-CC | set 1 → set 2 within one collection (different clothing) |
| `cross-collection_test` | Longitudinal | BRC1 field → BRC2 field |

**Longitudinal is open-set.** Only 39 of its 96 query identities appear in the gallery. mAP and Rank-k
are defined only for queries that have a match, so they are averaged over those 39 identities (1,198
query videos). The 57 unmatched query identities are excluded; `--mated-only` drops them when the
split is built. All 113 gallery identities, including those that never appear as queries, are kept as
distractors.

**2. Extract features** with each method's script, from inside its directory:

```bash
./extract_features.sh                                  # paperGR (default)
SPLIT=cross-collection_test ./extract_features.sh      # any other split above
```

These wrap `extract_caliper_features.py` (BoT: `tools/extract_features.py`), which takes
`--split` / `--dataset` / `--weight` / `--source` directly.

**3. Score** each feature set:

```bash
python -m caliper_eval.eval_from_features \
    --features-dir $REID_BENCH_ROOT/eval/combined_brc1_brc2/<split>/<method>/<source> \
    --pooling video --metric cosine --max-per-video 0
```

`python -m caliper_eval.results_table` then collects every eval result into one Markdown table.

## `caliper_eval` modules

**Evaluation** (numpy only):

| module | role |
|---|---|
| `paths.py` | benchmark paths under `REID_BENCH_ROOT` |
| `features_io.py` | the `.npz` feature-cache format every extractor writes |
| `eval_from_features.py` | mAP / CMC over a cached feature set, identical for every method |
| `eval_reid.py` | Market1501 metric (vendored from reid-strong-baseline) |
| `split_from_combined.py` | video-level split JSON → crop-level kwcoco |
| `stage_chips.py` | copies crops to a fast local filesystem (e.g. `/dev/shm`) for training |
| `results_table.py` | collects eval results into a Markdown table |

**Training** (adds torch; see [TRAINING.md](TRAINING.md)):

| module | role |
|---|---|
| `train_corpus.py` | `GroundFieldCorpus`, the labelled ground-field training items |
| `train_sampler.py` | P×K identity samplers, including the synthetic-mix variants |
| `train_entry.py` | shared CLI flags, checkpoint loading and corpus-composition checks for each `train_caliper.py` |
| `train_metrics.py` | per-range validation metrics, written to `val_metrics.jsonl` |
| `train_metadata.py` | records training settings, identity splits, and checkpoint iterations |

## License

Apache License 2.0 (see [LICENSE](LICENSE)) for the code in this repository (`caliper_eval/` and the
docs). Each fork under `methods/` keeps its own upstream license. Check that subdirectory before
reusing its code on its own.
