# CALIPeR-Experiments

Official benchmarking code for the WACV 2027 paper **"CALIPeR: Co-Temporal Atmospheric Long-range
Imagery for Person Re-identification."**

## Layout

- `caliper_eval/` reads splits and features, scores retrieval, and supplies the training corpus and samplers.
- `turbulence_eval/` generates simulated images and compares them with real images for [Table 3 and Figure 6](turbulence_eval/README.md).
- `methods/` contains the seven method implementations. BoT, SINet, CLIP-ReID, and PersonViT support fine-tuning. CAL, X-TFCLIP, and SapiensID support zero-shot evaluation. See the [upstream revisions](methods/UPSTREAM.md).

## Results

See the [published results](RESULTS.md) for zero-shot and fine-tuned retrieval scores. The
[training guide](TRAINING.md) gives the training settings and checkpoint epoch for each fine-tuned model.

## Install

```bash
python3 -m pip install -e .           # evaluation only (numpy)
python3 -m pip install -e '.[train]'  # also the training modules (torch)
```

Run these commands from the repository root in the Python environment used for your chosen method.
Each method also needs its own upstream dependencies; see its README for dependency and checkpoint
information. The upstream dataset and training examples describe those original projects; use the
CALIPeR commands below and in [TRAINING.md](TRAINING.md) for this benchmark.

Upstream examples use portable data, checkpoint and output paths. See
[the method path conventions](methods/PORTABILITY.md) before running those workflows.

## Data

| variable | default | what it points to |
|---|---|---|
| `CALIPER_DATA_ROOT` | `./data` | the CALIPeR release: `splits/` and `reid-chips/` |
| `REID_BENCH_ROOT` | `./reid-bench` | where this code writes crop-level splits, features and eval results |
| `CALIPER_EVAL_HOME` | the repository root | where method scripts find `caliper_eval` if it isn't installed |

Defaults are relative to the current directory unless stated otherwise. Before running any commands,
replace the example paths below and export **absolute paths** in the same shell. The method wrappers
change into their own directories, so relative roots will point somewhere different during extraction.

```bash
export CALIPER_DATA_ROOT="/absolute/path/to/caliper-data"
export REID_BENCH_ROOT="/absolute/path/to/caliper-benchmark-output"
```

Obtain the dataset and experiment manifests separately. Point `CALIPER_DATA_ROOT` at the
files listed below. Split conversion uses the existing person crops and chip index.

| input under `CALIPER_DATA_ROOT` | needed for |
|---|---|
| `splits/combined_brc1_brc2_test/*.json` | video-level evaluation splits listed below |
| `reid-chips/chip_index.jsonl` and `reid-chips/v4/` | mapping split videos to extracted crops |
| `train_test_ids.json` | results reporting; a JSON object with a nonempty `train` array of canonical subject strings |
| `splits/combined_brc1_brc2_train_corpus/train_manifest.csv` | fine-tuning; see [TRAINING.md](TRAINING.md) for the full corpus layout |
| `splits/combined_brc1_brc2_sim_train_corpus/sim_train_manifest.csv` and `reid-chips-sim/` | synthetic-data fine-tuning only |

`REID_BENCH_ROOT` is the output tree:

```
splits/<split>/{gallery,query_*}.kwcoco.json        crop-level splits
eval/<dataset>/<split>/<method>/<source>/           feature .npz files and eval .json results
```

## Reproducing the evaluation

**1. Build the crop-level splits** from the video-level split JSONs (CPU only). Run each command once;
building a split's gallery and queries in separate invocations renumbers subjects and breaks
matching.

```bash
S="$CALIPER_DATA_ROOT/splits/combined_brc1_brc2_test"
O="$REID_BENCH_ROOT/splits"
python3 -m caliper_eval.split_from_combined --split-json "$S/test_ground-range_combined_paperGR.json" \
    --out-dir "$O/paperGR" --gallery-token 380m --query-token-per-tok
python3 -m caliper_eval.split_from_combined --split-json "$S/test_cross-collection_combined.json" \
    --out-dir "$O/cross-collection_test" --gallery-token brc1 --query-token brc2 --mated-only
python3 -m caliper_eval.split_from_combined --split-json "$S/test_ground-aerial_combined.json" \
    --out-dir "$O/ground-aerial_test" --gallery-token ground --query-token aerial
python3 -m caliper_eval.split_from_combined --split-json "$S/test_enroll-field_combined.json" \
    --out-dir "$O/enroll-field_test" --gallery-token enroll --query-token field
python3 -m caliper_eval.split_from_combined --split-json "$S/test_clothing_combined_brc1.json" \
    --out-dir "$O/clothing-brc1_test" --gallery-token set1 --query-token set2
python3 -m caliper_eval.split_from_combined --split-json "$S/test_clothing_combined_brc2.json" \
    --out-dir "$O/clothing-brc2_test" --gallery-token set1 --query-token set2
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
are defined only for queries that have a match. With video pooling, they are averaged over the 1,198
valid query videos belonging to those 39 identities; this is not an average of per-identity scores.
The 57 unmatched query identities are excluded; `--mated-only` drops them when the
split is built. All 113 gallery identities, including those that never appear as queries, are kept as
distractors.

**2. Extract features and score** with each method's wrapper, from the working directory listed below.
Download the upstream checkpoints and place them at the paths configured near the top of that
method's `extract_features.sh`. The experiment checkpoints expected by these wrappers are not bundled.
Each wrapper also runs the shared evaluation after extraction.

```bash
./extract_features.sh                                  # paperGR (default)
SPLIT=cross-collection_test ./extract_features.sh      # any other split above
```

| method | working directory | Python extractor | checkpoint argument for direct Python invocation |
|---|---|---|---|
| BoT | `methods/reid-strong-baseline` | `tools/extract_features.py` | `TEST.WEIGHT` config override |
| SINet | `methods/SINet` | `extract_caliper_features.py` | `--weight` |
| CLIP-ReID | `methods/CLIP-ReID` | `extract_caliper_features.py` | `--weight` |
| PersonViT | `methods/PersonViT/transreid_pytorch` | `extract_caliper_features.py` | `--weight` |
| CAL | `methods/GEFF` | `extract_caliper_features.py` | `--weight` |
| X-TFCLIP | `methods/X-TFCLIP` | `extract_caliper_features.py` | `--weight` |
| SapiensID | `methods/sapiensid/tasks/sapiensID` | `extract_caliper_features.py` | `--ckpt-dir` (contains `model.yaml` and `model.pth`) |

All extractors accept `--split`, `--dataset`, and `--source`; use each extractor's `--help` for
method-specific options. These Python arguments are not forwarded by the shell wrappers.

**3. Score or rescore an existing feature set** directly, replacing the placeholders:

```bash
python3 -m caliper_eval.eval_from_features \
    --features-dir "$REID_BENCH_ROOT/eval/combined_brc1_brc2/<split>/<method>/<source>" \
    --pooling video --metric cosine --max-per-video 0
```

The installed `caliper-eval` command exposes the same evaluator. To collect the published splits
into a Markdown table, run:

```bash
python3 -m caliper_eval.results_table
```

The report is written to `$REID_BENCH_ROOT/eval/RESULTS.md`. It reads
`$CALIPER_DATA_ROOT/train_test_ids.json` to check for training identities in evaluation splits. If
that JSON is elsewhere, pass `--train-test-ids "/absolute/path/to/train_test_ids.json"`. The `train`
list must use the exact canonical subject names from the crop-level splits, with no leading or
trailing whitespace. Additional keys in the partition file are ignored.

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

Repository-authored shared code (`caliper_eval/`) and documentation are distributed under Apache
License 2.0 (see [LICENSE](LICENSE)). Third-party code under `methods/` retains its included license
notices; see [methods/UPSTREAM.md](methods/UPSTREAM.md) for source repositories. The GEFF and X-TFCLIP
top-level directories currently contain no license file. Consult the applicable upstream terms
before reusing third-party code.
