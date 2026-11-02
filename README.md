# CALIPeR-Experiments

Official benchmarking code for the WACV 2027 paper **"CALIPeR: Co-Temporal Atmospheric Long-range
Imagery for Person Re-identification."** Code accompanying our CALIPeR person re-identification (ReID)
publication.

## Layout

- **`caliper_eval/`** — shared, method-agnostic components for the CALIPeR ReID benchmark. Every method
  under `methods/` puts this package on `PYTHONPATH` so the parts that must be identical across
  methods — the evaluation metric, the feature-cache format, the training corpus/sampler — are
  written once.
- **`methods/`** — one subdirectory per method, each a fork of its upstream repo with CALIPeR
  integration added on top (full commit history preserved, including each fork's own README/LICENSE):
  `reid-strong-baseline` (BoT), `SINet`, `CLIP-ReID`, `PersonViT` (fine-tuned + zero-shot),
  `X-TFCLIP`, `GEFF` (CAL), `sapiensid` (zero-shot only).

## Results

See **[RESULTS.md](RESULTS.md)** for the published numbers (zero-shot survey, and the fine-tuning +
synthetic-augmentation headline table).

## caliper_eval modules

**Evaluation (torch-free, numpy only)** — the original purpose.

| module | role |
|---|---|
| `paths.py` | single source of truth for benchmark paths; override the tree with `REID_BENCH_ROOT` |
| `features_io.py` | the `.npz` feature-cache contract every extractor writes |
| `eval_from_features.py` | mAP / CMC over a cached feature set — run identically for every method |
| `eval_reid.py` | vendored Market1501 metric |
| `split_from_combined.py` | video-level split JSON → crop-level kwcoco |
| `stage_chips.py` | stage training/eval chips onto a faster local disk (e.g. `/dev/shm`) |
| `results_table.py` | regenerates the published results table from cached eval output — see [RESULTS.md](RESULTS.md) |

**Training (adds a torch dependency for the sampler only)** — see **[CALIPER_TRAINING.md](CALIPER_TRAINING.md)**.

| module | role |
|---|---|
| `train_corpus.py` | `GroundFieldCorpus` — the labelled ground-field training items (pure stdlib) |
| `train_sampler.py` | `ResamplingIdentitySampler` / `MixingIdentitySampler` / `MixingTrackletSampler` — P×K sampling, with the variants that support the synthetic-augmentation study |
| `train_entry.py` | shared CLI flags, checkpoint loading, and the corpus-composition assertions every method's `train_caliper.py` entry point uses |
| `train_metrics.py` | per-range validation metrics, written to `val_metrics.jsonl` during training |
| `train_metadata.py` | writes `run.json` next to each checkpoint (git SHA, config, explicit held-out id list) |

## Start here

- Finetuning a method on CALIPeR → **[CALIPER_TRAINING.md](CALIPER_TRAINING.md)**
- Scoring features → `python -m caliper_eval.eval_from_features --features-dir DIR --pooling video --metric cosine --max-per-video 0`
- Regenerating the published results table → `python -m caliper_eval.results_table`

## Install

```bash
pip install -e .          # or just: export PYTHONPATH=/path/to/caliper_eval:$PYTHONPATH
```

## Data

Reads `$CALIPER_DATA_ROOT/{splits,reid-chips}/` (default: `./data` relative to the current directory).
Override the dataset tree with `CALIPER_DATA_ROOT`, the separate eval/training-output tree with
`REID_BENCH_ROOT`, this package's own location (for the import fallback in method repos) with
`CALIPER_EVAL_HOME`, and the training chip location (e.g. a `/dev/shm` stage) with `--chip-root`.

## License

Apache License 2.0 (see [LICENSE](LICENSE)) for the code in this repo (`caliper_eval/` and the docs).
Each fork under `methods/` keeps its own upstream license — check that subdirectory before reusing
its code independently.
