# caliper_reid_eval

Code accompanying our CALIPeR person re-identification (ReID) publication. Shared, method-agnostic
components for the CALIPeR ReID benchmark: every method repo puts this package on `PYTHONPATH` so the
parts that must be identical across methods — the evaluation metric, the feature-cache format, the
training corpus/sampler — are written once.

## Results

Generate a report from evaluation outputs with `python -m caliper_eval.results_table`.

## Two halves

**Evaluation (torch-free, numpy only)** — the original purpose.

| module | role |
|---|---|
| `paths.py` | single source of truth for benchmark paths; override the tree with `REID_BENCH_ROOT` |
| `features_io.py` | the `.npz` feature-cache contract every extractor writes |
| `eval_from_features.py` | mAP / CMC over a cached feature set — run identically for every method |
| `eval_reid.py` | vendored Market1501 metric |
| `split_from_combined.py` | video-level split JSON → crop-level kwcoco |

**Training (adds a torch dependency for the sampler only)** — see **[CALIPER_TRAINING.md](CALIPER_TRAINING.md)**.

| module | role |
|---|---|
| `train_corpus.py` | `GroundFieldCorpus` — the labelled ground-field training items (pure stdlib) |
| `train_sampler.py` | `ResamplingIdentitySampler` — P×K with per-epoch frame re-draw |

## Start here

- Finetuning a method on CALIPeR → **[CALIPER_TRAINING.md](CALIPER_TRAINING.md)**
- Scoring features → `python -m caliper_eval.eval_from_features --features-dir DIR --pooling video --metric cosine --max-per-video 0`

## Install

```bash
pip install -e .          # or just: export PYTHONPATH=/path/to/caliper_reid_eval:$PYTHONPATH
```

## Data

Reads `$CALIPER_DATA_ROOT/{splits,reid-chips}/` (default: `./data` relative to the current directory).
Override the dataset tree with `CALIPER_DATA_ROOT`, the separate eval/training-output tree with
`REID_BENCH_ROOT`, this package's own location (for the import fallback in method repos) with
`CALIPER_EVAL_HOME`, and the training chip location (e.g. a `/dev/shm` stage) with `--chip-root`.
