# caliper_reid_eval

Shared, method-agnostic components for the CALIPeR ReID benchmark. Every method repo puts this package
on `PYTHONPATH` so the parts that must be identical across methods are written once.

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

Reads `./data/{splits,reid-chips}/`. Override with `REID_BENCH_ROOT` (benchmark tree),
`CALIPER_EVAL_HOME` (this package's location, for the import fallback in method repos), and
`--chip-root` (training chip location, e.g. a `/dev/shm` stage).
