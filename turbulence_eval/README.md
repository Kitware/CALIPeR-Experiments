# CALIPeR simulation

Generate synthetic long-range imagery with **pyBSM, DAATSim and P2S**, using
camera profiles for [Collect-1](collect1_cameras.py) and [Collect-2](collect2_cameras.py).
Use these workflows for the experiments in
*CALIPeR: Co-Temporal Atmospheric Long-range Imagery for Person Re-identification*:

- **Generate synthetic data:** simulate imagery at a target distance.
- **Table 3:** compare simulated and real imagery using image-quality metrics,
  with either person chips or full frames.
- **Figure 6:** evaluate Re-ID retrieval on real and simulated person chips
  using the existing model extractors and checkpoints.

## Install

Run from the repository root with Python 3.10:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r turbulence_eval/requirements.txt -e .
python -m pip check
```

CPU execution is supported; CUDA is optional. Before generating data, edit the
selected INI's dataset, model-asset and output paths for your machine. Paths
inside INIs resolve relative to the INI file. Use a new output directory for each
run. Image datasets, P2S model assets and Re-ID checkpoints are supplied separately;
see [setup and inputs](WORKFLOWS.md#environment-and-inputs).

## Table 3

Choose one base config and use it for both generation and analysis:

| Collection | Person chips | Full frames |
|---|---|---|
| Collect 1 | [chips_table3_brc1.ini](configs/chips_table3_brc1.ini) | [frames_table3_brc1.ini](configs/frames_table3_brc1.ini) |
| Collect 2 | [chips_table3_brc2.ini](configs/chips_table3_brc2.ini) | [frames_table3_brc2.ini](configs/frames_table3_brc2.ini) |

For example:

```bash
python -m turbulence_eval.generate_synthetic_data --config turbulence_eval/configs/chips_table3_brc1.ini --preview
python -m turbulence_eval.generate_synthetic_data --config turbulence_eval/configs/chips_table3_brc1.ini
python -m turbulence_eval.create_table3 --config turbulence_eval/configs/chips_table3_brc1.ini
```

The preview checks the selection without generating images. Generation saves the
synthetic dataset. Analysis writes `table3.md`, `table3.csv` and quality plots to
the configured report directory.

Chip configs read the existing indexed v4 person chips. Full-frame configs need
**pre-extracted JPEG/PNG images** in subject/camera folders. MP4-to-frame conversion
is an external preparation step and is not included in this repository.

Collect 2 full frames use the calibrated color source (`250m-c`) with grayscale
analysis. Indexed Collect 2 chips use the separate ground monochrome station
and retain proxy strengths. DAATSim outputs stay at their native square
resolution for metrics. See the [Table 3 workflow](WORKFLOWS.md#table-3-workflow-generate-data-then-analyze-it)
for input layouts, settings and repeated analysis runs.

## Figure 6

Figure 6 compares Re-ID retrieval on real and simulated person chips using a
fixed real 400m gallery. Choose a generation config:

| Collection | Person chips |
|---|---|
| Collect 1 | [chips_fig6_brc1.ini](configs/chips_fig6_brc1.ini) |
| Collect 2 | [chips_fig6_brc2.ini](configs/chips_fig6_brc2.ini) |
| Both collections | [chips_fig6_combined.ini](configs/chips_fig6_combined.ini) |

For example, generate the combined-collection chips:

```bash
python -m turbulence_eval.generate_synthetic_data --config turbulence_eval/configs/chips_fig6_combined.ini --preview
python -m turbulence_eval.generate_synthetic_data --config turbulence_eval/configs/chips_fig6_combined.ini
```

Use a model's extractor in `methods/` on the generated `eval/` directory, then
score its saved features with the shared `caliper_eval` evaluator and export the
figure (replace the example paths):

```bash
python -m caliper_eval.eval_from_features --features-dir /path/to/features --pooling video --fig6-generation-dir /path/to/chips_fig6_combined
python -m turbulence_eval.export_fig6 --eval-results bot-market1501=/path/to/features/eval_mean_cosine.json --scopes combined --output-dir /path/to/new-fig6-report
```

Generation degrades 500m chips to 1000m at three turbulence strengths. Evaluation
compares their retrieval curves with real queries; export saves PDF/PNG/SVG
figures and summary tables. See the
[Figure 6 workflow](WORKFLOWS.md#figure-6-workflow-prepare-evaluate-then-report)
for setup, feature-extraction commands, additional models, collection scopes,
and saved-result exports.

## Detailed guide

[WORKFLOWS.md](WORKFLOWS.md) covers setup and offline assets, input preparation,
generation and analysis stages, configuration options, output manifests,
model adapters, metric definitions, standalone APIs and troubleshooting.
