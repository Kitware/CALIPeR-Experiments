# CALIPeR simulation

Generate synthetic long-range imagery with **pyBSM, DAATSim and P2S**, using
camera profiles for [Collect-1](collect1_cameras.py) and [Collect-2](collect2_cameras.py).
Use these workflows for the experiments in
*CALIPeR: Co-Temporal Atmospheric Long-range Imagery for Person Re-identification*:

- **Generate synthetic data:** simulate imagery at a target distance.
- **Table 3:** compare simulated and real imagery using image-quality metrics,
  with either person chips or full frames.
- **Figure 6:** evaluate Re-ID retrieval on real and simulated person chips
  using your model adapters and checkpoints.

## Install

Run from this repository directory with Python 3.10:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -e .
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
python -m caliper_sim.generate_synthetic_data --config configs/chips_table3_brc1.ini --preview
python -m caliper_sim.generate_synthetic_data --config configs/chips_table3_brc1.ini
python -m caliper_sim.create_table3 --config configs/chips_table3_brc1.ini
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

Figure 6 uses **person chips only**, with separate generation and evaluation
configs for each collection:

| Collection | Generation | Evaluation |
|---|---|---|
| Collect 1 | [chips_fig6_brc1.ini](configs/chips_fig6_brc1.ini) | [fig6_brc1.ini](configs/fig6_brc1.ini) |
| Collect 2 | [chips_fig6_brc2.ini](configs/chips_fig6_brc2.ini) | [fig6_brc2.ini](configs/fig6_brc2.ini) |

Set your model adapter and checkpoint paths in the evaluation config, then run:

```bash
python -m caliper_sim.generate_synthetic_data --config configs/chips_fig6_brc1.ini
python -m caliper_sim.reid preview --config configs/fig6_brc1.ini
python -m caliper_sim.reid run --config configs/fig6_brc1.ini
```

Replace `brc1` with `brc2` for Collect 2. Generation simulates 500m queries at
three turbulence strengths; evaluation compares them with real queries using a
fixed 400m gallery. It saves retrieval scores, curves and summary tables.
This workflow evaluates existing models. See the
[Figure 6 workflow](WORKFLOWS.md#figure-6-workflow-prepare-evaluate-then-report)
for individual stages, combined-collection runs and the model adapter contract.

## Detailed guide

[WORKFLOWS.md](WORKFLOWS.md) covers setup and offline assets, input preparation,
generation and analysis stages, configuration options, output manifests,
model adapters, metric definitions, standalone APIs and troubleshooting.
