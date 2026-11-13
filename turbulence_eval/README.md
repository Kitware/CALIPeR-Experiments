# CALIPeR simulation

Generate synthetic long-range imagery with **pyBSM, DAATSim and P2S**, using
camera profiles for [Collect-1](collect1_cameras.py) and [Collect-2](collect2_cameras.py).
Use these workflows for the experiments in
*CALIPeR: Co-Temporal Atmospheric Long-range Imagery for Person Re-identification*:

- **Generate synthetic data:** simulate imagery at a target distance.
- **Table 3:** compare the image quality of simulated and real camera frames.
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

Simulation and Figure 6 require Python 3.10 or newer. The root distribution
installs `turbulence_eval` alongside `caliper_eval`; generation dependencies are
installed explicitly by the command above. Use separate method environments
for feature extraction. Core evaluation remains NumPy-only.

CPU execution is supported; CUDA is optional. Before generating data, edit the
selected INI's dataset, model-asset and output paths for your machine. Paths
inside INIs resolve relative to the INI file. Use a new output directory for each
run. Image datasets, P2S model assets and Re-ID checkpoints are supplied separately;
see [setup and inputs](WORKFLOWS.md#environment-and-inputs).

## Table 3

The paper compares whole camera frames. Prepare the [full-frame inputs](WORKFLOWS.md#prepare-full-frame-inputs-outside-this-repository)
as JPEG or PNG images in subject and camera folders. Use the same config for generation and analysis.

Run these commands for Collect 1 from the repository root.

```bash
CONFIG=turbulence_eval/configs/frames_table3_brc1.ini
python -m turbulence_eval.generate_synthetic_data --config "$CONFIG" --preview
python -m turbulence_eval.generate_synthetic_data --config "$CONFIG"
python -m turbulence_eval.create_table3 --config "$CONFIG"
```

The preview shows which frames were selected. Generation writes the simulated frames.
Analysis writes `table3.md`, `table3.csv` and plots to the report directory specified in the config.

For Collect 2, set `CONFIG=turbulence_eval/configs/frames_table3_brc2.ini` and run the same three commands.
Its source is the calibrated color camera (`250m-c`), and analysis uses grayscale.

### Measure image quality on person crops

To study cropped people, use [the Collect 1 chip config](configs/chips_table3_brc1.ini)
or [the Collect 2 chip config](configs/chips_table3_brc2.ini) with the same commands.
These configs read indexed v4 crops. Cropping changes the measured area and can change the simulator rankings.
Collect 2 chips come from the monochrome station and use proxy strengths. DAATSim outputs are measured
at their native square resolution. See the [Table 3 workflow](WORKFLOWS.md#table-3-workflow-generate-data-then-analyze-it)
for settings and repeat analyses.

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
