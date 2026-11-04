# Workflows and reference

Detailed setup, generation, analysis and integration guidance for CALIPeR.
Start with the [README](README.md) for the package overview and base configs.

Commands in this guide run from the `caliper_sim` repository directory after
installation. Paths inside INIs resolve relative to the INI file; command-line
path overrides resolve from the current directory. Generation reads the source
dataset and writes to a separate output directory. Analysis reads completed
generation outputs. Use a new output directory for each generation or report.

- [Environment and inputs](#environment-and-inputs)
- [Requirements by workflow](#requirements-by-workflow)
- [Table 3: generate, then analyze](#table-3-workflow-generate-data-then-analyze-it)
- [Current Collect 2 full-frame settings](#current-brc2-whole-frame-table-3)
- [Figure 6: generate, prepare, evaluate, report](#figure-6-workflow-prepare-evaluate-then-report)
- [Model adapter contract](#model-adapter-contract)
- [Other chip generation and configuration](#generate-from-the-current-chip-dataset-default)
- [Generated outputs and loader integration](#generated-chip-outputs-and-loader-integration)
- [Simulation models](#simulation-models)
- [Image-quality metrics and summaries](#image-quality-metrics)
- [Standalone pyBSM generation](#standalone-pybsm-generation)
- [Standalone quality analysis](#standalone-quality-analysis)
- [Troubleshooting](#troubleshooting)

## Environment and inputs

This repository installs independently. Use Python 3.10 and run setup from its
root directory:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -e .
python -m pip check
python -m caliper_sim.generate_synthetic_data --help
python -m caliper_sim.create_table3 --help
```

[`requirements.txt`](requirements.txt) installs the dependencies for image
generation and Table 3 evaluation, including public pyBSM and NRTK for radial
distortion. `-e .` installs this package so `python -m caliper_sim` works from
inside the repository or another directory. OpenCV is pinned to 4.11 to support
NumPy 1.26. CPU execution is supported; CUDA acceleration is optional.
For a specific CPU/CUDA Torch build, install the
matching Torch 2.12.0 and Torchvision 0.27.0 wheels before installing the requirements.

No separate DAATSim, P2S or modified pyBSM checkout is needed. DAATSim and P2S
kernels are included in `_vendor/`; pyBSM 0.15.0 is installed from PyPI.
The simulator repositories are
[P2S / TurbulenceSimulatorPython](https://github.com/Riponcs/TurbulenceSimulatorPython)
and [DAATSim](https://github.com/Riponcs/DAATSim).
[`adapter.py`](adapter.py) provides inverse radiometric mapping, midpoint
reflectance for constant frames, and rejection of non-finite noise inputs within
its simulator subclass. It uses internal pyBSM interfaces without modifying the
installed pyBSM classes; validate images and metrics when updating pyBSM.

The public pyBSM 0.15.0 wheel includes the MODTRAN atmospheric lookup database
(`fileDecoder.csv` and numbered `.bin` files in its installed `pybsm/atms/`
directory). No separate atmospheric download or MODTRAN executable is needed
with that wheel. Source-based installations must also supply those lookup files.
The 260m and 380m camera models interpolate between tabulated ranges.

### Requirements by workflow

[`requirements.txt`](requirements.txt) is the pinned dependency list for the
standalone package; `pyproject.toml` reads that file directly. The validated
environment uses Python 3.10. Package metadata also allows Python 3.11, which
is not part of the documented Python 3.10 setup. Model-specific packages belong to the
chosen Re-ID adapter's environment and are not bundled with the simulator.

| Workflow | Required inputs and assets beyond the Python packages |
|---|---|
| Table 3 on chips | Existing v4 chip tree, `chip_index.jsonl`, canonical train/test identity partitions, and P2S assets for the three-simulator presets. |
| Table 3 on full frames | Pre-extracted subject/camera JPEG/PNG tree and P2S assets. Collect 2's calibrated preset expects the frozen cleaned color-source pool. |
| Figure 6 generation | V4 chip tree/index, identity partitions, and a test split with a real 400m gallery and 500m/1000m queries. The presets use pyBSM only, so P2S assets are unnecessary. |
| Figure 6 evaluation | Completed Figure 6 generation, the existing method extractor and its dependencies, checkpoints, and any auxiliary model/aligner assets. |
| Figure 6 export from saved scores | Shared evaluator JSONs scored with `--fig6-generation-dir`, or original `results.json` archives; no image data or checkpoints are needed for export. |
| Table 3 quality analysis | Completed generation with its real inputs still accessible. BRISQUE weights are required unless `--skip-brisque` is selected. P2S assets are needed only during generation. |

P2S assets, the first-run BRISQUE cache and full-frame preparation are described
in the following sections. Datasets, model weights, MP4 conversion and calibration
sweeps are not included in the source distribution. No neighboring simulator or
Re-ID checkout is required for the standalone core.

Use this repository's virtual environment for its commands. If working from a
parent workspace, `caliper_sim/.venv/bin/python` selects it explicitly; a separate
workspace-level `.venv` may contain different dependency versions. For the
commands in this guide, first change into the `caliper_sim` directory and
activate its `.venv` as shown above.

### BRISQUE on the first run

BRISQUE is enabled by default and provided by **PIQ**, which is installed by
`requirements.txt`. The first quality-analysis run automatically downloads its
small SVM weights file (about 112 KiB) from PIQ's GitHub release. Torch prints a
`Downloading: .../brisque_svm_weights.pt` message while this happens. No manual
model download or additional configuration is needed.

The first download needs internet access. Subsequent runs reuse
`~/.cache/torch/hub/checkpoints/brisque_svm_weights.pt` and can run offline.
`TORCH_HOME` selects a different Torch cache location when needed. Image
generation does not download or use BRISQUE weights.

To download and check the weights before starting an analysis, optionally run:

```bash
python -c "import piq, torch; piq.brisque(torch.rand(1, 3, 256, 256)); print('BRISQUE ready')"
```

For an offline machine, populate the cache beforehand or copy the cached weights
file to its Torch Hub checkpoints directory. To intentionally omit this metric,
pass `--skip-brisque` to the analysis command; the report will contain five metric
rows instead of six.

### Real-image and P2S inputs

Whole-frame Table 3 uses the subject/camera layout below;
indexed chips use the [chip input layout](#generate-from-the-current-chip-dataset-default):

```text
stable_frames/
  <subject>/
    260m/*.jpg
    500m/*.jpg
    1000m/*.jpg
```

JPEG and PNG inputs are supported. P2S additionally requires `P2S_model.pt`,
`dictionary.npy` and `R-corr_-0.1.npy` in a separate model-asset directory.
Get the P2S assets from the
[TurbulenceSimulatorPython repository](https://github.com/Riponcs/TurbulenceSimulatorPython)
and its [data directory](https://github.com/Riponcs/TurbulenceSimulatorPython/tree/master/data).
Place all three files together and set `[paths] p2s_data_dir` in your generation
configuration to that directory. The base Table 3 configs use
`../data/p2s_simulator_data`, relative to the INI file. These assets are not
installed by pip. Set the real-image paths separately in the configuration.

### Prepare full-frame inputs outside this repository

Full-frame generation reads **JPEG/PNG images**, not MP4 files. Decode/sample the
videos with your own preparation pipeline, assign the intended subject identity
from the collection metadata, and place the images in this layout:

```text
<stable_frames_dir>/
  <subject>/
    <source-camera>/
      <clip-stem>_frame00000001.jpg
      <clip-stem>_frame00000002.jpg
    500m/
      <target-clip>_frame00000001.jpg
    1000m/
      <target-clip>_frame00000001.jpg
```

Set `[paths] stable_frames_dir` to the prepared root. The source folder must match
`[experiment] source_camera` exactly: `260m` for the Collect 1 full-frame base
config, or `250m-c` for Collect 2. Both use `500m` and `1000m` target folders.
JPEG (`.jpg`, `.jpeg`) and PNG files directly inside these folders are accepted.
MP4 conversion code, video downloads and annotation conversion are not included
in this package; generation never invokes them or requires another checkout.

Keep the images at their native sampling and use collision-free names across
clips. Naming them `<clip-stem>_frame<index>` also lets Collect 2 reset DAATSim's
temporal state at clip boundaries. Extracting a whole frame does not itself
identify a person: assign subject folders using verified dataset identities,
and exclude missing-person, corrupt and test-pattern frames upstream.

For the current Collect 2 result, supply the **frozen cleaned color-source pool**
described in [Current BRC2 whole-frame Table 3](#current-brc2-whole-frame-table-3).
The reported cohort uses 20 selected source frames per subject
and excludes 55 invalid 500m targets and 40 invalid 1000m targets. A fresh extraction
or a larger source pool can change both the sampled cohort and the results. The
INI does not detect or remove error patterns automatically. Do not mix the
monochrome `250m-m` station into the color `250m-c` source folder.

Run the normal generation and Table 3 commands after preparation:

```bash
python -m caliper_sim.generate_synthetic_data --config configs/frames_table3_brc1.ini --preview
python -m caliper_sim.generate_synthetic_data --config configs/frames_table3_brc1.ini
python -m caliper_sim.create_table3 --config configs/frames_table3_brc1.ini
```

Replace `brc1` with `brc2` for Collect 2. Each target cohort requires the source
folder and that target's folder; a subject can be eligible for only one distance.
The frame budget is pooled across clips per subject, rather than per video.

## Table 3 workflow: generate data, then analyze it

Choose one of the four base configs in the [README](README.md#table-3).
Use the same INI for generation and analysis. The examples and directory
layout below describe full-frame generation; indexed-chip Table 3 runs use
the [chip output manifests](#generated-chip-outputs-and-loader-integration).

Table 3 measures how closely simulated long-range imagery resembles real imagery
at the target range. For each subject, the source frames (260m by default) are
transformed to each target range (500m and 1000m) using pyBSM, DAATSim and P2S.
Generation samples up to **20 source frames per subject** with sampling seed
**42**. Quality analysis independently samples images per subject and group,
using analysis seed **0**: the chip base configs and generic `table3.ini` use
five, while both full-frame base configs use 20. Analysis uses the real-image pools and
the generated synthetic images. The analysis compares each simulator's
image-quality statistics with those of real frames at that target range.
It reports six differences for each of the six
simulator/range combinations; smaller differences indicate closer agreement.
The images do not need pixel registration. Source-image statistics are also
recorded for inspection, while the final table compares simulated and real target
images. This workflow does not train or evaluate a person re-identification model.

Run the commands from this repository directory in the installed environment.
The generic `configs/table3.ini` illustrates the full-frame schema; replace it
with `configs/frames_table3_brc1.ini` or `configs/frames_table3_brc2.ini` for the
collection-specific presets.

### 1. Generate synthetic data

Edit the paths and experiment settings in [`configs/table3.ini`](configs/table3.ini):

`[paths]` separates the large generated dataset from the smaller analysis reports.
For example, place the images and simulator caches on a large data drive and keep
the results locally:

```ini
output_data_path = /mnt/large-drive/caliper/synthetic_table3
output_dir = ../table3_runs/table3
```

Absolute paths can point to any mounted drive. Relative paths resolve against the
INI file's directory for both stages.

```bash
python -m caliper_sim.generate_synthetic_data \
  --config configs/table3.ini
```

This stage selects the cohort and frames, saves links to the real inputs, and
generates all three simulators' images at both target ranges. Source filenames
are preserved. The command finishes after saving the images and generation
metadata; it does not compute quality metrics, plots or Table 3. It requires the
P2S model assets but does not require BRISQUE weights.

Relative paths resolve against the INI file's directory. The sections specify:

- `[paths]`: real images, P2S assets, `output_data_path` for generated images and
  caches, and `output_dir` for analysis tables, CSVs, plots and report metadata.
- `[experiment]`: source and target cameras, subject selection, generation frame budget,
  seeds and device (`auto`, `cpu`, or a Torch CUDA device).
- `[target:500m]` and `[target:1000m]`: simulator turbulence strengths.
- `[pybsm]`, `[daatsim]`, `[p2s]`: simulation options.
- `[analysis]`: measurement frame budget (`num_images = 5`), BRISQUE, grayscale
  and resizing options saved for stage 2.

The example configuration uses source camera **260m**, pyBSM Cn2 **1e-12 at
500m** and **1e-13 at 1000m**, DAATSim r0 **0.035m**, and P2S r0 **0.10m**.
Use the collection-specific presets and match the paper’s cohort, calibration, preprocessing,
and sampling when comparing with Table 3.

Camera definitions live in [`collect1_cameras.py`](collect1_cameras.py) and
[`collect2_cameras.py`](collect2_cameras.py). Each module exports `CAMERA_CONFIGS`;
Collect 1 also exports its measured `M_PER_PIXEL` values. The shared
[`cameras.py`](cameras.py) resolves aliases and selects profiles. The **260m**
source label corresponds to the source range described as **250m** in the paper;
the additional **380m** model corresponds to the paper's **400m** description.
Effective focal lengths and subject-plane pixel scales include estimates based
on a 1.70m person height. Account for this mapping when comparing experiments.

Empty `subjects` and `num_subjects` select all eligible subjects per target.
Set `num_subjects = 2` and `num_images = 1` in a separate configuration for a
small end-to-end check. Explicit `subjects = K13553, K13554` selects those IDs;
missing required cameras raise an error. Use a new output directory for each run.

Generation sorts subjects and source filenames, then samples source frames
without replacement using NumPy's random generator. The same selected source
frames feed all simulators and both distances. Source sampling does not consume
random draws for target images. Each target comparison uses a common cohort
across all simulators.

Complete real source and target pools are linked into the run directory for
independent analysis sampling. `selection.json` records both the source frames
used for generation (`files`) and the full real-image pools (`analysis_files`).
Source images and model assets are read only. Keep the real input files available
for stage 2 because the symlinks refer to them.

```text
<output_data_path>/
  experiment.ini          # supplied configuration
  settings.json           # resolved paths and settings
  selection.json          # cohorts, generation source selection and full real-image pools
  generation.json         # completed generation and effective simulator parameters
  cache/p2s/              # computed tilt matrices
  500m/
    inputs/<subject>/260m/<source-filename>
    inputs/<subject>/500m/<target-filename>
    pybsm/<subject>/sim_260m_to_500m/<source-filename>
    daatsim/<subject>/sim_260m_to_500m/<source-filename>
    p2s/<subject>/sim_260m_to_500m/<source-filename>
  1000m/
```

`generation.json` is written only when every simulator and target has completed.
A failed run leaves partial files for inspection; choose a new output directory
for another generation attempt.

### 2. Create Table 3 from saved data

Once generation finishes, analyze that directory:

```bash
python -m caliper_sim.create_table3 \
  --config configs/table3.ini
```

Analysis samples the saved real and synthetic images, computes each metric per image, and
averages the results by subject. Keep the real images available through the links in the
generation directory. The saved cohort, sampling settings, and preprocessing defaults apply
even if you edit the INI later. Use `--config` to read the data and report paths from an INI,
or pass `--data-dir` directly. For BRISQUE, PIQ caches its weights on first use.

```text
<output_dir>/
  table3.csv              # six metric rows; mean and std columns per simulator/range
  table3.md
  run.json                # input dataset, effective settings and final values
  500m/quality/           # per-image/per-subject CSVs, plots and analysis.json
  1000m/quality/
```

You can override the generation location with `--output-data-path`. To analyze
an existing dataset without an INI, supply both paths explicitly:

```bash
python -m caliper_sim.create_table3 \
  --data-dir /mnt/large-drive/caliper/synthetic_table3 \
  --output-dir /tmp/new-table3-report
```

Choose a new report directory for each analysis. With `--config`, the default comes from
`output_dir`. With `--data-dir`, the default comes from the saved generation, or from
`<data-dir>/table3` for older datasets. Analysis requires a completed generation.

You can repeat stage 2 with the same images and a new output directory. For example,
add `--resize-512` or `--grayscale` to override the saved preprocessing for that
report; `--no-resize-512` and `--no-grayscale` disable those options.
`--num-images 10` overrides the measurement budget for that report without
regenerating images or changing the saved generation configuration.
`--skip-brisque` produces five metric rows. Each report records its effective
settings; none of these options modifies the generated images or saved defaults.

`--trim-border 1` excludes one native pixel from **each edge of every real and
synthetic image**, before resizing or metric computation. This is useful for
the dark final-row/final-column artifacts in BRC2 target frames. The default is
zero, preserving older results. The setting is also accepted as `[analysis]
trim_border = 1`, and is recorded in the report metadata and table footnote.
Compare reports using the same border policy; the original image files remain
unchanged. `--workers 6` measures independent frames concurrently while retaining
the same sampling and aggregation order. For CPU BRISQUE jobs, limiting Torch
threads per worker avoids oversubscription (for example `OMP_NUM_THREADS=1`).

Completed outputs from `python -m caliper_sim.table3` can also be passed to
`create_table3`. That command remains available to run generation and analysis
together using `output_data_path` for data and `output_dir` for reports. Older
INI files containing only `output_dir` retain their combined output layout.

## Current BRC2 whole-frame Table 3

Use [`configs/frames_table3_brc2.ini`](configs/frames_table3_brc2.ini)
for the current 250m color-source experiment, targeting 500m and 1000m. It
retains RGB during generation and computes whole-frame grayscale metrics with
one native pixel excluded from every edge. DAATSim saves and measures its native
512x512 results; restoring source proportions is a visualization-only operation.

| Target | pyBSM Cn² | DAATSim r₀ | P2S r₀ |
|---|---:|---:|---:|
| 500m | 1e-14 | 0.035 | 0.20 |
| 1000m | 1e-13 | 0.02 | 0.10 |

Paths in this config are local to this package. Supply the frozen cleaned frame
pool at `data/brc2_color_frames` and P2S assets at `data/p2s_simulator_data`, or
edit the INI paths for your machine. These datasets/assets and generated outputs
are ignored by Git. The prepared input pool contains the calibration's 20 selected
source frames per subject and its valid target frames, excluding 55 test-pattern
frames at 500m and 40 at 1000m. Using a broader raw input pool changes sampling and
can reintroduce those errors. Expected cohorts are 85 and 64 subjects respectively.

For the prepared saved simulations, run from this package directory:

```bash
python -m caliper_sim.create_table3 --config configs/frames_table3_brc2.ini
```

Each report requires a new output directory; use `--output-dir table3_runs/<new-name>`
to repeat it. To generate fresh simulations, choose a new destination and then
analyze it:

```bash
python -m caliper_sim.generate_synthetic_data \
  --config configs/frames_table3_brc2.ini \
  --output-data-path data/brc2_color_calibrated_fresh
python -m caliper_sim.create_table3 \
  --config configs/frames_table3_brc2.ini \
  --data-dir data/brc2_color_calibrated_fresh \
  --output-dir table3_runs/brc2_color_calibrated_fresh
```

`reset_between_clips = true` matches the calibration sweep's DAATSim temporal
boundaries. Clips are recognized by the extracted `*_frameNNNNNNNN` filenames;
other filenames fall back to their subject/camera group. Existing configurations
default to `false`, preserving their reset policy. `frames_table3_brc2_color_calibrated.ini` uses the same settings as `frames_table3_brc2.ini`.

## Figure 6 workflow: prepare, evaluate, then report

Run from the repository root with Python 3.10+. This workflow uses model
extractors in `methods/` and the shared evaluator in `caliper_eval/`. Install the
repository in each method's environment:

```bash
python -m pip install -e '.[fig6]'
```

Generation additionally needs `turbulence_eval/requirements.txt`; each method
needs its existing dependencies and released checkpoint (see the root README).
The examples below use BoT / Market1501 first.

### Generate the query chips

Edit dataset paths, `partition_file`, and
`output_data_path` in [chips_fig6_combined.ini](configs/chips_fig6_combined.ini).
It selects both collections with their own camera profiles. The BRC1 and BRC2
presets remain available for separate collection runs. INI paths resolve relative
to the INI file; output directories must be new.

```bash
python -m turbulence_eval.generate_synthetic_data --config turbulence_eval/configs/chips_fig6_combined.ini --preview
python -m turbulence_eval.generate_synthetic_data --config turbulence_eval/configs/chips_fig6_combined.ini
```

The generation's `eval/` directory contains one real 400m gallery, real 500m and
1000m queries, and three synthetic query files. The synthetic conditions use
500m source chips degraded to 1000m at Cn² `1e-13`, `4e-13`, and `1e-12`.
Use `max_images = -1` to preserve all source frames. Real 1000m queries remain a
different population; simulation and real 500m share exact source frames.

### Extract features

Set absolute paths in the same shell. Use a new feature
directory for each model/generation. `FIG6_GENERATION` must match the INI output.
Pass an **absolute checkpoint path**, since its saved path is also used at scoring.

```bash
export FIG6_GENERATION="/absolute/path/to/chips_fig6_combined"
export FIG6_FEATURES="/absolute/path/to/fig6/bot-market1501"
export FIG6_WEIGHT="/absolute/path/to/market_resnet50_model_120_rank1_945.pth"

(
  cd methods/reid-strong-baseline
  python tools/extract_features.py --config_file configs/softmax_triplet_with_center.yml \
    --split fig6 --source market1501 \
    --splits-dir "$FIG6_GENERATION/eval" --out-dir "$FIG6_FEATURES" \
    MODEL.PRETRAIN_CHOICE self TEST.NECK_FEAT after TEST.WEIGHT "$FIG6_WEIGHT"
)
```

For the other four paper models, use their existing Python extractors with the
same `--splits-dir` and separate `--out-dir` values. Keep their existing transforms,
checkpoint options and SINet sequence handling. Model labels for export are
`bot-market1501`, `sinet-lsvid`, `clip-reid-msmt17`, `personvit-msmt17`, and
`sapiensid-wb12m` (the saved `method-source` values).

### Score with the shared evaluator

Score the features saved by the extractor.

```bash
python -m caliper_eval.eval_from_features --features-dir "$FIG6_FEATURES" \
  --pooling video --pool-op mean --metric cosine --max-per-video 0 \
  --fig6-generation-dir "$FIG6_GENERATION"
```

Keep the images and checkpoint available when scoring. `--fig6-generation-dir` checks the
frame paths, identities, and tracklet grouping across all five query conditions. The output
records hashes for the generation, gallery, feature caches, and checkpoint file present at
scoring time.

The `caliper-features-fig6-v1` protocol averages raw features before cosine normalization.
It uses the shared evaluator’s tie ordering and excludes queries without a gallery match.

### Export the figure from saved results

One model is enough for a first run;
add more `MODEL=JSON` entries after `--eval-results` to include the other models.
This stage needs only the saved JSON files and Matplotlib.

```bash
python -m turbulence_eval.export_fig6 \
  --eval-results "bot-market1501=$FIG6_FEATURES/eval_mean_cosine.json" \
  --scopes combined --output-dir /absolute/path/to/new-fig6-report
```

Outputs include `sim_severity_diff_1000m.{pdf,png,svg}`, companion CMC plots,
CSV/Markdown summaries, and provenance. The main plot is
`CMC(simulated) - CMC(real 1000m)` at ranks 1–20. All models must use the same
generation/gallery. For a single-collection generation, use `--scopes brc1_only`
or `--scopes brc2_only`; a combined result cannot be relabeled as one collection.

Existing unannotated feature caches can be rescored with the flag above.
Original saved results still work with `--results-dir /path/to/results_synthetic`
in place of `--eval-results`.

For custom model adapters, use `reid.py`. Its `caliper-reid-v1` protocol normalizes frame features before averaging.
Results from different protocols must be reported separately.

### Model adapter contract

To use a custom encoder with `reid.py`, supply an INI in the format described in
`reid_config.py`.

`factory = my_reid_models.bot:load_encoder` imports an adapter function from an
installed Python package, or from the optional `python_path` directory. The
evaluator calls it as follows:

```python
encoder = load_encoder(checkpoint="/absolute/path/model.pth",
                       device="cuda:0", batch_size=32, **options)
features = encoder(crops)  # encoder.encode(crops) is also supported
```

`crops` is a list of RGB `uint8` NumPy arrays in frame order. The adapter owns
architecture construction, checkpoint loading, inference/eval mode, resizing,
normalization, alignment, and feature-layer selection. It must return a finite,
nonzero feature array or Torch tensor:

- `input_kind = image`: shape `(number_of_crops, feature_dimension)`. The
  evaluator calls the encoder in batches, normalizes each frame feature,
  averages over the tracklet, and normalizes the resulting vector.
- `input_kind = video`: shape `(1, feature_dimension)`. The encoder receives the
  complete ordered tracklet once and owns sequence sampling; its one returned
  feature is L2-normalized. Use this contract for SINet.

`options` is a JSON object passed unchanged as extra factory keyword arguments.
List auxiliary weights in `assets` and additional adapter/model source files or
directories in `code_paths`, one path per line. These paths resolve relative to
the INI file. Paths inside the arbitrary `options` object are passed unchanged;
use absolute paths there. The checkpoint, listed assets, factory source, listed
code, model options, device, batch size and core dependency versions contribute
to the model fingerprint. Include all preprocessing code/assets affecting
features. No reference model is substituted when imports or weights are missing.
An encoder object may provide `close()` to release its resources after evaluation.
Adapters should use deterministic inference; supply any required sampling seeds
through `options` so they are recorded in the cache fingerprint.

Tracklet features are cached by model fingerprint and ordered input-image hashes.
Changing a checkpoint or preprocessing settings creates a different cache entry.
Changing images after preparation is rejected, including on cache hits; prepare
a new run for changed data. Models with additional dependencies need those
dependencies installed in the evaluation environment.

### Retrieval and outputs

Scoring uses cosine similarity between normalized tracklet vectors, with one
gallery entry per tracklet. There is no camera exclusion or reranking. Unmatched
queries are excluded and counted explicitly. CMC uses each query's first correct
match; mAP averages query AP. Equal scores receive their average tied rank for
CMC/AP, matching the original evaluation; mINP uses the final positive's worst
tied position. Reported retrieval averages weight query tracklets equally.

```text
<evaluation output_dir>/
  prepared/conditions.json
  features/<model-fingerprint>/*.npy  # or the configured cache_dir
  results/<scope>/<condition-stem>__<model>/results.json
  report/
    sim_severity_diff_1000m.{png,pdf,svg}
    sim_severity_1000m.{png,pdf,svg}
    sim_severity_500m.{png,pdf,svg}
    sim_severity_<scope>.{csv,md}
    sim_severity_provenance.json
```

The difference plot shows `CMC(simulated) - CMC(real 1000m)` at ranks 1–20 by
default. Positive values mean simulated queries are easier; zero means equal
retrieval success at that rank. Colors identify models; dotted, solid and dashed
lines identify weak, medium and strong simulation. Other scopes receive a
`_<scope>` figure suffix. CSV/Markdown retrieval metrics are percentages; plotted
CMC differences are probabilities.

Export reports without loading models or accessing images:

```bash
python -m turbulence_eval.export_fig6 \
  --results-dir /path/to/saved/results \
  --output-dir /tmp/new-fig6-report \
  --scopes combined
```

This exporter also accepts the original `v4-sim1` result directories, without
the original exporter's external helper module.
It checks attribution, protocol/input fingerprints, checkpoint consistency,
query counts, CMC values and severity/source query pairing. Standalone results
add exact tracklet-pairing and gallery-content checks. The standalone protocol
is named `caliper-reid-v1`; newly evaluated data is not labeled as the archived
`v4-sim1` run.

## Generate from the current chip dataset (default)

Use the same protocol-v4 chip tree, identity partitions, training corpus, and
evaluation splits as the current Re-ID pipeline. Inputs are read directly;
generation does not create a staged input tree or write to the source dataset.
New generation INIs default to `mode = chips`. A partition must be explicit.

After installation, run from this package directory:

```bash
# Inspect the real selection before starting a simulation; writes nothing.
python -m caliper_sim.generate_synthetic_data --config configs/chips_train.ini --preview

# Generate all configured variants into a new output directory.
python -m caliper_sim.generate_synthetic_data --config configs/chips_train.ini
# Equivalent primary entry point:
python -m caliper_sim --config configs/chips_train.ini
```

Selection previews for new generation configs use only Python's standard library;
they do not load simulators, P2S assets, Torch, or quality-metric dependencies.
Paths in an INI resolve relative to that INI. Dataset mounts and output paths are
configurable; set the paths in the chip examples to your dataset and output directory.

| Configuration | Input and experiment |
|---|---|
| `configs/chips_train.ini` | Training corpus, 400m → 1000m, three pyBSM strengths |
| `configs/chips_test.ini` | PaperGR test split, 500m → 1000m queries, fixed real 400m gallery |
| `configs/chips_250_to_500.ini` | Training corpus, 250m → 500m, three pyBSM strengths |
| `configs/chips_table3_brc1.ini` | BRC1 chip quality comparison, 250m → 500m/1000m, three simulators |
| `configs/chips_table3_brc2.ini` | BRC2 ground monochrome chip quality comparison with BRC2 optics and proxy strengths |
| `configs/frames_table3_brc1.ini` | BRC1 pre-extracted full-frame quality comparison |
| `configs/frames_table3_brc2.ini` | Current calibrated BRC2 color-source full-frame quality comparison |
| `configs/chips_fig6_brc1.ini` | BRC1 Figure 6 test-chip generation, 500m → 1000m, three pyBSM severities |
| `configs/chips_fig6_brc2.ini` | BRC2 Figure 6 test-chip generation with BRC2 optics |
| `configs/chips_fig6_combined.ini` | Combined BRC1+BRC2 Figure 6 test-chip generation with collection-specific optics |
| `configs/images.ini` | Flat image directory; switch to `subject-tree` for the existing subject layout |
| `configs/table3.ini` | Original image-based Table 3 generation and measurement protocol |
| `configs/frames_table3_brc2_color_calibrated.ini` | Compatibility name for `frames_table3_brc2.ini` |

The three pyBSM strengths are `1e-13`, `4e-13`, and `1e-12`. Training input
selection follows the existing corpus's shared-source filtering, including its
coverage exception. Evaluation membership comes from `split_json`, never the
chip index's stale `side` field. Gallery-only distractors remain in the gallery.
Real queries are restricted to identities with selected source chips; the gallery
stays fixed. A source frame's canonical identity always comes from metadata,
including when its raw BRC1/BRC2 path ID differs.

`chip_root` is the directory containing `v4/`, not `v4/` itself. Chip paths resolve
as `<chip_root>/v4/<relpath-without-extension>/<frame:06d>.jpg`. For training,
`corpus_dir` must contain `train_chips.csv`, `train_tracklets.csv`,
`train_classes.json`, and (when deduplication is enabled) `train_manifest.csv`.
Evaluation uses the existing video-level split schema:
`{"gallery": {"<canonical-id>": ["<relpath>"]}, "query": {...}}`.
This generation mode covers ground field video chips; enrollment and elevated
camera simulation are outside its scope. Unusable/absent split records are
counted in the preview and run summary. Missing selected chip files, identity
disagreements, partition overlap, and empty selections raise errors.

All selected frames are generated by default (`max_images = -1`). An optional
`max_images` is a deterministic **per-subject** budget across that subject's
selected clips and collections; `num_subjects` and `subjects` allow smaller runs.
For a small trial run, copy a config, set `num_subjects = 1` and `max_images = 2`,
and choose a new output path. Changing the number of selected subjects does not
change another subject's frame draw.

For BRC1, dataset tokens `250m`/`400m` resolve to calibration labels
`260m`/`380m`. BRC2 uses its native `250m`/`400m` models; the legacy extractor
folder names `260m`/`380m` also resolve to those BRC2 stations.

To generate BRC2 chips (or ordinary image directories), set:

```ini
[input]
collections = brc2

[calibration]
brc2 = brc2

[experiment]
source_camera = 250m
```

Keep the remaining paths and variants from your existing INI, or use
[`configs/chips_table3_brc2.ini`](configs/chips_table3_brc2.ini) for chip quality
comparison. Mixed chip runs can use `collections = brc1 brc2` with `brc1 = brc1`
and `brc2 = brc2` in `[calibration]`; simulators are constructed per profile.
Setting `brc2 = brc1` still explicitly selects the old BRC1 proxy.

For legacy whole-frame Table 3 INIs, set `[experiment] camera_profile = brc2`.
[`configs/frames_table3_brc2.ini`](configs/frames_table3_brc2.ini) does this with
the separate `250m-c` color source folder. BRC1 remains the default profile.

BRC2 supports `156m`, `160m`, `250m`, `250m_lift`, `400m`, `500m`, and `1000m`
in the camera adapter. Indexed chip generation still requires matching ground
camera records; it does not add missing stations to the dataset. `250m_lift`
is a separate station and must not be collapsed into the monochrome `250m` pool.
BRC2 source metres per pixel are estimated as slant range × pixel pitch / focal
length, assuming native sensor sampling. Resized inputs need a corresponding
scale calibration. Sensor/lens proxies and atmospheric assumptions are documented
in the camera config file. The BRC2 full-frame preset uses its calibrated
strengths; the BRC2 chip-quality preset explicitly retains BRC1 proxy strengths.
pyBSM uses atmospheric interpolation directly. The scenario adapter handles pyBSM 0.15's
32.55 m versus 32.5 m lookup mismatch for the 3 m lift station.
Effective optics, source scale, and profiles are saved
with each run. DAATSim retains its bundled optics and uses the selected target
range; P2S uses the selected source aperture and target range.

Each `[variant:<name>]` selects one simulator and target camera:

```ini
[variant:daatsim_moderate]
simulator = daatsim
target_camera = 1000m
r0 = 0.035

[variant:p2s_mild]
simulator = p2s
target_camera = 1000m
r0 = 0.10
```

P2S variants additionally require `[paths] p2s_data_dir`; pyBSM-only and
DAATSim-only runs do not. Optional `[daatsim]` settings are `size`, `alpha_tilt`,
and `alpha_blur`; `[p2s] size` accepts `auto`, `128`, `256`, `512`, or `1024`.
DAATSim processes selected frames in temporal order and resets state at each
source tracklet. Its temporal coefficients are per selected frame, not adjusted
for gaps in frame numbers. P2S `auto` selects one supported size from the first
selected image, and records the effective size.

Chip-mode pyBSM defaults match the existing chip scripts: radiometric bridge
and source deconvolution enabled, resampling and radial distortion disabled.
The example source/target pairs and strengths reflect the paper and existing
generation scripts; reproducing published scores also requires matching the
training/evaluation configuration and cohort. These generation commands produce
data; the [Figure 6 workflow](#figure-6-workflow-prepare-evaluate-then-report) computes Re-ID scores.

## Generated-chip outputs and loader integration

```text
<output_data_path>/
  experiment.ini
  settings.json
  selection.json          # selected sources and coverage summary
  chips.jsonl             # authoritative output index, one record per synthetic chip
  extra_chips.csv         # training runs only; directly consumable via --extra-chips
  images/<variant>/<collection>/<original-relpath-stem>/<frame:06d>.jpg
  eval/                   # test runs only; full loader-compatible crop-level kwcoco
    gallery.kwcoco.json
    query_500m.kwcoco.json
    query_1000m.kwcoco.json
    query_500m_sim_1000m_pybsm_weak.kwcoco.json
    ...
  generation.json         # completion marker, effective parameters, counts and fingerprints
```

`chips.jsonl` stores each output's absolute `chip_path`, complete selected source
record, source calibration label, variant name, simulator, target range, and
strength. The source record retains canonical identity, collection, original
clip path, corpus/index tracklet ID, frame, physical camera, and DIVE track ID.
Output paths preserve source hierarchy, so repeated filenames across clips and
collections cannot overwrite one another.

`extra_chips.csv` has the existing six columns: `chip_path`, `tracklet_id`,
`src_tracklet_id`, `canonical_id`, `tok`, and `collection`. IDs take the form
`sim:<source-tracklet-id>:<variant>` so severities stay separate training units.
All configured variants are included in this manifest. It can be passed directly
to `GroundFieldCorpus(extra_chips=...)` or `tools/train_caliper.py --extra-chips ...`.

Evaluation documents include subjects, videos, tracks, images, annotations,
physical camera IDs, and original frame indices. All documents share the same
subject/camera registries. Synthetic queries preserve source clip grouping and
physical camera identity; the target condition is recorded in `camera_token`
and the query filename. Existing `features_io.load_crops_full` and role discovery
can consume the exported directory directly. Real images are referenced in place,
so the chip mount must remain available. Each variant produces the same selected
source frames, allowing comparison against the fixed real gallery and real queries.

Runs refuse existing output directories. A failed run leaves inspectable partial
files without `generation.json`; a completed run records input metadata SHA-256
fingerprints and dependency versions. Use a fresh output path for another run.
Quality comparison runs use `purpose = quality` and save complete real source
and target pools in `real_chips.jsonl`. `partition = all` includes all subjects
from the selected collection; `train` or `test` restricts the quality cohort.
These runs produce neither Re-ID training manifests nor gallery/query exports.
Each target uses the same source frames and common subject cohort across its
simulators. Ordinary Re-ID generation runs remain separate from quality analysis.

To compare BRC1 person chips, configure the P2S assets and run:

```bash
python -m caliper_sim --config configs/chips_table3_brc1.ini --preview
python -m caliper_sim --config configs/chips_table3_brc1.ini
python -m caliper_sim.create_table3 --config configs/chips_table3_brc1.ini
```

The defaults generate 20 source chips per subject and sample five real/synthetic
images per subject per group for measurement. Reports use the same metrics and
preprocessing described below, with manifest paths supplied directly to the
evaluator. No staged image tree is needed. Person-chip statistics describe the
cropped person regions and depend on those crop boundaries.

## Simulation models

**pyBSM** constructs source and target sensor/scenario models and applies the
regularized transfer OTF `Ht * Hs / (Hs**2 + 0.001)` when source deconvolution
is enabled. A Cn2 override applies to both cameras. The transmission band is
0.50–0.66 micrometres and the forward reflectance interval is [0.05, 0.5]. The
radiometric bridge uses the forward table's endpoints for inverse pixel mapping.
Configuration and caches belong to each simulator instance.

**[DAATSim](https://github.com/Riponcs/DAATSim)** uses uniform depth and the optics in
[`_vendor/daatsim_config.py`](_vendor/daatsim_config.py), including a 0.2m aperture.
Depth-estimation models are not required. The default simulation and saved output
size is 512x512. The adapter returns the native backend output without restoring
source dimensions. Metrics use that saved square image; visualization tools may
restore source proportions in memory before drawing a full frame or person crop.
`reset_between_subjects = false` carries temporal state across subjects; set it
to `true` to reset state for each subject. Temporal coefficients are explicit
even when source frames are randomly sampled. Effective parameters are recorded
in `generation.json` and the analysis report's `run.json`.

**[P2S](https://github.com/Riponcs/TurbulenceSimulatorPython)** uses the source camera's aperture, reflection padding to a square and the
supported simulation size nearest the input's shorter dimension when `size = auto`.
It resizes and crops the result back to the source dimensions. Tilt matrices are
cached inside the output directory with keys that include size, aperture, r0 and
propagation distance. Shared model assets are read only. DAATSim and P2S seeds
are isolated per call from the caller's Torch random state.

The P2S kernels retain attribution to Mao, Chimitt and Chan,
*Accelerating Atmospheric Turbulence Simulation via Learned Phase-to-Space
Transform* (ICCV 2021), and the CC BY-NC-SA 4.0 license in
[`_vendor/P2S_LICENSE`](_vendor/P2S_LICENSE).

## Image-quality metrics

The paper's Table 3 uses **whole camera images** for both simulation and quality
measurement. This choice reduces sensitivity to person pose, clothing, bounding-box
placement and crop aspect ratio, so the comparison emphasizes atmospheric and
camera effects across the scene. It does not require matching person crops or poses
between source and target images. Use whole-image inputs when reproducing this
experiment. Chip-based quality runs measure person crops instead and can produce
different values and simulator rankings; they are a separate evaluation.
Supply native full frames using the
[external input preparation layout](#prepare-full-frame-inputs-outside-this-repository).
Image data and download tools are supplied separately.

The six reported differences are sharpness (Laplacian variance), RMS contrast,
histogram entropy, BRISQUE, radial PSD L2 and radial PSD KL divergence. Scalar
differences use `mean_subjects(abs(mean_sim - mean_target))`, giving each subject
equal weight. Intensities use the 0–255 scale and entropy is in bits.

Final tables report **mean ± standard deviation** across the per-subject scalar
differences or PSD distances. This is population standard deviation (`ddof=0`);
a single-subject run reports zero. It describes variation between subjects.
CSV files keep numeric means and standard deviations in separate columns:
`<range> / <simulator>` and `<range> / <simulator> std` in `table3.csv`, and
`<simulator>` and `<simulator>_std` in each target's summary CSV. JSON reports
store means in `summary`, standard deviations in `summary_std`, and the convention
in `std_ddof` and `std_unit`.

The quality boxplots show pooled **per-image** metric distributions for each
group. Their boxes span the 25th to 75th percentiles with a median line; whiskers
reach the furthest observations within 1.5 times the interquartile range, and
points beyond are plotted as outliers. They do not display the table's
across-subject standard deviations, and plotted outliers are not automatically
invalid frames.

To summarize an existing completed Table 3 report with **medians across subjects**:

```bash
python -m caliper_sim.summarize_table3 \
  --report-dir table3_runs/frames_table3_brc1
```

This reads the saved per-subject scalar differences and PSD distances and writes
`median/table3.md`, `median/table3.csv`, and `median/run.json` inside the report.
An optional `--output-dir` selects a different new destination; `--overwrite`
updates an existing median report from the same source. Entries are
**median ± robust standard deviation**, with equal subject weights. Robust SD is
the normal-consistent median absolute deviation:
`1.4826 * median_subjects(abs(x - median_subjects(x)))`, using the unrounded
factor `1 / NormalDist().inv_cdf(0.75)`. This describes spread across subjects.
CSV columns store the median and `robust_sd` separately; JSON metadata records
the method and scaling factor. Scalar entries become
`median_subjects(abs(mean_frames_sim - mean_frames_target))`; within-subject frame
means and mean-spectrum PSD distances stay as originally computed. No images or
models are loaded. The original report is preserved, and input CSV hashes are
recorded in the median report's metadata.

PSD comparisons use per-subject mean spectra, with radial power interpolated
onto 127 nonzero frequency bins and normalized to unit sum. The reported PSD L2
is RMSE over those bins; KL is `KL(simulated || target)` in nats.

With `resize_512 = false`, real images match each subject's median pyBSM output
dimensions. Other simulators keep their saved output dimensions: normally 512x512
for DAATSim and source dimensions for pyBSM/P2S. With `resize_512 = true`, every group
is evaluated at 512x512. Real-image matching and the optional 512x512 resize use
OpenCV linear interpolation. Images load as RGB; sharpness, RMS contrast, entropy,
and PSD use grayscale intensities, while BRISQUE receives RGB scaled to [0, 1].

Analysis traverses subjects in sorted order and samples without replacement using
Python's `random.Random`. Within each subject the draw order is real source,
real target, pyBSM, DAATSim, then P2S. The default analysis seed is 0. Sampling is
independent between groups; spatial registration and pose matching are not required.
Resizing and grayscale conversion affect the statistics; record these settings
when comparing experiments.

Each target's `quality/` directory contains per-image metrics, per-subject means
and differences, summary tables, box plots and a radial PSD plot. `analysis.json`
records selected files, exclusions, frame counts, preprocessing, dependency
versions, spectra and final values. BRISQUE is enabled by default. Missing
dependencies or failed scores raise an error; disabling BRISQUE explicitly
produces a partial metric table.

## Standalone pyBSM generation

```bash
python -m caliper_sim \
  --input-dir /path/to/subject/500m \
  --source-camera 500m --target-camera 1000m \
  --cn2 4e-13 --no-resample --no-distortion \
  --noise-seed 1234 --output-dir /tmp/new-simulated-images
```

For a subject tree, replace `--input-dir` with `--stable-frames-dir`. Output then
uses `<output-dir>/<subject>/sim_500m_to_1000m/<source-filename>`. Filenames are
preserved. `--max-images` limits frames per subject, sampled with `--seed`
(default 42). `--noise-seed` separately controls sensor noise; omitting it permits
stochastic results. Defaults enable the radiometric bridge, source deconvolution,
resampling and radial distortion; use the corresponding `--no-*` flags to disable
them. Choose a new output directory.

The Python API accepts uint8 RGB arrays:

```python
from caliper_sim import CameraSimulator, SimulationOptions

simulator = CameraSimulator(
    "500m", "1000m",
    SimulationOptions(cn2=4e-13, resample=False, distortion=False, noise_seed=1234),
)
simulated_rgb = simulator.perturb(source_rgb)
```

## Standalone quality analysis

Compare one simulator's subject tree with real source and target images:

```bash
python -m caliper_sim.quality \
  --stable-frames-dir /path/to/stable_frames \
  --sim-dir /tmp/new-simulated-subjects \
  --source-camera 260m --target-camera 1000m \
  --num-images 5 --seed 0 --output-dir /tmp/new-quality-analysis
```

The simulated tree uses `<subject>/sim_260m_to_1000m/*.jpg`. Use `--dist-name`
to select another simulation folder. Only subjects with source, target and
simulated images are included. `--num-subjects` limits the cohort; frame sampling
defaults to at most five images per subject per group. Source and target frames
are sampled independently.

Real images match each subject's median simulated dimensions. `--resize-512`
evaluates every image at 512x512, `--grayscale` removes color, and
`--skip-brisque` explicitly omits BRISQUE. The analysis requires a new output
directory and writes the same quality artifacts described above.

## Troubleshooting

| Problem | Check |
|---|---|
| Output directory already exists | Choose a new generation destination or pass a fresh `--output-dir` for analysis/reporting. For Figure 6 evaluation, use the documented resume workflow. |
| Missing P2S assets | Set `p2s_data_dir` to a folder containing all three files listed under [inputs](#real-image-and-p2s-inputs). |
| BRISQUE cannot download weights | Populate the Torch cache before running offline; see [first-run setup](#brisque-on-the-first-run). |
| Figure 6 cannot import a model or find weights | Replace the evaluation INI's example factory, `python_path` and checkpoint with your adapter and weights. |
| Figure 6 rejects a generation directory | Use a completed Figure 6 test-chip generation with all three severities and its gallery/query exports. Table 3 quality outputs use a different protocol. |
| No eligible source/target images | Check the configured camera folders or chip metadata, collection filter, subject selection and test split. Full-frame source folder names must match the INI. |
