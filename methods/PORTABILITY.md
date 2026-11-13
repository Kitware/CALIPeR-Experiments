# Portable paths for upstream workflows

The method directories include their upstream training, data-preparation and
evaluation workflows alongside the CALIPeR integration. For benchmark runs, use
the repository [README](../README.md) and [training guide](../TRAINING.md).
Run upstream examples from the working directory in the method README. Relative paths follow
the process when a script changes directories.

| Location | Purpose |
|---|---|
| `./data/` | Input datasets and dataset preparation trees |
| `./weights/` | Pretrained checkpoints; retain each configured filename |
| `./outputs/` | Training logs and experiment outputs |
| `./outputs/debug/` | Optional diagnostic images and temporary inspection files |

Dataset preparation stages can read and write intermediate records under the
same `data/` tree. Their subdirectories and filenames retain the upstream layout.
For workflows that change directories between stages, pass absolute input and
output paths through their existing arguments or configuration fields. SapiensID's
WebBody examples also provide the existing `WEBBODY_DATA_ROOT` variable for this
purpose. Edit paths defined as module constants before running those scripts.

Supply the datasets and checkpoint files required by the selected workflow.
Existing command-line options and configuration overrides take precedence as
documented by that method. For optional diagnostics, create the output directory
from the working directory used by the script:

```bash
mkdir -p ./outputs/debug
```

CALIPeR continues to use `CALIPER_DATA_ROOT`, `REID_BENCH_ROOT` and
`CALIPER_EVAL_HOME` as described in the benchmark README. Export absolute roots
when using its extraction wrappers, which change working directories.

The upstream repository URLs, base commits, citations and included license
notices are recorded in [UPSTREAM.md](UPSTREAM.md). Portable path substitutions
are local modifications to those upstream snapshots; follow each method's
documentation for its dependencies, dataset access and checkpoint selection.
