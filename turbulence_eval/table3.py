"""Generate simulated frames and compare their image quality with real targets for Table 3."""

import argparse
from configparser import ConfigParser
from dataclasses import asdict
import gc
import json
from pathlib import Path
import re

import cv2
import numpy as np

from .adapter import CameraSimulator, SimulationOptions
from .cameras import camera_configs, camera_label, metres_per_pixel
from .generate import write_planned_images
from .quality import analyze_quality, _write_csv
from .simulators import DAATSimulator, P2SSimulator

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_SIMULATORS = ("pybsm", "daatsim", "p2s")
_SCHEMA = {
    "paths": {"stable_frames_dir", "p2s_data_dir", "output_data_path", "output_dir"},
    "experiment": {"source_camera", "target_cameras", "num_images", "num_subjects", "subjects",
                   "sample_seed", "analysis_seed", "noise_seed", "device", "camera_profile"},
    "analysis": {"num_images", "include_brisque", "resize_512", "grayscale", "trim_border", "workers"},
    "pybsm": {"bridge", "deconvolve", "resample", "distortion"},
    "daatsim": {"size", "alpha_tilt", "alpha_blur", "reset_between_subjects", "reset_between_clips"},
    "p2s": {"size"},
}


def _words(value):
    return [word for word in re.split(r"[,\s]+", value.strip()) if word]


def load_config(path):
    """Resolve paths relative to the INI file and reject misspelled settings."""
    path = Path(path).resolve()
    config = ConfigParser(interpolation=None)
    with path.open() as stream:
        config.read_file(stream)
    for section in _SCHEMA:
        if not config.has_section(section):
            raise ValueError(f"Missing INI section [{section}]")
    targets = _words(config["experiment"]["target_cameras"])
    source = config["experiment"]["source_camera"]
    profile = config["experiment"].get("camera_profile", "brc1")
    camera_configs(profile)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("Select a known source camera and distinct target cameras")
    camera_label(source, profile)
    for target in targets:
        camera_label(target, profile)
        if camera_label(target, profile) == camera_label(source, profile):
            raise ValueError(f"Invalid target camera {target!r}")
        if not config.has_section(f"target:{target}"):
            raise ValueError(f"Missing INI section [target:{target}]")
    for section in config.sections():
        allowed = _SCHEMA.get(section)
        if section in [f"target:{target}" for target in targets]:
            allowed = {"pybsm_cn2", "daatsim_r0", "p2s_r0"}
        if allowed is None:
            raise ValueError(f"Unknown INI section [{section}]")
        unknown = set(config[section]) - allowed
        if unknown:
            raise ValueError(f"Unknown settings in [{section}]: {sorted(unknown)}")
    def resolved(key):
        value = config["paths"][key].strip()
        if not value:
            raise ValueError(f"[paths] {key} must not be empty")
        value = Path(value).expanduser()
        return str((path.parent / value).resolve())
    paths = {key: resolved(key) for key in ("stable_frames_dir", "p2s_data_dir", "output_dir")}
    if config.has_option("paths", "output_data_path"):
        paths["output_data_path"] = resolved("output_data_path")
    experiment = config["experiment"]
    settings = dict(
        config_file=str(path), paths=paths,
        source_camera=source, target_cameras=targets, camera_profile=profile,
        num_images=experiment.getint("num_images", 20),
        num_subjects=experiment.getint("num_subjects") if experiment.get("num_subjects", "").strip() else None,
        subjects=_words(experiment.get("subjects", "")), sample_seed=experiment.getint("sample_seed", 42),
        analysis_seed=experiment.getint("analysis_seed", 0),
        noise_seed=experiment.getint("noise_seed") if experiment.get("noise_seed", "").strip() else None,
        device=experiment.get("device", "auto"),
        analysis=dict(num_images=config["analysis"].getint("num_images", 5),
                      trim_border=config["analysis"].getint("trim_border", 0),
                      workers=config["analysis"].getint("workers", 1),
                      **{key: config["analysis"].getboolean(key, fallback=(key == "include_brisque"))
                         for key in ("include_brisque", "resize_512", "grayscale")}),
        pybsm={key: config["pybsm"].getboolean(key, fallback=(key in ("bridge", "deconvolve")))
               for key in _SCHEMA["pybsm"]},
        daatsim=dict(size=config["daatsim"].getint("size", 512),
                     alpha_tilt=config["daatsim"].getfloat("alpha_tilt", 0.2),
                     alpha_blur=config["daatsim"].getfloat("alpha_blur", 0.2),
                     reset_between_subjects=config["daatsim"].getboolean("reset_between_subjects", False),
                     reset_between_clips=config["daatsim"].getboolean("reset_between_clips", False)),
        p2s_size=config["p2s"].get("size", "auto"),
        targets={target: {key: config[f"target:{target}"].getfloat(key)
                          for key in ("pybsm_cn2", "daatsim_r0", "p2s_r0")} for target in targets},
    )
    if (settings["num_images"] < 1 or settings["analysis"]["num_images"] < 1
            or (settings["num_subjects"] is not None and settings["num_subjects"] < 1)):
        raise ValueError("Generation/analysis num_images and num_subjects must be positive")
    if settings["analysis"]["trim_border"] < 0 or settings["analysis"]["workers"] < 1:
        raise ValueError("Analysis trim_border must be nonnegative and workers must be positive")
    for values in settings["targets"].values():
        if any(value is None or not np.isfinite(value) or value <= 0 for value in values.values()):
            raise ValueError("Each target needs positive finite pybsm_cn2, daatsim_r0 and p2s_r0")
    if settings["p2s_size"] != "auto" and settings["p2s_size"] not in ("128", "256", "512", "1024"):
        raise ValueError("P2S size must be auto, 128, 256, 512 or 1024")
    return settings


def _files(folder):
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_EXTS) if folder.is_dir() else []


def _temporal_group(original, subject, reset_between_clips):
    if not reset_between_clips:
        return subject
    match = re.match(r'(.+)_frame\d+$', Path(original).stem)
    return subject, match[1] if match else str(Path(original).parent)


def _select(settings):
    root = Path(settings["paths"]["stable_frames_dir"])
    if not root.is_dir():
        raise NotADirectoryError(root)
    source, targets = settings["source_camera"], settings["target_cameras"]
    requested = settings["subjects"]
    if len(set(requested)) != len(requested):
        raise ValueError("Explicit subjects must not contain duplicate IDs")
    subjects = requested or sorted(folder.name for folder in root.iterdir() if folder.is_dir())
    cohorts, excluded = {}, {}
    for target in targets:
        eligible, missing = [], []
        for subject in subjects:
            if Path(subject).name != subject or subject in (".", ".."):
                raise ValueError(f"Expected a subject folder name: {subject!r}")
            if _files(root / subject / source) and _files(root / subject / target):
                eligible.append(subject)
            else:
                missing.append(subject)
        if requested and missing:
            raise ValueError(f"Requested subjects missing {source}/{target} frames: {missing}")
        if settings["num_subjects"] is not None:
            eligible = eligible[:settings["num_subjects"]]
        if not eligible:
            raise ValueError(f"No subjects have both {source} and {target} frames")
        cohorts[target], excluded[target] = eligible, missing
    rng = np.random.default_rng(settings["sample_seed"])
    selection = {}
    included = set().union(*map(set, cohorts.values()))
    # Draw only source frames with the generation RNG. Real-image pools belong
    # to analysis; target availability must not change another subject's draw.
    for subject in sorted(subjects):
        paths = _files(root / subject / source)
        if len(paths) > settings["num_images"]:
            indices = sorted(rng.choice(len(paths), settings["num_images"], replace=False))
            paths = [paths[index] for index in indices]
        if subject not in included:
            continue
        selection[subject] = {source: [str(path.resolve()) for path in paths]}
        for target in targets:
            if subject in cohorts[target]:
                selection[subject][target] = [str(path.resolve()) for path in _files(root / subject / target)]
    return cohorts, excluded, selection


def _write_json(path, value):
    # DAATSim's Kolmogorov outer scale is infinity; JSON records it as a string.
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(value) for value in item]
        return str(item) if isinstance(item, float) and not np.isfinite(item) else item
    with path.open("x") as stream:
        json.dump(clean(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def _data_path(settings):
    paths = settings["paths"]
    return Path(paths["output_data_path"] if "output_data_path" in paths else paths["output_dir"])


def _report_path(settings, data_dir):
    if "output_data_path" in settings["paths"] and "output_dir" in settings["paths"]:
        return Path(settings["paths"]["output_dir"])
    return data_dir / "table3"


def generate_synthetic_data(config_file, *, output_data_path=None, output_dir=None):
    """Select real frames and save the configured simulator outputs."""
    settings = load_config(config_file)
    if output_data_path is not None and output_dir is not None:
        raise ValueError("Use output_data_path or its output_dir alias, not both")
    override = output_data_path if output_data_path is not None else output_dir
    if override is not None:
        key = "output_data_path" if "output_data_path" in settings["paths"] else "output_dir"
        settings["paths"][key] = str(Path(override).resolve())
    return _generate(settings)


def _generate(settings):
    output = _data_path(settings)
    if "output_data_path" in settings["paths"] and output == Path(settings["paths"]["output_dir"]):
        raise ValueError("output_data_path and output_dir must be different directories")
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    assets = Path(settings["paths"]["p2s_data_dir"])
    for asset in ("P2S_model.pt", "dictionary.npy", "R-corr_-0.1.npy"):
        if not (assets / asset).is_file():
            raise FileNotFoundError(assets / asset)
    cohorts, excluded, selection = _select(settings)
    # Keep complete real source/target pools for independent measurement draws.
    # Only selection[subject][source] is sent through the simulators.
    source = settings["source_camera"]
    root = Path(settings["paths"]["stable_frames_dir"])
    analysis_files = {subject: {**cameras, source: [str(path.resolve()) for path in _files(root / subject / source)]}
                      for subject, cameras in selection.items()}
    output.mkdir(parents=True, exist_ok=False)
    (output / "experiment.ini").write_text(Path(settings["config_file"]).read_text())
    _write_json(output / "settings.json", settings)
    _write_json(output / "selection.json", dict(cohorts=cohorts, excluded=excluded, files=selection,
                                                analysis_files=analysis_files))
    effective = {}
    for target in settings["target_cameras"]:
        target_root = output / target
        inputs = target_root / "inputs"
        for subject in cohorts[target]:
            for camera in (source, target):
                folder = inputs / subject / camera
                folder.mkdir(parents=True)
                for original in analysis_files[subject][camera]:
                    (folder / Path(original).name).symlink_to(original)
        first = selection[cohorts[target][0]][source][0]
        image = cv2.imread(first)
        if image is None:
            raise ValueError(f"Could not read source frame: {first}")
        p2s_size = (min((128, 256, 512, 1024), key=lambda n: abs(n - min(image.shape[:2])))
                    if settings["p2s_size"] == "auto" else int(settings["p2s_size"]))
        profile = settings["camera_profile"]
        configs = camera_configs(profile)
        source_config = configs[camera_label(source, profile)]
        target_config = configs[camera_label(target, profile)]
        distance = target_config["_camera_distance_m"]
        strengths = settings["targets"][target]
        effective[target] = {}
        for name in _SIMULATORS:
            print(f"\n=== {target}: {name} ===", flush=True)
            if name == "pybsm":
                options = SimulationOptions(cn2=strengths["pybsm_cn2"], noise_seed=settings["noise_seed"], **settings["pybsm"])
                simulator = CameraSimulator(source, target, options, profile=profile)
                effective[target][name] = dict(options=asdict(options), source_camera=source, target_camera=target,
                                               calibration_profile=profile, source_config=source_config, target_config=target_config,
                                               source_metres_per_pixel=metres_per_pixel(source, profile))
            elif name == "daatsim":
                parameters = {key: value for key, value in settings["daatsim"].items()
                              if key not in ("reset_between_subjects", "reset_between_clips")}
                simulator = DAATSimulator(r0=strengths["daatsim_r0"], distance=distance,
                                          device=settings["device"], seed=settings["noise_seed"], **parameters)
                effective[target][name] = simulator.settings
            else:
                simulator = P2SSimulator(source, distance=distance, data_dir=assets, cache_dir=output / "cache" / "p2s",
                                         r0=strengths["p2s_r0"], size=p2s_size, device=settings["device"], seed=settings["noise_seed"],
                                         aperture=source_config["D"])
                effective[target][name] = simulator.settings
            destination = target_root / name
            destination.mkdir()
            planned = ((Path(original), destination / subject / f"sim_{source}_to_{target}" / Path(original).name,
                        _temporal_group(original, subject, settings['daatsim']['reset_between_clips']))
                       for subject in cohorts[target] for original in selection[subject][source])
            for _ in write_planned_images(
                planned, simulator,
                reset_between_groups=(name == "daatsim" and (settings["daatsim"]["reset_between_subjects"]
                                                            or settings['daatsim']['reset_between_clips'])),
            ):
                pass
            del simulator
            gc.collect()
    generation = dict(format_version=2, kind="table3", settings=settings,
                      effective_simulators=effective, cohorts=cohorts)
    _write_json(output / "generation.json", generation)
    print(f"\nSynthetic data ready: {output}", flush=True)
    return generation


def _load_generation(data_dir):
    manifest = data_dir / "generation.json"
    if not manifest.is_file():
        # Completed runs from the combined command also contain this metadata.
        manifest = data_dir / "run.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"No completed generation found in {data_dir}. "
            "Run python -m caliper_sim.generate_synthetic_data first."
        )
    generation = json.loads(manifest.read_text())
    if generation.get("kind") in ("synthetic_chips", "synthetic_images"):
        raise ValueError("This is a general generation dataset, not a Table 3 image-quality dataset. Use a Table 3 configuration for quality analysis.")
    if any(key not in generation for key in ("settings", "cohorts", "effective_simulators")):
        raise ValueError(f"Incomplete generation metadata: {manifest}")
    settings = generation["settings"]
    saved_selection = json.loads((data_dir / "selection.json").read_text())
    selection = saved_selection["files"]
    analysis_files = saved_selection.get("analysis_files", selection)
    settings["analysis"].setdefault("num_images", 5)
    source = settings["source_camera"]
    profile = settings.get("camera_profile", "brc1")
    camera_label(source, profile)
    for target in settings["target_cameras"]:
        camera_label(target, profile)
        if not generation["cohorts"].get(target):
            raise ValueError(f"Missing or invalid target cohort in {manifest}: {target}")
        target_root = data_dir / target
        for subject in generation["cohorts"][target]:
            if Path(subject).name != subject or subject in (".", ".."):
                raise ValueError(f"Invalid subject folder in {manifest}: {subject!r}")
            source_files = _files(target_root / "inputs" / subject / source)
            target_files = _files(target_root / "inputs" / subject / target)
            if (not source_files or not target_files or
                {path.name for path in source_files} != {Path(path).name for path in analysis_files[subject][source]} or
                {path.name for path in target_files} != {Path(path).name for path in analysis_files[subject][target]}):
                raise ValueError(
                    f"Missing real input frames for {subject}/{target}; "
                    "keep the original frames available when using input symlinks."
                )
            expected = {Path(path).name for path in selection[subject][source]}
            for name in _SIMULATORS:
                folder = target_root / name / subject / f"sim_{source}_to_{target}"
                if {path.name for path in _files(folder)} != expected:
                    raise ValueError(f"Incomplete synthetic data in {folder}; expected one output per source frame")
    return generation


def _analyze_generation(data_dir, output, generation):
    settings, cohorts = generation["settings"], generation["cohorts"]
    source, results = settings["source_camera"], {}
    for target in settings["target_cameras"]:
        target_root = data_dir / target
        results[target] = analyze_quality(
            target_root / "inputs", target_root / "pybsm", source, target, output / target / "quality",
            seed=settings["analysis_seed"], **settings["analysis"],
            simulations={name: (target_root / name, f"sim_{source}_to_{target}") for name in _SIMULATORS},
        )
    _write_table3_report(data_dir, output, generation, results, cohorts)
    return results


def _write_table3_report(data_dir, output, generation, results, cohorts):
    settings = generation["settings"]
    first = next(iter(results.values()))["summary"]
    metrics = list(next(iter(first.values())))
    comparisons = [(target, name) for target in results for name in results[target]["summary"]]
    columns = ["metric", *(f"{target} / {name}" for target, name in comparisons)]
    rows = [dict(metric=metric, **{
        f"{target} / {name}" + suffix: results[target][statistic][name][metric]
        for target, name in comparisons
        for suffix, statistic in (("", "summary"), (" std", "summary_std"))}) for metric in metrics]
    _write_csv(output / "table3.csv", rows)
    table = ["| " + " | ".join(columns) + " |", "|---|" + "---:|" * (len(columns) - 1)]
    table.extend("| " + " | ".join(str(row[column]) if column == "metric" else
                                    f"{row[column]:.4g} ± {row[column + ' std']:.4g}"
                                    for column in columns) + " |" for row in rows)
    table += ["", "Scalar entries: mean_subjects(|mean_sim − mean_target|). PSD L2 = RMSE; KL is in nats.",
              "Entries are mean ± population standard deviation across subjects (ddof=0).",
              "Cohort sizes: " + ", ".join(f"{target}: {len(cohorts[target])}" for target in results) + ".",
              "Camera labels identify the calibration models used for this analysis.", ""]
    border = settings.get("analysis", {}).get("trim_border", 0)
    if border:
        table += [f"Excluded {border} native pixel(s) from every edge of every real and synthetic image before resizing and metrics.", ""]
    with (output / "table3.md").open("x") as stream:
        stream.write("\n".join(table))
    _write_json(output / "run.json", dict(settings=settings, effective_simulators=generation["effective_simulators"],
                                          data_dir=str(data_dir), cohorts=cohorts,
                                          summary={target: results[target]["summary"] for target in results},
                                          summary_std={target: results[target]["summary_std"] for target in results},
                                          std_ddof=0, std_unit="subject"))
    print("\n".join(table))


def create_table3(data_dir=None, *, config_file=None, output_dir=None,
                  include_brisque=None, resize_512=None, grayscale=None, num_images=None,
                  trim_border=None, workers=None):
    """Measure image quality in a completed generation and write Table 3."""
    if trim_border is not None and (not isinstance(trim_border, int) or trim_border < 0):
        raise ValueError("trim_border must be a nonnegative integer")
    if workers is not None and (not isinstance(workers, int) or workers < 1):
        raise ValueError("workers must be a positive integer")
    configured = None
    if config_file is not None:
        config = ConfigParser(interpolation=None)
        with Path(config_file).open() as stream:
            config.read_file(stream)
        if any(section.startswith("variant:") for section in config.sections()):
            from .pipeline import load_generation_config
            configured = load_generation_config(config_file)
        else:
            configured = load_config(config_file)
    if data_dir is None:
        if configured is None:
            raise ValueError("Provide data_dir or config_file to locate the generated dataset")
        data_dir = _data_path(configured)
    data_dir = Path(data_dir).resolve()
    output = Path(output_dir).resolve() if output_dir is not None else None
    if output is None and configured is not None:
        output = _report_path(configured, data_dir)
    if output is not None and output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    marker = data_dir / "generation.json"
    if marker.is_file():
        metadata = json.loads(marker.read_text())
        if metadata.get("kind") == "synthetic_chips" and metadata["settings"]["input"].get("purpose") == "quality":
            from .chip_quality import create_chip_table3
            return create_chip_table3(data_dir, output_dir=output, include_brisque=include_brisque,
                                      resize_512=resize_512, grayscale=grayscale, num_images=num_images,
                                      trim_border=trim_border, workers=workers)
    generation = _load_generation(data_dir)
    if output is None:
        output = _report_path(generation["settings"], data_dir)
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if output == data_dir:
        raise ValueError("The report output_dir must differ from the generated data directory")
    generation["settings"]["paths"].update(output_data_path=str(data_dir), output_dir=str(output))
    if num_images is not None and num_images < 1:
        raise ValueError("Analysis num_images must be positive")
    for key, value in dict(include_brisque=include_brisque, resize_512=resize_512,
                          grayscale=grayscale, num_images=num_images, trim_border=trim_border, workers=workers).items():
        if value is not None:
            generation["settings"]["analysis"][key] = value
    output.mkdir(parents=True, exist_ok=False)
    return _analyze_generation(data_dir, output, generation)


def run_table3(config_file, *, output_data_path=None, output_dir=None):
    """Generate the images and write the quality report to the configured directories."""
    settings = load_config(config_file)
    if output_data_path is not None:
        settings["paths"]["output_data_path"] = str(Path(output_data_path).resolve())
    if output_dir is not None:
        settings["paths"]["output_dir"] = str(Path(output_dir).resolve())
    if "output_data_path" in settings["paths"]:
        report = Path(settings["paths"]["output_dir"])
        if report.exists():
            raise FileExistsError(f"Output directory already exists: {report}")
        _generate(settings)
        return create_table3(_data_path(settings), output_dir=report)
    generation = _generate(settings)
    output = _data_path(generation["settings"])
    return _analyze_generation(output, output, _load_generation(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-data-path", type=Path, help="Override generated-data location; must be new.")
    parser.add_argument("--output-dir", type=Path, help="Override report location; must be new.")
    args = parser.parse_args()
    run_table3(args.config, output_data_path=args.output_data_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
