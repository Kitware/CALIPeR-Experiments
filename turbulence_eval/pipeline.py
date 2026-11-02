"""Generate synthetic crops or images for the configured cameras and turbulence strengths."""

from configparser import ConfigParser
from contextlib import nullcontext
import csv
import gc
import hashlib
import json
import math
from importlib import metadata
from pathlib import Path
import platform
import re

from .cameras import camera_configs, camera_label, metres_per_pixel, range_token
from .inputs import Selection, select_inputs, sources_for_variant


_SCHEMA = {
    "paths": {"chip_root", "chip_index", "partition_file", "corpus_dir", "split_json",
              "input_dir", "p2s_data_dir", "output_data_path", "output_dir"},
    "input": {"mode", "partition", "collections", "protocol_version", "source_role", "subjects",
              "max_images", "num_subjects", "deduplicate", "purpose"},
    "experiment": {"source_camera", "sample_seed", "noise_seed", "device"},
    "calibration": {"brc1", "brc2"},
    "pybsm": {"bridge", "deconvolve", "resample", "distortion"},
    "daatsim": {"size", "alpha_tilt", "alpha_blur"},
    "p2s": {"size"},
    "analysis": {"num_images", "seed", "include_brisque", "resize_512", "grayscale", "trim_border", "workers"},
}
_VARIANT_KEYS = {"simulator", "target_camera", "cn2", "r0"}


def _words(value):
    return [v for v in re.split(r"[,\s]+", value.strip()) if v]


def load_generation_config(config_file):
    """New configs default to indexed chips. Legacy INIs dispatch elsewhere."""
    path = Path(config_file).resolve()
    config = ConfigParser(interpolation=None)
    with path.open() as stream:
        config.read_file(stream)
    if config.defaults():
        raise ValueError("Generation INIs do not support [DEFAULT]; use explicit sections")
    for name in ("paths", "experiment"):
        if not config.has_section(name):
            raise ValueError(f"Missing INI section [{name}]")
    for section in config.sections():
        allowed = _VARIANT_KEYS if section.startswith("variant:") else _SCHEMA.get(section)
        if allowed is None or set(config[section]) - allowed:
            raise ValueError(f"Unknown generation section/settings: [{section}] {sorted(set(config[section]) - (allowed or set()))}")
    for section in _SCHEMA:
        if not config.has_section(section):
            config.add_section(section)
    paths = {}
    for key, value in config["paths"].items():
        if value.strip():
            paths[key] = str((path.parent / Path(value.strip()).expanduser()).resolve())
    inp, exp = config["input"], config["experiment"]
    mode = inp.get("mode", "chips")
    if mode not in ("chips", "images", "subject-tree"):
        raise ValueError("input.mode must be chips, images, or subject-tree")
    partition = inp.get("partition", "")
    purpose = inp.get("purpose", "reid")
    if purpose not in ("reid", "quality"):
        raise ValueError("input.purpose must be reid or quality")
    if purpose == "quality" and mode != "chips":
        raise ValueError("quality purpose requires indexed chips")
    required = {"output_data_path"}
    if mode == "chips":
        if partition not in (("train", "test", "all") if purpose == "quality" else ("train", "test")):
            raise ValueError("Chip generation requires explicit input.partition = train or test")
        required |= {"chip_root", "chip_index", "partition_file"}
        if purpose == "reid":
            required.add("corpus_dir" if partition == "train" else "split_json")
    else:
        required.add("input_dir")
    if required - set(paths):
        raise ValueError(f"Missing generation paths: {sorted(required - set(paths))}")
    collections = _words(inp.get("collections", "brc1"))
    calibration = dict(config["calibration"]) or {"brc1": "brc1"}
    for profile in calibration.values():
        camera_configs(profile)
    if mode != "chips" and set(collections) - set(calibration):
        raise ValueError("Explicit calibration profile required for image-directory collections")
    profiles = {calibration.get(collection, "brc1") for collection in collections}
    if mode != "chips" and len(profiles) != 1:
        raise ValueError("Image-directory generation requires one calibration profile")
    profile = calibration.get(collections[0], "brc1") if collections else "brc1"
    source = camera_label(exp["source_camera"], profile)
    for selected_profile in profiles:
        camera_label(source, selected_profile)
    settings = dict(
        config_file=str(path), paths=paths, source_camera=source, source_token=range_token(source),
        sample_seed=exp.getint("sample_seed", 42),
        noise_seed=exp.getint("noise_seed") if exp.get("noise_seed", "").strip() else None,
        device=exp.get("device", "auto"),
        input=dict(mode=mode, partition=partition, purpose=purpose, collections=collections,
                   protocol_version=inp.get("protocol_version", "v4"), source_role=inp.get("source_role", "query"),
                   subjects=_words(inp.get("subjects", "")), max_images=inp.getint("max_images", -1),
                   num_subjects=inp.getint("num_subjects") if inp.get("num_subjects", "").strip() else None,
                   deduplicate=inp.getboolean("deduplicate", True)),
        calibration=calibration, calibration_profile=profile,
        pybsm={key: config["pybsm"].getboolean(key, fallback=key in ("bridge", "deconvolve"))
               for key in _SCHEMA["pybsm"]},
        daatsim=dict(size=config["daatsim"].getint("size", 512),
                     alpha_tilt=config["daatsim"].getfloat("alpha_tilt", 0.2),
                     alpha_blur=config["daatsim"].getfloat("alpha_blur", 0.2)),
        p2s_size=config["p2s"].get("size", "auto"), variants=[],
        analysis=dict(num_images=config["analysis"].getint("num_images", 5),
                      trim_border=config["analysis"].getint("trim_border", 0),
                      workers=config["analysis"].getint("workers", 1),
                      seed=config["analysis"].getint("seed", 0),
                      **{key: config["analysis"].getboolean(key, fallback=key == "include_brisque")
                         for key in ("include_brisque", "resize_512", "grayscale")}),
    )
    options = settings["input"]
    if settings["analysis"]["num_images"] < 1:
        raise ValueError("Analysis num_images must be positive")
    if settings["analysis"]["trim_border"] < 0 or settings["analysis"]["workers"] < 1:
        raise ValueError("Analysis trim_border must be nonnegative and workers must be positive")
    if options["max_images"] != -1 and options["max_images"] < 1:
        raise ValueError("max_images must be positive or -1")
    if options["num_subjects"] is not None and options["num_subjects"] < 1:
        raise ValueError("num_subjects must be positive")
    if len(set(options["subjects"])) != len(options["subjects"]):
        raise ValueError("Duplicate requested subjects")
    if not options["collections"] or set(options["collections"]) - {"brc1", "brc2"}:
        raise ValueError("collections must contain brc1 and/or brc2")
    if options["source_role"] != "query":
        raise ValueError("Evaluation generation currently requires source_role = query")
    if options["protocol_version"] != "v4":
        raise ValueError("Only the v4 chip schema is supported")
    if settings["p2s_size"] not in ("auto", "128", "256", "512", "1024"):
        raise ValueError("P2S size must be auto, 128, 256, 512, or 1024")
    daat = settings["daatsim"]
    if daat["size"] < 32 or any(not 0 <= daat[key] <= 1 for key in ("alpha_tilt", "alpha_blur")):
        raise ValueError("Invalid DAATSim size or temporal coefficients")
    for section in config.sections():
        if not section.startswith("variant:"):
            continue
        name, values = section[len("variant:"):], config[section]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"Invalid variant name: {name!r}")
        simulator = values["simulator"]
        if simulator not in ("pybsm", "daatsim", "p2s"):
            raise ValueError(f"Unknown simulator: {simulator}")
        target = camera_label(values["target_camera"], profile)
        for selected_profile in profiles:
            camera_label(target, selected_profile)
        if target == source:
            raise ValueError("Source and target camera must differ")
        strength = "cn2" if simulator == "pybsm" else "r0"
        if ({"cn2", "r0"} - {strength}) & set(values):
            raise ValueError(f"[{section}] uses {strength}, not the other simulator's strength parameter")
        value = values.getfloat(strength)
        if value is None or not math.isfinite(value) or value <= 0:
            raise ValueError(f"[{section}] requires finite positive {strength}")
        settings["variants"].append(dict(name=name, simulator=simulator, target_camera=target,
                                         target_token=range_token(target), **{strength: value}))
    if not settings["variants"]:
        raise ValueError("Provide at least one [variant:<name>] section")
    return settings


def _clean_json(value):
    if isinstance(value, dict):
        return {k: _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _write_json(path, data):
    with path.open("x") as stream:
        json.dump(_clean_json(data), stream, indent=2, allow_nan=False)
        stream.write("\n")


def _fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def output_path(output, variant, frame, mode):
    if mode == "chips":
        return output / "images" / variant["name"] / frame.collection / Path(frame.relpath).with_suffix("") / f"{frame.frame:06d}.jpg"
    return output / "images" / variant["name"] / frame.relpath


def make_simulator(settings, variant, selection, output):
    """Construct only the requested simulator and load only its dependencies."""
    profile = settings["calibration"].get(selection.sources[0].collection, settings["calibration_profile"])
    configs = camera_configs(profile)
    source = camera_label(settings["source_camera"], profile)
    target = camera_label(variant["target_camera"], profile)
    distance = configs[target]["_camera_distance_m"]
    if variant["simulator"] == "pybsm":
        from dataclasses import asdict
        from .adapter import CameraSimulator, SimulationOptions
        options = SimulationOptions(cn2=variant["cn2"], noise_seed=settings["noise_seed"], **settings["pybsm"])
        simulator = CameraSimulator(source, target, options, profile=profile)
        effective = dict(options=asdict(options), source_config=configs[source], target_config=configs[target],
                         source_metres_per_pixel=metres_per_pixel(source, profile))
    elif variant["simulator"] == "daatsim":
        from .simulators import DAATSimulator
        simulator = DAATSimulator(r0=variant["r0"], distance=distance, device=settings["device"],
                                  seed=settings["noise_seed"], **settings["daatsim"])
        effective = dict(simulator.settings, reset_between_tracklets=True)
    else:
        import cv2
        from .simulators import P2SSimulator
        size = settings["p2s_size"]
        if size == "auto":
            image = cv2.imread(selection.sources[0].path)
            if image is None:
                raise ValueError(f"Unreadable source image: {selection.sources[0].path}")
            size = min((128, 256, 512, 1024), key=lambda n: abs(n - min(image.shape[:2])))
        simulator = P2SSimulator(source, aperture=configs[source]["D"], distance=distance, r0=variant["r0"], size=int(size),
                                 data_dir=settings["paths"]["p2s_data_dir"], cache_dir=output / "cache" / "p2s",
                                 device=settings["device"], seed=settings["noise_seed"])
        effective = simulator.settings
    return simulator, dict(effective, calibration_profile=profile)


def run_generation(config_file, *, output_data_path=None, preview=False):
    settings = load_generation_config(config_file)
    if output_data_path is not None:
        settings["paths"]["output_data_path"] = str(Path(output_data_path).resolve())
    output = Path(settings["paths"]["output_data_path"])
    if output.exists() and not preview:
        raise FileExistsError(f"Output directory already exists: {output}")
    selection = select_inputs(settings)
    summary = dict(selection.summary(), variants={v["name"]: len(sources_for_variant(selection, v)) for v in settings["variants"]})
    if any(count == 0 for count in summary["variants"].values()):
        raise ValueError("A configured target has no subjects with source and real target chips")
    print(json.dumps(summary, indent=2))
    if selection.references and not selection.references["gallery"]:
        raise ValueError("No usable gallery chips remain in the requested split/collections")
    if preview:
        return summary
    if any(v["simulator"] == "p2s" for v in settings["variants"]):
        if "p2s_data_dir" not in settings["paths"]:
            raise ValueError("P2S generation requires paths.p2s_data_dir")
        for name in ("P2S_model.pt", "dictionary.npy", "R-corr_-0.1.npy"):
            asset = Path(settings["paths"]["p2s_data_dir"]) / name
            if not asset.is_file():
                raise FileNotFoundError(asset)
    from .exports import (TRAIN_FIELDS, build_kwcoco, evaluation_registries,
                          synthetic_tracklet, write_evaluation_references)
    from .generate import write_planned_images

    output.mkdir(parents=True, exist_ok=False)
    (output / "experiment.ini").write_text(Path(config_file).read_text())
    _write_json(output / "settings.json", settings)
    _write_json(output / "selection.json", dict(summary=summary, sources=[r.to_dict() for r in selection.sources]))
    if selection.real_frames:
        with (output / "real_chips.jsonl").open("x") as stream:
            for row in selection.real_frames:
                stream.write(json.dumps(row.to_dict()) + "\n")
    fingerprints = {"config": _fingerprint(settings["config_file"])}
    if settings["input"]["mode"] == "chips":
        keys = ["chip_index", "partition_file"]
        if settings["input"]["purpose"] == "reid" and settings["input"]["partition"] == "test":
            keys.append("split_json")
        for key in keys:
            fingerprints[key] = _fingerprint(settings["paths"][key])
    if settings["input"]["mode"] == "chips" and settings["input"]["purpose"] == "reid" and settings["input"]["partition"] == "train":
        for name in ("train_chips.csv", "train_tracklets.csv", "train_classes.json", "train_manifest.csv"):
            path = Path(settings["paths"]["corpus_dir"]) / name
            if path.is_file():
                fingerprints[name] = _fingerprint(path)
    subjects, cameras = evaluation_registries(selection)
    eval_files = write_evaluation_references(output / "eval", selection, subjects, cameras)
    effective, counts = {}, {}
    training = (settings["input"]["mode"] == "chips" and settings["input"]["purpose"] == "reid"
                and settings["input"]["partition"] == "train")
    with (output / "chips.jsonl").open("x") as manifest:
        with ((output / "extra_chips.csv").open("x", newline="") if training else nullcontext(None)) as train_stream:
            writer = csv.DictWriter(train_stream, fieldnames=TRAIN_FIELDS) if training else None
            if writer:
                writer.writeheader()
            for variant in settings["variants"]:
                print(f"Generating {variant['name']} ({variant['simulator']})", flush=True)
                sources = sources_for_variant(selection, variant)
                groups = {}
                for frame in sources:
                    profile = settings["calibration"].get(frame.collection, settings["calibration_profile"])
                    groups.setdefault(profile, []).append(frame)
                per_profile = {}
                count = 0
                for profile, group in groups.items():
                    variant_selection = Selection(group, selection.references, selection.excluded, selection.real_frames)
                    simulator, per_profile[profile] = make_simulator(settings, variant, variant_selection, output)
                    planned = ((Path(r.path), output_path(output, variant, r, settings["input"]["mode"]),
                                (r.collection, r.tracklet_id)) for r in group)
                    generated = write_planned_images(planned, simulator, reset_between_groups=variant["simulator"] == "daatsim")
                    for frame, path in zip(group, generated):
                        record = dict(variant, source=frame.to_dict(), chip_path=str(path),
                                      calibration_profile=profile,
                                      source_camera=camera_label(settings["source_camera"], profile))
                        record["target_camera"] = camera_label(variant["target_camera"], profile)
                        manifest.write(json.dumps(record, allow_nan=False) + "\n")
                        if writer:
                            writer.writerow(dict(chip_path=str(path), tracklet_id=synthetic_tracklet(frame, variant["name"]),
                                                 src_tracklet_id=frame.tracklet_id, canonical_id=frame.canonical_id,
                                                 tok=variant["target_token"], collection=frame.collection))
                        count += 1
                        if count % 100 == 0 or count == len(sources):
                            print(f"{variant['name']}: {count}/{len(sources)} images", flush=True)
                    del simulator
                    gc.collect()
                effective[variant["name"]] = (next(iter(per_profile.values())) if len(per_profile) == 1
                                               else dict(by_profile=per_profile))
                counts[variant["name"]] = count
                if selection.references:
                    token = f"{settings['source_token']}_sim_{variant['target_token']}_{variant['name']}"
                    path = output / "eval" / f"query_{token}.kwcoco.json"
                    _write_json(path, build_kwcoco(selection.sources, subjects, cameras, token=token,
                                                  outputs=lambda r: output_path(output, variant, r, settings["input"]["mode"])))
                    eval_files.append(str(path))
    generation = dict(format_version=2, kind="synthetic_chips" if settings["input"]["mode"] == "chips" else "synthetic_images",
                      settings=settings, summary=summary, counts=counts, fingerprints=fingerprints,
                      effective_simulators=effective, evaluation_files=eval_files,
                      manifest="chips.jsonl", training_manifest="extra_chips.csv" if training else None)
    generation["versions"] = {"python": platform.python_version()}
    for package in ("pybsm", "numpy", "opencv-python", "torch", "nrtk"):
        try:
            generation["versions"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            generation["versions"][package] = None
    _write_json(output / "generation.json", generation)
    return generation
