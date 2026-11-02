"""Measure image quality and compare simulated images with real images at the target range."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from functools import partial
from importlib import metadata
import json
from pathlib import Path
import random

import cv2
import numpy as np

FREQUENCIES = np.linspace(0, 0.5, 128, endpoint=False)[1:]
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _validate_rgb(image):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an HxWx3 uint8 RGB image")
    if min(image.shape[:2]) < 2:
        raise ValueError("Image dimensions must be at least 2 pixels")


def compute_scalar_metrics(image: np.ndarray, *, include_brisque: bool = True) -> dict[str, float]:
    """Measure sharpness, contrast, and histogram entropy, with optional BRISQUE scoring."""
    _validate_rgb(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_float = gray.astype(np.float32)
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / gray.size
    values = {
        "sharpness": float(cv2.Laplacian(gray_float, cv2.CV_32F).var()),
        "rms_contrast": float(gray_float.std()),
        "entropy": float(-(probabilities * np.log2(probabilities)).sum()),
    }
    if include_brisque:
        try:
            import piq
            import torch
        except ImportError as error:
            raise ImportError("BRISQUE requires torch and piq; install them or use --skip-brisque") from error
        tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div(255).unsqueeze(0)
        with torch.no_grad():
            values["brisque"] = float(piq.brisque(tensor, data_range=1.0))
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("Image quality metrics contain non-finite values")
    return values


def compute_radial_psd(image: np.ndarray) -> np.ndarray:
    """Return normalized radial power on the shared spatial frequency grid."""
    _validate_rgb(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    power = np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))) ** 2
    height, width = gray.shape
    cy, cx = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    max_radius = min(cx, cy)
    mask = radius < max_radius
    counts = np.bincount(radius[mask], minlength=max_radius)
    sums = np.bincount(radius[mask], weights=power[mask], minlength=max_radius)
    radial = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    interpolated = np.interp(FREQUENCIES, np.arange(max_radius) / (2 * max_radius), radial)
    total = interpolated.sum()
    return interpolated / total if total > 0 else np.zeros_like(interpolated)


def psd_distances(simulated: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Table 3's PSD 'L2' is binwise RMSE; KL is KL(simulated || target), in nats."""
    p, q = np.clip(simulated, 1e-10, None), np.clip(target, 1e-10, None)
    return {
        "PSD L2": float(np.sqrt(np.mean((simulated - target) ** 2))),
        "PSD KL (nats)": float(np.sum(p * np.log(p / q))),
    }


def _sample_paths(folder, count, rng):
    paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
    return rng.sample(paths, min(count, len(paths)))


def _load_image(path, resize_512, grayscale, trim_border=0):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if trim_border:
        if min(image.shape[:2]) - 2 * trim_border < 2:
            raise ValueError(f"Border trim leaves fewer than two pixels: {path}")
        image = np.ascontiguousarray(image[trim_border:-trim_border, trim_border:-trim_border])
    if resize_512:
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
    if grayscale:
        image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    return image


def _measure_image(path, *, resize_512, grayscale, trim_border, reference_size, include_brisque):
    """Measure one image independently; callers aggregate results in input order."""
    try:
        image = _load_image(path, resize_512, grayscale, trim_border)
        resized = reference_size is not None and (image.shape[1], image.shape[0]) != reference_size
        if resized:
            image = cv2.resize(image, reference_size, interpolation=cv2.INTER_LINEAR)
        return (compute_scalar_metrics(image, include_brisque=include_brisque), compute_radial_psd(image),
                image.shape[1], image.shape[0], resized)
    except Exception as error:
        raise RuntimeError(f"Quality analysis failed for {path}: {error}") from error


def _write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_plots(output, suffix, frame_metrics, frame_psds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim_labels = [group for group in frame_psds if group not in ("real_source", "real_target")]
    groups = ("real_source", *sim_labels, "real_target")
    labels = {"real_source": "Real source", "simulated": "pyBSM", "real_target": "Real target",
              "pybsm": "pyBSM", "daatsim": "DAATSim", "p2s": "P2S"}
    labels = {group: labels.get(group, group) for group in groups}
    palette = ("#FF9800", "#E53935", "#9C27B0")
    colors = {"real_source": "#2196F3", "real_target": "#4CAF50"}
    colors.update({label: palette[i % len(palette)] for i, label in enumerate(sim_labels)})
    metrics = list(frame_metrics[0]["metrics"])
    figure, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), squeeze=False)
    for axis, metric in zip(axes[0], metrics):
        distributions = [[row["metrics"][metric] for row in frame_metrics if row["group"] == group]
                         for group in groups]
        boxes = axis.boxplot(distributions, patch_artist=True)
        for box, group in zip(boxes["boxes"], groups):
            box.set_facecolor(colors[group])
        axis.set_xticks(range(1, len(groups) + 1), [labels[group] for group in groups], rotation=25)
        axis.set_title(metric.replace("_", " "))
        axis.grid(axis="y", alpha=0.3)
    figure.suptitle(f"Reference-free image quality: {suffix}")
    figure.tight_layout()
    figure.savefig(output / f"metric_boxplots_{suffix}.png", dpi=130)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for group in groups:
        values = np.log10(np.clip(frame_psds[group], 1e-12, None))
        mean, std = values.mean(axis=0), values.std(axis=0)
        axis.plot(FREQUENCIES, 10 ** mean, color=colors[group], label=labels[group])
        axis.fill_between(FREQUENCIES, 10 ** (mean - std), 10 ** (mean + std),
                          color=colors[group], alpha=0.2)
    axis.set(xscale="log", yscale="log", xlabel="Spatial frequency (cycles/pixel)",
             ylabel="Normalized radial power", title=f"Radial PSD: {suffix}\nMean ± 1σ in log space")
    axis.legend()
    axis.grid(alpha=0.3, which="both")
    figure.tight_layout()
    figure.savefig(output / f"psd_comparison_{suffix}.png", dpi=130)
    plt.close(figure)


def analyze_quality(
    stable_frames_dir: str | Path,
    sim_dir: str | Path,
    source_camera: str,
    target_camera: str,
    output_dir: str | Path,
    *,
    num_images: int = 5,
    num_subjects: int | None = None,
    seed: int = 0,
    dist_name: str | None = None,
    include_brisque: bool = True,
    resize_512: bool = False,
    grayscale: bool = False,
    trim_border: int = 0,
    workers: int = 1,
    simulations: dict[str, tuple[str | Path, str]] | None = None,
    image_groups: dict[str, dict[str, list[str | Path]]] | None = None,
) -> dict:
    """Compare real and simulated images by subject and write the metric tables, plots, and
    sampling details."""
    real_root, simulated_root, output = map(Path, (stable_frames_dir, sim_dir, output_dir))
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if num_images < 1 or (num_subjects is not None and num_subjects < 1):
        raise ValueError("num_images and num_subjects must be positive")
    if not isinstance(trim_border, int) or trim_border < 0:
        raise ValueError("trim_border must be a nonnegative integer")
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    for root in (real_root, simulated_root):
        if not root.is_dir():
            raise NotADirectoryError(root)
    simulation_name = dist_name or f"sim_{source_camera}_to_{target_camera}"
    multiple = simulations is not None
    entries = simulations if multiple else {"simulated": (simulated_root, simulation_name)}
    if not entries:
        raise ValueError("At least one simulation is required")
    if any(label in ("real_source", "real_target") or not label or "|" in label for label in entries):
        raise ValueError("Simulation labels must be nonempty and distinct from real_source/real_target")
    entries = {label: (Path(root), name) for label, (root, name) in entries.items()}
    for root, _ in entries.values():
        if not root.is_dir():
            raise NotADirectoryError(root)
    group_names = ("real_source", "real_target", *entries)
    # Labels are folder names, never arbitrary paths outside the supplied roots.
    for name in (source_camera, target_camera, simulation_name, *(name for root, name in entries.values())):
        if not name or name in {".", ".."} or Path(name).name != name:
            raise ValueError(f"Expected a folder name, got {name!r}")
    rng = random.Random(seed)
    selected, skipped = {}, {}
    candidates = (sorted(image_groups) if image_groups is not None else
                  sorted({p.name for root, name in entries.values() for p in root.iterdir()
                          if p.is_dir() and (p / name).is_dir()}))
    for subject in candidates:
        if image_groups is not None:
            supplied = image_groups[subject]
            if set(supplied) != set(group_names):
                raise ValueError(f"Indexed image groups disagree for {subject}")
            pools = {group: sorted(Path(path) for path in supplied[group]) for group in group_names}
            for group, paths in pools.items():
                if len(paths) != len(set(paths)):
                    raise ValueError(f"Duplicate indexed paths for {subject}/{group}")
                for path in paths:
                    if not path.is_file():
                        raise FileNotFoundError(path)
        else:
            folders = {"real_source": real_root / subject / source_camera,
                       "real_target": real_root / subject / target_camera,
                       **{label: root / subject / name for label, (root, name) in entries.items()}}
            pools = {group: sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
                     if folder.is_dir() else [] for group, folder in folders.items()}
        # Check completeness before consuming sampling RNG state.
        missing = [group for group, paths in pools.items() if not paths]
        if missing:
            skipped[subject] = missing
            continue
        selected[subject] = {group: rng.sample(pools[group], min(num_images, len(pools[group]))) for group in group_names}
        if num_subjects is not None and len(selected) >= num_subjects:
            break
    if not selected:
        raise ValueError("No subjects have source, simulated and real target images")
    output.mkdir(parents=True, exist_ok=False)
    per_image, per_subject, agreements, spectra, subject_report = [], [], [], [], {}
    frame_psds = {group: [] for group in group_names}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for subject, groups in selected.items():
            reference_label = next(iter(entries))
            sizes = [_load_image(path, resize_512, grayscale, trim_border).shape[:2] for path in groups[reference_label]]
            reference_size = (int(np.median([w for h, w in sizes])), int(np.median([h for h, w in sizes])))
            means, mean_psds = {}, {}
            subject_report[subject] = {}
            for group, paths in groups.items():
                measurements, psds, resized_count = [], [], 0
                measure = partial(_measure_image, resize_512=resize_512, grayscale=grayscale,
                                  trim_border=trim_border, include_brisque=include_brisque,
                                  reference_size=reference_size if group in ("real_source", "real_target") else None)
                for path, (values, psd, width, height, resized) in zip(paths, pool.map(measure, paths)):
                    resized_count += int(resized)
                    measurements.append(values)
                    psds.append(psd)
                    frame_psds[group].append(psd)
                    per_image.append(dict(subject=subject, group=group, file=str(path.resolve()),
                                          width=width, height=height, metrics=values))
                means[group] = {metric: float(np.mean([row[metric] for row in measurements]))
                                for metric in measurements[0]}
                mean_psds[group] = np.mean(psds, axis=0)
                per_subject.append(dict(subject=subject, group=group, n_images=len(paths), **means[group]))
                subject_report[subject][group] = dict(files=[str(path.resolve()) for path in paths],
                                                    n_images=len(paths), resized_to_match_sim=resized_count,
                                                    means=means[group], mean_psd=mean_psds[group].tolist())
            differences, distances = {}, {}
            for label in entries:
                prefix = f"{label}|" if multiple else ""
                differences.update({prefix + metric: abs(value - means["real_target"][metric])
                                    for metric, value in means[label].items()})
                distances.update({prefix + metric: value for metric, value in
                                  psd_distances(mean_psds[label], mean_psds["real_target"]).items()})
            agreements.append(dict(subject=subject, **differences))
            spectra.append(dict(subject=subject, **distances))
            print(f"{subject}: " + ", ".join(f"{group}={len(paths)}" for group, paths in groups.items()), flush=True)
    averaged = {metric: float(np.mean([row[metric] for row in rows]))
                for rows in (agreements, spectra) for metric in rows[0] if metric != "subject"}
    deviations = {metric: float(np.std([row[metric] for row in rows], ddof=0))
                  for rows in (agreements, spectra) for metric in rows[0] if metric != "subject"}
    by_simulator = ({label: {key.split("|", 1)[1]: value for key, value in averaged.items()
                             if key.startswith(label + "|")} for label in entries}
                    if multiple else {"simulated": averaged})
    summary = by_simulator if multiple else averaged
    by_simulator_std = ({label: {key.split("|", 1)[1]: value for key, value in deviations.items()
                                 if key.startswith(label + "|")} for label in entries}
                        if multiple else {"simulated": deviations})
    summary_std = by_simulator_std if multiple else deviations
    suffix = f"{source_camera}_to_{target_camera}"
    _write_csv(output / f"per_image_metrics_{suffix}.csv",
               [{**{key: value for key, value in row.items() if key != "metrics"}, **row["metrics"]}
                for row in per_image])
    _write_csv(output / f"per_subject_metrics_{suffix}.csv", per_subject)
    _write_csv(output / f"agreement_per_subject_{suffix}.csv", agreements)
    _write_csv(output / f"psd_distance_{suffix}.csv", spectra)
    metric_names = list(next(iter(by_simulator.values())))
    display_columns = list(entries) if multiple else ["pybsm"]
    summary_rows = [dict(metric=metric, **{column: by_simulator[label][metric]
                                          for column, label in zip(display_columns, entries)},
                         **{column + "_std": by_simulator_std[label][metric]
                            for column, label in zip(display_columns, entries)},
                         n_subjects=len(selected)) for metric in metric_names]
    _write_csv(output / f"summary_table_{suffix}.csv", summary_rows)
    table = ["| Metric | " + " | ".join(display_columns) + " |",
             "|---|" + "---:|" * len(entries)]
    table.extend("| " + metric + " | " + " | ".join(
                     f"{row[column]:.4g} ± {row[column + '_std']:.4g}" for column in display_columns) + " |"
                 for metric, row in zip(metric_names, summary_rows))
    table.extend(["", f"Subjects: {len(selected)}. Scalar metrics: mean_subjects(|mean_sim − mean_target|).",
                  "Entries are mean ± population standard deviation across subjects (ddof=0).",
                  "PSD L2 = RMSE; KL = KL(simulated || target), in nats.", ""])
    if trim_border:
        table.extend([f"Excluded {trim_border} native pixel(s) from every edge of every real and synthetic image before resizing and metrics.", ""])
    (output / f"summary_table_{suffix}.md").write_text("\n".join(table))
    versions = {}
    for package in ("numpy", "opencv-python", "matplotlib", "piq", "torch"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    report = dict(source_camera=source_camera, target_camera=target_camera,
                  stable_frames_dir=str(real_root.resolve()), sim_dir=str(simulated_root.resolve()),
                  dist_name=simulation_name, seed=seed, num_images=num_images, num_subjects=num_subjects,
                  include_brisque=include_brisque, resize_512=resize_512, grayscale=grayscale,
                  trim_border=trim_border, trim_order="native_before_resize", workers=workers,
                  n_subjects=len(selected), skipped_subjects=skipped, summary=summary,
                  by_simulator=by_simulator, summary_std=summary_std,
                  by_simulator_std=by_simulator_std, std_ddof=0, std_unit="subject",
                  simulations={label: dict(root=str(root.resolve()), dist_name=name)
                               for label, (root, name) in entries.items()},
                  size_reference_simulator=next(iter(entries)),
                  sampling_order=list(group_names), resize_interpolation="INTER_LINEAR",
                  frequencies=FREQUENCIES.tolist(), subjects=subject_report, versions=versions)
    _save_plots(output, suffix, per_image, frame_psds)
    (output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("\n".join(table))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-frames-dir", required=True)
    parser.add_argument("--sim-dir", required=True)
    parser.add_argument("--source-camera", required=True)
    parser.add_argument("--target-camera", required=True)
    parser.add_argument("--output-dir", required=True, help="New directory; existing directories are refused.")
    parser.add_argument("--dist-name", help="Simulation subfolder; default: sim_<source>_to_<target>.")
    parser.add_argument("--num-images", type=int, default=5, help="Frames per subject per group (default: 5).")
    parser.add_argument("--num-subjects", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-brisque", action="store_true", help="Explicitly omit BRISQUE.")
    parser.add_argument("--resize-512", action="store_true", help="Resize all frames to 512x512 before analysis.")
    parser.add_argument("--grayscale", action="store_true", help="Convert frames to grayscale RGB before analysis.")
    parser.add_argument("--trim-border", type=int, default=0, help="Native pixels excluded from each edge, in all groups, before resizing.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel frame measurements; sampling and aggregation stay ordered.")
    args = parser.parse_args()
    analyze_quality(args.stable_frames_dir, args.sim_dir, args.source_camera, args.target_camera,
                    args.output_dir, num_images=args.num_images, num_subjects=args.num_subjects,
                    seed=args.seed, dist_name=args.dist_name, include_brisque=not args.skip_brisque,
                    resize_512=args.resize_512, grayscale=args.grayscale, trim_border=args.trim_border, workers=args.workers)


if __name__ == "__main__":
    main()
