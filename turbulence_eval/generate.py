"""Generate simulated images from a directory or a subject and camera tree."""

from pathlib import Path

import cv2
import numpy as np

from .adapter import CameraSimulator, SimulationOptions

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def write_planned_images(planned, simulator, *, reset_between_groups=False):
    """Generate and yield output paths in temporal-group order."""
    previous_group = None
    for original, output, group in planned:
        if reset_between_groups and group != previous_group:
            simulator.reset()
        previous_group = group
        bgr = cv2.imread(str(original))
        if bgr is None:
            raise ValueError(f"Could not read image: {original}")
        result = simulator.perturb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        success, encoded = cv2.imencode(output.suffix, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        if not success:
            raise OSError(f"Could not encode image: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(encoded.tobytes())
        yield output


def generate_images(
    input_dir: str | Path,
    source_camera: str,
    target_camera: str,
    output_dir: str | Path,
    *,
    subject_layout: bool = False,
    max_images: int = -1,
    seed: int = 42,
    options: SimulationOptions = SimulationOptions(),
    simulator=None,
    reset_between_subjects: bool = False,
) -> list[Path]:
    """Simulate selected source images and save them with their original filenames."""
    source, destination = Path(input_dir), Path(output_dir)
    if not source.is_dir():
        raise NotADirectoryError(source)
    if max_images != -1 and max_images < 1:
        raise ValueError("max_images must be positive or -1 (all images)")
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    rng = np.random.default_rng(seed)
    groups = (
        [(p.name, p / source_camera) for p in sorted(source.iterdir()) if p.is_dir()]
        if subject_layout else [("", source)]
    )
    planned = []
    for subject, image_dir in groups:
        if not image_dir.is_dir():
            continue
        paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
        if max_images != -1 and max_images < len(paths):
            indices = rng.choice(len(paths), size=max_images, replace=False)
            paths = [paths[i] for i in sorted(indices)]
        folder = Path(subject) / f"sim_{source_camera}_to_{target_camera}" if subject_layout else Path()
        planned.extend((path, destination / folder / path.name) for path in paths)
    if not planned:
        raise ValueError(f"No source images found in {source}")
    # Reserve a fresh directory; a second process cannot reuse it accidentally.
    destination.mkdir(parents=True, exist_ok=False)
    simulator = CameraSimulator(source_camera, target_camera, options) if simulator is None else simulator
    return list(write_planned_images(
        ((original, output, original.parent) for original, output in planned), simulator,
        reset_between_groups=reset_between_subjects,
    ))
