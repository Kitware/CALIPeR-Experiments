"""Load Figure 6 model adapters and cache their tracklet features."""

from contextlib import contextmanager, nullcontext
import hashlib
import importlib
from importlib import metadata
import inspect
from pathlib import Path
import sys
import tempfile
import os

import cv2
import numpy as np

from .reid_data import digest_json, sha256_file


def versions():
    result = {"python": sys.version.split()[0]}
    for name in ("numpy", "torch", "torchvision", "opencv-python", "Pillow"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def resolve_device(value):
    if value != "auto":
        return value
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@contextmanager
def adapter_imports(path):
    """Keep optional adapter dependencies available for lazy inference imports."""
    if path is not None:
        sys.path.insert(0, path)
    try:
        yield
    finally:
        if path is not None:
            sys.path.remove(path)


def load_model(name, settings, *, device, batch_size):
    """Load the configured image or video encoder from its factory and checkpoint."""
    files = [settings["checkpoint"], *settings["assets"]]
    fingerprints = {}
    for path in files:
        if not Path(path).is_file():
            raise FileNotFoundError(f"{name}: required model asset is missing: {path}")
        fingerprints[path] = sha256_file(path)
    search = settings["python_path"]
    if search is not None:
        if not Path(search).is_dir():
            raise NotADirectoryError(f"{name}: adapter python_path is missing: {search}")
        sys.path.insert(0, search)
    try:
        module_name, function_name = settings["factory"].split(":")
        try:
            module = importlib.import_module(module_name)
            factory = getattr(module, function_name)
        except (ImportError, AttributeError) as error:
            raise ImportError(f"{name}: cannot import model factory {settings['factory']}: {error}") from error
        code = inspect.getsourcefile(factory)
        if code:
            fingerprints[str(Path(code).resolve())] = sha256_file(code)
        else:
            raise ValueError(f"{name}: model factory must have a Python source file for provenance")
        for entry in settings["code_paths"]:
            root = Path(entry)
            paths = sorted(root.rglob('*.py')) if root.is_dir() else [root]
            if not paths:
                raise ValueError(f"{name}: code_paths entry has no Python source files: {root}")
            for p in paths:
                fingerprints[str(p)] = sha256_file(p)
        identity = dict(settings=settings, files=fingerprints, device=device,
                        batch_size=batch_size, versions=versions(), pooling="l2-mean-l2-v1")
        signature = digest_json(identity)
        try:
            encoder = factory(checkpoint=settings["checkpoint"], device=device,
                              batch_size=batch_size, **settings["options"])
        except Exception as error:
            raise RuntimeError(f"{name}: model factory failed: {error}") from error
        if not callable(encoder) and not callable(getattr(encoder, "encode", None)):
            raise TypeError(f"{name}: factory must return a callable encoder or an object with encode(crops)")
        return encoder, signature, identity
    finally:
        if search is not None:
            sys.path.remove(search)


def normalise(features):
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    if not np.isfinite(features).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Model features must be finite and have nonzero L2 norm")
    return features / norms


def _read_crop(frame):
    data = Path(frame["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != frame["sha256"]:
        raise ValueError(f"Image changed after preparation: {frame['path']}")
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot decode chip: {frame['path']}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _as_features(value, expected):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    features = np.asarray(value, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != expected or features.shape[1] == 0:
        raise ValueError(f"Expected encoder features shaped ({expected}, D), got {features.shape}")
    return normalise(features)


def tracklet_feature(tracklet, encoder, settings, signature, cache_dir, batch_size):
    key = digest_json(dict(model=signature, frames=[(f["frame"], f["sha256"]) for f in tracklet["frames"]]))
    destination = Path(cache_dir) / signature / (key + ".npy")
    if destination.exists():
        vector = np.load(destination, allow_pickle=False)
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1):
            raise ValueError(f"Invalid cached tracklet feature: {destination}")
        return vector
    encode = encoder if callable(encoder) else encoder.encode
    try:
        import torch
        context = torch.inference_mode
    except ImportError:
        context = nullcontext
    with adapter_imports(settings.get("python_path")), context():
        if settings["input_kind"] == "video":
            frames = [_read_crop(f) for f in tracklet["frames"]]
            features = _as_features(encode(frames), 1)
        else:
            batches = []
            for start in range(0, len(tracklet["frames"]), batch_size):
                frames = [_read_crop(f) for f in tracklet["frames"][start:start + batch_size]]
                batches.append(_as_features(encode(frames), len(frames)))
            features = np.concatenate(batches)
    vector = normalise(features.mean(axis=0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as stream:
        tmp = Path(stream.name)
        try:
            np.save(stream, vector, allow_pickle=False)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, destination)
    return vector
