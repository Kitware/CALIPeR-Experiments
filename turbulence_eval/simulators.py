"""DAATSim and P2S adapters for the Table 3 generation pipeline."""

from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch

from .cameras import CALIPER_CAMERA_CONFIGS


@contextmanager
def _seeded(seed, device):
    """Seed a call without changing the caller's Torch RNG state."""
    if seed is None:
        yield
        return
    device = torch.device(device)
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        if devices:
            torch.cuda.set_rng_state(torch.Generator(device=device).manual_seed(seed).get_state(), device)
        yield


def _device(value):
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)


def _validate(image):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2:
        raise ValueError("Expected an HxWx3 uint8 RGB image with dimensions >= 2")


class DAATSimulator:
    """Simulate uniform-depth turbulence on DAATSim’s native square grid."""

    def __init__(self, *, r0=0.035, distance=1000.0, size=512, device="auto", seed=None,
                 alpha_tilt=0.2, alpha_blur=0.2):
        from types import SimpleNamespace
        from ._vendor import daatsim_config
        from ._vendor.daatsim import DAATSimPerturber

        if not np.isfinite(r0) or r0 <= 0 or not np.isfinite(distance) or distance <= 0 or size < 32:
            raise ValueError("r0/distance must be finite and positive; DAATSim size must be >=32")
        if not 0 <= alpha_tilt <= 1 or not 0 <= alpha_blur <= 1:
            raise ValueError("DAATSim temporal coefficients must be in [0, 1]")
        cfg = SimpleNamespace(**{k: v for k, v in vars(daatsim_config).items() if not k.startswith('_')})
        cfg.SIZE_D_PARAM = size
        cfg.L_PROP_PARAM = distance
        cfg.ALPHA_TILT_PARAM = alpha_tilt
        cfg.ALPHA_BLUR_PARAM = alpha_blur
        self.device, self.seed, self.frame = _device(device), seed, 0
        self.model = DAATSimPerturber(r0=r0, cfg=cfg, device=self.device, use_depth=False,
                                     L_prop=distance, alpha_tilt=alpha_tilt, alpha_blur=alpha_blur)
        self.settings = dict(r0=r0, distance=distance, size=size, device=str(self.device),
                             seed=seed, alpha_tilt=alpha_tilt, alpha_blur=alpha_blur,
                             use_depth=False, physics=vars(cfg), output_geometry='simulation_grid')

    def perturb(self, image):
        _validate(image)
        with _seeded(None if self.seed is None else self.seed + self.frame, self.device), torch.no_grad():
            result = self.model.perturb(image)
        self.frame += 1
        return result

    def reset(self):
        """Clear temporal state between subjects; keep the random sequence advancing."""
        self.model.reset()


class P2SSimulator:
    """Simulate turbulence on padded images and restore their source dimensions."""

    def __init__(self, source_camera, *, distance, data_dir, cache_dir, r0=0.1,
                 size=1024, device="auto", seed=None, aperture=None):
        from ._vendor.p2s import Simulator
        from ._vendor.p2s_stats import tilt_mat

        if size not in (128, 256, 512, 1024):
            raise ValueError("P2S size must be 128, 256, 512 or 1024")
        diameter = CALIPER_CAMERA_CONFIGS[source_camera]["D"] if aperture is None else aperture
        if any(not np.isfinite(x) or x <= 0 for x in (r0, distance, diameter)):
            raise ValueError("P2S r0, distance and aperture must be finite and positive")
        assets, cache = Path(data_dir), Path(cache_dir)
        for name in ("P2S_model.pt", "dictionary.npy", "R-corr_-0.1.npy"):
            if not (assets / name).is_file():
                raise FileNotFoundError(assets / name)
        cache.mkdir(parents=True, exist_ok=True)
        # Float.hex preserves full parameter precision instead of rounding cache keys.
        key = f"N{size}_D{float(diameter).hex()}_r0{float(r0).hex()}_L{float(distance).hex()}"
        tilt_path = cache / f"{key}.npz"
        if tilt_path.exists():
            with np.load(tilt_path) as stored:
                tilt = {"s_half": stored["s_half"], "const": float(stored["const"])}
        else:
            print(f"Computing P2S tilt matrix: size={size}, D={diameter}, r0={r0}, L={distance}", flush=True)
            tilt = tilt_mat(size, diameter, r0, distance)
            with tilt_path.open("xb") as stream:
                np.savez(stream, **tilt)
        self.device, self.seed, self.frame, self.size = _device(device), seed, 0, size
        with _seeded(seed, self.device):
            self.model = Simulator(diameter / r0, size, data_path=str(assets),
                                   device=str(self.device), tilt_data=tilt).to(self.device, dtype=torch.float32).eval()
        self.settings = dict(source_camera=source_camera, r0=r0, distance=distance,
                             aperture=diameter, size=size, device=str(self.device), seed=seed,
                             data_dir=str(assets.resolve()), tilt_cache=str(tilt_path.resolve()), corr=-0.1)

    def perturb(self, image):
        _validate(image)
        height, width = image.shape[:2]
        side = max(height, width)
        top, left = (side - height) // 2, (side - width) // 2
        square = np.pad(image, ((top, side - height - top), (left, side - width - left), (0, 0)), mode="reflect")
        resized = cv2.resize(square, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.tensor(resized.transpose(2, 0, 1), device=self.device, dtype=torch.float32) / 255.0
        with _seeded(None if self.seed is None else self.seed + self.frame, self.device), torch.no_grad():
            result = self.model(tensor).detach().cpu().numpy()
        self.frame += 1
        pixels = np.clip(result.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        padded = cv2.resize(pixels, (side, side), interpolation=cv2.INTER_LINEAR)
        return padded[top:top + height, left:left + width]
