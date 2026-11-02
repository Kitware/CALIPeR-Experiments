"""Simulate changes in image quality between two ground cameras with pyBSM."""

import copy
from dataclasses import dataclass
import warnings

import numpy as np
from pybsm import utils as atmosphere_utils
from pybsm.otf.functional import otf_to_psf
from pybsm.simulation import Scenario, Sensor, SystemOTFSimulator
from scipy.interpolate import interp1d

from .cameras import camera_configs, camera_label, metres_per_pixel


@dataclass(frozen=True)
class SimulationOptions:
    """Camera-model options; sensor noise and frame sampling use separate seeds."""

    cn2: float | None = None
    bridge: bool = True
    deconvolve: bool = True
    resample: bool = True
    distortion: bool = True
    regularization: float = 0.001
    noise_seed: int | None = None

    def __post_init__(self):
        if self.cn2 is not None and (not np.isfinite(self.cn2) or self.cn2 <= 0):
            raise ValueError("cn2 must be finite and positive")
        if not np.isfinite(self.regularization) or self.regularization <= 0:
            raise ValueError("regularization must be finite and positive")


class _GroundScenario(Scenario):
    """Interpolate atmospheric parameters for low camera heights."""

    @property
    def atm(self):
        if not self._interp or not 2.0 < self.altitude < 32.5:
            return super().atm
        if self._atm is None:
            ranges = atmosphere_utils._get_ground_range_array(max_ground_range=301e3)
            if not ranges[0] <= self.ground_range <= ranges[-1]:
                raise IndexError("Ground range outside atmosphere database")
            index = np.searchsorted(ranges, self.ground_range)
            low = ranges[max(0, index - 1)]
            high = ranges[index]

            def at_height(height):
                lower = atmosphere_utils.load_database_atmosphere_no_interp(
                    altitude=height, ground_range=low, ihaze=self.ihaze)
                upper = atmosphere_utils.load_database_atmosphere_no_interp(
                    altitude=height, ground_range=high, ihaze=self.ihaze)
                weight = (self.ground_range - low) / (high - low) if high != low else 0.0
                return lower * (1 - weight) + upper * weight

            weight = (self.altitude - 2.0) / (32.5 - 2.0)
            self._atm = at_height(2.0) * (1 - weight) + at_height(32.5) * weight
        return self._atm


class _TransferSimulator(SystemOTFSimulator):
    """Supply a source-to-target PSF to pyBSM's existing simulation pipeline."""

    def __init__(self, *, source_otf=None, regularization=0.001, **kwargs):
        self.source_otf = source_otf
        self.regularization = regularization
        super().__init__(**kwargs)

    def apply_convolution(self, image, psf):
        """Map constant frames to midpoint reflectance before convolving."""
        if self._use_reflectance and image.min() == image.max():
            r1, r2 = self._reflectance_range
            reflectance = np.full(image.shape, (float(r1) + float(r2)) / 2.0)
            photoelectrons = self._reflect_to_photoelectrons(np.clip(reflectance, 0, 1))
            # A shallow copy bypasses a second reflectance conversion without
            # changing the simulator instance or pyBSM's installed classes.
            convolver = copy.copy(self)
            convolver._use_reflectance = False
            return SystemOTFSimulator.apply_convolution(convolver, photoelectrons, psf)
        return super().apply_convolution(image, psf)

    def apply_noise(self, image):
        """Reject non-finite Poisson inputs before entering pyBSM's noise kernels."""
        if self.add_noise and not np.isfinite(image).all():
            raise RuntimeError("Cannot apply sensor noise to non-finite photoelectron values")
        return super().apply_noise(image)

    def _get_psf_cached(self, gsd=None, use_default=False):
        if self.source_otf is None:
            return super()._get_psf_cached(gsd=gsd, use_default=use_default)
        key = ("transfer", round(gsd, 6) if gsd is not None else None)
        if key not in self._psf_cache:
            source = self.source_otf
            transfer = self._compute_otf() * source / (source ** 2 + self.regularization)
            dx_out = (
                2.0 * np.arctan(gsd / 2.0 / self._slant_range)
                if gsd is not None else self._ifov
            )
            self._psf_cache[key] = otf_to_psf(otf=transfer, df=self._df, dx_out=dx_out)
        return self._psf_cache[key]


def _build_simulator(camera, options, source_otf=None, *, profile="brc1"):
    cfg = camera_configs(profile)[camera_label(camera, profile)]
    sensor_keys = ("D", "f", "p_x", "int_time", "read_noise", "max_n",
                   "bit_depth", "s_x", "s_y")
    sensor = Sensor(
        name="Sensor",
        # Optical transmission band: 0.50–0.66 micrometres.
        opt_trans_wavelengths=np.array([0.58 - 0.08, 0.58 + 0.08]) * 1e-6,
        **{key: cfg[key] for key in sensor_keys},
    )
    scenario = _GroundScenario(
        name="Scenario",
        ihaze=cfg["ihaze"],
        altitude=cfg["altitude"],
        ground_range=cfg["ground_range"],
        interp=cfg["interp"],
        # Apply a Cn2 override when constructing both source and target models.
        cn2_at_1m=cfg["cn2_at_1m"] if options.cn2 is None else options.cn2,
    )
    simulator = _TransferSimulator(
        sensor=sensor, scenario=scenario, add_noise=True,
        use_reflectance=True, reflectance_range=np.array([0.05, 0.5]),
        rng=options.noise_seed, do_resample=options.resample,
        source_otf=source_otf, regularization=options.regularization,
    )
    # Set sensor wavefront error and invalidate PSFs before evaluating the OTF.
    simulator.sensor.pv = cfg.get("_pv", 0.0)
    simulator._psf_cache.clear()
    return simulator


class CameraSimulator:
    """Transform RGB frames between a fixed source and target camera."""

    def __init__(self, source_camera: str, target_camera: str,
                 options: SimulationOptions = SimulationOptions(), *, profile: str = "brc1"):
        source_camera = camera_label(source_camera, profile)
        target_camera = camera_label(target_camera, profile)
        self.profile = profile
        self.source_metres_per_pixel = metres_per_pixel(source_camera, profile)
        self.source_camera = source_camera
        self.target_camera = target_camera
        self.options = options
        source_otf = None
        if options.deconvolve:
            source_otf = _build_simulator(source_camera, options, profile=profile)._compute_otf()
        self._simulator = _build_simulator(target_camera, options, source_otf, profile=profile)
        self._inverse = None
        if options.bridge:
            fwd = self._simulator._reflect_to_photoelectrons
            if np.all(np.diff(fwd.y) > 0):
                self._inverse = interp1d(
                    fwd.y, fwd.x, bounds_error=False,
                    fill_value=(float(fwd.x[0]), float(fwd.x[-1])),
                )
            else:
                warnings.warn(
                    "Radiometric table is not strictly monotonic; using min-max normalization.",
                    stacklevel=2,
                )
        self._distortion = None
        if options.distortion:
            from nrtk.impls.perturb_image.optical import RadialDistortionPerturber

            self._distortion = RadialDistortionPerturber(
                k=camera_configs(profile)[target_camera]["_distortion_k"]
            )

    def perturb(self, image: np.ndarray) -> np.ndarray:
        """Return the simulated RGB image; resampling can change its dimensions."""
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an HxWx3 uint8 RGB image")
        _, _, simulated = self._simulator.simulate_image(image, self.source_metres_per_pixel)
        if not np.isfinite(simulated).all():
            raise RuntimeError("pyBSM returned non-finite pixel values")
        if self._inverse is not None:
            fwd = self._simulator._reflect_to_photoelectrons
            reflectance = self._inverse(np.clip(simulated, float(fwd.y.min()), float(fwd.y.max())))
            low, high = float(fwd.x[0]), float(fwd.x[-1])
            pixels = (reflectance - low) / (high - low + 1e-12) * 255.0
            result = np.clip(pixels, 0, 255).astype(np.uint8)
        else:
            low, high = float(simulated.min()), float(simulated.max())
            if high > low:
                simulated = (simulated - low) / (high - low) * 255.0
            result = simulated.astype(np.uint8)
        if self._distortion is not None:
            result, _ = self._distortion.perturb(image=result)
        return result
