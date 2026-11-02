"""Look up camera profiles, range aliases, and image scales for each collection."""

from math import hypot

# Keep the shared names available for existing camera/simulator callers.
from .collect1_cameras import CAMERA_CONFIGS as CALIPER_CAMERA_CONFIGS, M_PER_PIXEL
from .collect2_cameras import CAMERA_CONFIGS as BRC2_CAMERA_CONFIGS

# Dataset range buckets and calibration labels describe the same range station.
RANGE_TO_CAMERA = {"250m": "260m", "400m": "380m"}
CAMERA_TO_RANGE = {value: key for key, value in RANGE_TO_CAMERA.items()}
BRC2_CAMERA_ALIASES = {**CAMERA_TO_RANGE, "250m-c": "250m_lift"}


def camera_label(value: str, profile: str = "brc1") -> str:
    """Resolve a dataset bucket or native label to a simulation camera."""
    native = (RANGE_TO_CAMERA if profile == "brc1" else BRC2_CAMERA_ALIASES).get(value, value)
    if native not in camera_configs(profile):
        raise ValueError(f"Unknown simulation camera for {profile}: {value!r}")
    return native


def range_token(value: str) -> str:
    return CAMERA_TO_RANGE.get(value, value)


def camera_configs(profile: str = "brc1") -> dict[str, dict]:
    """Return a collection's optics without changing the BRC1 defaults."""
    if profile == "brc1":
        return CALIPER_CAMERA_CONFIGS
    if profile == "brc2":
        return BRC2_CAMERA_CONFIGS
    raise ValueError(f"Unknown camera calibration profile: {profile!r}")


def metres_per_pixel(camera: str, profile: str = "brc1") -> float:
    """Return the subject-plane pixel scale for the camera and collection."""
    camera = camera_label(camera, profile)
    if profile == "brc1":
        return M_PER_PIXEL[camera]
    cfg = camera_configs(profile)[camera]
    return hypot(cfg["ground_range"], cfg["altitude"]) * cfg["p_x"] / cfg["f"]
