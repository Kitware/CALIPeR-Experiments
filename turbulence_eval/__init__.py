"""Simulate imagery from CALIPeR ground cameras."""

import sys

if sys.version_info < (3, 10):
    raise ImportError("turbulence_eval requires Python 3.10 or newer; core caliper_eval remains available")

__all__ = ["CameraSimulator", "SimulationOptions", "generate_images"]


def __getattr__(name):
    # Selection previews do not need image/simulator dependencies installed.
    if name in ("CameraSimulator", "SimulationOptions"):
        from . import adapter
        return getattr(adapter, name)
    if name == "generate_images":
        from .generate import generate_images
        return generate_images
    raise AttributeError(name)
