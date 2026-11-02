"""Simulate imagery from CALIPeR ground cameras."""

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
