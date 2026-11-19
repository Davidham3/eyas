"""Eyas: train sharp-eyed prompts for multimodal models."""

from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth: the installed distribution's metadata, so the CLI
    # can never report a version that differs from the released wheel.
    __version__ = version("eyas")
except PackageNotFoundError:
    # Uninstalled source tree (no dist metadata): never fake a release number.
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
