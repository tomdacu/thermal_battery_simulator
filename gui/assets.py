"""Repository assets (icons, banner) resolved by path, never by cwd."""
from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Root of the checkout, one level above the ``gui`` package."""
    return Path(__file__).resolve().parents[1]


def asset(*parts: str):
    """Path of an asset if it exists, otherwise ``None`` (missing assets are not fatal)."""
    path = repo_root().joinpath(*parts)
    return path if path.exists() else None


def window_icon_path():
    """Application icon, tolerating the historical file-name variation."""
    for name in ("Icona Thermal Battery 2.png", "Icona Thermal Battery.png",
                 "Icon3.png"):
        found = asset("photo", name)
        if found is not None:
            return found
    return None
