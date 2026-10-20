"""
gui.assets
==========
Repository assets (icons, banner) resolved by path, never by cwd.

WHAT IT DOES
------------
Finds the window's icon files from the checkout, starting at this file's own
location, so a shortcut or a packaged run with another working directory still
finds them.

HOW IT WORKS
------------
-> REPOSITORY ROOT <-
   ``repo_root`` is the parent of the ``gui`` package: one level above this
   file.
-> MISSING ASSETS <-
   ``asset`` returns a path only when it exists, otherwise ``None``; a missing
   asset is not fatal.  The window icon is ``assets/icon.png`` first, else the
   historical names under ``photo/``.
"""
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
    """Application icon: the house-style icon, else the historical ones."""
    found = asset("assets", "icon.png")
    if found is not None:
        return found
    for name in ("Icona Thermal Battery 2.png", "Icona Thermal Battery.png",
                 "Icon3.png"):
        found = asset("photo", name)
        if found is not None:
            return found
    return None
