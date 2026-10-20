"""
run_gui
========
Launches the Thermal Battery Simulator GUI.

WHAT IT DOES
------------
Puts the repository root on ``sys.path`` so the ``gui`` and ``src`` packages
import from a checkout, calls :func:`gui.main_window.main` and exits with the
status it returns.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gui.main_window import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
