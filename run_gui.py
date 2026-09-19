"""Entry point: launch the Thermal Battery Simulator GUI."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gui.main_window import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
