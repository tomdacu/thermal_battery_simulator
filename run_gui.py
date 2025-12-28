"""
run_gui.py - Script per avviare la GUI Thermal Battery Simulation

Utilizzo:
    python run_gui.py
"""

import sys
from pathlib import Path

# Aggiungi la directory corrente al path
sys.path.insert(0, str(Path(__file__).parent))

from gui.main_window import main

if __name__ == "__main__":
    main()
