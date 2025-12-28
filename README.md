# Thermal Battery Simulator 3D

## Obiettivi del Progetto

Il **Thermal Battery Simulator** è un software avanzato per la progettazione e l'analisi di sistemi di accumulo termico a sabbia (Sand Batteries). L'obiettivo principale è fornire uno strumento flessibile e interattivo che permetta di:

1.  **Configurare Geometrie Complesse**: Definire dimensioni, strati isolanti e posizionamento di scambiatori e riscaldatori.
2.  **Simulare Scenari Operativi**: Analizzare il comportamento termico in regime stazionario (e futuro transitorio) variando potenze e temperature.
3.  **Ottimizzare il Design**: Valutare l'impatto di diversi materiali e configurazioni sull'efficienza energetica e sulle perdite termiche.
4.  **Accessibilità**: Rendere la simulazione numerica complessa accessibile tramite un'interfaccia grafica (GUI) intuitiva, eliminando la necessità di modificare il codice per ogni test.

## Funzionamento

Il sistema si basa su un motore di calcolo a **Differenze Finite (FDM)** che risolve l'equazione del calore in un dominio 3D. A differenza di modelli statici, questo simulatore permette di:
- Scegliere i materiali da un database integrato.
- Configurare le condizioni al contorno (aria, terreno, fluidi) direttamente dalla GUI.
- Visualizzare i risultati in tempo reale con strumenti di slicing 3D.

## Struttura della Documentazione

Per approfondire il funzionamento del programma, consulta i file nella cartella `docs/`:

1.  [01_THEORY.md](docs/01_THEORY.md): Fondamenti fisici ed equazioni del calore.
2.  [02_FDM_DISCRETIZATION.md](docs/02_FDM_DISCRETIZATION.md): Dettagli sulla discretizzazione FDM e indicizzazione.
3.  [03_GEOMETRY.md](docs/03_GEOMETRY.md): Modello geometrico e mapping sulla mesh.
4.  [04_GUI_DESIGN.md](docs/04_GUI_DESIGN.md): Struttura e design della GUI.

## Installazione

```bash
# 1. Clonare il repository
git clone https://github.com/PhyTom/Thermal_battery_simulator.git
cd Thermal_battery_simulator

# 2. Creare ambiente virtuale
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 3. Installare dipendenze
pip install -r requirements.txt
```

## Avvio Rapido

Per avviare l'interfaccia grafica:
```bash
python run_gui.py
```

Per eseguire una simulazione di test via script:
```bash
python main.py
```

## Requisiti

- Python 3.10+
- NumPy, SciPy (Calcolo numerico)
- PyVista, PyQt6 (Interfaccia e Visualizzazione)
- Numba (Accelerazione JIT)

## Licenza

MIT License
