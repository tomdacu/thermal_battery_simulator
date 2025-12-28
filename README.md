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

---

## 📖 Guida all'Uso

### Workflow Base

1. **Configura la Geometria** (Tab "Geometria")
   - Imposta le dimensioni del dominio (Lx, Ly, Lz)
   - Scegli la risoluzione mesh (punti per dimensione)
   - Definisci le dimensioni della batteria cilindrica

2. **Configura Resistenze e Tubi** (Tab "Resistenze" e "Tubi")
   - Seleziona il pattern di disposizione
   - Imposta potenza totale e numero di elementi
   - Usa "Anteprima" per verificare le posizioni

3. **Ottimizza il Solver** (Tab "Solver")
   - Vedi sezione "Ottimizzazione Performance" sotto

4. **Esegui**
   - Clicca "Costruisci Mesh"
   - Clicca "Esegui Simulazione"
   - Esplora i risultati con i controlli di visualizzazione

---

## ⚡ Ottimizzazione Performance

### Perché la simulazione è lenta?

Il tempo di calcolo dipende da:
- **Numero di celle**: $N = N_x \times N_y \times N_z$. Con 100×100×100 = 1 milione di celle!
- **Metodo di soluzione**: I metodi diretti sono $O(N^{1.5})$, gli iterativi $O(N)$
- **Tolleranza**: Tolleranze più strette richiedono più iterazioni

### Configurazione Consigliata per Scenario

| Scenario | Metodo | Precond. | Tolleranza | Tempo Est. |
|----------|--------|----------|------------|------------|
| Test rapido (debug) | cg | none | 1e-4 | ~1 sec |
| Visualizzazione | cg | jacobi | 1e-6 | ~5 sec |
| Precisione standard | cg | jacobi | 1e-8 | ~15 sec |
| Alta precisione | cg | jacobi | 1e-10 | ~30 sec |

### Metodi di Soluzione

| Metodo | Descrizione | Quando usarlo |
|--------|-------------|---------------|
| **bicgstab** | BiCGSTAB | ⭐ **CONSIGLIATO**. Robusto, funziona sempre |
| **cg** | Gradiente Coniugato | Veloce ma può non convergere con BC miste |
| **gmres** | GMRES | Ottima convergenza, usa più memoria |
| **direct** | LU diretto | Solo per mesh piccole (<30k celle) |

> ⚠️ **Nota su CG**: Il metodo CG richiede matrice simmetrica definita positiva. Con condizioni al contorno miste (convezione sui tubi + Dirichlet) la matrice può perdere simmetria → usa BiCGSTAB.

### Precondizionatori

| Precond. | Descrizione | Performance |
|----------|-------------|-------------|
| **jacobi** | Diagonale | ⭐ **CONSIGLIATO**. Multi-threaded, veloce |
| **none** | Nessuno | CG puro, sorprendentemente veloce! |
| **ilu** | Incomplete LU | ⚠️ Single-threaded, può essere LENTO |

> ⚠️ **Nota importante**: ILU usa SuperLU che è single-threaded. Per mesh grandi, Jacobi o nessun precondizionatore sono spesso più veloci!

### Tolleranza

| Valore | Uso | Note |
|--------|-----|------|
| 1e-10 | Alta precisione | Per validazione e analisi dettagliate |
| 1e-8 | Default | Buon compromesso velocità/precisione |
| 1e-6 | Veloce | Sufficiente per visualizzazione |
| 1e-4 | Molto veloce | Solo per test rapidi |

### Multi-Threading

- **Auto**: Usa tutti i core CPU → massima velocità, può rallentare il sistema
- **Tutti - 1**: ⭐ **Consigliato**. Lascia un core libero per la GUI
- **N core**: Limita a N core specifici

### Suggerimenti Pratici

1. **Inizia con mesh piccole** (30-40 punti) per test rapidi
2. **Usa CG + ILU** per la maggior parte dei casi
3. **Aumenta la mesh** solo per risultati finali
4. **Tolleranza 1e-6** è sufficiente per visualizzazione

---

## Requisiti

- Python 3.10+
- NumPy, SciPy (Calcolo numerico)
- PyVista, PyQt6 (Interfaccia e Visualizzazione)
- Numba (Accelerazione JIT - opzionale)

## Licenza

MIT License
