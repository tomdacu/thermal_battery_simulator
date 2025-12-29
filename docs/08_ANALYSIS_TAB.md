# 08_ANALYSIS_TAB.md - Documentazione Scheda Analisi

## Panoramica

La scheda **Analisi** permette di configurare simulazioni termiche avanzate:
- **Stazionaria**: Equilibrio termico con potenza costante
- **Perdite**: Calcola dispersioni termiche dato T nello storage
- **Transitoria**: Evoluzione temporale (carica/scarica)

## Struttura Sub-tabs

```
Analisi
├── Tipo          → Selezione tipo analisi + parametri specifici
├── Condizioni    → Condizione iniziale per transitorio
├── Potenza       → Profilo potenza resistenze
├── Estrazione    → Profilo estrazione dai tubi
└── Salvataggio   → Salva/carica stato simulazione
```

## 1. Tipo Analisi

### Stazionaria
Calcola la distribuzione di temperatura a regime:
- **Input**: Potenza totale resistenze [W]
- **Output**: Campo T, perdite, efficienza

### Analisi Perdite
Imposta T media nello storage e calcola dispersioni:
- **Input**: T media storage, T ambiente
- **Output**: Perdite (top, laterale, bottom), potenza necessaria per mantenerla

### Transitoria
Simula evoluzione temporale:
- **Durata**: Tempo totale simulazione [s]
- **dt**: Passo temporale [s] (raccomandato: 60s)
- **Intervallo salvataggio**: Frequenza salvataggio risultati
- **Schema numerico**: Backward Euler (implicito, stabile)

## 2. Condizioni Iniziali

### Temperatura Uniforme
Tutto il dominio a temperatura costante.

### Per Materiale
Temperature diverse per ogni materiale:
- Sabbia (storage)
- Isolamento
- Acciaio
- Aria
- Terreno
- Calcestruzzo

### Da File
Carica campo T da simulazione precedente (.h5)

### Da Stazionario
Prima calcola lo stazionario, poi usa come condizione iniziale per transitorio.

## 3. Profilo Potenza

### Spente
Resistenze disattivate (solo scarica/perdite)

### Costante
Potenza fissa durante tutta la simulazione.

### Schedulato
Tabella (tempo, potenza):
```
t [s]    P [W]
0        10000
3600     5000
7200     0
```

### Da CSV
File CSV con colonne `time,power`:
```csv
time,power
0,10000
3600,5000
7200,0
```

## 4. Estrazione Energia

### Off
Nessuna estrazione dai tubi

### Potenza Imposta
Specifica potenza da estrarre [W]

### Flow Rate
Specifica portata fluido [kg/s]:
- Per tubo o totale
- Fluido: acqua, olio termico, aria, glicole

### Temperatura Uscita
Impone T uscita desiderata [°C]

## 5. Salvataggio Stato

### Salva
- Nome simulazione
- Descrizione opzionale
- Formato: HDF5 (.h5) compresso

### Carica
- Sfoglia file .h5
- Verifica compatibilità geometria (hash)
- Carica campo T nella mesh

## Schema Numerico Transitorio

**Backward Euler** (implicito):

$$
\frac{\rho c_p}{\Delta t}(T^{n+1} - T^n) = \nabla \cdot (k \nabla T^{n+1}) + Q^{n+1}
$$

In forma matriciale:

$$
\left(\frac{M}{\Delta t} + A\right) T^{n+1} = \frac{M}{\Delta t} T^n + b
$$

**Vantaggi**:
- Incondizionatamente stabile (qualsiasi dt)
- Un sistema lineare per timestep
- Riutilizza precondizionatore AMG

## Formato File Stato (.h5)

```
/metadata
    version, timestamp, name, analysis_type
/geometry
    mesh_shape, geometry_hash, parameters
/state
    T (compressed)
    materials (compressed)
/results
    computed quantities
```

## Workflow Tipico

### Dimensionamento (Stazionario)
1. Configura geometria
2. Imposta potenza resistenze
3. Esegui stazionario
4. Verifica T max, perdite

### Analisi Ciclo (Transitorio)
1. Configura geometria
2. Condizione iniziale: uniforme 20°C
3. Potenza: schedule carica/scarica
4. Esegui transitorio (es. 24h)
5. Visualizza grafici T vs tempo
6. Esporta risultati CSV

### Verifica Perdite
1. Configura geometria completa
2. Tipo: Analisi Perdite
3. Imposta T storage = 400°C
4. Calcola → ottieni potenza dispersa
