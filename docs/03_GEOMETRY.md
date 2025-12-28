# Geometria della Thermal Battery

## 1. Panoramica

La geometria della batteria è definita nel modulo `src/core/geometry.py`. Il sistema utilizza un approccio semplificato a **4 zone concentriche**:
- Una **zona STORAGE** centrale che contiene il materiale di accumulo con tubi e resistenze immerse
- **Elementi discreti** per resistenze e tubi scambiatori posizionati DENTRO lo storage

---

## 2. Struttura a 4 Zone Concentriche

Il cilindro della batteria è suddiviso in **4 zone semplificate** (dal centro verso l'esterno):

| Zona | Descrizione | Materiale Default |
|------|-------------|-------------------|
| **STORAGE** | Materiale di accumulo termico con tubi e resistenze immerse | Sabbia / Steatite |
| **INSULATION** | Strato isolante termico | Lana di roccia |
| **STEEL** | Guscio strutturale esterno | Acciaio al carbonio |
| **AIR** | Aria esterna (fuori dal guscio) | Aria |

### Parametri geometrici principali:
- `r_storage`: Raggio della zona di accumulo
- `insulation_thickness`: Spessore dell'isolamento
- `shell_thickness`: Spessore del guscio in acciaio
- `phase_offset_deg`: Sfasamento angolare tra tubi e resistenze (per evitare sovrapposizioni)

---

## 3. Elementi Riscaldanti (Heaters)

Le resistenze elettriche sono **immerse nella zona STORAGE**. Possono essere modellate in due modi:
1.  **Zona Uniforme**: La potenza totale è distribuita uniformemente in tutto il volume storage.
2.  **Elementi Discreti**: Resistenze cilindriche verticali posizionate secondo un pattern specifico.

### Pattern disponibili:
- `UNIFORM_ZONE`: Distribuzione volumetrica continua.
- `GRID_VERTICAL`: Griglia rettangolare di resistenze.
- `RADIAL_ARRAY`: Resistenze disposte su anelli concentrici.
- `SPIRAL`: Disposizione a spirale dal centro.
- `CUSTOM`: Posizioni (x, y) definite manualmente.

### Sfasamento angolare:
Quando sia tubi che resistenze usano pattern radiali, è possibile impostare uno **sfasamento angolare** (phase_offset_deg) per evitare sovrapposizioni. Lo sfasamento viene applicato alle resistenze, mentre i tubi mantengono la posizione di riferimento.

---

## 4. Scambiatori di Calore (Tubes)

I tubi per l'estrazione del calore sono **immersi nella zona STORAGE** e posizionati secondo pattern geometrici:
- `CENTRAL_CLUSTER`: Gruppo di tubi al centro.
- `RADIAL_ARRAY`: Tubi disposti su anelli concentrici.
- `HEXAGONAL`: Pattern a massima densità (esagonale).
- `SINGLE_CENTRAL`: Un unico grande tubo centrale.

Ogni tubo è caratterizzato da:
- **Raggio**
- **Coefficiente convettivo ($h$)** del fluido interno.
- **Temperatura del fluido ($T_{fluid}$)**.

---

## 5. Integrazione con la Mesh

La classe `BatteryGeometry` si occupa di "mappare" queste entità geometriche sulla mesh 3D tramite il metodo `apply_to_mesh()`.

### Processo di mappatura:
1.  Per ogni cella della mesh, vengono calcolate le coordinate $(x, y, z)$.
2.  Si calcola il raggio $r = \sqrt{(x-x_c)^2 + (y-y_c)^2}$.
3.  Si determina la zona principale (STORAGE, INSULATION, STEEL, AIR).
4.  Per le celle nella zona STORAGE:
    - Si verifica se il punto appartiene a un elemento discreto (tubo o resistenza).
    - Se appartiene a un tubo: si assegnano proprietà e BC del tubo.
    - Se appartiene a una resistenza: si assegnano proprietà con sorgente di calore Q.
    - Altrimenti: si assegnano le proprietà del materiale di storage (sabbia/steatite).
5.  Vengono assegnate le proprietà termofisiche ($k, \rho, c_p$) e la sorgente di calore ($Q$) corrispondente.
6.  Vengono impostate le condizioni al contorno (BC) sulle facce esterne del dominio e sulle interfacce dei tubi.

---

## 6. Esempio di Configurazione (YAML)

```yaml
geometry:
  cylinder:
    r_storage: 3.5       # Raggio della zona di accumulo [m]
    insulation_thickness: 0.3  # Spessore isolamento [m]
    shell_thickness: 0.01      # Spessore guscio acciaio [m]
    height: 7.0                # Altezza [m]
    center_x: 5.0
    center_y: 5.0
    base_z: 0.5
    phase_offset_deg: 15.0     # Sfasamento angolare tubi-resistenze [gradi]
```
