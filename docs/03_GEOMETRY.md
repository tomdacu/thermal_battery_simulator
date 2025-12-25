# Geometria della Sand Battery

## 1. Panoramica

La geometria della batteria è definita nel modulo `src/core/geometry.py`. Il sistema utilizza un approccio ibrido:
- Una **geometria cilindrica** di base per definire le zone principali.
- **Elementi discreti** per resistenze e tubi scambiatori, che possono essere posizionati secondo vari pattern.

---

## 2. Zone Radiali

Il cilindro della batteria è suddiviso in zone concentriche (dall'interno verso l'esterno):

| Zona | Descrizione | Materiale Default |
|------|-------------|-------------------|
| **Tubes** | Area centrale per scambiatori | Aria / Sabbia |
| **Sand Inner** | Accumulo termico interno | Sabbia / Steatite |
| **Heaters** | Zona delle resistenze elettriche | Sabbia + Sorgente Q |
| **Sand Outer** | Accumulo termico esterno | Sabbia / Steatite |
| **Insulation** | Strato isolante termico | Lana di roccia |
| **Shell** | Guscio strutturale | Acciaio |

---

## 3. Elementi Riscaldanti (Heaters)

Le resistenze possono essere modellate in due modi:
1.  **Zona Uniforme**: La potenza totale è distribuita uniformemente in tutto il volume della zona "Heaters".
2.  **Elementi Discreti**: Resistenze cilindriche verticali posizionate secondo un pattern specifico.

### Pattern disponibili:
- `UNIFORM_ZONE`: Distribuzione volumetrica continua.
- `GRID_VERTICAL`: Griglia rettangolare di resistenze.
- `RADIAL_ARRAY`: Resistenze disposte su anelli concentrici.
- `SPIRAL`: Disposizione a spirale dal centro.
- `CUSTOM`: Posizioni (x, y) definite manualmente.

---

## 4. Scambiatori di Calore (Tubes)

I tubi per l'estrazione del calore sono posizionati nella zona centrale o secondo pattern geometrici:
- `CENTRAL_CLUSTER`: Gruppo di tubi al centro.
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
2.  Si verifica se il punto appartiene a un elemento discreto (tubo o resistenza).
3.  Se non appartiene a elementi discreti, si determina la zona radiale in base alla distanza dal centro $r = \sqrt{(x-x_c)^2 + (y-y_c)^2}$.
4.  Vengono assegnate le proprietà termofisiche ($k, \rho, c_p$) e la sorgente di calore ($Q$) corrispondente.
5.  Vengono impostate le condizioni al contorno (BC) sulle facce esterne del dominio e sulle interfacce dei tubi.

---

## 6. Esempio di Configurazione (YAML)

```yaml
geometry:
  cylinder:
    radius: 4.0
    height: 7.0
    center_x: 5.0
    center_y: 5.0
    base_z: 0.5
  layers:
    tubes_inner: 0.5
    sand_inner: 1.5
    heaters: 0.3
    sand_outer: 1.0
    insulation: 0.5
    steel_shell: 0.01
```
