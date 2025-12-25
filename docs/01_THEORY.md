# Teoria dello Scambio Termico - Sand Battery Simulation

## 1. Equazioni Fondamentali

### 1.1 Equazione del Calore (Generale)

L'equazione che governa la distribuzione della temperatura in un solido è:

$$\rho c_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + Q$$

Dove:
- $\rho$ = densità del materiale [kg/m³]
- $c_p$ = calore specifico a pressione costante [J/(kg·K)]
- $T$ = temperatura [K o °C]
- $t$ = tempo [s]
- $k$ = conducibilità termica [W/(m·K)]
- $Q$ = sorgente/pozzo di calore volumetrica [W/m³]

### 1.2 Caso Stazionario

Quando $\frac{\partial T}{\partial t} = 0$:

$$\nabla \cdot (k \nabla T) + Q = 0$$

Per conducibilità costante:

$$k \nabla^2 T + Q = 0$$

---

## 2. Meccanismi di Trasferimento del Calore

### 2.1 Conduzione (Legge di Fourier)

Il flusso di calore per conduzione è:

$$\vec{q} = -k \nabla T$$

In forma scalare unidimensionale:

$$q = -k \frac{dT}{dx}$$

### 2.2 Convezione (Legge di Newton)

Il flusso di calore per convezione tra una superficie e un fluido:

$$q = h (T_s - T_\infty)$$

Dove:
- $h$ = coefficiente di scambio termico convettivo [W/(m²·K)]
- $T_s$ = temperatura della superficie
- $T_\infty$ = temperatura del fluido lontano dalla superficie

**Valori tipici di h:**
| Condizione | h [W/(m²·K)] |
|------------|--------------|
| Convezione naturale aria | 5-25 |
| Convezione forzata aria | 25-250 |
| Convezione naturale acqua | 100-900 |
| Convezione forzata acqua | 50-20.000 |

### 2.3 Irraggiamento (Legge di Stefan-Boltzmann)

$$q = \epsilon \sigma (T_s^4 - T_{surr}^4)$$

Dove:
- $\epsilon$ = emissività della superficie (0-1)
- $\sigma$ = 5.67 × 10⁻⁸ W/(m²·K⁴)

**Nota:** Per le temperature tipiche della Sand Battery (< 600°C), l'irraggiamento 
è significativo ma spesso linearizzato o incluso nel coefficiente h effettivo.

---

## 3. Resistenze Termiche

### 3.1 Analogia Elettrica

Come in un circuito elettrico:
- Temperatura ↔ Tensione
- Flusso termico ↔ Corrente
- Resistenza termica ↔ Resistenza elettrica

$$q = \frac{\Delta T}{R_{th}}$$

### 3.2 Resistenze in Serie

$$R_{tot} = R_1 + R_2 + R_3 + ...$$

### 3.3 Tipi di Resistenza Termica

**Conduzione (parete piana):**
$$R_{cond} = \frac{L}{k \cdot A}$$

**Conduzione (cilindro):**
$$R_{cond,cyl} = \frac{\ln(r_2/r_1)}{2\pi k L}$$

**Convezione:**
$$R_{conv} = \frac{1}{h \cdot A}$$

---

## 4. Scambio Termico in Materiali Porosi

### 4.1 Conducibilità Termica Effettiva

Per la sabbia con aria nei pori, la conducibilità effettiva può essere stimata:

**Modello parallelo (limite superiore):**
$$k_{eff,\parallel} = \phi \cdot k_{fluido} + (1-\phi) \cdot k_{solido}$$

**Modello serie (limite inferiore):**
$$\frac{1}{k_{eff,serie}} = \frac{\phi}{k_{fluido}} + \frac{1-\phi}{k_{solido}}$$

**Media geometrica (buona approssimazione):**
$$k_{eff} = k_{solido}^{(1-\phi)} \cdot k_{fluido}^{\phi}$$

Dove $\phi$ = porosità (frazione di vuoti)

### 4.2 Capacità Termica Effettiva

$$(\rho c_p)_{eff} = \phi \cdot (\rho c_p)_{fluido} + (1-\phi) \cdot (\rho c_p)_{solido}$$

---

## 5. Condizioni al Contorno

### 5.1 Dirichlet (Temperatura Imposta)

$$T|_{\Gamma} = T_{prescritta}$$

Esempio: Base della batteria a contatto con terreno a temperatura costante.

### 5.2 Neumann (Flusso Imposto)

$$-k \frac{\partial T}{\partial n}\bigg|_{\Gamma} = q_{prescritta}$$

Esempio: Superficie adiabatica (q = 0) per simmetria.

### 5.3 Robin (Convezione)

$$-k \frac{\partial T}{\partial n}\bigg|_{\Gamma} = h(T_s - T_\infty)$$

Esempio: Superficie esterna a contatto con aria ambiente.

---

## 6. Adimensionalizzazione e Numeri Caratteristici

### 6.1 Numero di Biot

$$Bi = \frac{h \cdot L_c}{k}$$

- $Bi << 1$: Temperatura uniforme nel solido (lumped capacitance)
- $Bi >> 1$: Gradienti significativi nel solido

### 6.2 Numero di Fourier

$$Fo = \frac{\alpha \cdot t}{L_c^2}$$

Dove $\alpha = k/(\rho c_p)$ = diffusività termica [m²/s]

- Indica quanto il sistema è "vicino" all'equilibrio termico

---

## 7. Applicazione alla Sand Battery

### 7.1 Architettura della Batteria

La batteria è modellata come un cilindro verticale composto da diverse zone radiali concentriche:

1.  **Zona Tubi Centrali**: Area dedicata agli scambiatori di calore per l'estrazione dell'energia.
2.  **Sabbia Interna**: Primo strato di accumulo termico.
3.  **Zona Resistenze**: Area dove sono posizionati gli elementi riscaldanti elettrici.
4.  **Sabbia Esterna**: Secondo strato di accumulo termico.
5.  **Isolamento**: Strato di materiale a bassa conducibilità (es. lana di roccia) per minimizzare le perdite.
6.  **Guscio**: Rivestimento esterno in acciaio per protezione strutturale.

### 7.2 Bilancio Energetico Globale

**Stato di carica:**
$$\dot{E}_{in} = \dot{E}_{stored} + \dot{E}_{losses}$$

$$P_{resistenze} = \frac{d}{dt}(m \cdot c_p \cdot \bar{T}) + P_{dispersione}$$

**Stato di scarica:**
$$\dot{E}_{stored} = \dot{E}_{out} + \dot{E}_{losses}$$

$$\frac{d}{dt}(m \cdot c_p \cdot \bar{T}) = P_{tubi} + P_{dispersione}$$

### 7.3 Potenza delle Resistenze

La potenza termica generata per unità di volume ($Q$) è distribuita nella zona delle resistenze:

$$P_{res} = \sum_i Q_i \cdot V_i$$

Dove $V_i$ è il volume della cella i-esima appartenente alla zona riscaldante.

### 7.4 Potenza Estratta dai Tubi

Lo scambio termico con il fluido nei tubi è modellato tramite una condizione di convezione interna:

$$q_{tubi} = h_{fluido} (T_{parete} - T_{fluido})$$

### 7.5 Potenza Dispersa

La dispersione verso l'ambiente avviene per convezione sulle superfici esterne:

$$P_{disp} = \oint h(T_s - T_\infty) \, dA$$

E per conduzione verso il terreno alla base:

$$q_{base} = -k \frac{\partial T}{\partial z}\bigg|_{z=0}$$

---

## 8. Analisi Exergetica (Cenni)

### 8.1 Exergia Termica

L'exergia associata a un flusso di calore Q a temperatura T:

$$\dot{Ex} = \dot{Q} \cdot \left(1 - \frac{T_0}{T}\right)$$

Dove $T_0$ = temperatura di riferimento (ambiente)

### 8.2 Exergia Immagazzinata

$$Ex_{stored} = m \cdot c_p \cdot \left[(T - T_0) - T_0 \cdot \ln\left(\frac{T}{T_0}\right)\right]$$

### 8.3 Distruzione di Exergia

Per un sistema con temperature $T_1$ e $T_2$ che scambia calore Q:

$$\dot{Ex}_{distr} = T_0 \cdot \dot{Q} \cdot \left(\frac{1}{T_2} - \frac{1}{T_1}\right)$$

---

## 9. Riferimenti

1. Incropera, F.P., DeWitt, D.P. - "Fundamentals of Heat and Mass Transfer"
2. Çengel, Y.A. - "Heat Transfer: A Practical Approach"
3. Bejan, A. - "Advanced Engineering Thermodynamics" (per exergia)
4. Kaviany, M. - "Principles of Heat Transfer in Porous Media"
