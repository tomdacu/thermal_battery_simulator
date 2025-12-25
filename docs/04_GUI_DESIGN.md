# Interfaccia Utente (GUI)

## 1. Introduzione

L'interfaccia grafica è sviluppata in **PyQt6** e fornisce un ambiente integrato per la configurazione, l'esecuzione e l'analisi delle simulazioni. Il file principale è `gui/main_window.py`.

---

## 2. Struttura della Finestra

La GUI è suddivisa in tre aree principali:

### 2.1 Pannello Laterale (Input)
Permette di configurare tutti i parametri della simulazione:
- **Geometria**: Dimensioni del dominio e raggi delle zone cilindriche.
- **Mesh**: Risoluzione della griglia e target spacing.
- **Materiali**: Selezione del materiale di stoccaggio e isolamento.
- **Resistenze**: Potenza totale e pattern di distribuzione.
- **Solutore**: Scelta del metodo (Diretto, CG, BiCGStab) e tolleranza.

### 2.2 Area Centrale (Visualizzazione 3D)
Utilizza `PyVistaQt` per integrare un motore di rendering 3D interattivo:
- Visualizzazione del campo di temperatura.
- Visualizzazione della distribuzione dei materiali.
- Strumenti di **Slicing** (piani di sezione X, Y, Z) per ispezionare l'interno della batteria.
- Isosuperfici di temperatura.

### 2.3 Pannello Inferiore (Risultati)
Mostra i dati derivati dalla simulazione:
- Log del solutore (tempo di calcolo, residuo, iterazioni).
- Bilancio di potenza (P_in, P_out, Perdite).
- Energia totale immagazzinata [MWh].

---

## 3. Gestione della Simulazione

### 3.1 Threading
Per evitare il blocco dell'interfaccia durante i calcoli intensivi, la simulazione viene eseguita in un thread separato (`SimulationThread`). Questo permette di:
- Mantenere la visualizzazione 3D fluida.
- Aggiornare una barra di progresso in tempo reale.
- Interrompere la simulazione se necessario.

### 3.2 Workflow Utente
1.  **Configurazione**: L'utente imposta i parametri nei widget.
2.  **Preview Mesh**: (Opzionale) Visualizzazione della griglia prima del calcolo.
3.  **Run**: Pressione del tasto "Start Simulation".
4.  **Analisi**: Esplorazione dei risultati tramite i piani di sezione e i grafici.

---

## 4. Requisiti per la GUI

Per il corretto funzionamento della GUI sono necessari:
- `PyQt6`: Framework per le finestre.
- `pyvista`: Motore di rendering.
- `pyvistaqt`: Integrazione tra PyVista e Qt.

---

## 5. Sviluppi Futuri
- Grafici 2D dell'andamento temporale (per simulazioni transitorie).
- Esportazione dei risultati in formato VTK o CSV.
- Database materiali modificabile direttamente da interfaccia.
