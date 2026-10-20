# Schermate — Thermal Battery Simulator

Aggiornate il 30 settembre 2026. Immagini reali dell’applicazione, organizzate per la documentazione del progetto e per il portfolio personale. I nomi indicano il contenuto; gli originali forniti dall’autore rimangono nella cartella Screenshot.

| File | Contenuto |
| --- | --- |
| [thermal-overview.png](thermal-overview.png) | Standby a 500 °C: campo termico e bilancio del caso mostrato. Potenza di mantenimento di circa 5,91 kW. Schermata fornita dall’autore. |
| [thermal-circuit.webp](thermal-circuit.webp) | La geometria del caso simulato e il circuito del gas, acquisiti dal programma. |
| [thermal-current.webp](thermal-current.webp) | Carica a 200 kW per 24 ore: campo termico finale e campioni temporali nella GUI. Mesh di 3.032 celle, dt = 900 s. |
| [thermal-charge-chart.webp](thermal-charge-chart.webp) | Grafico ricavato dai 49 campioni esportati dalla simulazione. Caso illustrativo su mesh di 3.032 celle; non è una validazione sperimentale. |
| [thermal-energy.webp](thermal-energy.webp) | Pannello Energy balance al termine della carica, con il riepilogo cumulativo del transitorio. |

## Contesto e fonti

`thermal-overview.png` proviene dallo screenshot `Screenshot 2026-09-30 153728sd.png` fornito dall’autore: standby a 500 °C, potenza di mantenimento circa 5,91 kW. Le altre immagini rappresentano un caso separato di carica di 24 ore a 200 kW, da 20 °C, su mesh di 3.032 celle con dt = 900 s. Il CSV contiene 49 campioni esportati; il grafico è ricavato da questi dati. Sono esempi numerici, non misure sperimentali. Il portfolio ritaglia la cornice dello standby tramite CSS e conserva l’immagine originale.
