# DATABASE MATERIALI - THERMAL BATTERY SIMULATION
# ==============================================
# Estratto dal preliminary_dimensioning per uso nella simulazione termica
# Data: 14 Dicembre 2025

"""
Questo file contiene tutti i dati dei materiali necessari per la simulazione
termica 3D della Thermal Battery. I dati sono stati validati con impianti reali
di Polar Night Energy.

FONTE DEI DATI:
- Polar Night Energy Official Website
- Wikipedia: Thermal Energy Storage
- Engineering ToolBox
- Validazione con impianto Pornainen (100 MWh, 2000 t steatite)
"""

# =============================================================================
# PARAMETRI OPERATIVI DI RIFERIMENTO (POLAR NIGHT ENERGY)
# =============================================================================

PARAMETRI_OPERATIVI = {
    "T_max_stoccaggio": 600,      # °C - Temperatura massima
    "T_max_output": 400,          # °C - Temperatura output aria
    "T_min_district_heating": 90, # °C - Temperatura minima per DH
    "T_conservativa": 380,        # °C - Temperatura max conservativa
    "efficienza_roundtrip": {
        "piccola_scala": 0.60,    # 60-75% per impianti piccoli (8 MWh)
        "media_scala": 0.85,      # 85% per 2 MW
        "grande_scala": 0.90,     # 90% per 10 MW
    },
    "vita_utile_anni": 30,
    "ore_operative_annue": 8000,
}

# =============================================================================
# MATERIALI DI STOCCAGGIO TERMICO (SABBIE E SIMILI)
# =============================================================================

MATERIALI_STOCCAGGIO = {
    "sabbia_silicea": {
        "nome_completo": "Sabbia Silicea (Quarzo - SiO2)",
        "densita_kg_m3": 1600,           # kg/m³ (bulk, non compattata)
        "densita_solido_kg_m3": 2650,    # kg/m³ (solido puro)
        "cp_kJ_kgK": 0.82,               # kJ/(kg·K)
        "k_W_mK": 0.30,                  # W/(m·K) conducibilità termica
        "temp_max_C": 1200,              # °C
        "temp_fusione_C": 1700,          # °C
        "costo_relativo": 1.0,           # Riferimento
        "disponibilita": "alta",
        "note": "Economica, abbondante, chimicamente stabile"
    },
    "sabbia_olivina": {
        "nome_completo": "Sabbia di Olivina ((Mg,Fe)2SiO4)",
        "densita_kg_m3": 1700,
        "densita_solido_kg_m3": 3300,
        "cp_kJ_kgK": 0.94,
        "k_W_mK": 0.75,
        "temp_max_C": 1200,
        "costo_relativo": 1.5,
        "disponibilita": "media",
        "note": "Migliore capacità termica, buona conducibilità"
    },
    "steatite": {
        "nome_completo": "Steatite / Pietra Ollare (Soapstone)",
        "densita_kg_m3": 2700,           # Usata in blocchi
        "densita_solido_kg_m3": 2800,
        "cp_kJ_kgK": 0.95,
        "k_W_mK": 2.5,
        "temp_max_C": 1000,
        "costo_relativo": 2.0,
        "disponibilita": "media",
        "note": "Usata a Pornainen (100 MWh). Eccellente accumulo termico.",
        "validazione": {
            "impianto": "Pornainen",
            "capacita_MWh": 100,
            "massa_t": 2000,
        }
    },
    "basalto": {
        "nome_completo": "Basalto (frantumato)",
        "densita_kg_m3": 2750,
        "densita_solido_kg_m3": 3000,
        "cp_kJ_kgK": 0.87,
        "k_W_mK": 1.5,
        "temp_max_C": 1200,
        "costo_relativo": 1.3,
        "disponibilita": "alta",
        "note": "Usato da Siemens-Gamesa (130 MWh, 750°C)"
    },
    "granito": {
        "nome_completo": "Granito (frantumato)",
        "densita_kg_m3": 2650,
        "densita_solido_kg_m3": 2700,
        "cp_kJ_kgK": 0.82,
        "k_W_mK": 2.5,
        "temp_max_C": 800,
        "costo_relativo": 1.2,
        "disponibilita": "alta",
        "note": "Buona conducibilità, ampiamente disponibile"
    },
    "magnetite": {
        "nome_completo": "Magnetite (Fe3O4)",
        "densita_kg_m3": 4800,
        "densita_solido_kg_m3": 5200,
        "cp_kJ_kgK": 0.65,
        "k_W_mK": 3.0,
        "temp_max_C": 1500,
        "costo_relativo": 3.0,
        "disponibilita": "media",
        "note": "Altissima densità energetica volumetrica, molto pesante"
    },
    "ceramica_refrattaria": {
        "nome_completo": "Ceramica Refrattaria",
        "densita_kg_m3": 2200,
        "densita_solido_kg_m3": 2500,
        "cp_kJ_kgK": 0.95,
        "k_W_mK": 1.0,
        "temp_max_C": 1400,
        "costo_relativo": 2.5,
        "disponibilita": "media",
        "note": "Resistente alle altissime temperature"
    },
}

# =============================================================================
# MATERIALI STRUTTURALI (CONTENITORE, ISOLAMENTO)
# =============================================================================

MATERIALI_STRUTTURALI = {
    "acciaio_carbonio": {
        "nome_completo": "Acciaio al Carbonio",
        "densita_kg_m3": 7850,
        "cp_kJ_kgK": 0.50,
        "k_W_mK": 50.0,
        "temp_max_C": 500,
        "note": "Contenitore esterno e interno"
    },
    "acciaio_inox": {
        "nome_completo": "Acciaio Inossidabile 304/316",
        "densita_kg_m3": 8000,
        "cp_kJ_kgK": 0.50,
        "k_W_mK": 16.0,
        "temp_max_C": 800,
        "note": "Per componenti ad alta temperatura"
    },
    "lana_minerale": {
        "nome_completo": "Lana Minerale (Isolante)",
        "densita_kg_m3": 100,
        "cp_kJ_kgK": 0.84,
        "k_W_mK": 0.04,        # Molto basso = buon isolante
        "temp_max_C": 700,
        "spessore_tipico_m": 0.40,
        "note": "Isolamento principale"
    },
    "lana_ceramica": {
        "nome_completo": "Lana Ceramica (Isolante Alta Temp)",
        "densita_kg_m3": 130,
        "cp_kJ_kgK": 1.0,
        "k_W_mK": 0.08,
        "temp_max_C": 1200,
        "note": "Per zone ad altissima temperatura"
    },
    "cemite_refrattario": {
        "nome_completo": "Cemento Refrattario",
        "densita_kg_m3": 2100,
        "cp_kJ_kgK": 0.88,
        "k_W_mK": 1.0,
        "temp_max_C": 1400,
        "note": "Base/fondazione"
    },
}

# =============================================================================
# FLUIDI (ARIA, ACQUA, FLUIDI TERMOVETTORI)
# =============================================================================

FLUIDI = {
    "aria": {
        "nome_completo": "Aria",
        "densita_kg_m3": 1.2,            # A 20°C
        "cp_kJ_kgK": 1.005,
        "k_W_mK": 0.026,
        "viscosita_Pa_s": 1.8e-5,
        "temp_variabile": True,
        "note": "Fluido termovettore principale in Polar Night Energy",
        # Proprietà variabili con temperatura
        "densita_vs_T": lambda T: 1.2 * 293 / (T + 273),  # kg/m³
        "k_vs_T": lambda T: 0.026 + 0.00007 * T,          # W/(m·K)
    },
    "acqua": {
        "nome_completo": "Acqua",
        "densita_kg_m3": 1000,
        "cp_kJ_kgK": 4.18,
        "k_W_mK": 0.60,
        "viscosita_Pa_s": 1.0e-3,
        "temp_max_C": 100,               # Sotto pressione può essere più alta
        "note": "Fluido nei tubi di scambio termico"
    },
    "olio_diatermico": {
        "nome_completo": "Olio Diatermico",
        "densita_kg_m3": 850,
        "cp_kJ_kgK": 2.1,
        "k_W_mK": 0.12,
        "viscosita_Pa_s": 5.0e-3,
        "temp_max_C": 350,
        "note": "Alternativa per alte temperature"
    },
}

# =============================================================================
# TERRENO / AMBIENTE
# =============================================================================

AMBIENTE = {
    "terreno": {
        "nome_completo": "Terreno/Suolo",
        "densita_kg_m3": 1800,
        "cp_kJ_kgK": 0.84,
        "k_W_mK": 1.5,
        "temp_media_C": 10,              # Temperatura del terreno
        "note": "Sotto e intorno alla batteria"
    },
    "aria_ambiente": {
        "nome_completo": "Aria Ambiente",
        "temp_estate_C": 25,
        "temp_inverno_C": -10,
        "temp_media_C": 10,
        "coefficiente_convezione_W_m2K": 10,  # Convezione naturale
        "note": "Condizioni esterne Finlandia"
    },
}

# =============================================================================
# COMPONENTI ATTIVI (RESISTENZE, TUBI)
# =============================================================================

COMPONENTI_ATTIVI = {
    "resistenze_elettriche": {
        "nome_completo": "Resistenze Elettriche (Riscaldamento)",
        "materiale": "Kanthal/NiCr",
        "temp_max_superficie_C": 1200,
        "efficienza": 0.99,              # Quasi 100%
        "densita_potenza_kW_m": 5,       # kW per metro lineare
        "note": "Convertono elettricità in calore"
    },
    "tubi_scambiatore": {
        "nome_completo": "Tubi Scambiatore di Calore",
        "materiale_tubo": "acciaio_inox",
        "diametro_esterno_mm": 50,
        "spessore_mm": 3,
        "diametro_interno_mm": 44,
        "note": "Per estrazione del calore"
    },
}

# =============================================================================
# PARAMETRI DI IMPACCHETTAMENTO (PACKING)
# =============================================================================

PACKING = {
    "sabbia_sfusa": {
        "packing_factor": 0.60,          # 60% solido, 40% vuoti
        "porosita": 0.40,
    },
    "sabbia_compattata": {
        "packing_factor": 0.65,
        "porosita": 0.35,
    },
    "ghiaia_grossa": {
        "packing_factor": 0.55,
        "porosita": 0.45,
    },
    "sfere_uniformi": {
        "packing_factor": 0.64,          # Random packing
        "porosita": 0.36,
    },
}

# =============================================================================
# DIMENSIONI DI RIFERIMENTO (IMPIANTI REALI)
# =============================================================================

IMPIANTI_RIFERIMENTO = {
    "kankaanpaa_2022": {
        "nome": "Kankaanpää - Prima Sand Battery Commerciale",
        "anno": 2022,
        "potenza_kW": 200,
        "capacita_MWh": 8,
        "diametro_m": 4,
        "altezza_m": 7,
        "massa_t": 100,
        "materiale": "sabbia",
        "efficienza": 0.65,
    },
    "pornainen_2025": {
        "nome": "Pornainen - Steatite Battery",
        "anno": 2025,
        "potenza_kW": 1000,
        "capacita_MWh": 100,
        "diametro_m": 15,              # Stimato
        "altezza_m": 13,               # Stimato
        "massa_t": 2000,
        "materiale": "steatite",
        "efficienza": 0.75,
    },
    "sand_battery_2MW": {
        "nome": "Sand Battery 2 MW (Prodotto)",
        "potenza_kW": 2000,
        "capacita_MWh": 200,
        "footprint_m": "15 x 12",
        "efficienza": 0.85,
    },
    "sand_battery_10MW": {
        "nome": "Sand Battery 10 MW (Prodotto)",
        "potenza_kW": 10000,
        "capacita_MWh": 1000,
        "footprint_m": "30 x 12",
        "efficienza": 0.90,
    },
}

# =============================================================================
# FORMULE DI RIFERIMENTO
# =============================================================================

"""
FORMULA ENERGIA STOCCATA (Sensible Heat):
    E = m * cp * ΔT
    E = ρ * V * cp * ΔT
    
    Dove:
    - E = Energia [J] o [kWh]
    - m = massa [kg]
    - ρ = densità [kg/m³]
    - V = volume [m³]
    - cp = calore specifico [J/(kg·K)] o [kJ/(kg·K)]
    - ΔT = differenza temperatura [K] o [°C]
    
CONVERSIONI:
    1 kWh = 3600 kJ = 3.6 MJ
    1 MWh = 3600 MJ = 3.6 GJ

CONDUZIONE TERMICA (Legge di Fourier):
    q = -k * A * (dT/dx)
    
    Dove:
    - q = flusso termico [W]
    - k = conducibilità termica [W/(m·K)]
    - A = area [m²]
    - dT/dx = gradiente temperatura [K/m]

CONVEZIONE (Legge di Newton):
    q = h * A * (T_s - T_∞)
    
    Dove:
    - h = coefficiente di convezione [W/(m²·K)]
    - T_s = temperatura superficie
    - T_∞ = temperatura fluido

RESISTENZA TERMICA:
    R_cond = L / (k * A)     [K/W] - Conduzione
    R_conv = 1 / (h * A)     [K/W] - Convezione
"""
