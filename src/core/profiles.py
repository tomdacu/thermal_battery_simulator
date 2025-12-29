"""
profiles.py - Profili temporali per potenza e estrazione

=============================================================================
POWER AND EXTRACTION PROFILES
=============================================================================

Questo modulo definisce le classi per gestire:
- Profili di potenza delle resistenze nel tempo P(t)
- Profili di estrazione energia dai tubi (flow rate, potenza, temperatura)

Supporta tre modalità per ogni profilo:
1. Costante: valore fisso per tutta la simulazione
2. Schedulato: lista di (tempo, valore) con interpolazione
3. Da file CSV: importa profilo esterno

=============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
from pathlib import Path
import csv


@dataclass
class PowerProfile:
    """
    Profilo potenza resistenze nel tempo.
    
    Attributes:
        mode: Modalità di controllo
            - "off": resistenze spente
            - "constant": potenza costante
            - "schedule": profilo schedulato
            - "csv": da file CSV
        constant_power: Potenza costante [W] (se mode="constant")
        schedule: Lista [(t1, P1), (t2, P2), ...] con interpolazione lineare
        csv_path: Percorso file CSV con colonne (time, power)
        
    Note:
        Il profilo schedulato interpola linearmente tra i punti.
        Prima del primo punto e dopo l'ultimo, usa il valore più vicino.
    """
    mode: str = "constant"
    constant_power: float = 10000.0  # 10 kW default
    schedule: List[Tuple[float, float]] = field(default_factory=list)
    csv_path: Optional[str] = None
    
    # Cache per interpolazione veloce
    _times: np.ndarray = field(default=None, repr=False)
    _powers: np.ndarray = field(default=None, repr=False)
    
    def __post_init__(self):
        """Inizializza cache dopo creazione"""
        self._build_cache()
    
    def _build_cache(self):
        """Costruisce array numpy per interpolazione veloce"""
        if self.mode == "schedule" and self.schedule:
            self._times = np.array([t for t, p in self.schedule])
            self._powers = np.array([p for t, p in self.schedule])
        elif self.mode == "csv" and self.csv_path:
            self._load_csv()
    
    def _load_csv(self):
        """Carica profilo da file CSV"""
        try:
            data = np.loadtxt(self.csv_path, delimiter=',', skiprows=1)
            self._times = data[:, 0]
            self._powers = data[:, 1]
        except Exception as e:
            print(f"[ERRORE] Caricamento CSV potenza: {e}")
            self._times = np.array([0.0])
            self._powers = np.array([0.0])
    
    def get_power(self, t: float) -> float:
        """
        Restituisce la potenza al tempo t.
        
        Args:
            t: Tempo [s]
            
        Returns:
            Potenza [W]
        """
        if self.mode == "off":
            return 0.0
        elif self.mode == "constant":
            return self.constant_power
        elif self.mode in ("schedule", "csv"):
            if self._times is None or len(self._times) == 0:
                return 0.0
            return float(np.interp(t, self._times, self._powers))
        return 0.0
    
    def get_power_array(self, times: np.ndarray) -> np.ndarray:
        """
        Restituisce la potenza per un array di tempi (vettorizzato).
        
        Args:
            times: Array di tempi [s]
            
        Returns:
            Array di potenze [W]
        """
        if self.mode == "off":
            return np.zeros_like(times)
        elif self.mode == "constant":
            return np.full_like(times, self.constant_power)
        elif self.mode in ("schedule", "csv"):
            if self._times is None or len(self._times) == 0:
                return np.zeros_like(times)
            return np.interp(times, self._times, self._powers)
        return np.zeros_like(times)
    
    def add_schedule_point(self, t: float, power: float):
        """Aggiunge un punto al profilo schedulato"""
        self.schedule.append((t, power))
        self.schedule.sort(key=lambda x: x[0])  # Ordina per tempo
        self._build_cache()
    
    def clear_schedule(self):
        """Pulisce il profilo schedulato"""
        self.schedule = []
        self._times = None
        self._powers = None


@dataclass
class ExtractionProfile:
    """
    Profilo estrazione energia dai tubi.
    
    Attributes:
        mode: Modalità di controllo
            - "off": nessuna estrazione
            - "power": impongo potenza estratta, calcolo flow rate
            - "flow_rate": impongo flow rate, calcolo potenza
            - "temperature": impongo T outlet, calcolo flow rate
        
        power: Potenza da estrarre [W] (se mode="power")
        flow_rate: Portata massica [kg/s] per tubo (se mode="flow_rate")
        flow_rate_total: Portata massica totale [kg/s] (alternativa a per-tubo)
        T_inlet: Temperatura ingresso fluido [°C]
        T_outlet_target: Temperatura uscita target [°C] (se mode="temperature")
        
        fluid: Tipo di fluido ("water", "oil", "air")
        
    Note:
        Per mode="power", il solver calcola il flow rate necessario:
            flow_rate = P / (cp * (T_tube - T_inlet))
            
        Per mode="temperature", calcola il flow rate per raggiungere T_outlet:
            flow_rate = P_available / (cp * (T_outlet - T_inlet))
    """
    mode: str = "off"
    
    # Parametri per diverse modalità
    power: float = 0.0              # [W] potenza estratta
    flow_rate: float = 0.0          # [kg/s] per tubo
    flow_rate_total: float = 0.0    # [kg/s] totale
    T_inlet: float = 20.0           # [°C] temperatura ingresso
    T_outlet_target: float = 80.0   # [°C] temperatura uscita target
    
    # Fluido
    fluid: str = "water"
    
    # Profilo temporale (opzionale)
    time_profile: Optional[PowerProfile] = None  # Riusa PowerProfile per P(t) estrazione
    
    def get_fluid_properties(self) -> Tuple[float, float]:
        """
        Restituisce le proprietà del fluido.
        
        Returns:
            (cp [J/kg·K], rho [kg/m³])
        """
        fluids = {
            "water": (4186.0, 1000.0),      # Acqua
            "oil": (2000.0, 850.0),          # Olio termico
            "air": (1005.0, 1.2),             # Aria
            "glycol_30": (3600.0, 1050.0),   # Glicole 30%
        }
        return fluids.get(self.fluid, (4186.0, 1000.0))
    
    def calculate_extraction(self, T_tube_avg: float, n_tubes: int = 1) -> dict:
        """
        Calcola i parametri di estrazione dato lo stato corrente.
        
        Args:
            T_tube_avg: Temperatura media superficie tubi [°C]
            n_tubes: Numero di tubi
            
        Returns:
            Dict con:
                - power: potenza estratta [W]
                - flow_rate_per_tube: portata per tubo [kg/s]
                - flow_rate_total: portata totale [kg/s]
                - T_outlet: temperatura uscita [°C]
        """
        cp, rho = self.get_fluid_properties()
        result = {
            'power': 0.0,
            'flow_rate_per_tube': 0.0,
            'flow_rate_total': 0.0,
            'T_outlet': self.T_inlet
        }
        
        if self.mode == "off":
            return result
        
        # Differenza temperatura disponibile
        dT_available = max(T_tube_avg - self.T_inlet, 0.1)  # Minimo 0.1°C
        
        if self.mode == "power":
            # Impongo potenza, calcolo flow rate
            P = self.power
            m_dot_total = P / (cp * dT_available)
            result['power'] = P
            result['flow_rate_total'] = m_dot_total
            result['flow_rate_per_tube'] = m_dot_total / max(n_tubes, 1)
            result['T_outlet'] = self.T_inlet + dT_available
            
        elif self.mode == "flow_rate":
            # Impongo flow rate, calcolo potenza
            if self.flow_rate_total > 0:
                m_dot_total = self.flow_rate_total
            else:
                m_dot_total = self.flow_rate * n_tubes
            
            P = m_dot_total * cp * dT_available
            result['power'] = P
            result['flow_rate_total'] = m_dot_total
            result['flow_rate_per_tube'] = m_dot_total / max(n_tubes, 1)
            result['T_outlet'] = self.T_inlet + dT_available
            
        elif self.mode == "temperature":
            # Impongo T_outlet, calcolo flow rate necessario
            dT_target = self.T_outlet_target - self.T_inlet
            if dT_target <= 0:
                return result  # Configurazione non valida
            
            # Potenza disponibile basata su differenza con tubo
            P_available = 1e6  # Placeholder - dovrebbe venire dal solver
            
            m_dot_total = P_available / (cp * dT_target)
            result['power'] = P_available
            result['flow_rate_total'] = m_dot_total
            result['flow_rate_per_tube'] = m_dot_total / max(n_tubes, 1)
            result['T_outlet'] = self.T_outlet_target
        
        return result


@dataclass  
class InitialCondition:
    """
    Condizione iniziale per simulazione transitoria.
    
    Attributes:
        mode: Modalità di inizializzazione
            - "uniform": temperatura uniforme ovunque
            - "by_material": temperatura diversa per ogni materiale
            - "from_file": carica da file salvato
            - "from_steady": calcola da stazionario
            
        T_uniform: Temperatura uniforme [°C] (se mode="uniform")
        T_by_material: Dict {MaterialID: T} (se mode="by_material")
        file_path: Percorso file stato (se mode="from_file")
    """
    mode: str = "uniform"
    T_uniform: float = 20.0
    
    # Temperature per materiale
    T_sand: float = 20.0        # Sabbia/storage
    T_insulation: float = 20.0  # Isolamento
    T_steel: float = 20.0       # Acciaio shell
    T_air: float = 20.0         # Aria
    T_ground: float = 15.0      # Terreno
    T_concrete: float = 20.0    # Calcestruzzo
    
    # File path per caricamento
    file_path: Optional[str] = None
    
    def get_T_for_material(self, material_id: int) -> float:
        """
        Restituisce la temperatura iniziale per un materiale.
        
        Args:
            material_id: ID del materiale (da MaterialID enum)
            
        Returns:
            Temperatura [°C]
        """
        from ..core.mesh import MaterialID
        
        if self.mode == "uniform":
            return self.T_uniform
        
        # Mappa MaterialID -> temperatura
        material_temps = {
            MaterialID.SAND: self.T_sand,
            MaterialID.ITE_INSULATION: self.T_insulation,
            MaterialID.ITE_TOP_INSULATION: self.T_insulation,
            MaterialID.STEEL_SHELL: self.T_steel,
            MaterialID.AIR: self.T_air,
            MaterialID.GROUND: self.T_ground,
            MaterialID.SLAB_BOTTOM: self.T_concrete,
            MaterialID.SLAB_TOP: self.T_concrete,
            MaterialID.HEATER: self.T_sand,  # Resistenze = T sabbia
            MaterialID.TUBE: self.T_steel,    # Tubi = T acciaio
        }
        
        return material_temps.get(material_id, self.T_uniform)
    
    def apply_to_mesh(self, mesh) -> np.ndarray:
        """
        Applica la condizione iniziale alla mesh.
        
        Args:
            mesh: Mesh3D object
            
        Returns:
            Array temperatura [Nr, Ntheta, Nz]
        """
        if self.mode == "uniform":
            return np.full((mesh.Nr, mesh.Ntheta, mesh.Nz), self.T_uniform)
        
        elif self.mode == "by_material":
            T = np.zeros((mesh.Nr, mesh.Ntheta, mesh.Nz))
            for i in range(mesh.Nr):
                for j in range(mesh.Ntheta):
                    for k in range(mesh.Nz):
                        mat_id = mesh.material[i, j, k]
                        T[i, j, k] = self.get_T_for_material(mat_id)
            return T
        
        elif self.mode == "from_file":
            # Sarà gestito dal StateManager
            return None
        
        elif self.mode == "from_steady":
            # Sarà gestito dal solver
            return None
        
        return np.full((mesh.Nr, mesh.Ntheta, mesh.Nz), self.T_uniform)


@dataclass
class TransientConfig:
    """
    Configurazione completa per simulazione transitoria.
    
    Attributes:
        t_final: Tempo finale simulazione [s]
        dt: Passo temporale [s]
        dt_adaptive: Se True, usa dt adattivo
        dt_min: Passo temporale minimo [s] (se adattivo)
        dt_max: Passo temporale massimo [s] (se adattivo)
        
        save_interval: Intervallo salvataggio risultati [s]
        
        initial_condition: Condizione iniziale
        power_profile: Profilo potenza resistenze
        extraction_profile: Profilo estrazione tubi
    """
    # Tempo
    t_final: float = 3600.0  # 1 ora default
    dt: float = 60.0         # 1 minuto default
    dt_adaptive: bool = False
    dt_min: float = 1.0
    dt_max: float = 300.0
    
    # Salvataggio
    save_interval: float = 60.0  # Salva ogni minuto
    save_full_field: bool = False  # Salva campo T completo (usa memoria!)
    
    # Profili
    initial_condition: InitialCondition = field(default_factory=InitialCondition)
    power_profile: PowerProfile = field(default_factory=PowerProfile)
    extraction_profile: ExtractionProfile = field(default_factory=ExtractionProfile)
    
    def get_time_steps(self) -> np.ndarray:
        """Genera array dei tempi per la simulazione"""
        if self.dt_adaptive:
            # Per dt adattivo, restituiamo solo tempo iniziale
            # Il solver gestirà l'adattamento
            return np.array([0.0])
        else:
            return np.arange(0, self.t_final + self.dt, self.dt)
    
    def get_save_times(self) -> np.ndarray:
        """Genera array dei tempi di salvataggio"""
        return np.arange(0, self.t_final + self.save_interval, self.save_interval)
