"""
state_manager.py - Gestione salvataggio e caricamento stato simulazione

=============================================================================
STATE MANAGEMENT
=============================================================================

Questo modulo gestisce:
- Salvataggio dello stato completo della simulazione
- Caricamento e validazione di stati precedenti
- Verifica compatibilità geometria

FORMATO FILE: HDF5 (compatto e veloce)
Struttura:
    /metadata
        version, timestamp, name, analysis_type
    /geometry
        mesh_shape, geometry_hash, parameters
    /state
        T (temperature field)
        materials (material map)
    /results
        various computed quantities

=============================================================================
"""

import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json
import hashlib

# h5py è opzionale
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None


@dataclass
class SimulationState:
    """
    Stato completo di una simulazione.
    
    Contiene tutti i dati necessari per:
    - Riprendere una simulazione transitoria
    - Visualizzare risultati salvati
    - Verificare compatibilità con geometria corrente
    """
    # Identificazione
    version: str = "1.0"
    timestamp: str = ""
    name: str = "Untitled"
    description: str = ""
    
    # Tipo analisi
    analysis_type: str = "steady"  # "steady", "losses", "transient"
    simulation_time: float = 0.0   # Tempo simulazione [s] (0 per stazionaria)
    
    # Geometria (per verifica compatibilità)
    geometry_hash: str = ""
    mesh_shape: Tuple[int, int, int] = (0, 0, 0)
    
    # Parametri geometria (per riferimento)
    geometry_params: Dict[str, Any] = None
    
    # Stato termico
    T: np.ndarray = None           # Campo temperatura [Nr, Ntheta, Nz]
    materials: np.ndarray = None   # Mappa materiali [Nr, Ntheta, Nz]
    
    # Risultati calcolati
    results: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.geometry_params is None:
            self.geometry_params = {}
        if self.results is None:
            self.results = {}


class StateManager:
    """
    Manager per salvataggio e caricamento stati simulazione.
    
    Usa formato HDF5 per efficienza e compattezza.
    """
    
    FILE_EXTENSION = ".h5"
    CURRENT_VERSION = "1.0"
    
    @staticmethod
    def compute_geometry_hash(mesh) -> str:
        """
        Calcola hash della geometria per verifica compatibilità.
        
        L'hash è basato su:
        - Dimensioni mesh (Nr, Ntheta, Nz)
        - Coordinate radiali
        - Coordinate z
        - Distribuzione materiali
        
        Args:
            mesh: Mesh3D object
            
        Returns:
            Hash string (SHA256 troncato)
        """
        data = (
            f"{mesh.Nr},{mesh.Ntheta},{mesh.Nz},"
            f"{mesh.r[0]:.6f},{mesh.r[-1]:.6f},"
            f"{mesh.z[0]:.6f},{mesh.z[-1]:.6f},"
            f"{mesh.material.sum()}"
        ).encode()
        
        return hashlib.sha256(data).hexdigest()[:16]
    
    @staticmethod
    def save_state(state: SimulationState, filepath: str) -> bool:
        """
        Salva lo stato su file HDF5.
        
        Args:
            state: SimulationState da salvare
            filepath: Percorso file (estensione .h5 aggiunta se mancante)
            
        Returns:
            True se salvato con successo
        """
        if not HAS_H5PY:
            print("[ERRORE] h5py non installato. Esegui: pip install h5py")
            return False
        
        filepath = Path(filepath)
        if filepath.suffix != StateManager.FILE_EXTENSION:
            filepath = filepath.with_suffix(StateManager.FILE_EXTENSION)
        
        try:
            with h5py.File(filepath, 'w') as f:
                # === METADATA ===
                meta = f.create_group('metadata')
                meta.attrs['version'] = state.version
                meta.attrs['timestamp'] = state.timestamp
                meta.attrs['name'] = state.name
                meta.attrs['description'] = state.description
                meta.attrs['analysis_type'] = state.analysis_type
                meta.attrs['simulation_time'] = state.simulation_time
                
                # === GEOMETRY ===
                geom = f.create_group('geometry')
                geom.attrs['hash'] = state.geometry_hash
                geom.attrs['mesh_shape'] = state.mesh_shape
                
                # Parametri geometria come JSON
                if state.geometry_params:
                    geom.attrs['params_json'] = json.dumps(state.geometry_params)
                
                # === STATE ===
                state_grp = f.create_group('state')
                
                if state.T is not None:
                    # Comprimi con gzip per ridurre dimensione
                    state_grp.create_dataset(
                        'T', data=state.T, 
                        compression='gzip', compression_opts=4
                    )
                
                if state.materials is not None:
                    state_grp.create_dataset(
                        'materials', data=state.materials,
                        compression='gzip', compression_opts=4
                    )
                
                # === RESULTS ===
                if state.results:
                    results_grp = f.create_group('results')
                    for key, value in state.results.items():
                        if isinstance(value, np.ndarray):
                            results_grp.create_dataset(key, data=value)
                        elif isinstance(value, (int, float, str, bool)):
                            results_grp.attrs[key] = value
                        elif isinstance(value, dict):
                            results_grp.attrs[key] = json.dumps(value)
            
            print(f"[SAVE] Stato salvato: {filepath}")
            return True
            
        except Exception as e:
            print(f"[ERRORE] Salvataggio stato: {e}")
            return False
    
    @staticmethod
    def load_state(filepath: str) -> Optional[SimulationState]:
        """
        Carica lo stato da file HDF5.
        
        Args:
            filepath: Percorso file
            
        Returns:
            SimulationState caricato, o None se errore
        """
        if not HAS_H5PY:
            print("[ERRORE] h5py non installato. Esegui: pip install h5py")
            return None
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"[ERRORE] File non trovato: {filepath}")
            return None
        
        try:
            with h5py.File(filepath, 'r') as f:
                state = SimulationState()
                
                # === METADATA ===
                meta = f['metadata']
                state.version = meta.attrs['version']
                state.timestamp = meta.attrs['timestamp']
                state.name = meta.attrs['name']
                state.description = meta.attrs.get('description', '')
                state.analysis_type = meta.attrs['analysis_type']
                state.simulation_time = meta.attrs['simulation_time']
                
                # === GEOMETRY ===
                geom = f['geometry']
                state.geometry_hash = geom.attrs['hash']
                state.mesh_shape = tuple(geom.attrs['mesh_shape'])
                
                if 'params_json' in geom.attrs:
                    state.geometry_params = json.loads(geom.attrs['params_json'])
                
                # === STATE ===
                state_grp = f['state']
                
                if 'T' in state_grp:
                    state.T = state_grp['T'][:]
                
                if 'materials' in state_grp:
                    state.materials = state_grp['materials'][:]
                
                # === RESULTS ===
                state.results = {}
                if 'results' in f:
                    results_grp = f['results']
                    
                    # Dataset
                    for key in results_grp.keys():
                        state.results[key] = results_grp[key][:]
                    
                    # Attributi
                    for key, value in results_grp.attrs.items():
                        if isinstance(value, str) and value.startswith('{'):
                            state.results[key] = json.loads(value)
                        else:
                            state.results[key] = value
            
            print(f"[LOAD] Stato caricato: {filepath}")
            return state
            
        except Exception as e:
            print(f"[ERRORE] Caricamento stato: {e}")
            return None
    
    @staticmethod
    def verify_compatibility(state: SimulationState, mesh) -> Tuple[bool, str]:
        """
        Verifica se uno stato è compatibile con la mesh corrente.
        
        Args:
            state: Stato caricato
            mesh: Mesh3D corrente
            
        Returns:
            (compatibile: bool, messaggio: str)
        """
        # Verifica dimensioni
        current_shape = (mesh.Nr, mesh.Ntheta, mesh.Nz)
        if state.mesh_shape != current_shape:
            return False, (
                f"Dimensioni mesh non corrispondono!\n"
                f"Stato: {state.mesh_shape}\n"
                f"Corrente: {current_shape}"
            )
        
        # Verifica hash geometria
        current_hash = StateManager.compute_geometry_hash(mesh)
        if state.geometry_hash != current_hash:
            return False, (
                f"Geometria modificata dall'ultimo salvataggio.\n"
                f"Hash stato: {state.geometry_hash}\n"
                f"Hash corrente: {current_hash}\n\n"
                f"Vuoi continuare comunque? (Le temperature potrebbero non corrispondere ai materiali)"
            )
        
        return True, "Stato compatibile con geometria corrente."
    
    @staticmethod
    def create_state_from_mesh(
        mesh, 
        name: str = "Untitled",
        analysis_type: str = "steady",
        geometry_params: Dict = None
    ) -> SimulationState:
        """
        Crea uno stato dalla mesh corrente.
        
        Args:
            mesh: Mesh3D object con temperatura risolta
            name: Nome della simulazione
            analysis_type: Tipo di analisi
            geometry_params: Parametri geometria per riferimento
            
        Returns:
            SimulationState
        """
        state = SimulationState(
            name=name,
            analysis_type=analysis_type,
            geometry_hash=StateManager.compute_geometry_hash(mesh),
            mesh_shape=(mesh.Nr, mesh.Ntheta, mesh.Nz),
            geometry_params=geometry_params or {},
            T=mesh.T.copy() if mesh.T is not None else None,
            materials=mesh.material.copy()
        )
        return state
    
    @staticmethod
    def list_saved_states(directory: str) -> list:
        """
        Lista tutti gli stati salvati in una directory.
        
        Args:
            directory: Percorso directory
            
        Returns:
            Lista di dict con info su ogni file
        """
        directory = Path(directory)
        states = []
        
        for filepath in directory.glob(f"*{StateManager.FILE_EXTENSION}"):
            try:
                with h5py.File(filepath, 'r') as f:
                    meta = f['metadata']
                    states.append({
                        'path': str(filepath),
                        'name': meta.attrs['name'],
                        'timestamp': meta.attrs['timestamp'],
                        'analysis_type': meta.attrs['analysis_type'],
                        'simulation_time': meta.attrs['simulation_time']
                    })
            except Exception:
                continue
        
        # Ordina per timestamp (più recente prima)
        states.sort(key=lambda x: x['timestamp'], reverse=True)
        return states


# =============================================================================
# RISULTATI TRANSIENTI
# =============================================================================

@dataclass
class TransientResults:
    """
    Contenitore per risultati simulazione transitoria.
    
    Ogni campo è un array con un valore per ogni timestep salvato.
    """
    # Tempi
    times: np.ndarray = None  # [n_saves]
    
    # Temperature
    T_mean_storage: np.ndarray = None    # T media zona storage
    T_max: np.ndarray = None             # T massima
    T_min: np.ndarray = None             # T minima
    T_mean_shell: np.ndarray = None      # T media shell
    T_mean_insulation: np.ndarray = None # T media isolamento
    
    # Potenze [W]
    P_heaters: np.ndarray = None         # Potenza resistenze
    P_extracted: np.ndarray = None       # Potenza estratta tubi
    Q_losses_top: np.ndarray = None      # Perdite top
    Q_losses_bottom: np.ndarray = None   # Perdite bottom
    Q_losses_side: np.ndarray = None     # Perdite laterali
    Q_losses_total: np.ndarray = None    # Perdite totali
    
    # Energie cumulative [J]
    E_stored: np.ndarray = None          # Energia immagazzinata
    E_in_cumulative: np.ndarray = None   # Energia immessa cumulativa
    E_out_cumulative: np.ndarray = None  # Energia estratta cumulativa
    E_losses_cumulative: np.ndarray = None  # Perdite cumulative
    
    # Exergia [J]
    Ex_stored: np.ndarray = None         # Exergia immagazzinata
    Ex_destroyed: np.ndarray = None      # Exergia distrutta (irreversibilità)
    
    # Campi completi (opzionale, usa molta memoria)
    T_fields: list = None  # Lista di array T per ogni save time
    
    def __post_init__(self):
        if self.T_fields is None:
            self.T_fields = []
    
    def add_timestep(self, t: float, data: dict, T_field: np.ndarray = None):
        """
        Aggiunge i dati di un timestep.
        
        Args:
            t: Tempo corrente [s]
            data: Dict con tutti i valori calcolati
            T_field: Campo temperatura completo (opzionale)
        """
        # Inizializza array se necessario
        if self.times is None:
            self._init_arrays()
        
        # Aggiungi tempo
        self.times = np.append(self.times, t)
        
        # Aggiungi dati
        for key, value in data.items():
            if hasattr(self, key):
                arr = getattr(self, key)
                if arr is not None:
                    setattr(self, key, np.append(arr, value))
        
        # Campo completo (opzionale)
        if T_field is not None:
            self.T_fields.append(T_field.copy())
    
    def _init_arrays(self):
        """Inizializza tutti gli array vuoti"""
        self.times = np.array([])
        self.T_mean_storage = np.array([])
        self.T_max = np.array([])
        self.T_min = np.array([])
        self.T_mean_shell = np.array([])
        self.T_mean_insulation = np.array([])
        self.P_heaters = np.array([])
        self.P_extracted = np.array([])
        self.Q_losses_top = np.array([])
        self.Q_losses_bottom = np.array([])
        self.Q_losses_side = np.array([])
        self.Q_losses_total = np.array([])
        self.E_stored = np.array([])
        self.E_in_cumulative = np.array([])
        self.E_out_cumulative = np.array([])
        self.E_losses_cumulative = np.array([])
        self.Ex_stored = np.array([])
        self.Ex_destroyed = np.array([])
    
    def to_dict(self) -> dict:
        """Converte in dizionario per salvataggio"""
        return {
            'times': self.times,
            'T_mean_storage': self.T_mean_storage,
            'T_max': self.T_max,
            'T_min': self.T_min,
            'T_mean_shell': self.T_mean_shell,
            'T_mean_insulation': self.T_mean_insulation,
            'P_heaters': self.P_heaters,
            'P_extracted': self.P_extracted,
            'Q_losses_top': self.Q_losses_top,
            'Q_losses_bottom': self.Q_losses_bottom,
            'Q_losses_side': self.Q_losses_side,
            'Q_losses_total': self.Q_losses_total,
            'E_stored': self.E_stored,
            'E_in_cumulative': self.E_in_cumulative,
            'E_out_cumulative': self.E_out_cumulative,
            'E_losses_cumulative': self.E_losses_cumulative,
            'Ex_stored': self.Ex_stored,
            'Ex_destroyed': self.Ex_destroyed,
        }
    
    def export_csv(self, filepath: str):
        """Esporta risultati in formato CSV"""
        filepath = Path(filepath)
        
        # Prepara header e dati
        header = ['time_s']
        data = [self.times]
        
        for key, value in self.to_dict().items():
            if key != 'times' and value is not None and len(value) > 0:
                header.append(key)
                data.append(value)
        
        # Scrivi CSV
        with open(filepath, 'w', newline='') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(header)
            
            for i in range(len(self.times)):
                row = [d[i] if i < len(d) else '' for d in data]
                writer.writerow(row)
        
        print(f"[EXPORT] CSV salvato: {filepath}")
