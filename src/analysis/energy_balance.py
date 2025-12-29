"""
energy_balance.py - Calcolo bilancio energetico, perdite e exergia

=============================================================================
ENERGY AND EXERGY ANALYSIS
=============================================================================

Questo modulo calcola:
- Perdite termiche (top, bottom, laterali)
- Energia immagazzinata
- Exergia (disponibilità energetica)
- Efficienza energetica ed exergetica

EQUAZIONI PRINCIPALI:

Energia immagazzinata:
    E = ∫∫∫ ρ·cp·(T - T_ref) dV

Exergia immagazzinata:
    Ex = ∫∫∫ ρ·cp·[(T - T0) - T0·ln(T/T0)] dV

Exergia distrutta (irreversibilità):
    Ex_destroyed = T0 · S_gen
    dove S_gen = ∫∫∫ k·(∇T)²/T² dV

=============================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from ..core.mesh import Mesh3D, MaterialID, BoundaryType


@dataclass
class EnergyBalanceResult:
    """Risultato del calcolo del bilancio energetico"""
    
    # Potenze [W]
    P_input: float = 0.0            # Potenza immessa (resistenze)
    P_extracted: float = 0.0        # Potenza estratta (tubi)
    
    # Perdite [W]
    Q_losses_top: float = 0.0       # Perdite dalla superficie superiore
    Q_losses_bottom: float = 0.0    # Perdite dalla superficie inferiore
    Q_losses_side: float = 0.0      # Perdite laterali
    Q_losses_total: float = 0.0     # Perdite totali
    
    # Energie [J]
    E_stored: float = 0.0           # Energia termica immagazzinata
    E_storage_capacity: float = 0.0 # Capacità massima teorica
    
    # Temperature [°C]
    T_mean_storage: float = 0.0     # Temperatura media zona storage
    T_max: float = 0.0              # Temperatura massima
    T_min: float = 0.0              # Temperatura minima
    T_mean_shell: float = 0.0       # Temperatura media shell
    T_mean_insulation: float = 0.0  # Temperatura media isolamento
    
    # Exergia [J]
    Ex_stored: float = 0.0          # Exergia immagazzinata
    Ex_input: float = 0.0           # Exergia in ingresso
    Ex_destroyed: float = 0.0       # Exergia distrutta
    
    # Efficienze [-]
    eta_energy: float = 0.0         # Efficienza energetica
    eta_exergy: float = 0.0         # Efficienza exergetica
    
    # Dettagli per materiale
    E_by_material: Dict[int, float] = None
    
    def __post_init__(self):
        if self.E_by_material is None:
            self.E_by_material = {}


class EnergyBalanceAnalyzer:
    """
    Analizzatore del bilancio energetico e exergetico.
    
    Calcola perdite, energie e exergia basandosi sullo stato
    termico della mesh.
    """
    
    # Temperatura di riferimento per exergia [K]
    T0_KELVIN = 293.15  # 20°C
    
    def __init__(self, mesh: Mesh3D, T_ambient: float = 20.0):
        """
        Inizializza l'analizzatore.
        
        Args:
            mesh: Mesh3D con temperatura risolta
            T_ambient: Temperatura ambiente [°C]
        """
        self.mesh = mesh
        self.T_ambient = T_ambient
        self.T0 = T_ambient + 273.15  # Kelvin
    
    def compute_full_balance(self) -> EnergyBalanceResult:
        """
        Calcola il bilancio energetico completo.
        
        Returns:
            EnergyBalanceResult con tutti i calcoli
        """
        result = EnergyBalanceResult()
        
        # Calcola temperature
        result.T_mean_storage = self._compute_mean_T_storage()
        result.T_max = float(np.max(self.mesh.T))
        result.T_min = float(np.min(self.mesh.T))
        result.T_mean_shell = self._compute_mean_T_material(MaterialID.STEEL_SHELL)
        result.T_mean_insulation = self._compute_mean_T_insulation()
        
        # Calcola perdite
        result.Q_losses_top = self._compute_losses_top()
        result.Q_losses_bottom = self._compute_losses_bottom()
        result.Q_losses_side = self._compute_losses_side()
        result.Q_losses_total = (
            result.Q_losses_top + 
            result.Q_losses_bottom + 
            result.Q_losses_side
        )
        
        # Calcola potenza input (da resistenze)
        result.P_input = self._compute_power_input()
        
        # Calcola energia immagazzinata
        result.E_stored, result.E_by_material = self._compute_stored_energy()
        result.E_storage_capacity = self._compute_storage_capacity()
        
        # Calcola exergia
        result.Ex_stored = self._compute_stored_exergy()
        result.Ex_input = self._compute_input_exergy(result.P_input)
        result.Ex_destroyed = self._compute_destroyed_exergy()
        
        # Calcola efficienze
        if result.P_input > 0:
            result.eta_energy = 1.0 - result.Q_losses_total / result.P_input
        if result.Ex_input > 0:
            result.eta_exergy = result.Ex_stored / result.Ex_input
        
        return result
    
    def _compute_mean_T_storage(self) -> float:
        """Calcola temperatura media nella zona di storage (sabbia)"""
        mask = (self.mesh.material == MaterialID.SAND)
        if np.any(mask):
            return float(np.mean(self.mesh.T[mask]))
        return self.T_ambient
    
    def _compute_mean_T_material(self, material_id: int) -> float:
        """Calcola temperatura media per un materiale specifico"""
        mask = (self.mesh.material == material_id)
        if np.any(mask):
            return float(np.mean(self.mesh.T[mask]))
        return self.T_ambient
    
    def _compute_mean_T_insulation(self) -> float:
        """Calcola temperatura media dell'isolamento (tutti i tipi)"""
        mask = (
            (self.mesh.material == MaterialID.ITE_INSULATION) |
            (self.mesh.material == MaterialID.ITE_TOP_INSULATION)
        )
        if np.any(mask):
            return float(np.mean(self.mesh.T[mask]))
        return self.T_ambient
    
    def _compute_losses_top(self) -> float:
        """
        Calcola perdite dalla superficie superiore.
        
        Q = h · A · (T_surface - T_ambient)
        """
        # Trova celle alla superficie superiore
        k_top = self.mesh.Nz - 1
        
        Q_total = 0.0
        d = self.mesh.d
        
        for i in range(self.mesh.Nr):
            for j in range(self.mesh.Ntheta):
                bt = self.mesh.boundary_type[i, j, k_top]
                
                if bt == BoundaryType.CONVECTION:
                    T_surf = self.mesh.T[i, j, k_top]
                    h = self.mesh.h_conv[i, j, k_top]
                    
                    # Area della cella (approssimazione)
                    r = self.mesh.r[i]
                    dtheta = 2 * np.pi / self.mesh.Ntheta
                    A = r * dtheta * d
                    
                    Q_total += h * A * (T_surf - self.T_ambient)
        
        return float(Q_total)
    
    def _compute_losses_bottom(self) -> float:
        """Calcola perdite dalla superficie inferiore"""
        # Trova celle alla superficie inferiore
        k_bot = 0
        
        Q_total = 0.0
        d = self.mesh.d
        
        for i in range(self.mesh.Nr):
            for j in range(self.mesh.Ntheta):
                bt = self.mesh.boundary_type[i, j, k_bot]
                
                if bt in (BoundaryType.CONVECTION, BoundaryType.DIRICHLET):
                    T_surf = self.mesh.T[i, j, k_bot]
                    
                    if bt == BoundaryType.CONVECTION:
                        h = self.mesh.h_conv[i, j, k_bot]
                    else:
                        # Per Dirichlet, stima h da conducibilità
                        h = self.mesh.k[i, j, k_bot] / d
                    
                    r = self.mesh.r[i]
                    dtheta = 2 * np.pi / self.mesh.Ntheta
                    A = r * dtheta * d
                    
                    T_ref = self.mesh.T_bc.get((i, j, k_bot), self.T_ambient)
                    Q_total += h * A * (T_surf - T_ref)
        
        return float(Q_total)
    
    def _compute_losses_side(self) -> float:
        """Calcola perdite dalla superficie laterale"""
        # Celle al bordo esterno (r = r_max)
        i_ext = self.mesh.Nr - 1
        
        Q_total = 0.0
        d = self.mesh.d
        
        for j in range(self.mesh.Ntheta):
            for k in range(self.mesh.Nz):
                bt = self.mesh.boundary_type[i_ext, j, k]
                
                if bt == BoundaryType.CONVECTION:
                    T_surf = self.mesh.T[i_ext, j, k]
                    h = self.mesh.h_conv[i_ext, j, k]
                    
                    # Area laterale
                    r = self.mesh.r[i_ext]
                    dtheta = 2 * np.pi / self.mesh.Ntheta
                    A = r * dtheta * d
                    
                    Q_total += h * A * (T_surf - self.T_ambient)
        
        return float(Q_total)
    
    def _compute_power_input(self) -> float:
        """Calcola potenza totale immessa dalle resistenze"""
        # Somma Q nella zona heater
        mask = (self.mesh.material == MaterialID.HEATER)
        if np.any(mask):
            # Q è potenza volumetrica [W/m³]
            # Volume cella = d³
            V_cell = self.mesh.d ** 3
            Q_total = float(np.sum(self.mesh.Q[mask])) * V_cell
            return Q_total
        return 0.0
    
    def _compute_stored_energy(self) -> Tuple[float, Dict[int, float]]:
        """
        Calcola energia termica immagazzinata.
        
        E = ∫ ρ·cp·(T - T_ref) dV
        
        Returns:
            (energia_totale, dict per materiale)
        """
        T_ref = self.T_ambient
        V_cell = self.mesh.d ** 3
        
        E_total = 0.0
        E_by_material = {}
        
        # Calcola per ogni materiale
        for mat_id in np.unique(self.mesh.material):
            mask = (self.mesh.material == mat_id)
            
            rho = self.mesh.rho[mask]
            cp = self.mesh.cp[mask]
            T = self.mesh.T[mask]
            
            E_mat = float(np.sum(rho * cp * (T - T_ref))) * V_cell
            E_by_material[int(mat_id)] = E_mat
            E_total += E_mat
        
        return E_total, E_by_material
    
    def _compute_storage_capacity(self) -> float:
        """
        Calcola capacità massima teorica di storage.
        
        Assumendo T_max = 600°C per la sabbia
        """
        T_max = 600.0  # °C
        T_ref = self.T_ambient
        V_cell = self.mesh.d ** 3
        
        # Solo zona storage (sabbia)
        mask = (self.mesh.material == MaterialID.SAND)
        if np.any(mask):
            rho = self.mesh.rho[mask]
            cp = self.mesh.cp[mask]
            
            E_max = float(np.sum(rho * cp * (T_max - T_ref))) * V_cell
            return E_max
        return 0.0
    
    def _compute_stored_exergy(self) -> float:
        """
        Calcola exergia immagazzinata.
        
        Ex = ∫ ρ·cp·[(T - T0) - T0·ln(T/T0)] dV
        
        dove T0 è la temperatura ambiente in Kelvin
        """
        V_cell = self.mesh.d ** 3
        T0 = self.T0  # Kelvin
        
        # Solo zona storage
        mask = (self.mesh.material == MaterialID.SAND)
        if not np.any(mask):
            return 0.0
        
        rho = self.mesh.rho[mask]
        cp = self.mesh.cp[mask]
        T_K = self.mesh.T[mask] + 273.15  # Converti in Kelvin
        
        # Evita log di numeri <= 0
        T_K = np.maximum(T_K, 1.0)
        
        # Exergia specifica
        ex_specific = cp * ((T_K - T0) - T0 * np.log(T_K / T0))
        
        Ex_total = float(np.sum(rho * ex_specific)) * V_cell
        return Ex_total
    
    def _compute_input_exergy(self, P_input: float) -> float:
        """
        Calcola exergia in ingresso (da resistenze elettriche).
        
        Per energia elettrica: Ex = P (exergia = energia)
        """
        return P_input  # Elettricità è pura exergia
    
    def _compute_destroyed_exergy(self) -> float:
        """
        Calcola exergia distrutta per irreversibilità.
        
        Ex_destroyed = T0 · S_gen
        S_gen = ∫ k·(∇T)²/T² dV
        
        Approssimazione con differenze finite.
        """
        V_cell = self.mesh.d ** 3
        d = self.mesh.d
        T0 = self.T0
        
        S_gen = 0.0
        
        T_K = self.mesh.T + 273.15  # Kelvin
        
        # Calcola gradiente temperatura (differenze finite)
        for i in range(1, self.mesh.Nr - 1):
            for j in range(self.mesh.Ntheta):
                for k in range(1, self.mesh.Nz - 1):
                    T_P = T_K[i, j, k]
                    
                    if T_P < 1.0:
                        continue
                    
                    # Gradienti
                    dT_dr = (T_K[i+1, j, k] - T_K[i-1, j, k]) / (2 * d)
                    
                    j_prev = (j - 1) % self.mesh.Ntheta
                    j_next = (j + 1) % self.mesh.Ntheta
                    dT_dtheta = (T_K[i, j_next, k] - T_K[i, j_prev, k]) / (2 * d)
                    
                    dT_dz = (T_K[i, j, k+1] - T_K[i, j, k-1]) / (2 * d)
                    
                    # Magnitudine gradiente
                    grad_T_sq = dT_dr**2 + dT_dtheta**2 + dT_dz**2
                    
                    # Conducibilità
                    k_local = self.mesh.k[i, j, k]
                    
                    # Generazione entropia locale
                    S_gen += k_local * grad_T_sq / (T_P ** 2) * V_cell
        
        return float(T0 * S_gen)
    
    def compute_losses_analysis(self, T_storage_target: float) -> EnergyBalanceResult:
        """
        Analisi perdite con temperatura storage imposta.
        
        Utile per stimare le perdite a regime con una certa T media.
        
        Args:
            T_storage_target: Temperatura media target nella sabbia [°C]
            
        Returns:
            EnergyBalanceResult con perdite stimate
        """
        # Questa è un'analisi semplificata
        # Per essere precisi, bisognerebbe risolvere lo stazionario
        # con T imposta nel volume
        
        # Stima perdite con resistenza termica equivalente
        result = EnergyBalanceResult()
        result.T_mean_storage = T_storage_target
        
        dT = T_storage_target - self.T_ambient
        
        # Stima resistenza termica (da geometria)
        # R_th = ln(r_ext/r_int) / (2πkL) per cilindro
        
        # Per ora usa il calcolo standard
        return self.compute_full_balance()
