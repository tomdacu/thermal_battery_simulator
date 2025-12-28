"""
power_balance.py - Analisi del bilancio di potenza e energia

Calcola:
- Potenza immessa (resistenze)
- Potenza estratta (tubi)
- Perdite termiche
- Energia immagazzinata
- Analisi exergetica
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from ..core.mesh import BoundaryType


@dataclass
class PowerBalanceResult:
    """Risultato del bilancio di potenza"""
    # Potenze [W]
    P_input: float          # Potenza resistenze
    P_output: float         # Potenza estratta tubi
    P_loss_top: float       # Perdite superiori
    P_loss_lateral: float   # Perdite laterali
    P_loss_bottom: float    # Perdite inferiori
    P_loss_total: float     # Perdite totali
    
    # Rate of energy storage [W]
    dE_dt: float            # Variazione energia interna
    
    # Bilancio
    imbalance: float        # Errore di bilancio [W]
    imbalance_pct: float    # Errore percentuale
    
    def __repr__(self):
        return f"""PowerBalanceResult:
  P_input:       {self.P_input/1000:.2f} kW
  P_output:      {self.P_output/1000:.2f} kW
  P_loss_total:  {self.P_loss_total/1000:.2f} kW
    - top:       {self.P_loss_top/1000:.3f} kW
    - lateral:   {self.P_loss_lateral/1000:.3f} kW
    - bottom:    {self.P_loss_bottom/1000:.3f} kW
  dE/dt:         {self.dE_dt/1000:.2f} kW
  Imbalance:     {self.imbalance:.2f} W ({self.imbalance_pct:.2f}%)"""


@dataclass
class ExergyResult:
    """Risultato dell'analisi exergetica"""
    # Exergia [J o W]
    Ex_stored: float        # Exergia immagazzinata
    Ex_input: float         # Exergia in ingresso
    Ex_output: float        # Exergia in uscita
    Ex_destroyed: float     # Exergia distrutta (irreversibilità)
    Ex_loss: float          # Exergia persa (perdite termiche)
    
    # Efficienze
    eta_II: float           # Efficienza di secondo principio
    
    def __repr__(self):
        return f"""ExergyResult:
  Ex_stored:     {self.Ex_stored/1e6:.2f} MJ
  Ex_input:      {self.Ex_input/1000:.2f} kW
  Ex_output:     {self.Ex_output/1000:.2f} kW
  Ex_destroyed:  {self.Ex_destroyed/1000:.2f} kW
  Ex_loss:       {self.Ex_loss/1000:.2f} kW
  η_II:          {self.eta_II*100:.1f}%"""


class PowerBalanceAnalyzer:
    """
    Analizzatore del bilancio energetico per la Sand Battery.
    
    Calcola tutti i flussi di potenza attraverso i bordi e le sorgenti.
    """
    
    def __init__(self, mesh, T_ref: float = 20.0):
        """
        Args:
            mesh: Mesh3D con soluzione
            T_ref: Temperatura di riferimento per exergia [°C]
        """
        self.mesh = mesh
        self.T_ref = T_ref + 273.15  # Converti in K
    
    def compute_power_balance(self) -> PowerBalanceResult:
        """Calcola il bilancio di potenza completo"""
        mesh = self.mesh
        dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
        dV = dx * dy * dz
        
        # 1. Potenza in ingresso (sorgenti)
        P_input = np.sum(mesh.Q) * dV  # [W]
        
        # 2. Perdite attraverso le superfici
        P_loss_top = self._compute_surface_loss('z_max')
        P_loss_bottom = self._compute_surface_loss('z_min')
        P_loss_lateral = (
            self._compute_surface_loss('x_min') +
            self._compute_surface_loss('x_max') +
            self._compute_surface_loss('y_min') +
            self._compute_surface_loss('y_max')
        )
        P_loss_total = P_loss_top + P_loss_bottom + P_loss_lateral
        
        # 3. Potenza estratta dai tubi (se attivi)
        P_output = self._compute_internal_convection_output()
        
        # 4. Variazione energia (stazionario = 0)
        dE_dt = 0.0
        
        # 5. Bilancio: P_in = P_out + P_loss + dE/dt
        imbalance = P_input - P_output - P_loss_total - dE_dt
        P_ref = max(abs(P_input), abs(P_loss_total), 1e-10)
        imbalance_pct = 100 * abs(imbalance) / P_ref
        
        return PowerBalanceResult(
            P_input=P_input,
            P_output=P_output,
            P_loss_top=P_loss_top,
            P_loss_lateral=P_loss_lateral,
            P_loss_bottom=P_loss_bottom,
            P_loss_total=P_loss_total,
            dE_dt=dE_dt,
            imbalance=imbalance,
            imbalance_pct=imbalance_pct
        )

    def _compute_internal_convection_output(self) -> float:
        """Potenza scambiata da celle CONVECTION interne al dominio.

        Convenzione segni:
        - Ritorna un valore positivo quando il dominio cede calore al fluido (T > T_inf).

        Nota: esclude esplicitamente le facce esterne del dominio (già conteggiate in P_loss_*).
        """
        mesh = self.mesh

        # Maschera celle con BC convettiva
        conv = (mesh.boundary_type == BoundaryType.CONVECTION)
        if not np.any(conv):
            return 0.0

        # Escludi i bordi del dominio (per evitare doppio conteggio con le perdite superficiali)
        internal = conv.copy()
        internal[0, :, :] = False
        internal[-1, :, :] = False
        internal[:, 0, :] = False
        internal[:, -1, :] = False
        internal[:, :, 0] = False
        internal[:, :, -1] = False

        if not np.any(internal):
            return 0.0

        h = mesh.bc_h[internal]
        T = mesh.T[internal]
        T_inf = mesh.bc_T_inf[internal]

        # Coerente con il termine implementato nel solutore: a_conv = h/d con d ~ dx=dy=dz
        # La potenza equivalente per cella: h * A * (T - T_inf) con A ~ d^2.
        # (Se la mesh fosse non-cubica, questo andrebbe generalizzato.)
        dA = mesh.d * mesh.d
        q = h * (T - T_inf) * dA
        return float(np.sum(q))
    
    def _compute_surface_loss(self, face: str) -> float:
        """
        Calcola il flusso di calore attraverso una superficie.
        
        Positivo = calore uscente dal dominio
        """
        mesh = self.mesh
        dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
        
        if face == 'x_min':
            # Superficie a x = 0
            T_surf = mesh.T[0, :, :]
            T_inf = mesh.bc_T_inf[0, :, :]
            h = mesh.bc_h[0, :, :]
            k_surf = mesh.k[0, :, :]
            dA = dy * dz
            # Coefficiente effettivo coerente con lo schema Robin usato nel solutore
            h_eff = (2 * k_surf * h) / (2 * k_surf + h * dx + 1e-30)
            q = h_eff * (T_surf - T_inf) * dA
            return float(np.sum(q))
            
        elif face == 'x_max':
            T_surf = mesh.T[-1, :, :]
            T_inf = mesh.bc_T_inf[-1, :, :]
            h = mesh.bc_h[-1, :, :]
            k_surf = mesh.k[-1, :, :]
            dA = dy * dz
            h_eff = (2 * k_surf * h) / (2 * k_surf + h * dx + 1e-30)
            q = h_eff * (T_surf - T_inf) * dA
            return float(np.sum(q))
            
        elif face == 'y_min':
            T_surf = mesh.T[:, 0, :]
            T_inf = mesh.bc_T_inf[:, 0, :]
            h = mesh.bc_h[:, 0, :]
            k_surf = mesh.k[:, 0, :]
            dA = dx * dz
            h_eff = (2 * k_surf * h) / (2 * k_surf + h * dy + 1e-30)
            q = h_eff * (T_surf - T_inf) * dA
            return float(np.sum(q))
            
        elif face == 'y_max':
            T_surf = mesh.T[:, -1, :]
            T_inf = mesh.bc_T_inf[:, -1, :]
            h = mesh.bc_h[:, -1, :]
            k_surf = mesh.k[:, -1, :]
            dA = dx * dz
            h_eff = (2 * k_surf * h) / (2 * k_surf + h * dy + 1e-30)
            q = h_eff * (T_surf - T_inf) * dA
            return float(np.sum(q))
            
        elif face == 'z_min':
            # Base - conduzione verso terreno (z_min).
            # Coerente con l'approccio usato in steady_state.check_energy_balance:
            # flux_out ≈ k * (T[1]-T[0]) / dz * A
            if mesh.Nz < 2:
                return 0.0
            T0 = mesh.T[:, :, 0]
            T1 = mesh.T[:, :, 1]
            k0 = mesh.k[:, :, 0]
            k1 = mesh.k[:, :, 1]
            k_eff = (2 * k0 * k1) / (k0 + k1 + 1e-30)
            dA = dx * dy
            dT_dz = (T1 - T0) / dz
            q = k_eff * dT_dz * dA
            return float(np.sum(q))
            
        elif face == 'z_max':
            T_surf = mesh.T[:, :, -1]
            T_inf = mesh.bc_T_inf[:, :, -1]
            h = mesh.bc_h[:, :, -1]
            k_surf = mesh.k[:, :, -1]
            dA = dx * dy
            h_eff = (2 * k_surf * h) / (2 * k_surf + h * dz + 1e-30)
            q = h_eff * (T_surf - T_inf) * dA
            return float(np.sum(q))
        
        return 0.0
    
    def compute_stored_energy(self, T_ref: Optional[float] = None) -> Dict[str, float]:
        """
        Calcola l'energia termica immagazzinata.
        
        Args:
            T_ref: Temperatura di riferimento [°C], default ambiente
            
        Returns:
            Dict con energie in J, kWh, MWh
        """
        if T_ref is None:
            T_ref = self.T_ref - 273.15  # Converti da K a °C
        
        mesh = self.mesh
        dV = mesh.dx * mesh.dy * mesh.dz
        
        # Energia = ∫ ρ·cp·(T - T_ref) dV
        dT = mesh.T - T_ref
        E_J = np.sum(mesh.rho * mesh.cp * dT * dV)
        
        return {
            'E_J': E_J,
            'E_kWh': E_J / 3.6e6,
            'E_MWh': E_J / 3.6e9,
        }
    
    def compute_exergy_analysis(self) -> ExergyResult:
        """
        Esegue l'analisi exergetica del sistema.
        
        L'exergia è il lavoro massimo estraibile portando il sistema
        in equilibrio con l'ambiente.
        """
        mesh = self.mesh
        T0 = self.T_ref  # Temperatura ambiente [K]
        dV = mesh.dx * mesh.dy * mesh.dz
        
        # Temperatura in Kelvin
        T_K = mesh.T + 273.15
        
        # Exergia immagazzinata: Ex = m·cp·[(T-T0) - T0·ln(T/T0)]
        # Per ogni cella
        term1 = T_K - T0
        term2 = T0 * np.log(T_K / T0)
        ex_density = mesh.rho * mesh.cp * (term1 - term2)
        Ex_stored = np.sum(ex_density * dV)
        
        # Exergia in ingresso (resistenze): potenza elettrica = exergia pura
        P_input = np.sum(mesh.Q) * dV
        Ex_input = P_input  # Elettricità = exergia pura
        
        # Exergia persa attraverso le superfici
        # Ex_loss = Q_loss * (1 - T0/T_surf)
        balance = self.compute_power_balance()
        
        # Per semplificare, assumiamo T_surf medio
        T_surf_mean = np.mean(mesh.T) + 273.15
        Ex_loss = balance.P_loss_total * (1 - T0 / T_surf_mean)
        
        # Exergia in uscita (dai tubi)
        Ex_output = 0.0  # TODO quando tubi attivi
        
        # Exergia distrutta = Ex_in - Ex_out - Ex_loss - dEx/dt
        Ex_destroyed = Ex_input - Ex_output - Ex_loss
        
        # Efficienza di secondo principio
        # Se non c'è potenza in ingresso, l'efficienza non è definita
        if Ex_input > 1e-10:
            eta_II = (Ex_output + balance.dE_dt) / Ex_input
        else:
            eta_II = 0.0  # Nessun input = nessuna efficienza definita
        
        return ExergyResult(
            Ex_stored=Ex_stored,
            Ex_input=Ex_input,
            Ex_output=Ex_output,
            Ex_destroyed=Ex_destroyed,
            Ex_loss=Ex_loss,
            eta_II=eta_II
        )
    
    def compute_temperature_distribution_stats(self) -> Dict[str, float]:
        """Statistiche sulla distribuzione di temperatura"""
        T = self.mesh.T
        
        return {
            'T_min': float(T.min()),
            'T_max': float(T.max()),
            'T_mean': float(T.mean()),
            'T_std': float(T.std()),
            'T_median': float(np.median(T)),
            'T_10pct': float(np.percentile(T, 10)),
            'T_90pct': float(np.percentile(T, 90)),
        }
    
    def compute_radial_temperature_profile(self, 
                                           center_x: float, 
                                           center_y: float,
                                           z_position: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcola il profilo radiale di temperatura.
        
        Returns:
            r: Array delle distanze radiali [m]
            T: Array delle temperature [°C]
        """
        mesh = self.mesh
        
        # Trova l'indice z più vicino
        k = int(z_position / mesh.dz)
        k = max(0, min(k, mesh.Nz - 1))
        
        # Calcola distanza radiale per ogni punto
        r_vals = []
        T_vals = []
        
        for i in range(mesh.Nx):
            for j in range(mesh.Ny):
                x, y = mesh.x[i], mesh.y[j]
                r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                r_vals.append(r)
                T_vals.append(mesh.T[i, j, k])
        
        # Ordina per raggio
        idx = np.argsort(r_vals)
        r = np.array(r_vals)[idx]
        T = np.array(T_vals)[idx]
        
        return r, T
    
    def compute_vertical_temperature_profile(self,
                                              x: float,
                                              y: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcola il profilo verticale di temperatura.
        
        Returns:
            z: Array delle quote [m]
            T: Array delle temperature [°C]
        """
        mesh = self.mesh
        
        # Trova gli indici x, y più vicini
        i = int(x / mesh.dx)
        j = int(y / mesh.dy)
        i = max(0, min(i, mesh.Nx - 1))
        j = max(0, min(j, mesh.Ny - 1))
        
        return mesh.z.copy(), mesh.T[i, j, :].copy()


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("=== Test Power Balance ===")
    print("Eseguire da main.py per test completo con mesh configurata.")
