"""
transient.py - Solver per simulazioni transitorie

=============================================================================
TRANSIENT HEAT EQUATION SOLVER
=============================================================================

Risolve l'equazione del calore dipendente dal tempo:

    ρ·cp·∂T/∂t = ∇·(k∇T) + Q

SCHEMA NUMERICO:
    Backward Euler (implicito, incondizionatamente stabile):
    
    ρ·cp·(T^{n+1} - T^n)/dt = ∇·(k∇T^{n+1}) + Q^{n+1}
    
    Riorganizzando:
    (ρ·cp/dt)·T^{n+1} - ∇·(k∇T^{n+1}) = (ρ·cp/dt)·T^n + Q^{n+1}
    
    In forma matriciale:
    (M/dt + A)·T^{n+1} = (M/dt)·T^n + b

    dove:
    - M = matrice massa (diagonale: ρ·cp·V)
    - A = matrice rigidità (da FDM stazionario)
    - b = vettore sorgenti

VANTAGGI BACKWARD EULER:
    - Incondizionatamente stabile (dt può essere grande)
    - Un sistema lineare per timestep (riusa AMG)
    - Ideale per problemi diffusivi

=============================================================================
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
from dataclasses import dataclass
from typing import Optional, Callable
import time

from ..core.mesh import Mesh3D, MaterialID
from ..core.profiles import TransientConfig, PowerProfile, ExtractionProfile
from ..io.state_manager import TransientResults, SimulationState, StateManager
from ..analysis.energy_balance import EnergyBalanceAnalyzer
from .matrix_builder import build_steady_state_matrix
from .steady_state import SolverConfig, SolverResult, HAS_PYAMG

if HAS_PYAMG:
    import pyamg


@dataclass
class TransientSolverConfig(SolverConfig):
    """
    Configurazione estesa per solver transitorio.
    
    Eredita da SolverConfig e aggiunge parametri temporali.
    """
    # Parametri temporali
    dt: float = 60.0                # Passo temporale [s]
    t_final: float = 3600.0         # Tempo finale [s]
    
    # Salvataggio
    save_interval: float = 60.0     # Intervallo salvataggio [s]
    save_full_field: bool = False   # Salva campo T completo
    
    # Callbacks
    progress_callback: Optional[Callable] = None  # callback(t, T, results)


class TransientSolver:
    """
    Solver per simulazioni transitorie.
    
    Usa Backward Euler con risoluzione efficiente del sistema lineare.
    Riutilizza la gerarchia AMG tra timestep per efficienza.
    """
    
    def __init__(
        self, 
        mesh: Mesh3D, 
        config: TransientSolverConfig,
        transient_config: TransientConfig
    ):
        """
        Inizializza il solver.
        
        Args:
            mesh: Mesh3D object
            config: Configurazione solver (tolleranza, metodo, etc.)
            transient_config: Configurazione transitorio (dt, profili, etc.)
        """
        self.mesh = mesh
        self.config = config
        self.transient_config = transient_config
        
        # Matrici
        self._A_steady = None   # Matrice rigidità stazionaria
        self._b_steady = None   # Vettore sorgenti stazionario
        self._M = None          # Matrice massa (diagonale)
        self._A_transient = None  # Matrice sistema transitorio
        
        # Cache
        self._amg_hierarchy = None
        self._current_dt = None  # dt usato per costruire A_transient
        
        # Risultati
        self.results = TransientResults()
    
    def build_system(self):
        """
        Costruisce le matrici per il sistema transitorio.
        
        Sistema: (M/dt + A) · T^{n+1} = (M/dt) · T^n + b
        """
        if self.config.verbose:
            print("[BUILD] Costruzione sistema transitorio...")
        
        t_start = time.time()
        
        # 1. Costruisci matrice stazionaria (rigidità)
        self._A_steady, self._b_steady = build_steady_state_matrix(self.mesh)
        
        # 2. Costruisci matrice massa (diagonale)
        self._build_mass_matrix()
        
        # 3. Costruisci matrice transitoria
        self._build_transient_matrix(self.transient_config.dt)
        
        t_build = time.time() - t_start
        
        if self.config.verbose:
            print(f"[BUILD] Completato in {t_build:.3f} s")
            print(f"        Dimensione: {self._A_transient.shape[0]:,}")
    
    def _build_mass_matrix(self):
        """
        Costruisce la matrice massa M = ρ·cp·V.
        
        Per FDM con celle uniformi, M è diagonale.
        """
        N = self.mesh.N_total
        V_cell = self.mesh.d ** 3
        
        # Flatten mesh properties
        rho_flat = self.mesh.rho.flatten(order='F')
        cp_flat = self.mesh.cp.flatten(order='F')
        
        # Diagonale: ρ·cp·V
        m_diag = rho_flat * cp_flat * V_cell
        
        self._M = sparse.diags(m_diag, format='csr')
        
        if self.config.verbose:
            print(f"        Massa totale: {np.sum(m_diag):.2e} J/K")
    
    def _build_transient_matrix(self, dt: float):
        """
        Costruisce la matrice del sistema transitorio.
        
        A_trans = M/dt + A_steady
        """
        if self._current_dt == dt and self._A_transient is not None:
            return  # Già costruita con stesso dt
        
        self._A_transient = self._M / dt + self._A_steady
        self._current_dt = dt
        
        # Invalida cache AMG (matrice cambiata)
        self._amg_hierarchy = None
    
    def _get_preconditioner(self):
        """Costruisce precondizionatore (preferibilmente AMG con cache)"""
        
        if self.config.preconditioner.lower() in ("amg", "amg_rs", "amg_sa"):
            if not HAS_PYAMG:
                return None
            
            if self._amg_hierarchy is None:
                if self.config.verbose:
                    print("  [PRECOND] Costruzione gerarchia AMG...")
                
                t_start = time.time()
                
                if self.config.preconditioner.lower() == "amg_sa":
                    self._amg_hierarchy = pyamg.smoothed_aggregation_solver(
                        self._A_transient,
                        max_coarse=500,
                        max_levels=10
                    )
                else:
                    self._amg_hierarchy = pyamg.ruge_stuben_solver(
                        self._A_transient,
                        max_coarse=500,
                        max_levels=10,
                        strength='symmetric'
                    )
                
                if self.config.verbose:
                    t_amg = time.time() - t_start
                    print(f"  [PRECOND] AMG costruito in {t_amg:.2f} s")
            
            return self._amg_hierarchy.aspreconditioner()
        
        elif self.config.preconditioner.lower() == "jacobi":
            diag = self._A_transient.diagonal()
            diag[np.abs(diag) < 1e-10] = 1.0
            return sparse.diags(1.0 / diag)
        
        return None
    
    def set_initial_condition(self, T0: np.ndarray = None):
        """
        Imposta la condizione iniziale.
        
        Args:
            T0: Campo temperatura iniziale [Nr, Ntheta, Nz]
                Se None, usa la condizione da transient_config
        """
        if T0 is not None:
            self.mesh.T = T0.copy()
        else:
            ic = self.transient_config.initial_condition
            T0 = ic.apply_to_mesh(self.mesh)
            if T0 is not None:
                self.mesh.T = T0
    
    def solve(self) -> TransientResults:
        """
        Esegue la simulazione transitoria completa.
        
        Returns:
            TransientResults con tutti i dati calcolati
        """
        if self._A_transient is None:
            self.build_system()
        
        # Inizializza
        tc = self.transient_config
        dt = tc.dt
        t_final = tc.t_final
        
        t = 0.0
        n_steps = int(t_final / dt) + 1
        next_save_time = 0.0
        
        if self.config.verbose:
            print(f"[TRANSIENT] Inizio simulazione")
            print(f"            t_final={t_final:.0f}s, dt={dt:.1f}s, steps={n_steps}")
        
        t_start = time.time()
        
        # Costruisci precondizionatore (una volta sola se dt fisso)
        M_prec = self._get_preconditioner()
        
        # Prepara vettori
        T_flat = self.mesh.T.flatten(order='F')
        M_over_dt = self._M / dt
        
        # Analizzatore energia
        analyzer = EnergyBalanceAnalyzer(self.mesh, T_ambient=20.0)
        
        # Loop temporale
        step = 0
        E_in_cumulative = 0.0
        E_out_cumulative = 0.0
        E_losses_cumulative = 0.0
        
        while t <= t_final:
            # === SALVA RISULTATI ===
            if t >= next_save_time - 1e-6:
                self._save_timestep(t, analyzer, E_in_cumulative, E_out_cumulative, E_losses_cumulative)
                next_save_time += tc.save_interval
            
            if t >= t_final:
                break
            
            # === AGGIORNA SORGENTI ===
            # Potenza resistenze
            P_heaters = tc.power_profile.get_power(t)
            self._update_heater_power(P_heaters)
            
            # Estrazione tubi
            extraction = tc.extraction_profile.calculate_extraction(
                T_tube_avg=self._get_tube_temperature(),
                n_tubes=self._count_tubes()
            )
            P_extracted = extraction['power']
            
            # === RICOSTRUISCI b CON NUOVE SORGENTI ===
            # b = (M/dt) · T^n + b_sources
            b_sources = self._b_steady.copy()  # Contiene sorgenti aggiornate
            rhs = M_over_dt @ T_flat + b_sources
            
            # === RISOLVI SISTEMA ===
            if self.config.method == "cg":
                T_new, info = splinalg.cg(
                    self._A_transient, rhs, x0=T_flat, M=M_prec,
                    rtol=self.config.tolerance,
                    maxiter=self.config.max_iterations
                )
            elif self.config.method == "bicgstab":
                T_new, info = splinalg.bicgstab(
                    self._A_transient, rhs, x0=T_flat, M=M_prec,
                    rtol=self.config.tolerance,
                    maxiter=self.config.max_iterations
                )
            else:  # gmres
                T_new, info = splinalg.gmres(
                    self._A_transient, rhs, x0=T_flat, M=M_prec,
                    rtol=self.config.tolerance,
                    maxiter=self.config.max_iterations
                )
            
            # === AGGIORNA STATO ===
            T_flat = T_new
            self.mesh.T = self.mesh.unflatten_field(T_flat)
            
            # === CUMULA ENERGIE ===
            E_in_cumulative += P_heaters * dt
            E_out_cumulative += P_extracted * dt
            
            balance = analyzer.compute_full_balance()
            E_losses_cumulative += balance.Q_losses_total * dt
            
            # === AVANZA TEMPO ===
            t += dt
            step += 1
            
            # === CALLBACK PROGRESSO ===
            if self.config.progress_callback:
                progress = int(100 * t / t_final)
                self.config.progress_callback(progress, f"t = {t:.0f} s")
            
            if self.config.verbose and step % 10 == 0:
                T_mean = np.mean(self.mesh.T)
                print(f"  t={t:.0f}s: T_mean={T_mean:.1f}°C, P_in={P_heaters/1000:.1f}kW")
        
        t_total = time.time() - t_start
        
        if self.config.verbose:
            print(f"[TRANSIENT] Completato in {t_total:.1f} s ({n_steps} steps)")
        
        return self.results
    
    def _save_timestep(
        self, 
        t: float, 
        analyzer: EnergyBalanceAnalyzer,
        E_in_cum: float,
        E_out_cum: float,
        E_loss_cum: float
    ):
        """Salva i dati del timestep corrente"""
        
        # Calcola bilancio energia
        balance = analyzer.compute_full_balance()
        
        # Prepara dati
        data = {
            'T_mean_storage': balance.T_mean_storage,
            'T_max': balance.T_max,
            'T_min': balance.T_min,
            'T_mean_shell': balance.T_mean_shell,
            'T_mean_insulation': balance.T_mean_insulation,
            'P_heaters': balance.P_input,
            'P_extracted': 0.0,  # TODO: da extraction
            'Q_losses_top': balance.Q_losses_top,
            'Q_losses_bottom': balance.Q_losses_bottom,
            'Q_losses_side': balance.Q_losses_side,
            'Q_losses_total': balance.Q_losses_total,
            'E_stored': balance.E_stored,
            'E_in_cumulative': E_in_cum,
            'E_out_cumulative': E_out_cum,
            'E_losses_cumulative': E_loss_cum,
            'Ex_stored': balance.Ex_stored,
            'Ex_destroyed': balance.Ex_destroyed,
        }
        
        # Campo T completo (opzionale)
        T_field = self.mesh.T.copy() if self.transient_config.save_full_field else None
        
        self.results.add_timestep(t, data, T_field)
    
    def _update_heater_power(self, P_total: float):
        """
        Aggiorna la potenza delle resistenze nella mesh.
        
        Args:
            P_total: Potenza totale [W]
        """
        # Trova celle heater
        mask = (self.mesh.material == MaterialID.HEATER)
        n_heater_cells = np.sum(mask)
        
        if n_heater_cells == 0:
            return
        
        # Volume totale heaters
        V_cell = self.mesh.d ** 3
        V_heaters = n_heater_cells * V_cell
        
        # Potenza volumetrica
        Q_volumetric = P_total / V_heaters
        
        # Aggiorna
        self.mesh.Q[mask] = Q_volumetric
    
    def _get_tube_temperature(self) -> float:
        """Restituisce temperatura media dei tubi"""
        mask = (self.mesh.material == MaterialID.TUBE)
        if np.any(mask):
            return float(np.mean(self.mesh.T[mask]))
        return 20.0
    
    def _count_tubes(self) -> int:
        """Conta il numero di tubi"""
        # Approssimazione: conta cluster di celle TUBE
        mask = (self.mesh.material == MaterialID.TUBE)
        return max(1, np.sum(mask) // 100)  # Stima grossolana
    
    def solve_steady_as_initial(self) -> SolverResult:
        """
        Risolve lo stazionario per usarlo come condizione iniziale.
        
        Returns:
            SolverResult dallo steady state solver
        """
        from .steady_state import SteadyStateSolver
        
        if self.config.verbose:
            print("[INIT] Calcolo condizione iniziale da stazionario...")
        
        steady_solver = SteadyStateSolver(self.mesh, self.config)
        steady_solver.build_system()
        result = steady_solver.solve()
        
        return result


def run_transient_simulation(
    mesh: Mesh3D,
    solver_config: SolverConfig = None,
    t_final: float = 3600.0,
    dt: float = 60.0,
    save_interval: float = 60.0,
    power_profile: PowerProfile = None,
    extraction_profile: ExtractionProfile = None,
    initial_condition = None,
    callback: Callable = None
) -> TransientResults:
    """
    Funzione di convenienza per eseguire una simulazione transitoria.
    
    Args:
        mesh: Mesh3D inizializzata
        solver_config: Configurazione solver (opzionale)
        t_final: Tempo finale [s]
        dt: Passo temporale [s]
        save_interval: Intervallo salvataggio [s]
        power_profile: Profilo potenza resistenze
        extraction_profile: Profilo estrazione
        initial_condition: Condizione iniziale
        callback: Callback(step, t, T_mean) per ogni step
        
    Returns:
        TransientResults
    """
    from ..core.profiles import TransientConfig, InitialCondition
    
    if solver_config is None:
        solver_config = SolverConfig()
    
    if power_profile is None:
        power_profile = PowerProfile(mode="off")
    
    if extraction_profile is None:
        extraction_profile = ExtractionProfile(mode="off")
    
    if initial_condition is None:
        initial_condition = InitialCondition(mode="uniform", T_uniform=20.0)
    
    # Costruisci TransientConfig
    transient_config = TransientConfig(
        t_final=t_final,
        dt=dt,
        save_interval=save_interval,
        power_profile=power_profile,
        extraction_profile=extraction_profile,
        initial_condition=initial_condition
    )
    
    # Copia config e aggiungi callback
    config = TransientSolverConfig(
        method=solver_config.method,
        tolerance=solver_config.tolerance,
        max_iterations=solver_config.max_iterations,
        preconditioner=solver_config.preconditioner,
        n_threads=getattr(solver_config, 'n_threads', 0),
        verbose=getattr(solver_config, 'verbose', False),
        dt=dt,
        t_final=t_final,
        save_interval=save_interval,
        save_full_field=False,
        progress_callback=None
    )
    
    solver = TransientSolver(mesh, config, transient_config)
    
    # Imposta condizione iniziale
    solver.set_initial_condition()
    
    # Risultati
    results = TransientResults()
    
    # Loop semplificato per la GUI
    t = 0.0
    step = 0
    n_steps = int(t_final / dt) + 1
    
    if config.verbose:
        print(f"[TRANSIENT] Inizio: t_final={t_final:.0f}s, dt={dt:.1f}s")
    
    while t <= t_final:
        # Salva risultati
        T_mean = float(np.mean(mesh.T))
        T_min = float(np.min(mesh.T))
        T_max = float(np.max(mesh.T))
        
        if step % max(1, int(save_interval / dt)) == 0:
            results.times.append(t)
            results.T_mean.append(T_mean)
            results.T_min.append(T_min)
            results.T_max.append(T_max)
        
        if t >= t_final:
            break
        
        # Callback
        if callback:
            callback(step, t, T_mean)
        
        # Aggiorna potenza
        P_heaters = power_profile.get_power(t)
        _update_heater_power_simple(mesh, P_heaters)
        
        # Risolvi un timestep (semplificato)
        # Per ora usa un approccio esplicito semplice
        T_new = _solve_one_timestep(mesh, dt, config.tolerance)
        mesh.T[:] = T_new
        
        t += dt
        step += 1
    
    # Converti a numpy
    results.times = np.array(results.times)
    results.T_mean = np.array(results.T_mean)
    results.T_min = np.array(results.T_min)
    results.T_max = np.array(results.T_max)
    
    return results


def _update_heater_power_simple(mesh: Mesh3D, P_total: float):
    """Aggiorna potenza heaters nella mesh"""
    mask = (mesh.material_id == MaterialID.HEATER.value)
    n_cells = np.sum(mask)
    if n_cells == 0:
        return
    
    V_cell = mesh.d ** 3
    V_heaters = n_cells * V_cell
    Q_volumetric = P_total / V_heaters
    mesh.Q[mask] = Q_volumetric


def _solve_one_timestep(mesh: Mesh3D, dt: float, tol: float) -> np.ndarray:
    """
    Risolve un singolo timestep con Backward Euler.
    
    Sistema: (M/dt + A) · T^{n+1} = (M/dt) · T^n + b
    """
    # Costruisci sistema
    A_steady, b_steady = build_steady_state_matrix(mesh)
    
    # Matrice massa diagonale
    V_cell = mesh.d ** 3
    rho_flat = mesh.rho.flatten(order='F')
    cp_flat = mesh.cp.flatten(order='F')
    m_diag = rho_flat * cp_flat * V_cell
    M = sparse.diags(m_diag, format='csr')
    
    # Matrice transitoria
    M_over_dt = M / dt
    A_trans = M_over_dt + A_steady
    
    # RHS
    T_flat = mesh.T.flatten(order='F')
    rhs = M_over_dt @ T_flat + b_steady
    
    # Risolvi con CG + Jacobi
    diag = A_trans.diagonal()
    diag[np.abs(diag) < 1e-10] = 1.0
    M_prec = sparse.diags(1.0 / diag)
    
    T_new, info = splinalg.cg(A_trans, rhs, x0=T_flat, M=M_prec, rtol=tol, maxiter=2000)
    
    return mesh.unflatten_field(T_new)
