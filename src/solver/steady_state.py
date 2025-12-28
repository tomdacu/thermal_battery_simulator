"""
steady_state.py - Solutore per il caso stazionario

=============================================================================
MODULE OVERVIEW
=============================================================================

This module solves the steady-state heat equation:
    -∇·(k∇T) + Q = 0

with appropriate boundary conditions (Dirichlet, Neumann, Convection).

SOLVER METHODS AVAILABLE:
    - "direct": LU factorization via scipy.sparse.linalg.spsolve (best for small/medium systems)
    - "cg": Conjugate Gradient (for symmetric positive definite matrices)
    - "gmres": Generalized Minimal Residual (general sparse systems)
    - "bicgstab": BiConjugate Gradient Stabilized (general sparse systems)

PRECONDITIONERS:
    - "none": No preconditioning
    - "jacobi": Diagonal scaling (fast, weak)
    - "ilu": Incomplete LU factorization (robust, slower)

PERFORMANCE OPTIMIZATION:
    - n_threads: Number of threads for parallel operations (0 = auto)
    - Uses Intel MKL/OpenBLAS multi-threading when available
    - Numba JIT acceleration for matrix construction

DATA FLOW:
    GUI widgets
        → _build_battery_geometry_from_inputs()
        → BatteryGeometry.apply_to_mesh()
        → Mesh3D (k, rho, cp, Q, boundary conditions)
        → SteadyStateSolver.solve()
        → Temperature field T

ALL PARAMETERS FROM GUI:
    The mesh object contains all thermal properties and boundary conditions
    that were configured through the GUI. This solver has NO hardcoded
    thermal parameters.

USAGE:
    config = SolverConfig(method="direct", verbose=True, n_threads=4)
    solver = SteadyStateSolver(mesh, config)
    result = solver.solve()
    # result.T contains the temperature field in Fortran order
=============================================================================
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import time
import os

from ..core.mesh import Mesh3D
from .matrix_builder import build_steady_state_matrix

# Prova a importare PyAMG per multigrid
try:
    import pyamg
    HAS_PYAMG = True
except ImportError:
    HAS_PYAMG = False
    pyamg = None


def set_num_threads(n_threads: int):
    """
    Imposta il numero di thread per operazioni parallele.
    
    Args:
        n_threads: Numero di thread. 
                   0 = auto (tutti i core)
                   -1 = tutti - 1
                   N = esattamente N thread
    """
    import multiprocessing
    n_cpu = multiprocessing.cpu_count()
    
    if n_threads == 0:
        actual_threads = n_cpu
    elif n_threads == -1:
        actual_threads = max(1, n_cpu - 1)
    else:
        actual_threads = max(1, min(n_threads, n_cpu))
    
    # Imposta variabili d'ambiente per vari backend
    os.environ['OMP_NUM_THREADS'] = str(actual_threads)
    os.environ['MKL_NUM_THREADS'] = str(actual_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(actual_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(actual_threads)
    
    # Per Numba (se disponibile)
    try:
        import numba
        numba.set_num_threads(actual_threads)
    except (ImportError, AttributeError):
        pass
    
    return actual_threads


@dataclass
class SolverConfig:
    """Configurazione del solutore
    
    Attributes:
        method: Metodo di soluzione
            - "direct": Soluzione diretta LU. Robusto ma lento per mesh grandi (>50k celle).
            - "cg": Gradiente Coniugato. Il più veloce per matrici simmetriche (il nostro caso!).
            - "bicgstab": BiCGSTAB. Robusto per matrici generali, buon compromesso.
            - "gmres": GMRES. Ottima convergenza ma usa più memoria.
            
        tolerance: Tolleranza relativa per convergenza.
            - 1e-10: Alta precisione, più iterazioni
            - 1e-8: Default, buon compromesso
            - 1e-6: Veloce, sufficiente per visualizzazione
            - 1e-4: Molto veloce, solo stime grossolane
            
        max_iterations: Limite iterazioni per metodi iterativi.
            - Tipicamente 1000-10000
            - Se non converge entro max_iter, restituisce risultato parziale
            
        preconditioner: Precondizionatore per accelerare convergenza.
            - "jacobi": Diagonale. CONSIGLIATO! Veloce e multi-threaded.
            - "none": Nessuno. CG puro, sorprendentemente veloce per eq. calore.
            - "ilu": Incomplete LU. Single-threaded, può essere LENTO!
            - "amg": Algebraic Multigrid (PyAMG). OTTIMO per mesh grandi (>100k celle).
                     Complessità O(N), riduce iterazioni a ~10-20.
            
        n_threads: Numero di thread per parallelismo.
            - 0: Auto (tutti i core)
            - -1: Tutti i core meno uno (lascia il sistema reattivo)
            - N: Esattamente N thread
            
        verbose: Se True, stampa informazioni di progresso.
    """
    method: str = "bicgstab"        # "bicgstab", "cg", "gmres", "direct"
    tolerance: float = 1e-8         # Tolleranza per metodi iterativi
    max_iterations: int = 10000     # Max iterazioni
    preconditioner: str = "jacobi"  # "jacobi", "none", "ilu"
    n_threads: int = 0              # 0=auto, -1=all-1, N=N threads
    verbose: bool = True            # Stampa info


@dataclass
class SolverResult:
    """Risultato della soluzione"""
    T: np.ndarray                   # Campo di temperatura
    converged: bool                 # Convergenza raggiunta
    iterations: int                 # Numero di iterazioni
    residual: float                 # Residuo finale
    solve_time: float               # Tempo di soluzione [s]
    info: Dict[str, Any]            # Info aggiuntive


class SteadyStateSolver:
    """
    Solutore per l'equazione del calore stazionaria.
    
    Supporta metodi diretti e iterativi con precondizionamento.
    """
    
    def __init__(self, mesh: Mesh3D, config: Optional[SolverConfig] = None):
        """
        Inizializza il solutore.
        
        Args:
            mesh: Mesh3D con geometria e condizioni al contorno
            config: Configurazione del solutore
        """
        self.mesh = mesh
        self.config = config if config is not None else SolverConfig()
        
        # Imposta numero di thread
        self._actual_threads = set_num_threads(self.config.n_threads)
        
        # Matrice e RHS (costruiti al primo solve)
        self._A: Optional[sparse.csr_matrix] = None
        self._b: Optional[np.ndarray] = None
        self._preconditioner = None
        
    def build_system(self):
        """Costruisce il sistema lineare A*T = b"""
        if self.config.verbose:
            print(f"Costruzione sistema lineare (usando {self._actual_threads} thread)...")
        
        t_start = time.time()
        self._A, self._b = build_steady_state_matrix(self.mesh)
        t_build = time.time() - t_start
        
        if self.config.verbose:
            print(f"  Tempo costruzione: {t_build:.2f} s")
            print(f"  Dimensione: {self._A.shape[0]} x {self._A.shape[1]}")
            print(f"  Non-zero: {self._A.nnz} ({100*self._A.nnz/self._A.shape[0]**2:.3f}%)")
    
    def solve(self, rebuild: bool = False) -> SolverResult:
        """
        Risolve il sistema.
        
        Args:
            rebuild: Se True, ricostruisce la matrice anche se esiste
            
        Returns:
            SolverResult con temperatura e info
        """
        # Costruisci sistema se necessario
        if self._A is None or rebuild:
            self.build_system()
        
        if self.config.verbose:
            print(f"Risoluzione con metodo: {self.config.method}")
        
        t_start = time.time()
        
        if self.config.method == "direct":
            result = self._solve_direct()
        else:
            result = self._solve_iterative()
        
        t_solve = time.time() - t_start
        result.solve_time = t_solve
        
        # Aggiorna la mesh con la soluzione
        self.mesh.T = self.mesh.unflatten_field(result.T)
        
        if self.config.verbose:
            print(f"  Tempo soluzione: {t_solve:.2f} s")
            print(f"  Convergenza: {result.converged}")
            if result.iterations > 0:
                print(f"  Iterazioni: {result.iterations}")
            print(f"  Residuo: {result.residual:.2e}")
        
        return result
    
    def _solve_direct(self) -> SolverResult:
        """Risolve con metodo diretto (LU sparse)"""
        try:
            T = splinalg.spsolve(self._A, self._b)
            
            # Calcola residuo
            r = self._A @ T - self._b
            residual = np.linalg.norm(r) / max(np.linalg.norm(self._b), 1e-12)
            
            return SolverResult(
                T=T,
                converged=True,
                iterations=0,
                residual=residual,
                solve_time=0.0,
                info={'method': 'direct_spsolve'}
            )
        except Exception as e:
            print(f"Errore solutore diretto: {e}")
            return SolverResult(
                T=np.zeros(self.mesh.N_total),
                converged=False,
                iterations=0,
                residual=np.inf,
                solve_time=0.0,
                info={'error': str(e)}
            )
    
    def _solve_iterative(self) -> SolverResult:
        """Risolve con metodo iterativo"""
        
        # Costruisci precondizionatore
        M = self._get_preconditioner()
        
        # Punto iniziale (temperatura corrente o uniforme)
        # NOTE: la matrice è costruita con l'indicizzazione lineare della mesh
        # (equivalente a ravel(order='F')). Manteniamo coerenza.
        x0 = self.mesh.flatten_field(self.mesh.T)
        
        # Callback per contare iterazioni
        self._iter_count = 0
        def callback(xk):
            self._iter_count += 1
        
        # Seleziona metodo
        method = self.config.method.lower()
        
        def run_solver(M_prec):
            if method == "cg":
                return splinalg.cg(
                    self._A, self._b, x0=x0, M=M_prec,
                    rtol=self.config.tolerance,
                    maxiter=self.config.max_iterations,
                    callback=callback
                )
            elif method == "gmres":
                return splinalg.gmres(
                    self._A, self._b, x0=x0, M=M_prec,
                    rtol=self.config.tolerance,
                    maxiter=self.config.max_iterations,
                    callback=callback
                )
            elif method == "bicgstab":
                return splinalg.bicgstab(
                    self._A, self._b, x0=x0, M=M_prec,
                    rtol=self.config.tolerance,
                    maxiter=self.config.max_iterations,
                    callback=callback
                )
            else:
                raise ValueError(f"Metodo non supportato: {method}")

        try:
            T, info = run_solver(M)
            
            # Calcola residuo
            r = self._A @ T - self._b
            residual = np.linalg.norm(r) / max(np.linalg.norm(self._b), 1e-12)
            converged = (info == 0)

            # Fallback: in alcuni casi ILU può causare breakdown/non-convergenza.
            # In tal caso, ritenta con Jacobi per robustezza.
            if (not converged) and (self.config.preconditioner.lower() == "ilu"):
                try:
                    self._iter_count = 0
                    M_retry = self._get_preconditioner_jacobi()
                    T_retry, info_retry = run_solver(M_retry)
                    r_retry = self._A @ T_retry - self._b
                    residual_retry = np.linalg.norm(r_retry) / max(np.linalg.norm(self._b), 1e-12)
                    if info_retry == 0:
                        T, info, residual, converged = T_retry, info_retry, residual_retry, True
                except Exception:
                    pass
            
            return SolverResult(
                T=T,
                converged=converged,
                iterations=self._iter_count,
                residual=residual,
                solve_time=0.0,
                info={'method': method, 'info_code': info}
            )
            
        except Exception as e:
            print(f"Errore solutore iterativo: {e}")
            return SolverResult(
                T=x0,
                converged=False,
                iterations=self._iter_count,
                residual=np.inf,
                solve_time=0.0,
                info={'error': str(e)}
            )
    
    def _get_preconditioner(self) -> Optional[splinalg.LinearOperator]:
        """Costruisce il precondizionatore"""
        
        prec_type = self.config.preconditioner.lower()
        
        if prec_type == "none":
            return None
            
        elif prec_type == "jacobi":
            # Precondizionatore diagonale (Jacobi)
            diag = self._A.diagonal()
            diag[np.abs(diag) < 1e-10] = 1.0  # Evita divisione per zero
            M_inv = sparse.diags(1.0 / diag)
            return M_inv
            
        elif prec_type == "ilu":
            # Incomplete LU
            try:
                ilu = splinalg.spilu(
                    self._A.tocsc(),
                    drop_tol=1e-4,
                    fill_factor=10
                )
                M = splinalg.LinearOperator(
                    self._A.shape, 
                    matvec=ilu.solve
                )
                return M
            except Exception as e:
                print(f"Errore ILU, uso Jacobi: {e}")
                return self._get_preconditioner_jacobi()
        
        elif prec_type == "amg":
            # Algebraic Multigrid (PyAMG)
            if not HAS_PYAMG:
                print("PyAMG non installato. Installa con: pip install pyamg")
                print("Fallback a Jacobi...")
                return self._get_preconditioner_jacobi()
            
            try:
                if self.config.verbose:
                    print("  Costruzione gerarchia AMG...")
                
                # Smoothed Aggregation è il più robusto per equazioni ellittiche
                ml = pyamg.smoothed_aggregation_solver(
                    self._A,
                    max_coarse=500,  # Dimensione minima mesh grossolana
                    max_levels=10    # Massimo livelli di gerarchia
                )
                
                if self.config.verbose:
                    print(f"  AMG: {ml.levels} livelli, coarse size: {ml.levels[-1].A.shape[0]}")
                
                # Restituisci come precondizionatore
                M = ml.aspreconditioner(cycle='V')  # V-cycle è il più comune
                return M
                
            except Exception as e:
                print(f"Errore AMG, uso Jacobi: {e}")
                return self._get_preconditioner_jacobi()
        
        else:
            print(f"Precondizionatore non riconosciuto: {prec_type}")
            return None
    
    def _get_preconditioner_jacobi(self):
        """Fallback a Jacobi"""
        diag = self._A.diagonal()
        diag[np.abs(diag) < 1e-10] = 1.0
        return sparse.diags(1.0 / diag)
    
    # =========================================================================
    # METODI DI ANALISI
    # =========================================================================
    
    def compute_residual(self, T: Optional[np.ndarray] = None) -> float:
        """Calcola il residuo ||A*T - b||"""
        if T is None:
            T = self.mesh.flatten_field(self.mesh.T)
        r = self._A @ T - self._b
        return np.linalg.norm(r) / max(np.linalg.norm(self._b), 1e-12)
    
    def get_temperature_stats(self) -> Dict[str, float]:
        """Restituisce statistiche sul campo di temperatura"""
        T = self.mesh.T
        return {
            'T_min': float(T.min()),
            'T_max': float(T.max()),
            'T_mean': float(T.mean()),
            'T_std': float(T.std()),
        }
    
    def check_energy_balance(self) -> Dict[str, float]:
        """
        Verifica il bilancio energetico.
        
        Per caso stazionario: P_in = P_out
        """
        dx, dy, dz = self.mesh.dx, self.mesh.dy, self.mesh.dz
        dV = dx * dy * dz
        
        # Potenza sorgenti interne
        P_source = np.sum(self.mesh.Q) * dV  # [W]
        
        # Potenza uscente dai bordi (approssimata)
        T = self.mesh.T
        k = self.mesh.k
        
        # Flusso attraverso ogni faccia
        # Faccia z_max
        q_z_max = np.sum(self.mesh.bc_h[:, :, -1] * 
                        (T[:, :, -1] - self.mesh.bc_T_inf[:, :, -1])) * dx * dy
        
        # Faccia z_min (conduzione verso terreno)
        dT_dz = (T[:, :, 1] - T[:, :, 0]) / dz
        q_z_min = np.sum(k[:, :, 0] * dT_dz) * dx * dy
        
        # Facce laterali x_min, x_max, y_min, y_max
        q_x_min = np.sum(self.mesh.bc_h[0, :, :] * 
                        (T[0, :, :] - self.mesh.bc_T_inf[0, :, :])) * dy * dz
        q_x_max = np.sum(self.mesh.bc_h[-1, :, :] * 
                        (T[-1, :, :] - self.mesh.bc_T_inf[-1, :, :])) * dy * dz
        q_y_min = np.sum(self.mesh.bc_h[:, 0, :] * 
                        (T[:, 0, :] - self.mesh.bc_T_inf[:, 0, :])) * dx * dz
        q_y_max = np.sum(self.mesh.bc_h[:, -1, :] * 
                        (T[:, -1, :] - self.mesh.bc_T_inf[:, -1, :])) * dx * dz
        q_lateral = q_x_min + q_x_max + q_y_min + q_y_max
        
        P_out = q_z_max + q_z_min + q_lateral
        
        return {
            'P_source_W': P_source,
            'P_out_W': P_out,
            'imbalance_W': P_source - P_out,
            'imbalance_pct': 100 * abs(P_source - P_out) / max(abs(P_source), 1e-10)
        }


# =============================================================================
# FUNZIONI DI UTILITÀ
# =============================================================================

def solve_steady_state(mesh: Mesh3D, 
                       method: str = "direct",
                       verbose: bool = True) -> SolverResult:
    """
    Funzione wrapper per risolvere rapidamente il caso stazionario.
    
    Args:
        mesh: Mesh3D configurata
        method: Metodo di soluzione
        verbose: Stampa info
        
    Returns:
        SolverResult
    """
    config = SolverConfig(method=method, verbose=verbose)
    solver = SteadyStateSolver(mesh, config)
    return solver.solve()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETATO")
    print("=" * 60)
