"""
benchmark_ab_comparison.py - Confronto A/B tra versioni ottimizzate e baseline

Questo script confronta:
A) Versione BASELINE: NumPy puro, CG + Jacobi, niente AMG
B) Versione OTTIMIZZATA: Numba, CG + AMG con cache

Per identificare esattamente cosa ha causato il rallentamento.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.mesh import Mesh3D, BoundaryType
from src.solver import matrix_builder
from scipy.sparse import linalg as splinalg


def create_test_mesh(Nx, Ny, Nz, L=1.0):
    """Crea una mesh di test."""
    Lx = L
    Ly = L
    Lz = L * Nz / Nx
    
    mesh = Mesh3D(Lx=Lx, Ly=Ly, Lz=Lz, Nx=Nx, Ny=Ny, Nz=Nz, uniform=False)
    
    mesh.k[:] = 0.3
    mesh.rho[:] = 1500
    mesh.cp[:] = 830
    mesh.Q[:] = 0
    
    # Bordi
    for arr in [mesh.boundary_type]:
        arr[0, :, :] = BoundaryType.CONVECTION
        arr[-1, :, :] = BoundaryType.CONVECTION
        arr[:, 0, :] = BoundaryType.CONVECTION
        arr[:, -1, :] = BoundaryType.CONVECTION
        arr[:, :, 0] = BoundaryType.CONVECTION
        arr[:, :, -1] = BoundaryType.CONVECTION
    
    mesh.bc_h[:] = 10.0
    mesh.bc_T_inf[:] = 20.0
    
    # Sorgente
    cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
    r = max(1, min(Nx, Ny, Nz) // 10)
    for i in range(max(0, cx-r), min(Nx, cx+r+1)):
        for j in range(max(0, cy-r), min(Ny, cy+r+1)):
            for k in range(max(0, cz-r), min(Nz, cz+r+1)):
                mesh.Q[i, j, k] = 50000
    
    return mesh


def solve_baseline(A, b, tol=1e-6, maxiter=1000):
    """
    BASELINE: CG + Jacobi (semplice, veloce da setup)
    """
    diag = A.diagonal()
    diag[diag == 0] = 1.0
    M = splinalg.LinearOperator(A.shape, matvec=lambda x: x / diag)
    
    iters = [0]
    def callback(xk):
        iters[0] += 1
    
    t0 = time.perf_counter()
    x, info = splinalg.cg(A, b, rtol=tol, maxiter=maxiter, M=M, callback=callback)
    t1 = time.perf_counter()
    
    return x, t1 - t0, iters[0]


def solve_amg(A, b, tol=1e-6, maxiter=1000, ml_cache=None):
    """
    OTTIMIZZATO: CG + AMG Ruge-Stuben
    Se ml_cache è fornito, riutilizza il solver AMG (warm start)
    """
    import pyamg
    
    t_setup = 0
    if ml_cache is None:
        t0 = time.perf_counter()
        ml = pyamg.ruge_stuben_solver(A)
        t_setup = time.perf_counter() - t0
    else:
        ml = ml_cache
    
    M = ml.aspreconditioner()
    
    iters = [0]
    def callback(xk):
        iters[0] += 1
    
    t0 = time.perf_counter()
    x, info = splinalg.cg(A, b, rtol=tol, maxiter=maxiter, M=M, callback=callback)
    t1 = time.perf_counter()
    
    return x, t1 - t0 + t_setup, iters[0], ml, t_setup


def benchmark_matrix_construction(mesh, use_numba=True):
    """Benchmark costruzione matrice con/senza Numba."""
    original_has_numba = matrix_builder.HAS_NUMBA
    
    try:
        matrix_builder.HAS_NUMBA = use_numba and original_has_numba
        
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            A, b = matrix_builder.build_steady_state_matrix(mesh)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        
        return np.mean(times), A, b
    finally:
        matrix_builder.HAS_NUMBA = original_has_numba


def run_ab_comparison(Nx, Ny, Nz):
    """Esegue confronto A/B."""
    N = Nx * Ny * Nz
    print(f"\n{'='*70}")
    print(f"CONFRONTO A/B - MESH {Nx}x{Ny}x{Nz} = {N:,} celle")
    print(f"{'='*70}")
    
    # Crea mesh
    mesh = create_test_mesh(Nx, Ny, Nz)
    
    # A) BASELINE: NumPy + CG/Jacobi
    print("\n[A] BASELINE (NumPy + CG/Jacobi)")
    print("-" * 40)
    
    t_build_numpy, A, b = benchmark_matrix_construction(mesh, use_numba=False)
    print(f"    Costruzione matrice: {t_build_numpy:.3f}s")
    
    _, t_solve_jacobi, iters_jacobi = solve_baseline(A, b)
    print(f"    Solver CG+Jacobi:    {t_solve_jacobi:.3f}s ({iters_jacobi} iter)")
    
    t_total_baseline = t_build_numpy + t_solve_jacobi
    print(f"    TOTALE BASELINE:     {t_total_baseline:.3f}s")
    
    # B) OTTIMIZZATO: Numba + AMG
    print("\n[B] OTTIMIZZATO (Numba + CG/AMG)")
    print("-" * 40)
    
    # Warmup Numba se necessario
    if matrix_builder.HAS_NUMBA and N > 10000:
        _ = benchmark_matrix_construction(mesh, use_numba=True)
    
    t_build_numba, A, b = benchmark_matrix_construction(mesh, use_numba=True)
    print(f"    Costruzione matrice: {t_build_numba:.3f}s")
    
    try:
        _, t_solve_amg, iters_amg, ml, t_amg_setup = solve_amg(A, b)
        print(f"    Solver CG+AMG:       {t_solve_amg:.3f}s ({iters_amg} iter)")
        print(f"      - di cui setup:    {t_amg_setup:.3f}s")
        
        t_total_opt = t_build_numba + t_solve_amg
    except ImportError:
        print("    AMG non disponibile")
        t_total_opt = t_build_numba + t_solve_jacobi
    
    print(f"    TOTALE OTTIMIZZATO:  {t_total_opt:.3f}s")
    
    # C) IBRIDO: NumPy + AMG con cache (warm start simulato)
    print("\n[C] IBRIDO (NumPy build + AMG con cache)")
    print("-" * 40)
    
    t_build_numpy2, A2, b2 = benchmark_matrix_construction(mesh, use_numba=False)
    print(f"    Costruzione matrice: {t_build_numpy2:.3f}s")
    
    try:
        # Prima chiamata con setup
        _, t_solve_amg1, iters_amg1, ml, t_setup1 = solve_amg(A2, b2)
        # Seconda chiamata con cache (warm start)
        _, t_solve_amg2, iters_amg2, _, t_setup2 = solve_amg(A2, b2, ml_cache=ml)
        print(f"    Solver AMG (cold):   {t_solve_amg1:.3f}s (setup={t_setup1:.3f}s)")
        print(f"    Solver AMG (warm):   {t_solve_amg2:.3f}s (setup={t_setup2:.3f}s)")
        
        t_total_hybrid = t_build_numpy2 + t_solve_amg2
    except ImportError:
        t_total_hybrid = t_build_numpy2 + t_solve_jacobi
    
    print(f"    TOTALE IBRIDO:       {t_total_hybrid:.3f}s")
    
    # Sommario
    print("\n" + "="*70)
    print("SOMMARIO")
    print("="*70)
    print(f"  [A] BASELINE:    {t_total_baseline:.3f}s (riferimento)")
    print(f"  [B] OTTIMIZZATO: {t_total_opt:.3f}s ({t_total_baseline/t_total_opt:.2f}x)")
    print(f"  [C] IBRIDO:      {t_total_hybrid:.3f}s ({t_total_baseline/t_total_hybrid:.2f}x)")
    
    if t_total_opt > t_total_baseline:
        print(f"\n  ⚠️  ATTENZIONE: Ottimizzato è {t_total_opt/t_total_baseline:.2f}x PIÙ LENTO!")
        print(f"      Causa principale: AMG setup time ({t_amg_setup:.3f}s)")
    
    return {
        'baseline': t_total_baseline,
        'optimized': t_total_opt,
        'hybrid': t_total_hybrid
    }


if __name__ == '__main__':
    print("="*70)
    print(" CONFRONTO A/B: BASELINE vs OTTIMIZZATO")
    print("="*70)
    
    # Warmup
    print("\n>>> Warmup...")
    _ = run_ab_comparison(20, 20, 10)
    
    # Test varie dimensioni
    for dims in [(30, 30, 15), (50, 50, 25), (70, 70, 35), (100, 100, 50)]:
        run_ab_comparison(*dims)
    
    print("\n" + "="*70)
    print(" CONCLUSIONI")
    print("="*70)
    print("""
PROBLEMI IDENTIFICATI:
1. AMG Setup Time: Per mesh < 200k celle, il tempo di setup AMG
   supera il risparmio sulle iterazioni. AMG conviene solo per
   mesh molto grandi o risoluzioni ripetute (warm start).

2. Numba per mesh piccole: L'overhead di scheduling thread e
   JIT compilation non viene ammortizzato per mesh < 50k celle.

SOLUZIONI PROPOSTE:
- Usare Jacobi per mesh < 200k celle
- Usare AMG solo per mesh > 200k celle O con warm start
- Disabilitare Numba per mesh < 50k celle
- Implementare soglie automatiche nel solver
""")
