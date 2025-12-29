"""
benchmark_phases.py - Benchmark per identificare rallentamenti nelle ottimizzazioni

Questo script testa separatamente le diverse fasi del calcolo per una mesh 100x100x50
per identificare quale componente ha causato il rallentamento.

COMPONENTI TESTATI:
1. Creazione mesh e allocazione memoria
2. Costruzione matrice FDM (Fase 2: kernel Numba unificato)
3. Conversione COO -> CSR
4. Risoluzione sistema lineare (Fase 1: AMG, warm start)
5. Tempo totale

CONFRONTO:
- Versione Numba vs NumPy puro
- Versione COO pre-allocata vs liste Python
"""

import numpy as np
import time
import sys
import os

# Aggiungi il path del progetto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.mesh import Mesh3D, BoundaryType
from src.solver import matrix_builder
from src.solver.steady_state import SteadyStateSolver, SolverConfig

def create_test_mesh(Nx, Ny, Nz, L=1.0):
    """Crea una mesh di test con parametri realistici."""
    # Usa uniform=False per poter specificare Nx, Ny, Nz esattamente
    Lx = L
    Ly = L
    Lz = L * Nz / Nx
    
    mesh = Mesh3D(Lx=Lx, Ly=Ly, Lz=Lz, Nx=Nx, Ny=Ny, Nz=Nz, uniform=False)
    
    # Imposta proprietà materiale (sabbia)
    mesh.k[:] = 0.3  # W/(m·K)
    mesh.rho[:] = 1500  # kg/m³
    mesh.cp[:] = 830  # J/(kg·K)
    mesh.Q[:] = 0
    
    # Condizioni al contorno: convezione su tutti i bordi
    # Bordi x
    mesh.boundary_type[0, :, :] = BoundaryType.CONVECTION
    mesh.boundary_type[-1, :, :] = BoundaryType.CONVECTION
    mesh.bc_h[0, :, :] = 10.0  # W/(m²·K)
    mesh.bc_h[-1, :, :] = 10.0
    mesh.bc_T_inf[0, :, :] = 20.0  # °C
    mesh.bc_T_inf[-1, :, :] = 20.0
    
    # Bordi y
    mesh.boundary_type[:, 0, :] = BoundaryType.CONVECTION
    mesh.boundary_type[:, -1, :] = BoundaryType.CONVECTION
    mesh.bc_h[:, 0, :] = 10.0
    mesh.bc_h[:, -1, :] = 10.0
    mesh.bc_T_inf[:, 0, :] = 20.0
    mesh.bc_T_inf[:, -1, :] = 20.0
    
    # Bordi z
    mesh.boundary_type[:, :, 0] = BoundaryType.CONVECTION
    mesh.boundary_type[:, :, -1] = BoundaryType.CONVECTION
    mesh.bc_h[:, :, 0] = 10.0
    mesh.bc_h[:, :, -1] = 10.0
    mesh.bc_T_inf[:, :, 0] = 20.0
    mesh.bc_T_inf[:, :, -1] = 20.0
    
    # Sorgente di calore al centro (simula riscaldatore)
    cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
    r = max(1, min(Nx, Ny, Nz) // 10)
    for i in range(max(0, cx-r), min(Nx, cx+r+1)):
        for j in range(max(0, cy-r), min(Ny, cy+r+1)):
            for k in range(max(0, cz-r), min(Nz, cz+r+1)):
                mesh.Q[i, j, k] = 50000  # W/m³
    
    return mesh


def benchmark_matrix_construction(mesh, n_runs=3, use_numba=True, use_preallocated=True):
    """Benchmark della costruzione matrice."""
    original_has_numba = matrix_builder.HAS_NUMBA
    original_use_vectorized = matrix_builder.USE_VECTORIZED
    
    try:
        # Forza configurazione
        matrix_builder.HAS_NUMBA = use_numba and original_has_numba
        matrix_builder.USE_VECTORIZED = use_preallocated
        
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            A, b = matrix_builder.build_steady_state_matrix(mesh)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        
        return {
            'mean': np.mean(times),
            'min': np.min(times),
            'max': np.max(times),
            'nnz': A.nnz,
            'shape': A.shape
        }
    finally:
        # Ripristina
        matrix_builder.HAS_NUMBA = original_has_numba
        matrix_builder.USE_VECTORIZED = original_use_vectorized


def benchmark_solver(mesh, A, b, method='cg', precond='amg', n_runs=3):
    """Benchmark del solver."""
    from scipy.sparse import linalg as splinalg
    
    times = []
    iters_list = []
    
    for run in range(n_runs):
        t0 = time.perf_counter()
        
        if method == 'direct':
            x = splinalg.spsolve(A, b)
            iters = 1
        else:
            # Costruisci precondizionatore
            if precond == 'amg':
                try:
                    import pyamg
                    ml = pyamg.ruge_stuben_solver(A)
                    M = ml.aspreconditioner()
                except ImportError:
                    M = None
            elif precond == 'jacobi':
                diag = A.diagonal()
                diag[diag == 0] = 1.0
                M = splinalg.LinearOperator(A.shape, matvec=lambda x: x / diag)
            elif precond == 'ilu':
                try:
                    ilu = splinalg.spilu(A.tocsc())
                    M = splinalg.LinearOperator(A.shape, matvec=ilu.solve)
                except Exception:
                    M = None
            else:
                M = None
            
            # Risolvi
            iters = [0]
            def callback(xk):
                iters[0] += 1
            
            if method == 'cg':
                x, info = splinalg.cg(A, b, rtol=1e-6, maxiter=1000, M=M, callback=callback)
            elif method == 'bicgstab':
                x, info = splinalg.bicgstab(A, b, rtol=1e-6, maxiter=1000, M=M, callback=callback)
            elif method == 'gmres':
                x, info = splinalg.gmres(A, b, rtol=1e-6, maxiter=1000, M=M, callback=callback)
            
            iters = iters[0]
        
        t1 = time.perf_counter()
        times.append(t1 - t0)
        iters_list.append(iters)
    
    return {
        'mean': np.mean(times),
        'min': np.min(times),
        'max': np.max(times),
        'iters': np.mean(iters_list)
    }


def run_full_benchmark(Nx, Ny, Nz):
    """Esegue benchmark completo."""
    N = Nx * Ny * Nz
    print(f"\n{'='*70}")
    print(f"BENCHMARK MESH {Nx}x{Ny}x{Nz} = {N:,} celle ({N/1e6:.2f}M)")
    print(f"{'='*70}")
    
    # 1. Creazione mesh
    print("\n[1] CREAZIONE MESH...")
    t0 = time.perf_counter()
    mesh = create_test_mesh(Nx, Ny, Nz)
    t_mesh = time.perf_counter() - t0
    print(f"    Tempo: {t_mesh:.3f}s")
    print(f"    Memoria mesh: ~{mesh.k.nbytes * 4 / 1024**2:.1f} MB")
    
    # 2. Costruzione matrice - confronto versioni
    print("\n[2] COSTRUZIONE MATRICE FDM...")
    
    # 2a. NumPy puro (baseline)
    print("    2a. NumPy puro (baseline)...")
    result_numpy = benchmark_matrix_construction(mesh, use_numba=False, use_preallocated=True)
    print(f"        Tempo: {result_numpy['mean']:.3f}s (min={result_numpy['min']:.3f}, max={result_numpy['max']:.3f})")
    
    # 2b. Numba unificato (Fase 2)
    if matrix_builder.HAS_NUMBA:
        print("    2b. Numba unificato (Fase 2)...")
        # Prima chiamata per compilazione JIT
        _ = benchmark_matrix_construction(mesh, use_numba=True, use_preallocated=True, n_runs=1)
        # Chiamate effettive
        result_numba = benchmark_matrix_construction(mesh, use_numba=True, use_preallocated=True)
        print(f"        Tempo: {result_numba['mean']:.3f}s (min={result_numba['min']:.3f}, max={result_numba['max']:.3f})")
        print(f"        Speedup vs NumPy: {result_numpy['mean'] / result_numba['mean']:.2f}x")
    else:
        print("    2b. Numba NON disponibile!")
        result_numba = result_numpy
    
    print(f"    Matrice: {result_numba['shape'][0]:,} x {result_numba['shape'][1]:,}, nnz = {result_numba['nnz']:,}")
    
    # 3. Costruisci matrice finale per test solver
    print("\n[3] COSTRUZIONE MATRICE FINALE...")
    t0 = time.perf_counter()
    A, b = matrix_builder.build_steady_state_matrix(mesh)
    t_build = time.perf_counter() - t0
    print(f"    Tempo costruzione: {t_build:.3f}s")
    
    # 4. Test solver con diversi metodi
    print("\n[4] BENCHMARK SOLVER...")
    
    # 4a. Diretto (baseline per sistemi piccoli)
    if N < 200000:
        print("    4a. Diretto (LU)...")
        result_direct = benchmark_solver(mesh, A, b, method='direct', precond='none')
        print(f"        Tempo: {result_direct['mean']:.3f}s")
    else:
        print("    4a. Diretto (LU) - SKIPPED (sistema troppo grande)")
        result_direct = {'mean': float('inf')}
    
    # 4b. CG + Jacobi
    print("    4b. CG + Jacobi...")
    result_cg_jacobi = benchmark_solver(mesh, A, b, method='cg', precond='jacobi')
    print(f"        Tempo: {result_cg_jacobi['mean']:.3f}s, Iterazioni: {result_cg_jacobi['iters']:.0f}")
    
    # 4c. CG + AMG (Fase 1)
    try:
        import pyamg
        print("    4c. CG + AMG Ruge-Stuben (Fase 1)...")
        result_cg_amg = benchmark_solver(mesh, A, b, method='cg', precond='amg')
        print(f"        Tempo: {result_cg_amg['mean']:.3f}s, Iterazioni: {result_cg_amg['iters']:.0f}")
        
        # AMG setup time separato
        print("    4d. AMG setup time...")
        t0 = time.perf_counter()
        ml = pyamg.ruge_stuben_solver(A)
        t_amg_setup = time.perf_counter() - t0
        print(f"        Setup AMG: {t_amg_setup:.3f}s")
        
    except ImportError:
        print("    4c. AMG - SKIPPED (pyamg non installato)")
        result_cg_amg = {'mean': float('inf'), 'iters': 0}
        t_amg_setup = 0
    
    # 4e. BiCGSTAB + ILU
    print("    4e. BiCGSTAB + ILU...")
    result_bicg_ilu = benchmark_solver(mesh, A, b, method='bicgstab', precond='ilu')
    print(f"        Tempo: {result_bicg_ilu['mean']:.3f}s, Iterazioni: {result_bicg_ilu['iters']:.0f}")
    
    # 5. Analisi memoria
    print("\n[5] ANALISI MEMORIA...")
    mem_A = A.data.nbytes + A.indices.nbytes + A.indptr.nbytes
    mem_b = b.nbytes
    print(f"    Matrice A (CSR): {mem_A / 1024**2:.1f} MB")
    print(f"    Vettore b: {mem_b / 1024:.1f} KB")
    
    # 6. Sommario
    print("\n[6] SOMMARIO TEMPI")
    print("-" * 50)
    t_total_best = t_mesh + t_build + min(result_cg_jacobi['mean'], 
                                          result_cg_amg.get('mean', float('inf')),
                                          result_bicg_ilu['mean'])
    print(f"    Creazione mesh:     {t_mesh:.3f}s")
    print(f"    Costruzione matrix: {t_build:.3f}s")
    print(f"    Solver (best):      {min(result_cg_jacobi['mean'], result_cg_amg.get('mean', float('inf')), result_bicg_ilu['mean']):.3f}s")
    print(f"    TOTALE STIMATO:     {t_total_best:.3f}s")
    
    return {
        't_mesh': t_mesh,
        't_build': t_build,
        'result_numpy': result_numpy,
        'result_numba': result_numba,
        'result_cg_jacobi': result_cg_jacobi,
        'result_cg_amg': result_cg_amg,
        'result_bicg_ilu': result_bicg_ilu
    }


def test_memory_allocation_pattern(Nx, Ny, Nz):
    """
    Testa il pattern di allocazione memoria nella costruzione matrice.
    
    Questo può identificare problemi di:
    - Allocazioni temporanee eccessive
    - Memory fragmentation
    - Cache thrashing
    """
    print(f"\n{'='*70}")
    print(f"TEST ALLOCAZIONE MEMORIA {Nx}x{Ny}x{Nz}")
    print(f"{'='*70}")
    
    mesh = create_test_mesh(Nx, Ny, Nz)
    N = mesh.N_total
    d = mesh.d
    d2 = d * d
    
    # 1. Test flatten array
    print("\n[1] Flatten array (ravel)...")
    t0 = time.perf_counter()
    k_flat = mesh.k.ravel(order='F')
    Q_flat = mesh.Q.ravel(order='F')
    bc_type_flat = mesh.boundary_type.ravel(order='F')
    t1 = time.perf_counter()
    print(f"    Tempo: {(t1-t0)*1000:.1f}ms")
    
    # 2. Test meshgrid
    print("\n[2] Meshgrid + indici...")
    t0 = time.perf_counter()
    ii, jj, kk = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij')
    ii = ii.ravel(order='F')
    jj = jj.ravel(order='F')
    kk = kk.ravel(order='F')
    t1 = time.perf_counter()
    print(f"    Tempo: {(t1-t0)*1000:.1f}ms")
    print(f"    Memoria meshgrid: {3 * ii.nbytes / 1024**2:.1f} MB")
    
    # 3. Test indici vicini
    print("\n[3] Calcolo indici vicini...")
    t0 = time.perf_counter()
    p_all = ii + jj * Nx + kk * Nx * Ny
    p_W = np.clip(ii - 1, 0, Nx - 1) + jj * Nx + kk * Nx * Ny
    p_E = np.clip(ii + 1, 0, Nx - 1) + jj * Nx + kk * Nx * Ny
    p_S = ii + np.clip(jj - 1, 0, Ny - 1) * Nx + kk * Nx * Ny
    p_N = ii + np.clip(jj + 1, 0, Ny - 1) * Nx + kk * Nx * Ny
    p_D = ii + jj * Nx + np.clip(kk - 1, 0, Nz - 1) * Nx * Ny
    p_U = ii + jj * Nx + np.clip(kk + 1, 0, Nz - 1) * Nx * Ny
    t1 = time.perf_counter()
    print(f"    Tempo: {(t1-t0)*1000:.1f}ms")
    print(f"    Memoria indici: {7 * p_all.nbytes / 1024**2:.1f} MB")
    
    # 4. Test maschere bordi
    print("\n[4] Maschere bordi...")
    t0 = time.perf_counter()
    is_x_min = (ii == 0)
    is_x_max = (ii == Nx - 1)
    is_y_min = (jj == 0)
    is_y_max = (jj == Ny - 1)
    is_z_min = (kk == 0)
    is_z_max = (kk == Nz - 1)
    t1 = time.perf_counter()
    print(f"    Tempo: {(t1-t0)*1000:.1f}ms")
    
    # 5. Test lookup conducibilità
    print("\n[5] Lookup conducibilità vicini...")
    t0 = time.perf_counter()
    k_P = k_flat[p_all]
    k_W = k_flat[p_W]
    k_E = k_flat[p_E]
    k_S = k_flat[p_S]
    k_N = k_flat[p_N]
    k_D = k_flat[p_D]
    k_U = k_flat[p_U]
    t1 = time.perf_counter()
    print(f"    Tempo: {(t1-t0)*1000:.1f}ms")
    print(f"    Memoria k arrays: {7 * k_P.nbytes / 1024**2:.1f} MB")
    
    # 6. Test kernel Numba
    eps = 1e-20
    if matrix_builder.HAS_NUMBA:
        print("\n[6] Kernel Numba (unificato)...")
        # Warmup
        _ = matrix_builder._compute_fdm_coefficients_unified_numba(
            k_P[:1000], k_W[:1000], k_E[:1000], k_S[:1000], k_N[:1000], k_D[:1000], k_U[:1000],
            d2, eps,
            is_x_min[:1000], is_x_max[:1000], is_y_min[:1000], is_y_max[:1000], is_z_min[:1000], is_z_max[:1000]
        )
        
        t0 = time.perf_counter()
        a_W, a_E, a_S, a_N, a_D, a_U, a_P = matrix_builder._compute_fdm_coefficients_unified_numba(
            k_P, k_W, k_E, k_S, k_N, k_D, k_U,
            d2, eps,
            is_x_min, is_x_max, is_y_min, is_y_max, is_z_min, is_z_max
        )
        t1 = time.perf_counter()
        print(f"    Tempo: {(t1-t0)*1000:.1f}ms")
    else:
        print("\n[6] NumPy medie armoniche...")
        t0 = time.perf_counter()
        k_w, k_e, k_s, k_n, k_d, k_u = matrix_builder._compute_harmonic_means_numpy(
            k_P, k_W, k_E, k_S, k_N, k_D, k_U, eps
        )
        a_W = k_w / d2
        a_E = k_e / d2
        a_S = k_s / d2
        a_N = k_n / d2
        a_D = k_d / d2
        a_U = k_u / d2
        a_P = a_W + a_E + a_S + a_N + a_D + a_U
        t1 = time.perf_counter()
        print(f"    Tempo: {(t1-t0)*1000:.1f}ms")
    
    # Memoria totale stimata
    total_mem = (
        mesh.k.nbytes * 4 +  # 4 array mesh
        3 * ii.nbytes +  # meshgrid
        7 * p_all.nbytes +  # indici
        7 * k_P.nbytes +  # conducibilità
        7 * a_W.nbytes  # coefficienti
    )
    print(f"\n    MEMORIA TOTALE STIMATA: {total_mem / 1024**2:.1f} MB")


if __name__ == '__main__':
    print("="*70)
    print(" BENCHMARK OTTIMIZZAZIONI FASE 1 & 2")
    print("="*70)
    print(f"Numba disponibile: {matrix_builder.HAS_NUMBA}")
    try:
        import pyamg
        print(f"PyAMG disponibile: True")
    except ImportError:
        print(f"PyAMG disponibile: False")
    
    # Test mesh piccola per warmup
    print("\n>>> WARMUP con mesh 20x20x10...")
    _ = run_full_benchmark(20, 20, 10)
    
    # Test mesh 50x50x25 (62,500 celle)
    print("\n\n>>> TEST MESH MEDIA 50x50x25...")
    results_medium = run_full_benchmark(50, 50, 25)
    
    # Test mesh 100x100x50 (500,000 celle) - richiesta dall'utente
    print("\n\n>>> TEST MESH GRANDE 100x100x50...")
    results_large = run_full_benchmark(100, 100, 50)
    
    # Test allocazione memoria per mesh grande
    test_memory_allocation_pattern(100, 100, 50)
    
    # Sommario finale
    print("\n" + "="*70)
    print(" SOMMARIO FINALE")
    print("="*70)
    print("\nTempo costruzione matrice (media):")
    print(f"  50x50x25:   NumPy={results_medium['result_numpy']['mean']:.3f}s, Numba={results_medium['result_numba']['mean']:.3f}s")
    print(f"  100x100x50: NumPy={results_large['result_numpy']['mean']:.3f}s, Numba={results_large['result_numba']['mean']:.3f}s")
    
    print("\nTempo solver (CG+AMG):")
    print(f"  50x50x25:   {results_medium['result_cg_amg'].get('mean', 'N/A')}")
    print(f"  100x100x50: {results_large['result_cg_amg'].get('mean', 'N/A')}")
