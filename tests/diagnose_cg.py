"""
diagnose_cg.py - Diagnosi del problema di convergenza CG

Questo script analizza perché CG non converge senza warm start.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.mesh import Mesh3D, BoundaryType
from src.solver import matrix_builder
from scipy.sparse import linalg as splinalg
from scipy import sparse


def create_test_mesh(Nx=50, Ny=50, Nz=25):
    """Crea mesh di test"""
    mesh = Mesh3D(Lx=1.0, Ly=1.0, Lz=0.5, Nx=Nx, Ny=Ny, Nz=Nz, uniform=False)
    
    mesh.k[:] = 0.3
    mesh.rho[:] = 1500
    mesh.cp[:] = 830
    mesh.Q[:] = 0
    
    # BC convezione su bordi
    for bc_type in [mesh.boundary_type]:
        bc_type[0, :, :] = BoundaryType.CONVECTION
        bc_type[-1, :, :] = BoundaryType.CONVECTION
        bc_type[:, 0, :] = BoundaryType.CONVECTION
        bc_type[:, -1, :] = BoundaryType.CONVECTION
        bc_type[:, :, 0] = BoundaryType.CONVECTION
        bc_type[:, :, -1] = BoundaryType.CONVECTION
    
    mesh.bc_h[:] = 10.0
    mesh.bc_T_inf[:] = 20.0
    
    # Sorgente al centro
    cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
    r = max(1, min(Nx, Ny, Nz) // 10)
    for i in range(max(0, cx-r), min(Nx, cx+r+1)):
        for j in range(max(0, cy-r), min(Ny, cy+r+1)):
            for k in range(max(0, cz-r), min(Nz, cz+r+1)):
                mesh.Q[i, j, k] = 100000  # 100 kW/m³
    
    return mesh


def analyze_matrix(A, b):
    """Analizza proprietà della matrice"""
    print("\n" + "="*60)
    print("ANALISI MATRICE")
    print("="*60)
    
    N = A.shape[0]
    print(f"Dimensione: {N:,} x {N:,}")
    print(f"Non-zero: {A.nnz:,} ({100*A.nnz/N**2:.4f}%)")
    
    # Diagonale
    diag = A.diagonal()
    print(f"\nDiagonale:")
    print(f"  min: {diag.min():.6e}")
    print(f"  max: {diag.max():.6e}")
    print(f"  zero/piccoli (<1e-10): {np.sum(np.abs(diag) < 1e-10)}")
    
    # Simmetria
    A_diff = A - A.T
    sym_error = sparse.linalg.norm(A_diff) / sparse.linalg.norm(A)
    print(f"\nSimmetria:")
    print(f"  ||A - A^T|| / ||A|| = {sym_error:.2e}")
    is_symmetric = sym_error < 1e-10
    print(f"  Simmetrica: {'✅ Sì' if is_symmetric else '❌ No'}")
    
    # Definitezza positiva (approssimata tramite autovalori estremi)
    print(f"\nAutovalori (stima):")
    try:
        # Autovalore più piccolo
        eigvals_small = splinalg.eigsh(A, k=3, which='SM', return_eigenvectors=False)
        print(f"  3 più piccoli: {eigvals_small}")
        
        # Autovalore più grande
        eigvals_large = splinalg.eigsh(A, k=3, which='LM', return_eigenvectors=False)
        print(f"  3 più grandi: {eigvals_large}")
        
        lambda_min = eigvals_small.min()
        lambda_max = eigvals_large.max()
        cond_number = lambda_max / max(lambda_min, 1e-15)
        print(f"\n  Numero di condizione stimato: {cond_number:.2e}")
        
        if lambda_min <= 0:
            print("  ⚠️ ATTENZIONE: Autovalori non positivi! CG potrebbe non convergere.")
        else:
            print("  ✅ Matrice positiva definita")
    except Exception as e:
        print(f"  Errore calcolo autovalori: {e}")
    
    # RHS
    print(f"\nVettore b:")
    print(f"  min: {b.min():.6e}")
    print(f"  max: {b.max():.6e}")
    print(f"  ||b||: {np.linalg.norm(b):.6e}")
    
    return is_symmetric


def test_initial_guesses(A, b, tol=1e-6, maxiter=1000):
    """Testa diverse strategie di initial guess"""
    print("\n" + "="*60)
    print("TEST INITIAL GUESS")
    print("="*60)
    
    N = len(b)
    
    # Precondizionatore Jacobi
    diag = A.diagonal()
    diag[np.abs(diag) < 1e-10] = 1.0
    M = sparse.diags(1.0 / diag)
    
    strategies = {
        "1. Zero": np.zeros(N),
        "2. Uniforme 20°C": np.ones(N) * 20.0,
        "3. Uniforme 100°C": np.ones(N) * 100.0,
        "4. Uniforme 300°C (media stimata)": np.ones(N) * 300.0,
        "5. Media di b/diag": np.ones(N) * np.mean(b / A.diagonal()),
        "6. b/diag (soluzione Jacobi 1 iter)": b / A.diagonal(),
        "7. Random [-100, 100]": np.random.uniform(-100, 100, N),
    }
    
    # Prima calcola la soluzione "vera" con BiCGSTAB (più robusto)
    print("\nCalcolo soluzione di riferimento con BiCGSTAB...")
    x_ref, info_ref = splinalg.bicgstab(A, b, rtol=1e-10, maxiter=5000, M=M)
    if info_ref == 0:
        print(f"  ✅ BiCGSTAB convergito, T range: [{x_ref.min():.1f}, {x_ref.max():.1f}]°C")
        strategies["8. Soluzione vera (warm start)"] = x_ref.copy()
    else:
        print(f"  ❌ BiCGSTAB non convergito (info={info_ref})")
        x_ref = None
    
    print("\nTest CG con diverse initial guess:")
    print("-" * 60)
    
    for name, x0 in strategies.items():
        iters = [0]
        residuals = []
        
        def callback(xk):
            iters[0] += 1
            r = A @ xk - b
            residuals.append(np.linalg.norm(r))
        
        try:
            x, info = splinalg.cg(A, b, x0=x0.copy(), M=M, rtol=tol, maxiter=maxiter, callback=callback)
            
            r_final = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
            
            if info == 0:
                status = f"✅ {iters[0]:4d} iter"
            else:
                status = f"❌ {iters[0]:4d} iter (info={info})"
            
            # Divergenza?
            if len(residuals) > 1:
                if residuals[-1] > residuals[0] * 10:
                    status += " DIVERGE!"
            
            print(f"  {name:40s}: {status}, residuo={r_final:.2e}")
            
        except Exception as e:
            print(f"  {name:40s}: ❌ Errore: {e}")
    
    return x_ref


def test_cg_vs_bicgstab(A, b):
    """Confronta CG e BiCGSTAB"""
    print("\n" + "="*60)
    print("CONFRONTO CG vs BiCGSTAB")
    print("="*60)
    
    N = len(b)
    x0 = np.ones(N) * 20.0
    
    diag = A.diagonal()
    diag[np.abs(diag) < 1e-10] = 1.0
    M = sparse.diags(1.0 / diag)
    
    for method_name, method_func in [("CG", splinalg.cg), ("BiCGSTAB", splinalg.bicgstab), ("GMRES", splinalg.gmres)]:
        iters = [0]
        def callback(xk):
            iters[0] += 1
        
        try:
            x, info = method_func(A, b, x0=x0.copy(), M=M, rtol=1e-6, maxiter=1000, callback=callback)
            r_final = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
            
            if info == 0:
                print(f"  {method_name:10s}: ✅ {iters[0]:4d} iter, residuo={r_final:.2e}")
            else:
                print(f"  {method_name:10s}: ❌ info={info}, {iters[0]} iter, residuo={r_final:.2e}")
        except Exception as e:
            print(f"  {method_name:10s}: ❌ Errore: {e}")


if __name__ == '__main__':
    print("="*60)
    print(" DIAGNOSI PROBLEMA CONVERGENZA CG")
    print("="*60)
    
    # Test con diverse configurazioni
    test_cases = [
        {"name": "Piccola (50x50x25)", "size": (50, 50, 25), "Q": 100000},
        {"name": "Grande (100x100x50)", "size": (100, 100, 50), "Q": 100000},
        {"name": "Potenza alta (50x50x25)", "size": (50, 50, 25), "Q": 1000000},
        {"name": "Potenza molto alta", "size": (50, 50, 25), "Q": 10000000},
    ]
    
    for case in test_cases:
        print(f"\n{'#'*60}")
        print(f" TEST: {case['name']}")
        print(f"{'#'*60}")
        
        # Crea mesh
        Nx, Ny, Nz = case['size']
        mesh = create_test_mesh(Nx, Ny, Nz)
        
        # Modifica potenza
        cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
        r = max(1, min(Nx, Ny, Nz) // 10)
        mesh.Q[:] = 0
        for i in range(max(0, cx-r), min(Nx, cx+r+1)):
            for j in range(max(0, cy-r), min(Ny, cy+r+1)):
                for k in range(max(0, cz-r), min(Nz, cz+r+1)):
                    mesh.Q[i, j, k] = case['Q']
        
        print(f"Mesh: {Nx}x{Ny}x{Nz}, Q={case['Q']:.0e} W/m³")
        
        A, b = matrix_builder.build_steady_state_matrix(mesh)
        analyze_matrix(A, b)
        test_initial_guesses(A, b, maxiter=2000)
        test_cg_vs_bicgstab(A, b)
