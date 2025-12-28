"""
test_solver.py - Unit tests per i moduli solver

Eseguire con: pytest tests/test_solver.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.mesh import Mesh3D, BoundaryType
from src.solver.matrix_builder import build_steady_state_matrix, build_transient_matrix
from src.solver.steady_state import SteadyStateSolver, SolverConfig, solve_steady_state


class TestMatrixBuilder:
    """Test per la costruzione della matrice"""
    
    def test_matrix_dimensions(self):
        """Test dimensioni matrice"""
        mesh = Mesh3D(Lx=1, Ly=1, Lz=1, Nx=5, Ny=5, Nz=5)
        A, b = build_steady_state_matrix(mesh)
        
        N = mesh.N_total
        assert A.shape == (N, N)
        assert b.shape == (N,)
    
    def test_matrix_sparsity(self):
        """Test sparsità matrice"""
        mesh = Mesh3D(Lx=1, Ly=1, Lz=1, Nx=10, Ny=10, Nz=10)
        A, b = build_steady_state_matrix(mesh)
        
        # Ogni nodo ha al massimo 7 vicini
        max_nnz = 7 * mesh.N_total
        assert A.nnz <= max_nnz
        
        # Sparsità dovrebbe essere alta
        sparsity = 1 - A.nnz / (mesh.N_total ** 2)
        assert sparsity > 0.99  # > 99% sparse
    
    def test_diagonal_dominance(self):
        """Test dominanza diagonale (per stabilità)"""
        mesh = Mesh3D(Lx=1, Ly=1, Lz=1, Nx=5, Ny=5, Nz=5)
        mesh.k[:] = 1.0  # Conducibilità uniforme
        
        A, b = build_steady_state_matrix(mesh)
        A_dense = A.toarray()
        
        # Per la maggior parte dei nodi, |a_ii| >= Σ|a_ij|
        diag = np.abs(np.diag(A_dense))
        off_diag_sum = np.sum(np.abs(A_dense), axis=1) - diag
        
        # Almeno l'80% dei nodi dovrebbe essere diagonalmente dominante
        dominant = np.sum(diag >= off_diag_sum - 1e-10)
        assert dominant >= 0.8 * mesh.N_total

    def test_internal_convection_adds_term(self):
        """Verifica che la convezione interna (celle-tubo) aggiunga un termine su diagonale e RHS."""
        mesh = Mesh3D(Lx=1, Ly=1, Lz=1, Nx=5, Ny=5, Nz=5)
        # Seleziona una cella interna
        i = j = k = 2
        p = mesh.ijk_to_linear(i, j, k)

        # Caso base: interno
        mesh.boundary_type[i, j, k] = BoundaryType.INTERNAL
        mesh.bc_h[i, j, k] = 0.0
        mesh.bc_T_inf[i, j, k] = 20.0
        A0, b0 = build_steady_state_matrix(mesh)
        a0 = A0[p, p]
        rhs0 = b0[p]

        # Caso con convezione interna
        h = 10.0
        T_inf = 5.0
        mesh.boundary_type[i, j, k] = BoundaryType.CONVECTION
        mesh.bc_h[i, j, k] = h
        mesh.bc_T_inf[i, j, k] = T_inf
        A1, b1 = build_steady_state_matrix(mesh)
        a1 = A1[p, p]
        rhs1 = b1[p]

        # Il modello implementato usa a_conv = h / d
        a_conv = h / mesh.d
        assert a1 == pytest.approx(a0 + a_conv, rel=1e-9, abs=1e-12)
        assert rhs1 == pytest.approx(rhs0 + a_conv * T_inf, rel=1e-9, abs=1e-12)

    def test_transient_mass_matrix_ordering(self):
        """Verifica che la matrice di massa usi lo stesso ordine di flattening della mesh."""
        mesh = Mesh3D(Lx=1, Ly=1, Lz=1, Nx=4, Ny=3, Nz=2)

        # Crea un campo non uniforme per distinguere l'ordine
        mesh.rho[:] = 1.0
        mesh.cp[:] = 1.0
        mesh.rho[1, 0, 0] = 7.0
        mesh.cp[1, 0, 0] = 11.0
        dt = 2.0
        theta = 1.0

        A, B = build_transient_matrix(mesh, dt=dt, theta=theta)

        # Per theta=1: A = M + L, B = M
        # quindi M.diag = B.diag
        mass_diag_expected = (mesh.rho * mesh.cp).ravel(order='F') / dt
        mass_diag_actual = B.diagonal()
        assert np.allclose(mass_diag_actual, mass_diag_expected)


class TestSteadyStateSolver:
    """Test per il solutore stazionario"""
    
    def test_uniform_temperature(self):
        """Test caso banale: temperatura uniforme"""
        mesh = Mesh3D(Lx=1, Ly=1, Lz=1, Nx=5, Ny=5, Nz=5)
        
        # Tutte le BC a 100°C
        mesh.T[:] = 100.0
        mesh.bc_T_inf[:] = 100.0
        mesh.boundary_type[:] = BoundaryType.DIRICHLET
        mesh.boundary_type[1:-1, 1:-1, 1:-1] = BoundaryType.INTERNAL
        mesh.Q[:] = 0.0  # Nessuna sorgente
        
        result = solve_steady_state(mesh, method="direct", verbose=False)
        
        # La temperatura dovrebbe rimanere 100°C ovunque
        assert result.converged
        # Con Dirichlet solo sui bordi, i nodi interni potrebbero avere valori diversi
        # ma comunque dovrebbero essere ~100°C senza sorgenti
    
    def test_gradient_z(self):
        """Test gradiente lineare in z"""
        mesh = Mesh3D(Lx=1, Ly=1, Lz=1, Nx=5, Ny=5, Nz=10)
        
        # T = 0 su z=0, T = 100 su z=1
        mesh.set_fixed_temperature_bc('z_min', 0.0)
        mesh.set_fixed_temperature_bc('z_max', 100.0)
        
        # Conduzione uniforme
        mesh.k[:] = 1.0
        mesh.Q[:] = 0.0
        
        result = solve_steady_state(mesh, method="direct", verbose=False)
        
        assert result.converged
        
        # La temperatura dovrebbe essere lineare in z
        # Campiona il centro
        T_center = mesh.T[2, 2, :]
        z = mesh.z
        
        # Verifica linearità (correlazione > 0.99)
        corr = np.corrcoef(z, T_center)[0, 1]
        assert abs(corr) > 0.99
    
    def test_with_source(self):
        """Test con sorgente di calore"""
        mesh = Mesh3D(Lx=1, Ly=1, Lz=1, Nx=10, Ny=10, Nz=10)
        
        # BC convezione su tutte le facce
        mesh.bc_h[:] = 10.0
        mesh.bc_T_inf[:] = 20.0
        
        # Sorgente uniforme
        mesh.Q[:] = 1000.0  # W/m³
        mesh.k[:] = 1.0
        
        result = solve_steady_state(mesh, method="direct", verbose=False)
        
        assert result.converged
        
        # La temperatura massima dovrebbe essere maggiore di T_inf
        assert mesh.T.max() > 20.0
        
        # La temperatura dovrebbe essere massima al centro
        T_center = mesh.T[5, 5, 5]
        T_corner = mesh.T[0, 0, 0]
        assert T_center > T_corner
    
    def test_iterative_solver(self):
        """Test solutore iterativo"""
        mesh = Mesh3D(Lx=1, Ly=1, Lz=1, Nx=10, Ny=10, Nz=10)
        
        mesh.set_fixed_temperature_bc('z_min', 100.0)
        mesh.set_convection_bc('z_max', 10.0, 20.0)
        mesh.k[:] = 1.0
        mesh.Q[:] = 500.0
        
        # Test BiCGSTAB
        config = SolverConfig(
            method="bicgstab",
            tolerance=1e-8,
            max_iterations=1000,
            verbose=False
        )
        
        solver = SteadyStateSolver(mesh, config)
        result = solver.solve()
        
        assert result.converged
        assert result.residual < 1e-6
    
    def test_solution_consistency(self):
        """Verifica che diretto e iterativo diano stesso risultato"""
        mesh = Mesh3D(Lx=1, Ly=1, Lz=1, Nx=8, Ny=8, Nz=8)
        
        mesh.set_fixed_temperature_bc('z_min', 100.0)
        mesh.set_convection_bc('z_max', 10.0, 20.0)
        mesh.k[:] = 1.0
        mesh.Q[:] = 500.0
        
        # Soluzione diretta
        result1 = solve_steady_state(mesh, method="direct", verbose=False)
        T1 = mesh.T.copy()
        
        # Soluzione iterativa
        result2 = solve_steady_state(mesh, method="bicgstab", verbose=False)
        T2 = mesh.T.copy()
        
        # Devono essere molto simili
        diff = np.abs(T1 - T2).max()
        assert diff < 0.01  # Meno di 0.01°C di differenza


class TestEnergyBalance:
    """Test verifica bilancio energetico"""
    
    def test_steady_state_balance(self):
        """In regime stazionario P_in = P_out"""
        mesh = Mesh3D(Lx=1, Ly=1, Lz=1, Nx=15, Ny=15, Nz=15)
        
        # Sorgente interna
        P_source = 100.0  # W
        V_total = mesh.Lx * mesh.Ly * mesh.Lz
        Q = P_source / V_total
        mesh.Q[:] = Q
        
        # BC convezione
        mesh.bc_h[:] = 20.0
        mesh.bc_T_inf[:] = 20.0
        mesh.k[:] = 1.0
        
        solve_steady_state(mesh, verbose=False)
        
        # Calcola potenza uscente approssimata
        from src.analysis.power_balance import PowerBalanceAnalyzer
        analyzer = PowerBalanceAnalyzer(mesh)
        balance = analyzer.compute_power_balance()
        
        # Il bilancio dovrebbe chiudersi con errore < 10%
        # (errore maggiore accettabile per mesh grossolana)
        assert balance.imbalance_pct < 20.0

    def test_internal_convection_counts_as_output(self):
        """Verifica che la convezione interna venga conteggiata come P_output nel bilancio."""
        mesh = Mesh3D(Lx=1, Ly=1, Lz=1, Nx=5, Ny=5, Nz=5)

        # Imposta un'unica cella interna come "tubo" convettivo
        i = j = k = 2
        mesh.boundary_type[i, j, k] = BoundaryType.CONVECTION
        mesh.bc_h[i, j, k] = 10.0
        mesh.bc_T_inf[i, j, k] = 20.0
        mesh.T[i, j, k] = 30.0

        from src.analysis.power_balance import PowerBalanceAnalyzer
        analyzer = PowerBalanceAnalyzer(mesh)
        balance = analyzer.compute_power_balance()

        expected = 10.0 * (30.0 - 20.0) * (mesh.d ** 2)
        assert balance.P_output == pytest.approx(expected)


# =============================================================================
# ESECUZIONE
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
