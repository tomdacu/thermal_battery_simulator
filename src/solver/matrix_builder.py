"""
matrix_builder.py - Costruzione della matrice FDM

=============================================================================
MODULE OVERVIEW
=============================================================================

This module assembles the sparse linear system A·T = b for solving the 
heat equation using the Finite Difference Method (FDM).

KEY EQUATIONS:
    Steady-state:   -∇·(k∇T) + Q = 0
    Transient:      ρcp ∂T/∂t = ∇·(k∇T) + Q  (future)

DISCRETIZATION:
    Uses a 7-point stencil on a uniform 3D Cartesian grid (dx = dy = dz = d).
    For each node P with neighbors W, E, S, N, D, U:
    
        a_P·T_P + a_W·T_W + a_E·T_E + a_S·T_S + a_N·T_N + a_D·T_D + a_U·T_U = b_P
    
    The coefficients depend on:
    - Thermal conductivity k (harmonic mean at faces)
    - Boundary type (Dirichlet, Neumann, Convection, Internal)
    - Heat source Q

INDEXING CONVENTION:
    Uses Fortran-order (column-major) linear indexing:
        p = i + j*Nx + k*Nx*Ny
    
    This ensures contiguous memory access when iterating z-first.
    
ALL PARAMETERS FROM GUI:
    All thermal properties (k, rho, cp, Q) come from the mesh object,
    which is populated by BatteryGeometry.apply_to_mesh() based on GUI input.
=============================================================================
"""

import numpy as np
from scipy import sparse
from typing import Tuple, Optional
from ..core.mesh import Mesh3D, BoundaryType


def build_steady_state_matrix(mesh: Mesh3D) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """
    Costruisce la matrice sparsa e il vettore RHS per il caso stazionario.
    
    Equazione discretizzata per nodo interno (mesh uniforme con d = dx = dy = dz):
    
    Per conduzione: -∇·(k∇T) = Q
    Discretizzazione: Σ k_face * (T_neighbor - T_P) / d² = Q_P
    
    Returns:
        A: Matrice sparsa CSR
        b: Vettore dei termini noti
    """
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    N = mesh.N_total
    
    # Spaziatura UNIFORME (dx = dy = dz = d)
    d = mesh.d
    d2 = d * d
    
    # Verifica uniformità
    assert abs(mesh.dx - mesh.dy) < 1e-10, f"Mesh non uniforme: dx={mesh.dx}, dy={mesh.dy}"
    assert abs(mesh.dx - mesh.dz) < 1e-10, f"Mesh non uniforme: dx={mesh.dx}, dz={mesh.dz}"
    
    # Liste per costruzione matrice COO (più efficiente per costruzione)
    row_indices = []
    col_indices = []
    data = []
    
    # Vettore termini noti
    b = np.zeros(N, dtype=np.float64)
    
    # Itera su tutti i nodi
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Indice lineare del nodo corrente
                p = mesh.ijk_to_linear(i, j, k)
                
                # Conducibilità del nodo corrente
                k_P = mesh.k[i, j, k]
                
                # Tipo di condizione al contorno
                bc_type = mesh.boundary_type[i, j, k]
                
                if bc_type == BoundaryType.DIRICHLET:
                    # Temperatura fissa
                    row_indices.append(p)
                    col_indices.append(p)
                    data.append(1.0)
                    b[p] = mesh.bc_T_inf[i, j, k]

                elif bc_type == BoundaryType.INTERNAL:
                    # Nodo interno - schema a 7 punti
                    # Bilancio: Σ k_eff/d² * (T_neighbor - T_P) + Q = 0
                    # Riarrangiato: Σ(k_eff/d²)*T_P - Σ(k_eff/d²*T_neighbor) = Q
                    
                    coeffs, rhs = _get_internal_coefficients_v2(mesh, i, j, k, d, d2)
                    
                    # Diagonale principale
                    row_indices.append(p)
                    col_indices.append(p)
                    data.append(coeffs['P'])
                    
                    # Vicini
                    neighbors = [
                        (i-1, j, k, 'W'),
                        (i+1, j, k, 'E'),
                        (i, j-1, k, 'S'),
                        (i, j+1, k, 'N'),
                        (i, j, k-1, 'D'),
                        (i, j, k+1, 'U'),
                    ]
                    
                    for ni, nj, nk, direction in neighbors:
                        if 0 <= ni < Nx and 0 <= nj < Ny and 0 <= nk < Nz:
                            if direction in coeffs:
                                p_neighbor = mesh.ijk_to_linear(ni, nj, nk)
                                row_indices.append(p)
                                col_indices.append(p_neighbor)
                                data.append(coeffs[direction])
                    
                    b[p] = rhs

                elif bc_type == BoundaryType.CONVECTION:
                    # Nodo con convezione.
                    # - Se è sul bordo del dominio: usa schema Robin di bordo.
                    # - Se è interno (es. celle-tubo): aggiunge un termine di scambio
                    #   locale verso T_inf (sink/source) mantenendo i contributi conduttivi.
                    is_domain_boundary = (
                        i == 0 or i == Nx - 1 or
                        j == 0 or j == Ny - 1 or
                        k == 0 or k == Nz - 1
                    )

                    if is_domain_boundary:
                        coeffs, rhs = _get_boundary_coefficients(mesh, i, j, k, d, d2)
                    else:
                        coeffs, rhs = _get_internal_coefficients_v2(mesh, i, j, k, d, d2)

                        h_local = float(mesh.bc_h[i, j, k])
                        T_inf_local = float(mesh.bc_T_inf[i, j, k])

                        # Modello semplice di scambio interno:
                        # aggiunge a_conv = h * A/V con A≈d^2 e V=d^3 => a_conv = h/d
                        if h_local > 0:
                            a_conv = h_local / d
                            coeffs['P'] = coeffs.get('P', 0.0) + a_conv
                            rhs += a_conv * T_inf_local

                    # Diagonale principale
                    row_indices.append(p)
                    col_indices.append(p)
                    data.append(coeffs['P'])

                    # Vicini (solo quelli interni al dominio)
                    neighbors = [
                        (i-1, j, k, 'W'),
                        (i+1, j, k, 'E'),
                        (i, j-1, k, 'S'),
                        (i, j+1, k, 'N'),
                        (i, j, k-1, 'D'),
                        (i, j, k+1, 'U'),
                    ]

                    for ni, nj, nk, direction in neighbors:
                        if 0 <= ni < Nx and 0 <= nj < Ny and 0 <= nk < Nz:
                            if direction in coeffs:
                                p_neighbor = mesh.ijk_to_linear(ni, nj, nk)
                                row_indices.append(p)
                                col_indices.append(p_neighbor)
                                data.append(coeffs[direction])

                    b[p] = rhs

                else:
                    # Nodo di bordo con condizione convettiva o mista
                    coeffs, rhs = _get_boundary_coefficients(mesh, i, j, k, d, d2)
                    
                    # Diagonale principale
                    row_indices.append(p)
                    col_indices.append(p)
                    data.append(coeffs['P'])
                    
                    # Vicini (solo quelli interni al dominio)
                    neighbors = [
                        (i-1, j, k, 'W'),
                        (i+1, j, k, 'E'),
                        (i, j-1, k, 'S'),
                        (i, j+1, k, 'N'),
                        (i, j, k-1, 'D'),
                        (i, j, k+1, 'U'),
                    ]
                    
                    for ni, nj, nk, direction in neighbors:
                        if 0 <= ni < Nx and 0 <= nj < Ny and 0 <= nk < Nz:
                            if direction in coeffs:
                                p_neighbor = mesh.ijk_to_linear(ni, nj, nk)
                                row_indices.append(p)
                                col_indices.append(p_neighbor)
                                data.append(coeffs[direction])
                    
                    b[p] = rhs
    
    # Costruisci matrice sparsa
    A = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(N, N))
    A = A.tocsr()
    
    return A, b


def _get_internal_coefficients(mesh: Mesh3D, i: int, j: int, k: int, d2: float) -> dict:
    """
    DEPRECATO - Usa _get_internal_coefficients_v2
    """
    return _get_internal_coefficients_v2(mesh, i, j, k, np.sqrt(d2), d2)[0]


def _get_internal_coefficients_v2(mesh: Mesh3D, i: int, j: int, k: int, 
                                   d: float, d2: float) -> Tuple[dict, float]:
    """
    Calcola i coefficienti per un nodo interno.
    
    Bilancio energetico sul volume di controllo:
    
    Σ_faces [k_eff * (T_neighbor - T_P) / d] * A_face + Q * V = 0
    
    Con A_face = d² e V = d³:
    Σ [k_eff / d * d² * (T_neighbor - T_P)] + Q * d³ = 0
    Σ [k_eff * d * (T_neighbor - T_P)] + Q * d³ = 0
    
    Dividendo per d:
    Σ [k_eff * (T_neighbor - T_P)] + Q * d² = 0
    
    Oppure dividendo per d³:
    Σ [k_eff / d² * (T_neighbor - T_P)] + Q = 0
    
    Usiamo la seconda forma (coefficienti in W/(m³·K)):
    a_P * T_P - Σ(a_neighbor * T_neighbor) = Q
    
    dove a_neighbor = k_eff / d²
    
    Returns:
        coeffs: Dizionario dei coefficienti
        rhs: Termine noto (Q)
    """
    k_P = mesh.k[i, j, k]
    Q_P = mesh.Q[i, j, k]
    
    coeffs = {}
    a_P = 0.0
    
    # Conducibilità ai nodi vicini
    k_E = mesh.k[i+1, j, k]
    k_W = mesh.k[i-1, j, k]
    k_N = mesh.k[i, j+1, k]
    k_S = mesh.k[i, j-1, k]
    k_U = mesh.k[i, j, k+1]
    k_D = mesh.k[i, j, k-1]
    
    # Media armonica per conducibilità interfaccia
    k_e = 2 * k_P * k_E / (k_P + k_E) if (k_P + k_E) > 0 else 0
    k_w = 2 * k_P * k_W / (k_P + k_W) if (k_P + k_W) > 0 else 0
    k_n = 2 * k_P * k_N / (k_P + k_N) if (k_P + k_N) > 0 else 0
    k_s = 2 * k_P * k_S / (k_P + k_S) if (k_P + k_S) > 0 else 0
    k_u = 2 * k_P * k_U / (k_P + k_U) if (k_P + k_U) > 0 else 0
    k_d = 2 * k_P * k_D / (k_P + k_D) if (k_P + k_D) > 0 else 0
    
    # Coefficienti (W/(m³·K))
    a_E = k_e / d2
    a_W = k_w / d2
    a_N = k_n / d2
    a_S = k_s / d2
    a_U = k_u / d2
    a_D = k_d / d2
    
    coeffs['E'] = -a_E
    coeffs['W'] = -a_W
    coeffs['N'] = -a_N
    coeffs['S'] = -a_S
    coeffs['U'] = -a_U
    coeffs['D'] = -a_D
    
    # Coefficiente diagonale
    a_P = a_E + a_W + a_N + a_S + a_U + a_D
    coeffs['P'] = a_P
    
    return coeffs, Q_P


def _get_boundary_coefficients(mesh: Mesh3D, i: int, j: int, k: int, 
                                d: float, d2: float) -> Tuple[dict, float]:
    """
    Calcola i coefficienti per un nodo di bordo con convezione.
    
    Bilancio energetico sul volume di controllo al bordo:
    
    Per ogni faccia interna: q = k_eff * (T_neighbor - T_P) / d  [W/m²]
    Per ogni faccia esterna: q = h * (T_inf - T_P)               [W/m²]
    
    Bilancio: Σ q_facce * A_faccia + Q * V = 0
    
    Con A_faccia = d² e V = d³, dividendo per d²:
    Σ k_eff/d * (T_neighbor - T_P) + h*(T_inf - T_P) + Q*d = 0
    
    Returns:
        coeffs: Dizionario dei coefficienti (normalizzati)
        rhs: Termine noto (normalizzato)
    """
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    k_P = mesh.k[i, j, k]
    h = mesh.bc_h[i, j, k]
    T_inf = mesh.bc_T_inf[i, j, k]
    Q_P = mesh.Q[i, j, k]
    
    coeffs = {}
    a_P = 0.0  # Coefficiente diagonale
    rhs = 0.0
    
    # Identifica quali bordi
    is_x_min = (i == 0)
    is_x_max = (i == Nx - 1)
    is_y_min = (j == 0)
    is_y_max = (j == Ny - 1)
    is_z_min = (k == 0)
    is_z_max = (k == Nz - 1)
    
    # === DIREZIONE X ===
    if not is_x_min:
        # Faccia interna verso West
        k_W = mesh.k[i-1, j, k]
        k_eff = 2 * k_P * k_W / (k_P + k_W) if (k_P + k_W) > 0 else 0
        a_W = k_eff / d2
        coeffs['W'] = -a_W
        a_P += a_W
    else:
        # Bordo x=0: convezione
        # Coefficiente effettivo: 1/(1/(2k/d) + 1/h) = 2*k*h/(2*k + h*d) per unità di area
        h_bc = mesh.bc_h[i, j, k]
        T_bc = mesh.bc_T_inf[i, j, k]
        if h_bc > 0:
            a_bc = 2 * k_P * h_bc / (2 * k_P + h_bc * d) / d
            a_P += a_bc
            rhs += a_bc * T_bc
    
    if not is_x_max:
        k_E = mesh.k[i+1, j, k]
        k_eff = 2 * k_P * k_E / (k_P + k_E) if (k_P + k_E) > 0 else 0
        a_E = k_eff / d2
        coeffs['E'] = -a_E
        a_P += a_E
    else:
        h_bc = mesh.bc_h[i, j, k]
        T_bc = mesh.bc_T_inf[i, j, k]
        if h_bc > 0:
            a_bc = 2 * k_P * h_bc / (2 * k_P + h_bc * d) / d
            a_P += a_bc
            rhs += a_bc * T_bc
    
    # === DIREZIONE Y ===
    if not is_y_min:
        k_S = mesh.k[i, j-1, k]
        k_eff = 2 * k_P * k_S / (k_P + k_S) if (k_P + k_S) > 0 else 0
        a_S = k_eff / d2
        coeffs['S'] = -a_S
        a_P += a_S
    else:
        h_bc = mesh.bc_h[i, j, k]
        T_bc = mesh.bc_T_inf[i, j, k]
        if h_bc > 0:
            a_bc = 2 * k_P * h_bc / (2 * k_P + h_bc * d) / d
            a_P += a_bc
            rhs += a_bc * T_bc
    
    if not is_y_max:
        k_N = mesh.k[i, j+1, k]
        k_eff = 2 * k_P * k_N / (k_P + k_N) if (k_P + k_N) > 0 else 0
        a_N = k_eff / d2
        coeffs['N'] = -a_N
        a_P += a_N
    else:
        h_bc = mesh.bc_h[i, j, k]
        T_bc = mesh.bc_T_inf[i, j, k]
        if h_bc > 0:
            a_bc = 2 * k_P * h_bc / (2 * k_P + h_bc * d) / d
            a_P += a_bc
            rhs += a_bc * T_bc
    
    # === DIREZIONE Z ===
    if not is_z_min:
        k_D = mesh.k[i, j, k-1]
        k_eff = 2 * k_P * k_D / (k_P + k_D) if (k_P + k_D) > 0 else 0
        a_D = k_eff / d2
        coeffs['D'] = -a_D
        a_P += a_D
    else:
        # Questo è tipicamente Dirichlet (terreno) - non dovremmo arrivare qui
        h_bc = mesh.bc_h[i, j, k]
        T_bc = mesh.bc_T_inf[i, j, k]
        if h_bc > 0:
            a_bc = 2 * k_P * h_bc / (2 * k_P + h_bc * d) / d
            a_P += a_bc
            rhs += a_bc * T_bc
    
    if not is_z_max:
        k_U = mesh.k[i, j, k+1]
        k_eff = 2 * k_P * k_U / (k_P + k_U) if (k_P + k_U) > 0 else 0
        a_U = k_eff / d2
        coeffs['U'] = -a_U
        a_P += a_U
    else:
        h_bc = mesh.bc_h[i, j, k]
        T_bc = mesh.bc_T_inf[i, j, k]
        if h_bc > 0:
            a_bc = 2 * k_P * h_bc / (2 * k_P + h_bc * d) / d
            a_P += a_bc
            rhs += a_bc * T_bc
    
    # Coefficiente diagonale
    coeffs['P'] = a_P
    
    # Termine sorgente (Q è già in W/m³)
    rhs += Q_P
    
    return coeffs, rhs


def build_transient_matrix(mesh: Mesh3D, dt: float, theta: float = 1.0
                           ) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """
    Costruisce la matrice per il caso transitorio con schema theta.
    
    theta = 0: Euler esplicito (instabile per Fo > 1/6)
    theta = 0.5: Crank-Nicolson (secondo ordine)
    theta = 1: Euler implicito (incondizionatamente stabile)
    
    Equazione:
    (ρcp/dt) * T^{n+1} - θ∇·(k∇T^{n+1}) = (ρcp/dt) * T^n + (1-θ)∇·(k∇T^n) + Q
    
    Returns:
        A: Matrice per il lato sinistro
        M: Matrice di massa per il lato destro
    """
    # Prima ottieni la matrice stazionaria (laplaciano discreto)
    L, _ = build_steady_state_matrix(mesh)
    
    # Matrice di massa
    N = mesh.N_total
    d3 = mesh.dx * mesh.dy * mesh.dz
    
    # Capacità termica volumetrica per ogni nodo
    # IMPORTANT: la matrice L è costruita usando l'indicizzazione lineare della mesh
    # (equivalente a Fortran order). Manteniamo lo stesso ordine anche qui.
    rho_cp = (mesh.rho * mesh.cp).ravel(order='F')
    
    # Coefficiente temporale
    mass_diag = rho_cp / dt
    M = sparse.diags(mass_diag)
    
    # Matrice lato sinistro: M + θ*L
    A = M + theta * L
    
    # Matrice lato destro: M - (1-θ)*L
    B = M - (1 - theta) * L
    
    return A.tocsr(), B.tocsr()


# =============================================================================
# VERSIONE OTTIMIZZATA CON NUMBA (opzionale)
# =============================================================================

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def _compute_coefficients_numba(k_field, Nx, Ny, Nz, d2):
        """
        Versione Numba per calcolo parallelo dei coefficienti.
        
        Restituisce array di coefficienti per tutti i nodi.
        """
        N = Nx * Ny * Nz
        
        # Array per coefficienti (7 per nodo: P, E, W, N, S, U, D)
        coeffs = np.zeros((N, 7), dtype=np.float64)
        
        for p in prange(N):
            # Converti a indici 3D (uso kz per evitare confusione con k_field)
            kz = p // (Nx * Ny)
            remainder = p % (Nx * Ny)
            j = remainder // Nx
            i = remainder % Nx
            
            k_P = k_field[i, j, kz]
            
            # Evita divisione per zero se k_P è troppo piccolo
            if k_P < 1e-10:
                k_P = 1e-10
            
            # Conducibilità vicini (con clamp ai bordi)
            k_E = k_field[min(i+1, Nx-1), j, kz]
            k_W = k_field[max(i-1, 0), j, kz]
            k_N = k_field[i, min(j+1, Ny-1), kz]
            k_S = k_field[i, max(j-1, 0), kz]
            k_U = k_field[i, j, min(kz+1, Nz-1)]
            k_D = k_field[i, j, max(kz-1, 0)]
            
            # Media armonica
            k_e = 2 * k_P * k_E / (k_P + k_E) if (k_P + k_E) > 0 else 0
            k_w = 2 * k_P * k_W / (k_P + k_W) if (k_P + k_W) > 0 else 0
            k_n = 2 * k_P * k_N / (k_P + k_N) if (k_P + k_N) > 0 else 0
            k_s = 2 * k_P * k_S / (k_P + k_S) if (k_P + k_S) > 0 else 0
            k_u = 2 * k_P * k_U / (k_P + k_U) if (k_P + k_U) > 0 else 0
            k_d = 2 * k_P * k_D / (k_P + k_D) if (k_P + k_D) > 0 else 0
            
            # Coefficienti normalizzati
            coeffs[p, 1] = -k_e / k_P  # E
            coeffs[p, 2] = -k_w / k_P  # W
            coeffs[p, 3] = -k_n / k_P  # N
            coeffs[p, 4] = -k_s / k_P  # S
            coeffs[p, 5] = -k_u / k_P  # U
            coeffs[p, 6] = -k_d / k_P  # D
            
            # Diagonale
            coeffs[p, 0] = -(coeffs[p, 1] + coeffs[p, 2] + coeffs[p, 3] + 
                            coeffs[p, 4] + coeffs[p, 5] + coeffs[p, 6])
        
        return coeffs


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    from ..core.mesh import Mesh3D
    
    print("=== Test Matrix Builder ===")
    
    # Mesh piccola per test
    mesh = Mesh3D(Lx=1.0, Ly=1.0, Lz=1.0, Nx=10, Ny=10, Nz=10)
    
    # Imposta conducibilità uniforme
    mesh.k[:] = 1.0
    
    # Imposta BC
    mesh.set_fixed_temperature_bc('z_min', 100.0)
    mesh.set_convection_bc('z_max', 10.0, 20.0)
    
    # Costruisci matrice
    print("Costruzione matrice...")
    A, b = build_steady_state_matrix(mesh)
    
    print(f"Dimensione matrice: {A.shape}")
    print(f"Elementi non-zero: {A.nnz}")
    print(f"Sparsità: {100 * (1 - A.nnz / (A.shape[0]**2)):.2f}%")
    print(f"Memoria matrice: {A.data.nbytes / 1e6:.2f} MB")
    
    # Verifica simmetria (approssimativa)
    diff = A - A.T
    max_asym = np.abs(diff).max()
    print(f"Max asimmetria: {max_asym:.2e}")
    
    print("\n=== Test Completato ===")
