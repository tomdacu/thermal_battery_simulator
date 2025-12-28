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

# Flag per usare la versione ottimizzata
USE_VECTORIZED = True

# Prova a importare Numba per accelerazione
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# =============================================================================
# FUNZIONI NUMBA (se disponibile)
# =============================================================================

if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _compute_harmonic_means_numba(k_P, k_W, k_E, k_S, k_N, k_D, k_U, eps):
        """
        Calcola le medie armoniche delle conducibilità alle interfacce.
        Versione Numba parallelizzata.
        """
        N = len(k_P)
        k_w = np.empty(N, dtype=np.float64)
        k_e = np.empty(N, dtype=np.float64)
        k_s = np.empty(N, dtype=np.float64)
        k_n = np.empty(N, dtype=np.float64)
        k_d = np.empty(N, dtype=np.float64)
        k_u = np.empty(N, dtype=np.float64)
        
        for i in prange(N):
            k_w[i] = 2 * k_P[i] * k_W[i] / (k_P[i] + k_W[i] + eps)
            k_e[i] = 2 * k_P[i] * k_E[i] / (k_P[i] + k_E[i] + eps)
            k_s[i] = 2 * k_P[i] * k_S[i] / (k_P[i] + k_S[i] + eps)
            k_n[i] = 2 * k_P[i] * k_N[i] / (k_P[i] + k_N[i] + eps)
            k_d[i] = 2 * k_P[i] * k_D[i] / (k_P[i] + k_D[i] + eps)
            k_u[i] = 2 * k_P[i] * k_U[i] / (k_P[i] + k_U[i] + eps)
        
        return k_w, k_e, k_s, k_n, k_d, k_u
    
    @njit(parallel=True, cache=True)
    def _compute_coefficients_numba(k_w, k_e, k_s, k_n, k_d, k_u, d2,
                                     is_x_min, is_x_max, is_y_min, is_y_max,
                                     is_z_min, is_z_max):
        """
        Calcola i coefficienti della matrice sparsa.
        Versione Numba parallelizzata.
        """
        N = len(k_w)
        a_W = np.empty(N, dtype=np.float64)
        a_E = np.empty(N, dtype=np.float64)
        a_S = np.empty(N, dtype=np.float64)
        a_N = np.empty(N, dtype=np.float64)
        a_D = np.empty(N, dtype=np.float64)
        a_U = np.empty(N, dtype=np.float64)
        a_P = np.empty(N, dtype=np.float64)
        
        for i in prange(N):
            # Coefficienti base
            aw = k_w[i] / d2 if not is_x_min[i] else 0.0
            ae = k_e[i] / d2 if not is_x_max[i] else 0.0
            as_ = k_s[i] / d2 if not is_y_min[i] else 0.0
            an = k_n[i] / d2 if not is_y_max[i] else 0.0
            ad = k_d[i] / d2 if not is_z_min[i] else 0.0
            au = k_u[i] / d2 if not is_z_max[i] else 0.0
            
            a_W[i] = aw
            a_E[i] = ae
            a_S[i] = as_
            a_N[i] = an
            a_D[i] = ad
            a_U[i] = au
            a_P[i] = aw + ae + as_ + an + ad + au
        
        return a_W, a_E, a_S, a_N, a_D, a_U, a_P


def _compute_harmonic_means_numpy(k_P, k_W, k_E, k_S, k_N, k_D, k_U, eps):
    """Versione NumPy (fallback)"""
    k_w = 2 * k_P * k_W / (k_P + k_W + eps)
    k_e = 2 * k_P * k_E / (k_P + k_E + eps)
    k_s = 2 * k_P * k_S / (k_P + k_S + eps)
    k_n = 2 * k_P * k_N / (k_P + k_N + eps)
    k_d = 2 * k_P * k_D / (k_P + k_D + eps)
    k_u = 2 * k_P * k_U / (k_P + k_U + eps)
    return k_w, k_e, k_s, k_n, k_d, k_u


def build_steady_state_matrix(mesh: Mesh3D) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """
    Costruisce la matrice sparsa e il vettore RHS per il caso stazionario.
    
    Equazione discretizzata per nodo interno (mesh uniforme con d = dx = dy = dz):
    
    Per conduzione: -∇·(k∇T) = Q
    Discretizzazione: Σ k_face * (T_neighbor - T_P) / d² = Q_P
    
    Usa la versione vettorizzata per performance ottimali.
    
    Returns:
        A: Matrice sparsa CSR
        b: Vettore dei termini noti
    """
    if USE_VECTORIZED:
        return _build_steady_state_matrix_vectorized(mesh)
    else:
        return _build_steady_state_matrix_loop(mesh)


def _build_steady_state_matrix_vectorized(mesh: Mesh3D) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """
    Versione vettorizzata della costruzione matrice - 10-50x più veloce.
    
    Usa NumPy broadcasting invece di loop Python.
    """
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    N = mesh.N_total
    
    d = mesh.d
    d2 = d * d
    
    # Flatten degli array in ordine Fortran per coerenza con mesh.ijk_to_linear
    k_flat = mesh.k.ravel(order='F')
    Q_flat = mesh.Q.ravel(order='F')
    bc_type_flat = mesh.boundary_type.ravel(order='F')
    bc_h_flat = mesh.bc_h.ravel(order='F')
    bc_T_inf_flat = mesh.bc_T_inf.ravel(order='F')
    
    # Pre-calcola indici lineari per tutti i nodi
    # p = i + j*Nx + k*Nx*Ny (Fortran order)
    ii, jj, kk = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij')
    ii = ii.ravel(order='F')
    jj = jj.ravel(order='F')
    kk = kk.ravel(order='F')
    p_all = ii + jj * Nx + kk * Nx * Ny
    
    # Indici dei vicini (con clipping per evitare out-of-bounds)
    p_W = np.clip(ii - 1, 0, Nx - 1) + jj * Nx + kk * Nx * Ny
    p_E = np.clip(ii + 1, 0, Nx - 1) + jj * Nx + kk * Nx * Ny
    p_S = ii + np.clip(jj - 1, 0, Ny - 1) * Nx + kk * Nx * Ny
    p_N = ii + np.clip(jj + 1, 0, Ny - 1) * Nx + kk * Nx * Ny
    p_D = ii + jj * Nx + np.clip(kk - 1, 0, Nz - 1) * Nx * Ny
    p_U = ii + jj * Nx + np.clip(kk + 1, 0, Nz - 1) * Nx * Ny
    
    # Maschera per bordi
    is_x_min = (ii == 0)
    is_x_max = (ii == Nx - 1)
    is_y_min = (jj == 0)
    is_y_max = (jj == Ny - 1)
    is_z_min = (kk == 0)
    is_z_max = (kk == Nz - 1)
    
    # Conducibilità ai vicini
    k_P = k_flat[p_all]
    k_W = k_flat[p_W]
    k_E = k_flat[p_E]
    k_S = k_flat[p_S]
    k_N = k_flat[p_N]
    k_D = k_flat[p_D]
    k_U = k_flat[p_U]
    
    # Media armonica per conducibilità interfaccia
    eps = 1e-20  # Evita divisione per zero
    
    # Usa Numba se disponibile e mesh grande (>50k celle)
    USE_NUMBA_HERE = HAS_NUMBA and N > 50000
    
    if USE_NUMBA_HERE:
        # Versione Numba parallelizzata
        k_w, k_e, k_s, k_n, k_d, k_u = _compute_harmonic_means_numba(
            k_P, k_W, k_E, k_S, k_N, k_D, k_U, eps
        )
        a_W, a_E, a_S, a_N, a_D, a_U, a_P = _compute_coefficients_numba(
            k_w, k_e, k_s, k_n, k_d, k_u, d2,
            is_x_min, is_x_max, is_y_min, is_y_max, is_z_min, is_z_max
        )
    else:
        # Versione NumPy vettorizzata
        k_w, k_e, k_s, k_n, k_d, k_u = _compute_harmonic_means_numpy(
            k_P, k_W, k_E, k_S, k_N, k_D, k_U, eps
        )
        
        # Coefficienti base (W/(m³·K))
        a_W = k_w / d2
        a_E = k_e / d2
        a_S = k_s / d2
        a_N = k_n / d2
        a_D = k_d / d2
        a_U = k_u / d2
        
        # Azzera coefficienti per bordi (non ci sono vicini)
        a_W[is_x_min] = 0
        a_E[is_x_max] = 0
        a_S[is_y_min] = 0
        a_N[is_y_max] = 0
        a_D[is_z_min] = 0
        a_U[is_z_max] = 0
        
        # Coefficiente diagonale base
        a_P = a_W + a_E + a_S + a_N + a_D + a_U
    
    # RHS base (sorgente)
    b = Q_flat.copy()
    
    # === CONDIZIONI AL CONTORNO ===
    
    # Dirichlet: T = T_inf
    is_dirichlet = (bc_type_flat == BoundaryType.DIRICHLET)
    
    # Convezione ai bordi: aggiungi contributo h*(T_inf - T)
    # Coefficiente effettivo: 2*k*h/(2*k + h*d) / d
    h_bc = bc_h_flat
    T_inf_bc = bc_T_inf_flat
    
    # Calcola contributi convettivi per ogni bordo
    def add_convection_at_boundary(mask, h, T_inf, k_local):
        """Aggiunge contributo convettivo per celle al bordo."""
        valid = mask & (h > 0)
        a_conv = np.zeros(N)
        b_conv = np.zeros(N)
        
        # Coefficiente Robin: 2*k*h / (2*k + h*d) / d
        denom = 2 * k_local[valid] + h[valid] * d + eps
        a_conv[valid] = 2 * k_local[valid] * h[valid] / denom / d
        b_conv[valid] = a_conv[valid] * T_inf[valid]
        
        return a_conv, b_conv
    
    # Aggiungi convezione a tutti i bordi
    for mask in [is_x_min, is_x_max, is_y_min, is_y_max, is_z_min, is_z_max]:
        a_conv, b_conv = add_convection_at_boundary(mask, h_bc, T_inf_bc, k_P)
        a_P += a_conv
        b += b_conv
    
    # Gestione nodi Dirichlet (T = T_inf fissata)
    a_P[is_dirichlet] = 1.0
    a_W[is_dirichlet] = 0
    a_E[is_dirichlet] = 0
    a_S[is_dirichlet] = 0
    a_N[is_dirichlet] = 0
    a_D[is_dirichlet] = 0
    a_U[is_dirichlet] = 0
    b[is_dirichlet] = T_inf_bc[is_dirichlet]
    
    # Gestione nodi interni con convezione (es. tubi)
    # Questi hanno bc_type = CONVECTION ma non sono sul bordo del dominio
    is_internal_conv = (bc_type_flat == BoundaryType.CONVECTION) & ~(
        is_x_min | is_x_max | is_y_min | is_y_max | is_z_min | is_z_max
    )
    valid_internal_conv = is_internal_conv & (h_bc > 0)
    
    # Aggiungi scambio verso T_inf: a_conv = h/d
    a_conv_internal = np.zeros(N)
    a_conv_internal[valid_internal_conv] = h_bc[valid_internal_conv] / d
    a_P += a_conv_internal
    b[valid_internal_conv] += a_conv_internal[valid_internal_conv] * T_inf_bc[valid_internal_conv]
    
    # === COSTRUZIONE MATRICE SPARSA ===
    
    # Costruisci liste per COO format
    # Diagonale principale
    row_list = [p_all]
    col_list = [p_all]
    data_list = [a_P]
    
    # Vicino West (solo se non al bordo x_min)
    valid = ~is_x_min
    row_list.append(p_all[valid])
    col_list.append(p_W[valid])
    data_list.append(-a_W[valid])
    
    # Vicino East
    valid = ~is_x_max
    row_list.append(p_all[valid])
    col_list.append(p_E[valid])
    data_list.append(-a_E[valid])
    
    # Vicino South
    valid = ~is_y_min
    row_list.append(p_all[valid])
    col_list.append(p_S[valid])
    data_list.append(-a_S[valid])
    
    # Vicino North
    valid = ~is_y_max
    row_list.append(p_all[valid])
    col_list.append(p_N[valid])
    data_list.append(-a_N[valid])
    
    # Vicino Down
    valid = ~is_z_min
    row_list.append(p_all[valid])
    col_list.append(p_D[valid])
    data_list.append(-a_D[valid])
    
    # Vicino Up
    valid = ~is_z_max
    row_list.append(p_all[valid])
    col_list.append(p_U[valid])
    data_list.append(-a_U[valid])
    
    # Concatena e costruisci matrice
    rows = np.concatenate(row_list)
    cols = np.concatenate(col_list)
    data = np.concatenate(data_list)
    
    A = sparse.coo_matrix((data, (rows, cols)), shape=(N, N))
    A = A.tocsr()
    
    return A, b


def _build_steady_state_matrix_loop(mesh: Mesh3D) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """
    Versione originale con loop Python (più lenta, mantenuta per riferimento).
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
