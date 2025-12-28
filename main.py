"""
main.py - Script CLI per testing/scripting (NON PRODUCTION)

=============================================================================
⚠️  WARNING: THIS FILE IS FOR TESTING AND DEVELOPMENT ONLY
=============================================================================

This script contains HARDCODED VALUES for quick testing of the simulation
engine from the command line. It does NOT respect the GUI-driven design.

FOR PRODUCTION USE:
    Run the GUI with: python run_gui.py
    
    The GUI reads ALL parameters from user input widgets - no hardcoded values.
    See docs/05_ARCHITECTURE.md for the proper data flow.

WHY THIS FILE EXISTS:
    - Quick testing of solver changes without launching GUI
    - Automated regression testing
    - Scripted batch simulations
    - Development and debugging

The values in create_test_battery() and run_simulation() are examples only
and do NOT represent real production configurations.
=============================================================================

Workflow:
1. Create 3D mesh
2. Apply geometry and materials
3. Solve steady-state heat equation
4. Analyze energy balance
5. Visualize results (optional)
"""

import numpy as np
import sys
import time
from pathlib import Path

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.mesh import Mesh3D, MaterialID
from src.core.materials import MaterialManager
from src.core.geometry import BatteryGeometry, CylinderGeometry, HeaterConfig, TubeConfig
from src.solver.steady_state import SteadyStateSolver, SolverConfig
from src.analysis.power_balance import PowerBalanceAnalyzer


def create_test_battery():
    """Crea una batteria di test piccola per sviluppo"""
    
    cylinder = CylinderGeometry(
        center_x=3.0,
        center_y=3.0,
        base_z=0.3,
        height=4.0,
        r_tubes=0.3,
        r_sand_inner=1.0,
        r_heaters=1.2,
        r_sand_outer=2.0,
        r_insulation=2.3,
        r_shell=2.32,
    )
    
    return BatteryGeometry(
        cylinder=cylinder,
        heaters=HeaterConfig(power_total=50),  # 50 kW
        tubes=TubeConfig(n_tubes=6, active=False),
        storage_material="steatite",
        insulation_material="rock_wool",
        shell_material="carbon_steel",
        packing_fraction=0.63,
    )


def run_simulation():
    """Esegue una simulazione completa"""
    
    print("=" * 70)
    print("SAND BATTERY THERMAL SIMULATION")
    print("=" * 70)
    
    # =========================================================================
    # 1. CONFIGURAZIONE
    # =========================================================================
    print("\n[1/5] Configurazione...")
    
    # Parametri mesh - UNIFORME con dx = dy = dz
    Lx, Ly, Lz = 6.0, 6.0, 5.0  # Dimensioni dominio [m]
    target_spacing = 0.2  # [m] - spaziatura uniforme target (~30 celle per 6m)
    
    # Crea geometria
    geom = create_test_battery()
    mat_manager = MaterialManager()
    
    # Stima energia
    energy = geom.estimate_energy_capacity(mat_manager, T_high=380, T_low=90)
    print(f"  Capacità stimata: {energy['E_usable_MWh']:.2f} MWh")
    print(f"  Massa sabbia: {energy['mass_sand_tonnes']:.1f} t")
    
    # =========================================================================
    # 2. CREAZIONE MESH
    # =========================================================================
    print("\n[2/5] Creazione mesh...")
    
    t_start = time.time()
    
    # Crea mesh con spaziatura UNIFORME (dx = dy = dz)
    mesh = Mesh3D(
        Lx=Lx, Ly=Ly, Lz=Lz,
        target_spacing=target_spacing,
        uniform=True  # Forza dx = dy = dz
    )
    
    print(f"  Dimensioni dominio: {mesh.Lx:.2f} x {mesh.Ly:.2f} x {mesh.Lz:.2f} m")
    print(f"  Celle: {mesh.Nx} x {mesh.Ny} x {mesh.Nz}")
    print(f"  Spaziatura uniforme: d = {mesh.d:.4f} m (dx = dy = dz)")
    print(f"  Nodi totali: {mesh.N_total:,}")
    print(f"  Spaziatura: dx={mesh.dx:.3f} m, dy={mesh.dy:.3f} m, dz={mesh.dz:.3f} m")
    print(f"  Memoria stimata: {mesh._estimate_memory()/1e6:.1f} MB")
    
    # =========================================================================
    # 3. APPLICAZIONE GEOMETRIA
    # =========================================================================
    print("\n[3/5] Applicazione geometria e materiali...")
    
    geom.apply_to_mesh(mesh, mat_manager)
    
    # Statistiche materiali
    unique, counts = np.unique(mesh.material_id, return_counts=True)
    print("  Distribuzione materiali:")
    for mat_id, count in zip(unique, counts):
        pct = 100 * count / mesh.N_total
        name = MaterialID(mat_id).name if mat_id in [m.value for m in MaterialID] else f"ID={mat_id}"
        print(f"    {name}: {count:,} celle ({pct:.1f}%)")
    
    # Verifica sorgenti
    Q_total = np.sum(mesh.Q) * mesh.dx * mesh.dy * mesh.dz
    print(f"  Potenza totale sorgenti: {Q_total/1000:.1f} kW")
    
    t_setup = time.time() - t_start
    print(f"  Tempo setup: {t_setup:.2f} s")
    
    # =========================================================================
    # 4. SOLUZIONE
    # =========================================================================
    print("\n[4/5] Risoluzione equazione del calore...")
    
    # Configura solutore
    config = SolverConfig(
        method="direct",  # "direct", "bicgstab", "gmres"
        tolerance=1e-8,
        max_iterations=10000,
        preconditioner="ilu",
        verbose=True
    )
    
    solver = SteadyStateSolver(mesh, config)
    result = solver.solve()
    
    # Statistiche temperatura
    stats = solver.get_temperature_stats()
    print("\n  Statistiche temperatura:")
    print(f"    T_min: {stats['T_min']:.1f} °C")
    print(f"    T_max: {stats['T_max']:.1f} °C")
    print(f"    T_mean: {stats['T_mean']:.1f} °C")
    print(f"    T_std: {stats['T_std']:.1f} °C")
    
    # =========================================================================
    # 5. ANALISI
    # =========================================================================
    print("\n[5/5] Analisi bilancio energetico...")
    
    analyzer = PowerBalanceAnalyzer(mesh)
    
    # Bilancio di potenza
    balance = analyzer.compute_power_balance()
    print(f"\n  Bilancio di potenza:")
    print(f"    P_input (resistenze): {balance.P_input/1000:.2f} kW")
    print(f"    P_loss_total: {balance.P_loss_total/1000:.2f} kW")
    print(f"      - Superiore: {balance.P_loss_top/1000:.3f} kW")
    print(f"      - Laterale: {balance.P_loss_lateral/1000:.3f} kW")
    print(f"      - Inferiore: {balance.P_loss_bottom/1000:.3f} kW")
    print(f"    Imbalance: {balance.imbalance:.2f} W ({balance.imbalance_pct:.2f}%)")
    
    # Energia immagazzinata
    stored = analyzer.compute_stored_energy()
    print(f"\n  Energia immagazzinata:")
    print(f"    {stored['E_kWh']:.1f} kWh = {stored['E_MWh']:.3f} MWh")
    
    # Profilo radiale al centro
    cyl = geom.cylinder
    r, T_r = analyzer.compute_radial_temperature_profile(
        cyl.center_x, cyl.center_y, cyl.base_z + cyl.height/2
    )
    print(f"\n  Profilo radiale T(r) a z = {cyl.base_z + cyl.height/2:.1f} m:")
    # Campiona alcuni punti
    for r_val in [0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        idx = np.argmin(np.abs(r - r_val))
        print(f"    r = {r[idx]:.2f} m: T = {T_r[idx]:.1f} °C")
    
    # =========================================================================
    # VISUALIZZAZIONE (opzionale)
    # =========================================================================
    try:
        from src.visualization.renderer import BatteryRenderer, VisualizationConfig
        
        print("\n" + "=" * 70)
        print("VISUALIZZAZIONE")
        print("=" * 70)
        
        viz_config = VisualizationConfig(
            colormap="coolwarm",
            T_min=0,
            T_max=500,
        )
        renderer = BatteryRenderer(viz_config)
        
        # Visualizza sezione orizzontale a metà altezza
        print("  Visualizzazione sezione Z...")
        renderer.plot_slice(mesh, axis='z', position=cyl.base_z + cyl.height/2)
        
        # Visualizza sezione verticale Y
        print("  Visualizzazione sezione Y...")
        renderer.plot_slice(mesh, axis='y', position=cyl.center_y)
        
        # Visualizza campo materiali
        print("  Visualizzazione materiali...")
        viz_config_mat = VisualizationConfig(
            colormap="tab10",
            T_min=0,
            T_max=8,
        )
        renderer_mat = BatteryRenderer(viz_config_mat)
        renderer_mat.plot_slice(mesh, axis='z', position=cyl.base_z + cyl.height/2, field="Material")
        
    except ImportError as e:
        print(f"\nNota: Visualizzazione non disponibile: {e}")
        print("  Installare con: pip install pyvista")
    
    # =========================================================================
    # SOMMARIO
    # =========================================================================
    print("\n" + "=" * 70)
    print("SOMMARIO")
    print("=" * 70)
    print(f"  Tempo totale: {time.time() - t_start:.2f} s")
    print(f"  Convergenza: {'✓' if result.converged else '✗'}")
    print(f"  T range: {stats['T_min']:.1f} - {stats['T_max']:.1f} °C")
    print(f"  Potenza: {balance.P_input/1000:.1f} kW in, {balance.P_loss_total/1000:.1f} kW persi")
    print(f"  Energia: {stored['E_kWh']:.0f} kWh")
    print("=" * 70)
    
    return mesh, solver, analyzer


def run_quick_test():
    """Test rapido con mesh molto piccola"""
    print("=== Quick Test ===")
    
    mesh = Mesh3D(Lx=2, Ly=2, Lz=2, Nx=10, Ny=10, Nz=10)
    
    # Imposta temperatura fissa sul fondo
    mesh.set_fixed_temperature_bc('z_min', 100.0)
    mesh.set_convection_bc('z_max', 10.0, 20.0)
    
    # Sorgente uniforme
    mesh.Q[:] = 1000.0  # W/m³
    mesh.k[:] = 1.0
    
    solver = SteadyStateSolver(mesh, SolverConfig(verbose=True))
    result = solver.solve()
    
    print(f"T range: {mesh.T.min():.1f} - {mesh.T.max():.1f} °C")
    
    return mesh


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sand Battery Simulation")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    parser.add_argument("--visualize", action="store_true", help="Show visualization")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        mesh, solver, analyzer = run_simulation()
        
        if args.visualize:
            try:
                from src.visualization import quick_slice
                quick_slice(mesh, axis='z', position=2.5)
            except ImportError:
                print("PyVista non disponibile")
