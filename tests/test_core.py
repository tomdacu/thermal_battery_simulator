"""
test_core.py - Unit tests per i moduli core

Eseguire con: pytest tests/test_core.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.mesh import Mesh3D, MaterialID, BoundaryType
from src.core.materials import MaterialManager, MaterialType
from src.core.geometry import BatteryGeometry, CylinderGeometry, create_small_test_geometry


class TestMesh3D:
    """Test per la classe Mesh3D"""
    
    def test_creation(self):
        """Test creazione base"""
        mesh = Mesh3D(Lx=1.0, Ly=1.0, Lz=1.0, Nx=10, Ny=10, Nz=10)
        
        assert mesh.Nx == 10
        assert mesh.Ny == 10
        assert mesh.Nz == 10
        assert mesh.N_total == 1000
        assert mesh.dx == pytest.approx(0.1)
    
    def test_indexing(self):
        """Test conversione indici"""
        mesh = Mesh3D(Lx=1.0, Ly=1.0, Lz=1.0, Nx=10, Ny=10, Nz=10)
        
        # Test round-trip
        for i, j, k in [(0, 0, 0), (5, 5, 5), (9, 9, 9), (3, 7, 2)]:
            p = mesh.ijk_to_linear(i, j, k)
            i2, j2, k2 = mesh.linear_to_ijk(p)
            assert (i, j, k) == (i2, j2, k2)
    
    def test_coordinates(self):
        """Test coordinate centri cella"""
        mesh = Mesh3D(Lx=1.0, Ly=1.0, Lz=1.0, Nx=10, Ny=10, Nz=10)
        
        # Prima cella
        x, y, z = mesh.get_position(0, 0, 0)
        assert x == pytest.approx(0.05)
        assert y == pytest.approx(0.05)
        assert z == pytest.approx(0.05)
        
        # Ultima cella
        x, y, z = mesh.get_position(9, 9, 9)
        assert x == pytest.approx(0.95)
        assert y == pytest.approx(0.95)
        assert z == pytest.approx(0.95)
    
    def test_find_cell(self):
        """Test ricerca cella da coordinate"""
        mesh = Mesh3D(Lx=1.0, Ly=1.0, Lz=1.0, Nx=10, Ny=10, Nz=10)
        
        i, j, k = mesh.find_cell(0.5, 0.5, 0.5)
        assert (i, j, k) == (5, 5, 5)
    
    def test_boundary_detection(self):
        """Test rilevamento bordi"""
        mesh = Mesh3D(Lx=1.0, Ly=1.0, Lz=1.0, Nx=10, Ny=10, Nz=10)
        
        # Bordo z=0 dovrebbe essere Dirichlet
        assert mesh.boundary_type[5, 5, 0] == BoundaryType.DIRICHLET
        
        # Bordo z=max dovrebbe essere convezione
        assert mesh.boundary_type[5, 5, 9] == BoundaryType.CONVECTION
        
        # Nodo interno
        assert mesh.boundary_type[5, 5, 5] == BoundaryType.INTERNAL
    
    def test_memory_estimate(self):
        """Test stima memoria"""
        mesh = Mesh3D(Lx=1.0, Ly=1.0, Lz=1.0, Nx=10, Ny=10, Nz=10)
        
        mem = mesh._estimate_memory()
        assert mem > 0
        assert mem < 1e9  # < 1GB per mesh piccola


class TestMaterialManager:
    """Test per MaterialManager"""
    
    def test_get_material(self):
        """Test recupero materiale"""
        manager = MaterialManager()
        
        steatite = manager.get("steatite")
        assert steatite.name == "Steatite (Pietra Ollare)"
        assert steatite.k > 0
        assert steatite.rho > 0
        assert steatite.cp > 0
    
    def test_list_materials(self):
        """Test lista materiali"""
        manager = MaterialManager()
        
        storage = manager.list_materials(MaterialType.STORAGE)
        assert "steatite" in storage
        assert "silica_sand" in storage
        assert len(storage) >= 5
    
    def test_effective_properties(self):
        """Test calcolo proprietà effettive"""
        manager = MaterialManager()
        
        eff = manager.compute_packed_bed_properties("steatite", packing_fraction=0.63)
        
        # Conducibilità effettiva < conducibilità solida
        solid = manager.get("steatite")
        assert eff.k < solid.k
        
        # Densità effettiva < densità solida
        assert eff.rho < solid.rho
    
    def test_unknown_material(self):
        """Test materiale sconosciuto"""
        manager = MaterialManager()
        
        with pytest.raises(KeyError):
            manager.get("materiale_inventato")
    
    def test_energy_density(self):
        """Test calcolo densità energetica"""
        manager = MaterialManager()
        
        ed = manager.get_energy_density("steatite", T_high=400, T_low=100, packing_fraction=0.63)
        
        # Dovrebbe essere positiva e ragionevole
        assert ed > 100  # MJ/m³
        assert ed < 1000  # MJ/m³


class TestGeometry:
    """Test per BatteryGeometry"""
    
    def test_create_geometry(self):
        """Test creazione geometria"""
        geom = create_small_test_geometry()
        
        assert geom.cylinder.height > 0
        assert geom.cylinder.r_shell > geom.cylinder.r_insulation
    
    def test_zone_volumes(self):
        """Test calcolo volumi"""
        geom = create_small_test_geometry()
        
        volumes = geom.get_zone_volumes()
        
        assert volumes['sand_total'] > 0
        assert volumes['insulation'] > 0
        assert volumes['total'] > volumes['sand_total']
    
    def test_zone_masses(self):
        """Test calcolo masse"""
        geom = create_small_test_geometry()
        manager = MaterialManager()
        
        masses = geom.get_zone_masses(manager)
        
        assert masses['sand_total'] > 0
        assert masses['insulation'] > 0
    
    def test_energy_capacity(self):
        """Test stima capacità energetica"""
        geom = create_small_test_geometry()
        manager = MaterialManager()
        
        energy = geom.estimate_energy_capacity(manager, T_high=380, T_low=90)
        
        assert energy['E_usable_MWh'] > 0
        assert energy['E_usable_kWh'] == energy['E_usable_MWh'] * 1000
    
    def test_apply_to_mesh(self):
        """Test applicazione geometria a mesh"""
        geom = create_small_test_geometry()
        manager = MaterialManager()
        
        mesh = Mesh3D(Lx=6, Ly=6, Lz=5, Nx=20, Ny=20, Nz=15)
        geom.apply_to_mesh(mesh, manager)
        
        # Verifica che siano stati assegnati diversi materiali
        unique_materials = np.unique(mesh.material_id)
        assert len(unique_materials) >= 3  # Almeno aria, sabbia, isolamento
        
        # Verifica sorgenti di calore nella zona resistenze
        assert np.any(mesh.Q > 0)

    def test_tubes_inactive_do_not_set_internal_convection(self):
        """Se i tubi sono inattivi, le celle-tubo non devono diventare CONVECTION con h>0."""
        geom = create_small_test_geometry()
        geom.tubes.active = False
        manager = MaterialManager()

        mesh = Mesh3D(Lx=6, Ly=6, Lz=5, Nx=20, Ny=20, Nz=15)
        geom.apply_to_mesh(mesh, manager)

        # Non devono esserci celle interne CONVECTION con h>0
        internal = np.ones(mesh.boundary_type.shape, dtype=bool)
        internal[0, :, :] = False
        internal[-1, :, :] = False
        internal[:, 0, :] = False
        internal[:, -1, :] = False
        internal[:, :, 0] = False
        internal[:, :, -1] = False

        conv_internal = (mesh.boundary_type == BoundaryType.CONVECTION) & internal
        assert not np.any(conv_internal & (mesh.bc_h > 0))


class TestSliceExtraction:
    """Test estrazione sezioni"""
    
    def test_z_slice(self):
        """Test sezione orizzontale"""
        mesh = Mesh3D(Lx=1, Ly=1, Lz=1, Nx=10, Ny=10, Nz=10)
        mesh.T[:] = np.arange(mesh.Nz).reshape(1, 1, -1)  # Gradiente in z
        
        Y, Z, T = mesh.get_temperature_slice('z', 0.5)
        
        assert T.shape == (10, 10)
        assert np.all(T == T[0, 0])  # Uniforme su questo piano
    
    def test_y_slice(self):
        """Test sezione verticale Y"""
        mesh = Mesh3D(Lx=1, Ly=1, Lz=1, Nx=10, Ny=10, Nz=10)
        mesh.T[:] = 100.0
        
        X, Z, T = mesh.get_temperature_slice('y', 0.5)
        
        assert T.shape == (10, 10)


# =============================================================================
# ESECUZIONE
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
