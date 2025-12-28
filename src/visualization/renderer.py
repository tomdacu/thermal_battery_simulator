"""
renderer.py - Visualizzazione 3D con PyVista

Fornisce:
- Rendering 3D del campo di temperatura
- Sezioni (slices) interattive
- Isosuperfici
- Export VTK
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import warnings

# Importa PyVista con gestione errore
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    pv = None
    PYVISTA_AVAILABLE = False
    warnings.warn("PyVista non disponibile. Installare con: pip install pyvista")


@dataclass
class VisualizationConfig:
    """Configurazione della visualizzazione"""
    colormap: str = "coolwarm"
    T_min: Optional[float] = None   # Auto se None
    T_max: Optional[float] = None   # Auto se None
    opacity: float = 0.8
    show_mesh: bool = False
    show_edges: bool = False
    show_axes: bool = True
    show_colorbar: bool = True
    background_color: str = "white"
    window_size: Tuple[int, int] = (1200, 800)


class BatteryRenderer:
    """
    Renderer 3D per la visualizzazione della Sand Battery.
    
    Utilizza PyVista per il rendering interattivo.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista non è installato. Installare con: pip install pyvista")
        
        self.config = config if config is not None else VisualizationConfig()
        self._plotter = None
        self._grid = None
        
    def create_grid_from_mesh(self, mesh):
        """Crea un UniformGrid PyVista dalla mesh."""
        grid = pv.ImageData(
            dimensions=(mesh.Nx + 1, mesh.Ny + 1, mesh.Nz + 1),
            spacing=(mesh.dx, mesh.dy, mesh.dz),
            origin=(0, 0, 0)
        )
        
        grid.cell_data["Temperature"] = mesh.T.ravel(order='F')
        grid.cell_data["Material"] = mesh.material_id.ravel(order='F').astype(float)
        grid.cell_data["k"] = mesh.k.ravel(order='F')
        grid.cell_data["Q"] = mesh.Q.ravel(order='F')
        
        self._grid = grid
        return grid
    
    def plot_3d(self, mesh, field: str = "Temperature",
                clip_box: Optional[Tuple[float, float, float, float, float, float]] = None,
                show: bool = True):
        """Visualizza il campo 3D con possibile clipping."""
        grid = self.create_grid_from_mesh(mesh)
        
        pl = pv.Plotter(window_size=self.config.window_size)
        pl.set_background(self.config.background_color)
        
        data = grid.cell_data[field]
        clim = [
            self.config.T_min if self.config.T_min is not None else data.min(),
            self.config.T_max if self.config.T_max is not None else data.max()
        ]
        
        if clip_box is not None:
            grid = grid.clip_box(clip_box, invert=False)
        
        pl.add_mesh(
            grid, scalars=field, cmap=self.config.colormap,
            clim=clim, opacity=self.config.opacity,
            show_edges=self.config.show_edges,
            scalar_bar_args={'title': f'{field} [°C]', 'vertical': True}
        )
        
        if self.config.show_axes:
            pl.add_axes()
        
        if show:
            pl.show()
        else:
            self._plotter = pl
            return pl
    
    def plot_slice(self, mesh, axis: str = "z", position: float = None,
                   field: str = "Temperature", show: bool = True):
        """Visualizza una sezione 2D."""
        grid = self.create_grid_from_mesh(mesh)
        
        if position is None:
            position = {'x': mesh.Lx/2, 'y': mesh.Ly/2, 'z': mesh.Lz/2}[axis]
        
        normals = {'x': (1,0,0), 'y': (0,1,0), 'z': (0,0,1)}
        origins = {
            'x': (position, mesh.Ly/2, mesh.Lz/2),
            'y': (mesh.Lx/2, position, mesh.Lz/2),
            'z': (mesh.Lx/2, mesh.Ly/2, position)
        }
        
        sliced = grid.slice(normal=normals[axis], origin=origins[axis])
        
        data = grid.cell_data[field]
        clim = [
            self.config.T_min if self.config.T_min is not None else data.min(),
            self.config.T_max if self.config.T_max is not None else data.max()
        ]
        
        pl = pv.Plotter(window_size=self.config.window_size)
        pl.set_background(self.config.background_color)
        
        pl.add_mesh(sliced, scalars=field, cmap=self.config.colormap,
                   clim=clim, show_edges=self.config.show_edges)
        
        pl.add_text(f"Slice {axis.upper()} = {position:.2f} m", 
                   position='upper_left', font_size=12)
        
        if self.config.show_axes:
            pl.add_axes()
        
        if show:
            pl.show()
        else:
            return pl
    
    def export_vtk(self, mesh, filename: str):
        """Esporta i dati in formato VTK."""
        grid = self.create_grid_from_mesh(mesh)
        grid.save(filename)
        print(f"Salvato: {filename}")


# =============================================================================
# FUNZIONI DI UTILITÀ
# =============================================================================

def quick_plot(mesh, field: str = "Temperature", **kwargs):
    """Plot rapido del campo"""
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista non è installato")
    renderer = BatteryRenderer()
    renderer.plot_3d(mesh, field=field, **kwargs)


def quick_slice(mesh, axis: str = "z", position: float = None, **kwargs):
    """Sezione rapida"""
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista non è installato")
    renderer = BatteryRenderer()
    renderer.plot_slice(mesh, axis=axis, position=position, **kwargs)
