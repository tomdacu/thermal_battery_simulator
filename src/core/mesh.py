"""Structured Cartesian 3D mesh for the FDM heat solver.

Indexing contract: node ``(i, j, k)`` has linear index ``i + j*Nx + k*Nx*Ny``
(Fortran order), matching ``field.ravel(order="F")`` used everywhere else.

Unit contract: ``T``, ``bc_T_inf`` and every ``FaceBC`` temperature are KELVIN.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from ..constants import (
    CP_AIR,
    DEFAULT_SPACING,
    K_AIR,
    MIN_CELLS_PER_AXIS,
    RHO_AIR,
    T_AMBIENT_DEFAULT,
    T_GROUND_DEFAULT,
    T_INITIAL_DEFAULT,
)
from ..units import check_kelvin
from .refinement import (GridSpec, edges_to_centers,
                        edges_to_sizes, size_at, worst_ratio)

FACES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


def _find(edges: np.ndarray, position: float) -> int:
    """Index of the cell of ``edges`` containing ``position``, clamped."""
    return int(np.clip(np.searchsorted(edges, position, side="right") - 1,
                       0, edges.size - 2))


class MaterialID(IntEnum):
    AIR = 0
    SAND = 1
    INSULATION = 2
    STEEL = 3
    TUBES = 4
    HEATERS = 5
    GROUND = 6
    CONCRETE = 7


class BoundaryType(IntEnum):
    INTERNAL = 0
    DIRICHLET = 1
    NEUMANN = 2
    CONVECTION = 3
    SYMMETRY = 4


@dataclass
class FaceBC:
    """Boundary condition of one domain face.

    ``kind`` INTERNAL means adiabatic (zero flux); CONVECTION uses ``h`` and
    ``value`` (fluid temperature, K); DIRICHLET fixes ``value`` (K); NEUMANN
    imposes the heat flux ``value`` [W/m^2] flowing INTO the domain.
    """

    kind: BoundaryType = BoundaryType.INTERNAL
    h: float = 0.0
    value: float = T_AMBIENT_DEFAULT
    emissivity: float = 0.0

    def is_active(self) -> bool:
        if self.kind == BoundaryType.DIRICHLET:
            return True
        if self.kind == BoundaryType.CONVECTION:
            return self.h > 0.0 or self.emissivity > 0.0
        if self.kind == BoundaryType.NEUMANN:
            return self.value != 0.0
        return False


@dataclass
class NodeProperties:
    """Thermal properties of a single material."""

    k: float
    rho: float
    cp: float

    @property
    def alpha(self) -> float:
        """Thermal diffusivity [m^2/s]."""
        return self.k / (self.rho * self.cp)


@dataclass
class Mesh3D:
    """Structured Cartesian grid, uniform or graded.

    Two ways to describe it:

    * ``spacing`` - legacy uniform grid ``dx = dy = dz = spacing``, snapped to the
      box (``snapped`` reports the leftover);
    * ``grid`` - a :class:`~src.core.refinement.GridSpec`, i.e. per-axis *physical*
      refinement targets: cells are small where the gradients are and coarse in the
      far field.

    Either way the public contract is the same: ``dx``/``dy``/``dz`` are per-axis
    cell sizes and ``V``/``Ax``/``Ay``/``Az`` the cell volume and face areas.  The
    scalar shortcuts ``d``, ``V_cell`` and ``A_cell`` stay available for uniform
    grids only: on a graded grid a single cell size does not exist and asking for
    one raises instead of silently using the first cell.
    """

    Lx: float = 10.0
    Ly: float = 10.0
    Lz: float = 8.0
    spacing: float | None = None
    grid: GridSpec | None = None

    # derived geometry (filled in __post_init__)
    Nx: int = field(init=False)
    Ny: int = field(init=False)
    Nz: int = field(init=False)
    N_total: int = field(init=False)
    snapped: dict[str, float] = field(init=False, repr=False)
    uniform: bool = field(init=False)

    #: cell sizes per axis, ``(Nx,)``-shaped
    dx: np.ndarray = field(init=False, repr=False)
    dy: np.ndarray = field(init=False, repr=False)
    dz: np.ndarray = field(init=False, repr=False)
    #: cell edges per axis, ``(Nx+1,)``-shaped
    edges_x: np.ndarray = field(init=False, repr=False)
    edges_y: np.ndarray = field(init=False, repr=False)
    edges_z: np.ndarray = field(init=False, repr=False)

    # coordinates and grids
    x: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)
    z: np.ndarray = field(init=False, repr=False)
    X: np.ndarray = field(init=False, repr=False)
    Y: np.ndarray = field(init=False, repr=False)
    Z: np.ndarray = field(init=False, repr=False)

    # material / boundary description
    material_id: np.ndarray = field(init=False, repr=False)
    boundary_type: np.ndarray = field(init=False, repr=False)
    face_bc: dict[str, FaceBC] = field(init=False, repr=False)

    # fields
    k: np.ndarray = field(init=False, repr=False)
    rho: np.ndarray = field(init=False, repr=False)
    cp: np.ndarray = field(init=False, repr=False)
    Q_source: np.ndarray = field(init=False, repr=False)   # volumetric sources [W/m^3] (>= 0)
    Q_sink: np.ndarray = field(init=False, repr=False)     # volumetric sinks [W/m^3] (<= 0)
    source_mask: np.ndarray = field(init=False, repr=False)  # cells whose Q_source is driven by the power profile
    T: np.ndarray = field(init=False, repr=False)          # [K]
    bc_h: np.ndarray = field(init=False, repr=False)       # internal (tube) convection [W/m^2/K]
    bc_T_inf: np.ndarray = field(init=False, repr=False)   # internal fluid temperature [K]

    def __post_init__(self) -> None:
        if self.grid is not None and self.spacing is not None:
            raise ValueError("give either spacing (uniform) or grid (graded), not both")
        if self.grid is not None:
            self._build_graded()
        else:
            self._build_uniform(float(self.spacing if self.spacing is not None
                                      else DEFAULT_SPACING))

        self.x = edges_to_centers(self.edges_x)
        self.y = edges_to_centers(self.edges_y)
        self.z = edges_to_centers(self.edges_z)
        if self.uniform:
            # a uniform grid has *identical* cells by construction: taking the sizes
            # from the edges would leave them differing by an ulp, which breaks the
            # exact symmetry of the operator
            self.dx = np.full(self.Nx, self._step)
            self.dy = np.full(self.Ny, self._step)
            self.dz = np.full(self.Nz, self._step)
        else:
            self.dx = edges_to_sizes(self.edges_x)
            self.dy = edges_to_sizes(self.edges_y)
            self.dz = edges_to_sizes(self.edges_z)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")

        # per-cell volumes and face areas, both (Nx, Ny, Nz).  A face area does not
        # depend on the index along its own axis (the two cells sharing the face have
        # the same extent in the other two axes), so the areas are broadcast views
        # and cost no memory.
        self.V = (self.dx[:, None, None] * self.dy[None, :, None]
                  * self.dz[None, None, :])
        self.Ax = np.broadcast_to(self.dy[None, :, None] * self.dz[None, None, :],
                                  self.V.shape)
        self.Ay = np.broadcast_to(self.dx[:, None, None] * self.dz[None, None, :],
                                  self.V.shape)
        self.Az = np.broadcast_to(self.dx[:, None, None] * self.dy[None, :, None],
                                  self.V.shape)

        shape = (self.Nx, self.Ny, self.Nz)
        self.material_id = np.zeros(shape, dtype=np.int8)
        self.boundary_type = np.zeros(shape, dtype=np.int8)
        self.k = np.full(shape, K_AIR)
        self.rho = np.full(shape, RHO_AIR)
        self.cp = np.full(shape, CP_AIR)
        self.Q_source = np.zeros(shape)
        self.Q_sink = np.zeros(shape)
        #: cells that are NOT part of the thermal problem (the air around the vessel):
        #: no conduction into them, and their interface carries the outside film
        self.excluded = np.zeros(shape, dtype=bool)
        #: film coefficient of the outer surface [W/(m^2 K)] (natural + wind)
        self.h_out = 0.0
        #: ambient temperature driving that film [K]
        self.t_ambient = T_AMBIENT_DEFAULT
        #: contact conductance between materials [W/(m^2 K)]: real interfaces are not
        #: perfect, so a finite resistance sits in series with the half cells
        self.h_contact = 0.0
        self.source_mask = np.zeros(shape, dtype=bool)
        self.T = np.full(shape, T_INITIAL_DEFAULT)
        self.bc_h = np.zeros(shape)
        self.bc_T_inf = np.full(shape, T_AMBIENT_DEFAULT)

        # default domain BCs: insulated sides, ground at fixed temperature
        self.face_bc = {f: FaceBC() for f in FACES}
        self.face_bc["z_min"] = FaceBC(BoundaryType.DIRICHLET, value=T_GROUND_DEFAULT)

    # ------------------------------------------------------------ grid build
    def _build_uniform(self, d: float) -> None:
        """Uniform grid: one cell size, box snapped to a whole number of cells."""
        if d <= 0:
            raise ValueError(f"spacing must be > 0, got {d}")
        nx = max(MIN_CELLS_PER_AXIS, int(round(self.Lx / d)))
        step = self.Lx / nx
        self.Nx = nx
        self.Ny = max(MIN_CELLS_PER_AXIS, int(round(self.Ly / step)))
        self.Nz = max(MIN_CELLS_PER_AXIS, int(round(self.Lz / step)))
        self.snapped = {"Ly": self.Ly - self.Ny * step, "Lz": self.Lz - self.Nz * step}
        # edges from a constant step (not linspace): every cell of a uniform grid is
        # then bit-identical, so the face coefficients are exactly symmetric
        self._step = step
        self.edges_x = np.arange(self.Nx + 1) * step
        self.edges_y = np.arange(self.Ny + 1) * step
        self.edges_z = np.arange(self.Nz + 1) * step
        self.Lx = self.Nx * step
        self.Ly = self.Ny * step
        self.Lz = self.Nz * step
        self.uniform = True
        self.N_total = self.Nx * self.Ny * self.Nz

    def _build_graded(self) -> None:
        """Graded path: the spec decides the edges, the box is spanned exactly."""
        self.edges_x, self.edges_y, self.edges_z = self.grid.edges(self.Lx, self.Ly,
                                                                  self.Lz)
        self.Nx, self.Ny, self.Nz = (e.size - 1 for e in
                                     (self.edges_x, self.edges_y, self.edges_z))
        # the graded walk always lands exactly on the box: nothing to snap
        self.snapped = {"Ly": 0.0, "Lz": 0.0}
        self.uniform = False
        self.N_total = self.Nx * self.Ny * self.Nz

    # ------------------------------------------------------------------ props
    def _require_uniform(self, what: str) -> None:
        if not self.uniform:
            raise ValueError(
                f"{what} does not exist on a graded mesh: use the per-axis arrays "
                f"(dx/dy/dz, V, Ax/Ay/Az) or a local size (size_x/y/z, cell_size_at)")

    @property
    def d(self) -> float:
        """Cell size of a *uniform* mesh [m]; raises on a graded one."""
        self._require_uniform("a single cell size")
        return float(self.dx[0])

    @property
    def V_cell(self) -> float:
        """Cell volume of a *uniform* mesh [m^3]; raises on a graded one."""
        self._require_uniform("a single cell volume")
        return float(self.dx[0] * self.dy[0] * self.dz[0])

    @property
    def A_cell(self) -> float:
        """Cell face area of a *uniform* mesh [m^2]; raises on a graded one."""
        self._require_uniform("a single face area")
        return float(self.dx[0] * self.dy[0])

    @property
    def h_char(self) -> np.ndarray:
        """Characteristic cell size ``V^(1/3)`` [m], for lumped exchange models."""
        return np.cbrt(self.V)

    def axis_size(self, axis: int) -> np.ndarray:
        """Cell size along ``axis`` (0=x, 1=y, 2=z) broadcast to every cell."""
        sizes = (self.dx, self.dy, self.dz)[axis]
        shape = [1, 1, 1]
        shape[axis] = sizes.size
        return np.broadcast_to(sizes.reshape(shape), self.T.shape)

    def size_x(self, position: float) -> float:
        """Cell size in x at the coordinate ``position`` [m]."""
        return size_at(self.edges_x, position)

    def size_y(self, position: float) -> float:
        return size_at(self.edges_y, position)

    def size_z(self, position: float) -> float:
        return size_at(self.edges_z, position)

    def cell_size_at(self, x: float, y: float, z: float) -> float:
        """Characteristic size of the cell containing ``(x, y, z)`` [m]."""
        if self.uniform:
            return float(self.dx[0])
        i, j, k = self.find_cell(x, y, z)
        return float(self.V[i, j, k]) ** (1.0 / 3.0)

    def size_label(self) -> str:
        """Cell size as text: one value when uniform, a range when graded."""
        summary = self.grid_summary()
        if self.uniform:
            return f"{summary['min_size']:.3f} m"
        return (f"{summary['min_size']:.3f}-{summary['max_size']:.3f} m "
                f"(ratio {summary['worst_ratio']:.2f})")

    def grid_summary(self) -> dict:
        """Sizes of the realised grid (GUI summary and log)."""
        if self.uniform:
            d = float(self.dx[0])
            return {"cells_axis": (self.Nx, self.Ny, self.Nz), "cells": self.N_total,
                    "min_size": d, "max_size": d, "worst_ratio": 1.0}
        return {
            "cells_axis": (self.Nx, self.Ny, self.Nz), "cells": self.N_total,
            "min_size": float(min(e.min() for e in (self.dx, self.dy, self.dz))),
            "max_size": float(max(e.max() for e in (self.dx, self.dy, self.dz))),
            "worst_ratio": max(worst_ratio(e) for e in
                               (self.edges_x, self.edges_y, self.edges_z)),
        }

    # ------------------------------------------------------- index arithmetic
    def ijk_to_linear(self, i: int, j: int, k: int) -> int:
        return i + j * self.Nx + k * self.Nx * self.Ny

    def linear_to_ijk(self, p: int) -> tuple[int, int, int]:
        k, rem = divmod(int(p), self.Nx * self.Ny)
        j, i = divmod(rem, self.Nx)
        return i, j, k

    def find_cell(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        """Index of the cell containing (x, y, z); clamped to the domain."""
        return (_find(self.edges_x, x), _find(self.edges_y, y), _find(self.edges_z, z))

    # ------------------------------------------------- boundary conditions
    def set_convection_bc(self, face: str, h: float, T_inf: float) -> None:
        """Convection on a domain face; ``T_inf`` in KELVIN."""
        self._check_face(face)
        check_kelvin(T_inf, f"set_convection_bc({face}) T_inf")
        self.face_bc[face] = FaceBC(BoundaryType.CONVECTION, h=float(h), value=float(T_inf))

    def set_fixed_temperature_bc(self, face: str, T: float) -> None:
        """Dirichlet temperature on a domain face; ``T`` in KELVIN."""
        self._check_face(face)
        check_kelvin(T, f"set_fixed_temperature_bc({face}) T")
        self.face_bc[face] = FaceBC(BoundaryType.DIRICHLET, value=float(T))

    def set_heat_flux_bc(self, face: str, q: float, T_inf: float | None = None) -> None:
        """Imposed heat flux [W/m^2] entering the domain through ``face``."""
        self._check_face(face)
        self.face_bc[face] = FaceBC(BoundaryType.NEUMANN, h=1.0, value=float(q))
        if T_inf is not None:
            check_kelvin(T_inf, f"set_heat_flux_bc({face}) T_inf")

    def set_adiabatic(self, face: str) -> None:
        """Zero-flux (insulated) domain face."""
        self._check_face(face)
        self.face_bc[face] = FaceBC(BoundaryType.INTERNAL)

    def set_internal_convection(self, mask: np.ndarray, h: float, T_fluid: float) -> None:
        """Convection on interior cells (heat-exchanger tubes); ``T_fluid`` in KELVIN."""
        check_kelvin(T_fluid, "set_internal_convection T_fluid")
        m = np.asarray(mask, dtype=bool)
        self.bc_h[m] = float(h)
        self.bc_T_inf[m] = float(T_fluid)
        if h > 0:
            self.boundary_type[m] = BoundaryType.CONVECTION
        else:
            self.boundary_type[m] = BoundaryType.INTERNAL

    @staticmethod
    def _check_face(face: str) -> None:
        if face not in FACES:
            raise ValueError(f"invalid face {face!r}; expected one of {FACES}")

    # ---------------------------------------------------------- conversions
    def unflatten_field(self, vector: np.ndarray) -> np.ndarray:
        return np.asarray(vector).reshape((self.Nx, self.Ny, self.Nz), order="F")

    # ---------------------------------------------------------- validation
    def validate(self, check_temperature: bool = True) -> None:
        """Reject non-physical fields before an expensive solve."""
        for name, arr in (("k", self.k), ("rho", self.rho), ("cp", self.cp)):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name}: non-finite values")
            if arr.min() <= 0:
                raise ValueError(f"{name}: values must be > 0, min = {arr.min():.3g}")
        if not np.all(np.isfinite(self.Q_source)) or not np.all(np.isfinite(self.Q_sink)):
            raise ValueError("Q fields: non-finite values")
        if np.any(self.Q_source < 0):
            raise ValueError("Q_source must be >= 0 (use Q_sink for extractions)")
        if np.any(self.Q_sink > 0):
            raise ValueError("Q_sink must be <= 0")
        if check_temperature:
            check_kelvin(self.T, "mesh.T")
            for face, bc in self.face_bc.items():
                if bc.kind in (BoundaryType.DIRICHLET, BoundaryType.CONVECTION):
                    check_kelvin(bc.value, f"face_bc[{face}].value")
        for face, bc in self.face_bc.items():
            if bc.kind == BoundaryType.CONVECTION and bc.h < 0:
                raise ValueError(f"face_bc[{face}].h must be >= 0")

    # ------------------------------------------------------------------ info
    def get_info(self) -> dict[str, object]:
        summary = self.grid_summary()
        return {
            "dimensions": (self.Lx, self.Ly, self.Lz),
            "cells": (self.Nx, self.Ny, self.Nz),
            "uniform": self.uniform,
            "cell_size_min": summary["min_size"],
            "cell_size_max": summary["max_size"],
            "worst_ratio": summary["worst_ratio"],
            "total_nodes": self.N_total,
            "memory_MB": self._estimate_memory() / 1e6,
            "snap_correction_m": dict(self.snapped),
            "face_bc": {f: bc.kind.name for f, bc in self.face_bc.items()},
        }

    def _estimate_memory(self) -> float:
        n_float64 = 12  # k, rho, cp, Q_source, Q_sink, T, bc_h, bc_T_inf, X, Y, Z, V
        n_small = 2     # material_id, boundary_type
        return self.N_total * (n_float64 * 8 + n_small * 1)
