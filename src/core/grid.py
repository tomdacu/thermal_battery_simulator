"""Index tables shared by the assembler, the flux integrals and the analysis.

Living in :mod:`src.core` keeps the dependency graph acyclic: ``analysis`` needs
the same neighbour tables as the solver, but must not import the solver package.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mesh import FACES, BoundaryType, Mesh3D

FACE_SLICES: dict[str, tuple] = {
    "x_min": (0, slice(None), slice(None)),
    "x_max": (-1, slice(None), slice(None)),
    "y_min": (slice(None), 0, slice(None)),
    "y_max": (slice(None), -1, slice(None)),
    "z_min": (slice(None), slice(None), 0),
    "z_max": (slice(None), slice(None), -1),
}

#: slice of the first cell inside the domain, per face (for one-sided gradients)
INNER_SLICES: dict[str, tuple] = {
    "x_min": (1, slice(None), slice(None)),
    "x_max": (-2, slice(None), slice(None)),
    "y_min": (slice(None), 1, slice(None)),
    "y_max": (slice(None), -2, slice(None)),
    "z_min": (slice(None), slice(None), 1),
    "z_max": (slice(None), slice(None), -2),
}


#: axis of the face normal, per domain face
FACE_AXIS: dict[str, int] = {"x_min": 0, "x_max": 0, "y_min": 1, "y_max": 1,
                             "z_min": 2, "z_max": 2}


def dirichlet_mask(mesh: Mesh3D, index: GridIndex) -> np.ndarray:
    """(N,) bool mask of the nodes whose temperature is fixed by a Dirichlet face.

    Sources and sinks deposited there never reach the solver: the rows are replaced
    by the identity, so the energy balance must not count them as input.
    """
    mask = np.zeros(mesh.N_total, dtype=bool)
    for face in FACES:
        if mesh.face_bc[face].kind == BoundaryType.DIRICHLET:
            mask |= index.on_face[face]
    return mask


@dataclass(frozen=True)
class GridIndex:
    """Linear-index tables of a mesh, built once and reused."""

    flat: np.ndarray                     # (N,) node ids in Fortran order
    neighbours: tuple[np.ndarray, ...]   # (W, E, S, N, D, U) node ids
    on_face: dict[str, np.ndarray]       # bool mask per domain face
    interior_tube: np.ndarray            # CONVECTION cells strictly inside the box
    #: cell size along each axis broadcast to every node and flattened (Fortran):
    #: the assembly and the flux integrals need it on every face, so it is built once
    sizes: tuple[np.ndarray, np.ndarray, np.ndarray]
    #: face area normal to each axis, same layout (V is not cached: ``mesh.V`` is
    #: a real array while the areas are broadcast views)
    areas: tuple[np.ndarray, np.ndarray, np.ndarray]
    #: cell volume flattened (Fortran order)
    volume: np.ndarray
    #: scratch space for the lazily built tables below (the dataclass is frozen)
    _cache: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def face_factors(self) -> tuple[np.ndarray, ...]:
        """``A / (d_centers V)`` of the six faces, in the FACES order.

        The geometric part of the face coefficients: the solver then only needs the
        harmonic mean of the conductivities.  Built on first use, because a caller
        that only wants the masks must not pay for six full-length arrays.
        """
        factors = self._cache.get("face_factors")
        if factors is None:
            volume = self.volume
            factors = tuple(
                self.areas[FACE_AXIS[face]]
                / (0.5 * (self.sizes[FACE_AXIS[face]] + self.sizes[FACE_AXIS[face]][nb])
                   * volume)
                for face, nb in zip(FACES, self.neighbours, strict=True))
            self._cache["face_factors"] = factors
        return factors

    def face_rows(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                 np.ndarray]:
        """The conservative face list of the structured mesh, in the layout of a tree's.

        One entry per *interior* interface: the ``+axis`` neighbour of every cell that
        has one, with the area [m^2] of that face and the centre-to-centre distance [m]
        of the pair - exactly what :meth:`src.core.mesh_api.MeshAPI.faces` returns for an
        adaptive mesh (``Octree.faces``, in the same ``(cell_i, cell_j, axis, area,
        d_centers)`` layout).  Between the two, a consumer of the flux report can read
        either mesh's own faces instead of re-deriving them from the index arithmetic.

        The box faces are not in the list: the neighbour tables clip there, and their
        exchange is the ``face_bc`` film the assembly applies to the wall cells.  Each
        interface appears once, from its lower cell along the axis, so the heat rate
        ``k_face A / d (T_i - T_j)`` is the one number both cells share.
        """
        cell_i, cell_j, axes, areas, distances = [], [], [], [], []
        for axis in range(3):
            nb = self.neighbours[2 * axis + 1]
            inside = np.flatnonzero(~self.on_face[FACES[2 * axis + 1]])
            other = nb[inside]
            size = self.sizes[axis]
            cell_i.append(inside)
            cell_j.append(other)
            axes.append(np.full(inside.size, axis, dtype=int))
            areas.append(self.areas[axis][inside])
            distances.append(0.5 * (size[inside] + size[other]))
        return (np.concatenate(cell_i), np.concatenate(cell_j),
                np.concatenate(axes), np.concatenate(areas),
                np.concatenate(distances))

    @classmethod
    def from_mesh(cls, mesh: Mesh3D) -> GridIndex:
        nx, ny, nz = mesh.Nx, mesh.Ny, mesh.Nz
        ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        ii, jj, kk = (v.ravel(order="F") for v in (ii, jj, kk))
        flat = ii + jj * nx + kk * nx * ny
        neighbours = tuple(
            a + b * nx + c * nx * ny
            for a, b, c in (
                (np.clip(ii - 1, 0, nx - 1), jj, kk),
                (np.clip(ii + 1, 0, nx - 1), jj, kk),
                (ii, np.clip(jj - 1, 0, ny - 1), kk),
                (ii, np.clip(jj + 1, 0, ny - 1), kk),
                (ii, jj, np.clip(kk - 1, 0, nz - 1)),
                (ii, jj, np.clip(kk + 1, 0, nz - 1)),
            )
        )
        on_face = {
            "x_min": ii == 0, "x_max": ii == nx - 1,
            "y_min": jj == 0, "y_max": jj == ny - 1,
            "z_min": kk == 0, "z_max": kk == nz - 1,
        }
        exposed = np.logical_or.reduce(list(on_face.values()))
        tube = mesh.boundary_type.ravel(order="F") == BoundaryType.CONVECTION
        sizes = tuple(mesh.axis_size(axis).ravel(order="F") for axis in range(3))
        areas = tuple((mesh.Ax, mesh.Ay, mesh.Az)[axis].ravel(order="F")
                      for axis in range(3))
        return cls(flat=flat, neighbours=neighbours, on_face=on_face,
                   interior_tube=tube & ~exposed, sizes=sizes, areas=areas,
                   volume=mesh.V.ravel(order="F"))
