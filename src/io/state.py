"""Persistent simulation state (HDF5).

The file records everything needed to decide whether a stored field is still
compatible with the current model: the grid, the material map, a content hash of
the geometry description and the temperature unit.  ``load_state`` refuses a
file whose geometry hash does not match the mesh it is applied to, instead of
silently mixing two different models.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from ..constants import T0
from ..core.mesh import FaceBC, Mesh3D

try:
    import h5py

    HAS_H5PY = True
except ImportError:  # pragma: no cover
    h5py = None
    HAS_H5PY = False

FORMAT_VERSION = 2
TEMPERATURE_UNIT = "K"


class StateError(RuntimeError):
    """Raised when a state file is missing, corrupt or incompatible."""


@dataclass
class SimulationState:
    """Snapshot of a mesh plus the metadata needed to validate it."""

    name: str = "state"
    created: float = 0.0
    grid: dict[str, float] = field(default_factory=dict)
    geometry_hash: str = ""
    geometry_params: dict[str, float] = field(default_factory=dict)
    temperature_unit: str = TEMPERATURE_UNIT
    version: int = FORMAT_VERSION
    material_id: np.ndarray | None = None
    T: np.ndarray | None = None
    Q_source: np.ndarray | None = None
    face_bc: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def temperature_celsius(self) -> np.ndarray:
        return self.T - T0


def _canonical(value):
    """Stable representation of a geometry parameter of any simple type."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    return str(value)


def geometry_hash(mesh: Mesh3D, geometry_params: dict[str, object] | None = None) -> str:
    """Content hash of everything that changes the physics of a stored field.

    Unlike the old ``(shape, sum of material ids)`` digest, this also covers the
    domain size, the cell spacing, the per-material cell counts and the geometry
    parameters, so moving a heater or changing a material is detected.  Parameter
    values may be numbers or strings (material names).
    """
    payload = {
        "shape": [mesh.Nx, mesh.Ny, mesh.Nz],
        "size": [round(float(mesh.Lx), 6), round(float(mesh.Ly), 6), round(float(mesh.Lz), 6)],
        "edges": [[round(float(v), 9) for v in edges]
                  for edges in (mesh.edges_x, mesh.edges_y, mesh.edges_z)],
        "counts": {int(k): int(v) for k, v in
                   zip(*np.unique(mesh.material_id, return_counts=True), strict=True)},
        "source_cells": int(np.count_nonzero(mesh.source_mask)),
        "geometry": {str(k): _canonical(v)
                     for k, v in sorted((geometry_params or {}).items())},
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


class StateManager:
    """Read/write :class:`SimulationState` objects as HDF5."""

    def __init__(self, directory: str = "results/states") -> None:
        self.directory = directory

    # ------------------------------------------------------------------ save
    def save_state(self, mesh: Mesh3D, name: str = "state",
                   geometry_params: dict[str, float] | None = None,
                   notes: list[str] | None = None, path: str | None = None) -> str:
        if not HAS_H5PY:
            raise StateError("h5py is required to save a state (pip install h5py)")
        mesh.validate()
        target = path or os.path.join(self.directory, f"{name}.h5")
        os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)
        digest = geometry_hash(mesh, geometry_params)
        with h5py.File(target, "w") as f:
            f.attrs["version"] = FORMAT_VERSION
            f.attrs["temperature_unit"] = TEMPERATURE_UNIT
            f.attrs["created"] = time.time()
            f.attrs["geometry_hash"] = digest
            f.attrs["geometry_params"] = json.dumps(geometry_params or {})
            f.attrs["notes"] = json.dumps(notes or [])
            f.create_dataset("material_id", data=mesh.material_id, compression="gzip")
            f.create_dataset("T", data=mesh.T, compression="gzip")
            f.create_dataset("Q_source", data=mesh.Q_source, compression="gzip")
            f.create_dataset("grid", data=np.array(
                [mesh.Lx, mesh.Ly, mesh.Lz,
                 float(mesh.dx.min()), float(mesh.dx.max())], dtype=float))
            for axis, edges in zip("xyz", (mesh.edges_x, mesh.edges_y, mesh.edges_z),
                                   strict=True):
                f.create_dataset(f"edges_{axis}", data=edges, compression="gzip")
            f.create_dataset("face_bc", data=np.array(
                [f"{k}:{mesh.face_bc[k].kind.name}" for k in mesh.face_bc], dtype=h5py.string_dtype()))
        return target

    # ------------------------------------------------------------------ load
    def load_state(self, path: str) -> SimulationState:
        if not HAS_H5PY:
            raise StateError("h5py is required to load a state (pip install h5py)")
        if not os.path.isfile(path):
            raise StateError(f"state file not found: {path}")
        try:
            with h5py.File(path, "r") as f:
                version = int(f.attrs.get("version", 0))
                if version > FORMAT_VERSION:
                    raise StateError(
                        f"{path} was written by a newer version ({version} > {FORMAT_VERSION})")
                unit = str(f.attrs.get("temperature_unit", "degC"))
                grid = np.asarray(f["grid"][:], dtype=float) if "grid" in f else np.zeros(4)
                edges = {axis: np.asarray(f[f"edges_{axis}"][:], dtype=float)
                         for axis in "xyz" if f"edges_{axis}" in f}
                face_bc = {}
                if "face_bc" in f:
                    for raw in f["face_bc"][:]:
                        text = raw.decode() if isinstance(raw, bytes) else str(raw)
                        face, _, kind = text.partition(":")
                        face_bc[face] = kind
                state = SimulationState(
                    name=os.path.splitext(os.path.basename(path))[0],
                    created=float(f.attrs.get("created", 0.0)),
                    grid={"Lx": grid[0], "Ly": grid[1], "Lz": grid[2], "spacing": grid[3],
                          "cell_size_max": grid[4] if grid.size > 4 else grid[3],
                          "edges": edges},
                    geometry_hash=str(f.attrs.get("geometry_hash", "")),
                    geometry_params=json.loads(f.attrs.get("geometry_params", "{}")),
                    temperature_unit=unit,
                    version=version,
                    material_id=np.asarray(f["material_id"][:]),
                    T=np.asarray(f["T"][:], dtype=float),
                    Q_source=np.asarray(f["Q_source"][:]) if "Q_source" in f else None,
                    face_bc=face_bc,
                    notes=json.loads(f.attrs.get("notes", "[]")),
                )
        except OSError as exc:
            raise StateError(f"cannot read {path}: {exc}") from exc
        if state.temperature_unit != TEMPERATURE_UNIT:
            state.T = state.T + T0          # upgrade a degC file to the Kelvin contract
            state.notes.append(f"converted from {state.temperature_unit} to K")
            state.temperature_unit = TEMPERATURE_UNIT
        return state

    # -------------------------------------------------------------- checking
    def verify(self, mesh: Mesh3D, state: SimulationState,
               geometry_params: dict[str, float] | None = None) -> list[str]:
        """Problems that make ``state`` unusable for ``mesh`` ([] if compatible)."""
        problems = []
        if state.T is None:
            problems.append("state has no temperature field")
            return problems
        if state.T.shape != mesh.T.shape:
            problems.append(f"grid mismatch: state {state.T.shape} vs mesh {mesh.T.shape}")
        if state.geometry_hash and state.geometry_hash != geometry_hash(mesh, geometry_params):
            problems.append("geometry hash mismatch: the model changed since the state was saved")
        if state.material_id is not None and not np.array_equal(state.material_id, mesh.material_id):
            problems.append("material map differs from the current geometry")
        return problems

    def apply(self, mesh: Mesh3D, state: SimulationState,
              geometry_params: dict[str, float] | None = None) -> list[str]:
        """Restore the field into ``mesh``; raises :class:`StateError` if invalid."""
        problems = self.verify(mesh, state, geometry_params)
        if problems:
            raise StateError("; ".join(problems))
        mesh.T = np.array(state.T, dtype=float)
        if state.Q_source is not None and state.Q_source.shape == mesh.Q_source.shape:
            mesh.Q_source = np.array(state.Q_source, dtype=float)
        return list(state.notes)

    def list_saved(self) -> list[str]:
        if not os.path.isdir(self.directory):
            return []
        return sorted(os.path.join(self.directory, f) for f in os.listdir(self.directory)
                      if f.endswith(".h5"))


def face_bc_from_state(state: SimulationState) -> dict[str, FaceBC]:
    """Rebuild FaceBC objects from the stored face descriptors."""
    from ..core.mesh import BoundaryType

    out = {}
    for face, descriptor in state.face_bc.items():
        kind = descriptor.split(":", 1)[-1]
        try:
            out[face] = FaceBC(BoundaryType[kind])
        except KeyError:
            continue
    return out
