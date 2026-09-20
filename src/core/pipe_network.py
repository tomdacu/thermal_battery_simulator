"""Buried pipe networks of a cylindrical vessel: layouts, headers and voxelisation.

Polar Night Energy's published architecture (see ``docs/13_REDESIGN.md``) is a closed
air loop through pipes buried in the sand: the electric resistors heat the *air*, the
air heats the bed through the pipe walls and on discharge the same loop delivers the
heat to an exchanger, so the pipes are the only heat-transfer surface.  A silo holds a
bundle of vertical tubes; a vessel large enough to matter cannot be fed by one header,
so the bundle is organised as a **network**: risers in parallel between a bottom
*distributor* and a top *collector*, and two lateral ducts crossing the vessel wall.

This module extends :mod:`src.core.pipes` and keeps its two invariants:

* every pipe - riser, ring, header, jumper, duct - is a
  :class:`~src.core.pipes.PipeRun`, so its wetted area is the geometric ``pi d L`` of
  its centreline.  The area is **never** read off the voxel mask: the mask is a
  staircase and its surface is a mesh artefact, not a heat-transfer surface;
* the length rasterised into each cell sums to the centreline length to machine
  precision, so ``sum(area per cell) = pi d L_total`` for every run and for the whole
  network.  The collectors carry their area and their pressure loss like any other run.

Layouts (the plan inside the circle of radius ``radius - wall_clearance``):

``staggered``
    equilateral lattice, alternate rows shifted by half a pitch: the arrangement the
    bundle literature recommends;
``grid``
    square lattice on the same rows;
``rings``
    concentric rings - **every ring carries its own risers** and is linked to its
    neighbours by radial jumpers, so a ring with no riser or with no link is a
    configuration error (``PipeNetwork.validate``);
``radial``
    radial files of risers, spaced by the horizontal pitch along the radius and with
    the files pitched by the vertical pitch at the mean radius (the inner radius of the
    files is the pitch circle, where two neighbouring files are exactly one horizontal
    pitch apart, so no other riser of a file can sit closer than the pitch).

Collection modes (how the flow is distributed and collected):

``distributor_collector``
    the reference: cold in at the bottom, hot out at the top, and the collector
    discharges at the *same* end of the ladder the distributor is fed from (*direct
    return*: the branch closest to the inlet is also closest to the outlet, so its path
    is the shortest and the flow favours it);
``reverse_return``
    the collector discharges at the *far* end of the ladder (Tichelmann): every branch
    then travels the same total header length, which is what keeps the flow between
    the risers uniform;
``central_header``
    the two nozzles sit below and above the two header levels and the flow reaches
    them through vertical ducts on the axis; the headers are take-offs (*derivazioni*)
    that branch off those ducts, and the return trunk crosses the bundle;
``two_level_rings``
    the concentric ring headers alternate between two elevations one duct diameter
    apart, so adjacent rings clear each other and the droppers between them have a
    defined length; the chain is walked the balancing way (the two levels are a
    plumbing variant of the reverse return).

Design rules, with the sources collected in ``docs/15_PIPE_NETWORKS.md``:

* pitches as multiples of the outer diameter: ``2.0 d`` horizontal, ``2.5 d``
  vertical, ``sqrt(3) d`` for an equilateral lattice (:mod:`src.core.pipes`);
* header span (the distance between the two header elevations): below 1 m no care is
  needed, 1-3 m needs flow-distribution care, above 3 m the bundle must be split into
  parallel modules;
* a module keeps its bed below 4 m of height;
* the nozzles must leave through the *wall*: an outlet that would go through the roof
  is refused, because the roof carries the insulation and the cone, not a nozzle band.

Every length is metres and every elevation is measured from the domain floor, exactly
as in :mod:`src.core.geometry`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mesh import Mesh3D
from .pipes import (HEADER_LIMIT, HEADER_SAFE, PITCH_HORIZONTAL, PITCH_TRIANGULAR,
                    PITCH_VERTICAL, PipeRun, rasterize_pipe)

# ---------------------------------------------------------------------- vocabulary
LAYOUT_STAGGERED = "staggered"
LAYOUT_GRID = "grid"
LAYOUT_RINGS = "rings"
LAYOUT_RADIAL = "radial"
LAYOUTS = (LAYOUT_STAGGERED, LAYOUT_GRID, LAYOUT_RINGS, LAYOUT_RADIAL)

COLLECTION_DIRECT = "distributor_collector"
COLLECTION_REVERSE = "reverse_return"
COLLECTION_CENTRAL = "central_header"
COLLECTION_TWO_LEVEL = "two_level_rings"
COLLECTIONS = (COLLECTION_DIRECT, COLLECTION_REVERSE, COLLECTION_CENTRAL,
               COLLECTION_TWO_LEVEL)

#: the collections that run the return in the *same* order as the feed (Tichelmann):
#: every branch then travels the same header length.  The direct return walks it the
#: other way round, which is what makes the first branch the favoured one.
BALANCED_COLLECTIONS = (COLLECTION_REVERSE, COLLECTION_TWO_LEVEL)

SPLIT_EQUAL = "equal"
SPLIT_PATH = "path"
SPLITS = (SPLIT_EQUAL, SPLIT_PATH)

#: prefix of the non-blocking messages returned by ``validate()``
WARNING = "warning: "

#: how far the nozzle stub reaches outside the vessel wall [m]: the duct must be seen
#: to cross the wall in the voxel mask
WALL_STUB = 0.2

#: distance between the two ring levels of ``two_level_rings``, in duct diameters:
#: two crossing pipes need one diameter, so two leave one diameter of clearance
TWO_LEVEL_GAP = 2.0

#: a module keeps its bed below this height [m] (``docs/13_REDESIGN.md``)
MODULE_HEIGHT_LIMIT = 4.0


# ------------------------------------------------------------------- configuration
@dataclass
class PipeNetworkConfig:
    """Geometry and plumbing of a buried pipe network; the grid is not involved.

    Elevations are absolute in the domain.  The *active band* is
    ``base_z + band_bottom .. base_z + band_top`` and it is what the sand uses: the
    distributor sits at its bottom end and the collector at its top end, so the risers
    span exactly the band.

    Everything else is derived unless given: the pitches follow the published practice
    for a bundle in a granular bed, the clearance keeps the outer risers off the wall,
    the two levels of the ring collection are ``2 d_duct`` apart and the nozzle
    elevations sit below the distributor (cold side) and at the collector (hot side).
    """

    radius: float = 2.0                     # [m] inner radius of the vessel
    height: float = 7.0                     # [m] wall height of the vessel
    base_z: float = 0.0                     # [m] vessel floor in the domain
    band_bottom: float = 0.3                # [m] active band, from the floor
    band_top: float | None = None           # [m]; None = height - band_bottom
    diameter: float = 0.05                  # [m] outer diameter of the tubes
    horizontal_pitch: float | None = None   # [m] spacing inside a row / ring / file
    vertical_pitch: float | None = None     # [m] spacing between rows / rings / files
    layout: str = LAYOUT_STAGGERED
    collection: str = COLLECTION_DIRECT
    n_rings: int | None = None              # None = as many as the pitch fits
    n_files: int | None = None              # None = as many as the pitch fits
    duct_diameter: float | None = None      # [m]; None = the tube diameter
    wall_clearance: float | None = None     # [m]; None = two tube diameters
    azimuth_in: float = 0.0                 # [deg] azimuth of the inlet nozzle
    azimuth_out: float | None = None        # [deg]; None = mode dependent
    elevation_in: float | None = None       # [m]; None = below the distributor
    elevation_out: float | None = None      # [m]; None = at the collector
    split_mode: str = SPLIT_EQUAL

    # ------------------------------------------------------------- derived sizes
    @property
    def pitch_h(self) -> float:
        """Spacing between neighbouring risers inside a row, a ring or a file [m]."""
        if self.horizontal_pitch is not None:
            return float(self.horizontal_pitch)
        factor = PITCH_TRIANGULAR if self.layout == LAYOUT_STAGGERED else PITCH_HORIZONTAL
        return float(factor * self.diameter)

    @property
    def pitch_v(self) -> float:
        """Spacing between rows, rings or files [m]."""
        if self.vertical_pitch is not None:
            return float(self.vertical_pitch)
        factor = PITCH_TRIANGULAR if self.layout == LAYOUT_STAGGERED else PITCH_VERTICAL
        return float(factor * self.diameter)

    @property
    def clearance(self) -> float:
        """Distance kept between the outermost riser and the vessel wall [m]."""
        if self.wall_clearance is not None:
            return float(self.wall_clearance)
        return 2.0 * self.diameter

    @property
    def inner_radius(self) -> float:
        """Radius available to the bundle [m]."""
        return self.radius - self.clearance

    @property
    def duct_d(self) -> float:
        """Outer diameter of the headers, the ducts and the jumpers [m]."""
        if self.duct_diameter is None:
            return self.diameter
        return float(self.duct_diameter)

    @property
    def z_bottom(self) -> float:
        """Elevation of the distributor (bottom of the active band) [m]."""
        return self.base_z + self.band_bottom

    @property
    def z_top(self) -> float:
        """Elevation of the collector (top of the active band) [m]."""
        top = self.height - self.band_bottom if self.band_top is None else self.band_top
        return self.base_z + top

    @property
    def roof_z(self) -> float:
        """Elevation of the vessel roof [m]: the wall ends here."""
        return self.base_z + self.height

    @property
    def level_gap(self) -> float:
        """Elevation difference between the two ring levels [m]."""
        return TWO_LEVEL_GAP * self.duct_d

    @property
    def inlet_elevation(self) -> float:
        """Elevation of the inlet nozzle [m]: below the distributor (cold side)."""
        if self.elevation_in is not None:
            return float(self.elevation_in)
        return self.base_z + 0.5 * (self.z_bottom - self.base_z)

    @property
    def outlet_elevation(self) -> float:
        """Elevation of the outlet nozzle [m]: at the collector (hot side)."""
        if self.elevation_out is not None:
            return float(self.elevation_out)
        if self.collection == COLLECTION_CENTRAL:
            return 0.5 * (self.z_top + self.level_gap + self.roof_z)
        if self.collection == COLLECTION_TWO_LEVEL:
            return self.z_top + self.level_gap
        return self.z_top

    @property
    def outlet_azimuth(self) -> float:
        """Azimuth of the outlet nozzle [deg]: opposite the inlet where it helps."""
        if self.azimuth_out is not None:
            return float(self.azimuth_out)
        if self.collection == COLLECTION_DIRECT:
            return self.azimuth_in
        return self.azimuth_in + 180.0

    # ---------------------------------------------------------------- validation
    def validate(self) -> list[str]:
        """Actionable problems of the configuration ([] when it is usable).

        Blocking problems (no prefix) make the network unrepresentable; the messages
        prefixed with :data:`WARNING` describe designs that build but break one of the
        rules of the module docstring.
        """
        problems: list[str] = []
        if self.radius <= 0:
            problems.append(f"the vessel radius must be > 0, got {self.radius:.3f} m")
        if self.height <= 0:
            problems.append(f"the vessel height must be > 0, got {self.height:.3f} m")
        if self.diameter <= 0:
            problems.append(
                f"the tube outer diameter must be > 0, got {self.diameter:.4f} m")
        if self.duct_diameter is not None and self.duct_diameter <= 0:
            problems.append(
                f"the duct diameter must be > 0, got {self.duct_diameter:.4f} m")
        if self.layout not in LAYOUTS:
            problems.append(f"unknown layout {self.layout!r}: expected one of {LAYOUTS}")
        if self.collection not in COLLECTIONS:
            problems.append(
                f"unknown collection {self.collection!r}: expected one of {COLLECTIONS}")
        if self.split_mode not in SPLITS:
            problems.append(
                f"unknown split mode {self.split_mode!r}: expected one of {SPLITS}")
        if self.collection == COLLECTION_TWO_LEVEL and self.layout != LAYOUT_RINGS:
            problems.append(
                f"the two-level collection stacks concentric rings, so it needs "
                f"layout={LAYOUT_RINGS!r}, not {self.layout!r}")
        if self.pitch_h < self.diameter:
            problems.append(
                f"the horizontal pitch ({self.pitch_h:.4f} m) is smaller than the tube "
                f"diameter ({self.diameter:.4f} m): the tubes would touch")
        if self.pitch_v < self.diameter:
            problems.append(
                f"the vertical pitch ({self.pitch_v:.4f} m) is smaller than the tube "
                f"diameter ({self.diameter:.4f} m): the tubes would touch")
        if self.band_bottom < 0:
            problems.append(f"band_bottom must be >= 0, got {self.band_bottom:.3f} m")
        if self.z_top <= self.z_bottom:
            problems.append(
                f"the active band ({self.z_bottom:.2f}..{self.z_top:.2f} m) has no "
                f"height: raise band_top above band_bottom")
        if self.z_top + 0.5 * self.diameter > self.roof_z:
            problems.append(
                f"the collector at {self.z_top:.2f} m would sit above the vessel wall "
                f"({self.roof_z:.2f} m): lower band_top below "
                f"{self.height - 0.5 * self.diameter:.2f} m or raise the vessel")
        if self.collection == COLLECTION_TWO_LEVEL and (
                self.z_top + self.level_gap + 0.5 * self.diameter > self.roof_z):
            problems.append(
                f"the raised ring level reaches {self.z_top + self.level_gap:.2f} m, "
                f"above the vessel wall ({self.roof_z:.2f} m): lower band_top or use a "
                f"single level")
        if self.clearance < 0:
            problems.append(
                f"the wall clearance must be >= 0, got {self.clearance:.3f} m: the "
                f"risers would sit outside the wall")
        elif self.inner_radius <= 0:
            problems.append(
                f"the wall clearance ({self.clearance:.3f} m) leaves no bundle inside "
                f"a vessel of radius {self.radius:.3f} m")
        else:
            plan = _plan(self)
            if not plan.points:
                problems.append(
                    f"the {self.layout} layout places no riser inside the vessel: "
                    f"reduce the pitches ({self.pitch_h * 1000:.0f}/"
                    f"{self.pitch_v * 1000:.0f} mm) or enlarge the vessel")
            elif plan.family == "rings":
                fits = int(np.floor(self.inner_radius / self.pitch_v + 1e-12))
                if self.n_rings is not None and self.n_rings > fits:
                    problems.append(
                        f"only {fits} rings fit inside a vessel of radius "
                        f"{self.radius:.2f} m at a ring pitch of "
                        f"{self.pitch_v * 1000:.0f} mm, but n_rings is "
                        f"{self.n_rings}: reduce n_rings to {fits} or the vertical "
                        f"pitch")
        if self.inlet_elevation - 0.5 * self.duct_d < self.base_z:
            problems.append(
                f"the inlet duct at {self.inlet_elevation:.2f} m does not fit inside "
                f"the vessel (floor at {self.base_z:.2f} m): raise elevation_in")
        if self.outlet_elevation + 0.5 * self.duct_d > self.roof_z:
            problems.append(
                f"the outlet duct at {self.outlet_elevation:.2f} m would leave through "
                f"the roof (the wall ends at {self.roof_z:.2f} m): lower elevation_out "
                f"below {self.roof_z - 0.5 * self.duct_d:.2f} m or raise the vessel")
        if self.outlet_elevation <= self.inlet_elevation:
            problems.append(
                f"the outlet ({self.outlet_elevation:.2f} m) must sit above the inlet "
                f"({self.inlet_elevation:.2f} m): the network is fed cold at the "
                f"bottom and collects hot at the top")
        if self.inlet_elevation > self.z_bottom + 1e-12:
            problems.append(
                f"{WARNING}the inlet duct at {self.inlet_elevation:.2f} m sits above "
                f"the distributor ({self.z_bottom:.2f} m): the duct then crosses the "
                f"active band and short-circuits part of the sand")
        return problems


# ---------------------------------------------------------------------------- plan
@dataclass
class _Plan:
    """Riser plan of a configuration: positions, groups and the arcs a path needs."""

    points: list[tuple[float, float]]     # riser positions, branch order, vessel-local
    group: list[int]                      # row / ring / file index of each riser
    forward: list[float]                  # arc inside the group from its entry vertex
    perimeter: list[float]                # full perimeter of the group (0 for a lattice)
    radii: list[float]                    # ring radius [m]; [] for the ladder layouts

    @property
    def family(self) -> str:
        """``"rings"`` when the groups are rings, ``"ladder"`` otherwise."""
        return "rings" if self.radii else "ladder"

    @property
    def n_groups(self) -> int:
        """Number of header groups: rings for the ring family, taps otherwise."""
        return len(self.radii) if self.radii else len(self.points)


def _serpentine(groups: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    """Flatten groups so a ladder run visits them all without a wasted jump.

    The header walks along a row (or a radial file), steps over to the next one at the
    end and comes back the other way: the shortest single pipe that touches every riser
    once.
    """
    points: list[tuple[float, float]] = []
    for index, group in enumerate(groups):
        points.extend(group[::-1] if index % 2 else group)
    return points


def _lattice_groups(config: PipeNetworkConfig) -> list[list[tuple[float, float]]]:
    """Riser positions of the lattice layouts, one list per row (or per radial file)."""
    p_h, p_v, r_eff = config.pitch_h, config.pitch_v, config.inner_radius
    if config.layout == LAYOUT_RADIAL:
        n_files = int(config.n_files) if config.n_files else \
            max(3, int(round(np.pi * r_eff / p_v)))
        n_files = max(n_files, 3)
        # the first file points at the inlet nozzle, so the feed duct is short
        start = float(np.deg2rad(config.azimuth_in))
        # the pitch circle: at this radius two neighbouring files are exactly one
        # horizontal pitch apart, and every larger radius keeps them further apart
        radius = p_h / (2.0 * np.sin(np.pi / n_files))
        radii = []
        while radius <= r_eff + 1e-12:
            radii.append(radius)
            radius += p_h
        groups = []
        for file_index in range(n_files):
            angle = start + 2.0 * np.pi * file_index / n_files
            group = [(radius * np.cos(angle), radius * np.sin(angle))
                     for radius in radii]
            if group:
                groups.append(group)
        return groups

    rows = int(np.floor(2.0 * r_eff / p_v + 1e-12)) + 1
    y0 = -0.5 * (rows - 1) * p_v
    groups = []
    for row in range(rows):
        y = y0 + row * p_v
        offset = 0.5 * p_h if (config.layout == LAYOUT_STAGGERED and row % 2) else 0.0
        half = float(np.sqrt(max(r_eff ** 2 - y ** 2, 0.0)))
        # one lattice origin for the whole vessel: the rows of a grid share their
        # columns and the alternate rows of a staggered bundle are half a pitch across
        first = int(np.ceil((-half - offset) / p_h - 1e-12))
        last = int(np.floor((half - offset) / p_h + 1e-12))
        group = [(offset + k * p_h, y) for k in range(first, last + 1)]
        group = [(x, y) for x, y in group if x * x + y * y <= r_eff ** 2 + 1e-12]
        if group:
            groups.append(group)
    return groups


def _plan(config: PipeNetworkConfig) -> _Plan:
    """Riser plan of ``config``: pure, so the same configuration always builds the same.

    ``points`` is the *branch order*: the order in which the flow visits the risers,
    which is the serpentine walk for the ladder layouts and, for the rings, ring by
    ring with each ring starting at its own entry vertex.
    """
    p_h, p_v = config.pitch_h, config.pitch_v
    r_eff = config.inner_radius
    if r_eff <= 0 or p_h <= 0 or p_v <= 0:
        return _Plan([], [], [], [], [])

    if config.layout == LAYOUT_RINGS:
        theta_f = float(np.deg2rad(config.azimuth_in))
        fits = int(np.floor(r_eff / p_v + 1e-12))
        count = fits if config.n_rings is None else min(int(config.n_rings), fits)
        points, group, forward, perimeter, radii = [], [], [], [], []
        for index in range(max(count, 0)):
            radius = (index + 1) * p_v
            # an even number of taps makes the ring symmetric about the diameter that
            # joins its feed point to its exit point, so every tap travels half the
            # ring whichever way round it is fed; the floor keeps the arc spacing
            # between neighbouring taps at or above the horizontal pitch
            taps = 2 * max(int(np.floor(np.pi * radius / p_h + 1e-12)), 1)
            spacing = 2.0 * np.pi * radius / taps
            # the entry azimuths alternate so that consecutive rings, which the chain
            # visits one after the other, are joined by a short radial hop
            start = theta_f + np.pi * ((count - 1 - index) % 2)
            radii.append(float(radius))
            perimeter.append(2.0 * np.pi * radius)
            for tap in range(taps):
                angle = start + tap * spacing / radius
                points.append((radius * np.cos(angle), radius * np.sin(angle)))
                group.append(index)
                forward.append(tap * spacing)
        return _Plan(points, group, forward, perimeter, radii)

    points = _serpentine(_lattice_groups(config))
    count = len(points)
    return _Plan(points, list(range(count)), [0.0] * count, [0.0] * count, [])


def _levels(config: PipeNetworkConfig, n_groups: int) -> list[float]:
    """Elevation added to each group's headers: 0 except on the two-level collection."""
    if config.collection != COLLECTION_TWO_LEVEL:
        return [0.0] * n_groups
    return [config.level_gap * (index % 2) for index in range(n_groups)]


# --------------------------------------------------------------- branch and network
@dataclass(frozen=True)
class Branch:
    """One parallel branch: a riser and the header arcs that feed and return it."""

    index: int
    riser: PipeRun
    distributor: PipeRun
    collector: PipeRun
    feed_length: float          # [m] from the inlet nozzle to the foot of the riser
    return_length: float        # [m] from the head of the riser to the outlet nozzle

    @property
    def path_length(self) -> float:
        """Centreline length of the whole branch [m]."""
        return self.feed_length + self.riser.total_length + self.return_length


@dataclass
class PipeNetwork:
    """A buried pipe network: risers, headers, ducts and where the flow goes.

    ``risers``, ``distributors``, ``collectors``, ``connectors``, ``inlet`` and
    ``outlet`` are all :class:`~src.core.pipes.PipeRun`: the wetted area, the length
    and the pressure loss of every pipe of the design are carried by the run itself,
    so nothing is lost between the geometry and the fluid solve.  The thermal march of
    the gas uses the risers with the branch split
    (``FluidLoop(runs=network.risers, split=network.split())``); the headers and the
    ducts stay available for the hydraulics and for the voxelisation.
    """

    config: PipeNetworkConfig
    center: tuple[float, float]
    risers: list[PipeRun]
    distributors: list[PipeRun]
    collectors: list[PipeRun]
    connectors: list[PipeRun]
    inlet: PipeRun
    outlet: PipeRun
    branches: list[Branch] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- geometry
    @property
    def runs(self) -> list[PipeRun]:
        """Every run of the network, in a fixed order."""
        return [self.inlet, *self.distributors, *self.risers, *self.collectors,
                *self.connectors, self.outlet]

    @property
    def headers(self) -> list[PipeRun]:
        """Distributor and collector runs together: the collectors of the design."""
        return [*self.distributors, *self.collectors]

    @property
    def n_risers(self) -> int:
        """Number of risers (and branches) of the network."""
        return len(self.risers)

    @property
    def total_length(self) -> float:
        """Centreline length of every pipe of the network [m]."""
        return float(sum(run.total_length for run in self.runs))

    @property
    def total_area(self) -> float:
        """Geometric wetted area of the network [m^2]: ``pi d L``, never the mask."""
        return float(sum(run.total_area for run in self.runs))

    @property
    def riser_length(self) -> float:
        """Centreline length of the risers alone [m]."""
        return float(sum(run.total_length for run in self.risers))

    @property
    def riser_area(self) -> float:
        """Wetted area of the risers alone [m^2]: the surface inside the active band."""
        return float(sum(run.total_area for run in self.risers))

    @property
    def bed_volume(self) -> float:
        """Volume of the active band [m^3]: what the surface is sized against."""
        band = max(self.config.z_top - self.config.z_bottom, 0.0)
        return float(np.pi * self.config.radius ** 2 * band)

    @property
    def specific_area(self) -> float:
        """Wetted area per unit active volume [m^2/m^3] - the sizing number."""
        return self.total_area / self.bed_volume if self.bed_volume > 0 else 0.0

    def plan_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """Plan coordinates of the risers, relative to the vessel axis [m]."""
        if not self.risers:
            return np.empty(0), np.empty(0)
        x = np.asarray([run.points[0, 0] for run in self.risers]) - self.center[0]
        y = np.asarray([run.points[0, 1] for run in self.risers]) - self.center[1]
        return x, y

    @property
    def bundle_width(self) -> float:
        """Plan extent of the bundle along x [m] (centrelines)."""
        x, _ = self.plan_xy()
        return float(np.max(x) - np.min(x)) if x.size else 0.0

    @property
    def bundle_height(self) -> float:
        """Plan extent of the bundle along y [m] (centrelines)."""
        _, y = self.plan_xy()
        return float(np.max(y) - np.min(y)) if y.size else 0.0

    @property
    def bundle_radius(self) -> float:
        """Radius of the outermost riser [m]."""
        x, y = self.plan_xy()
        return float(np.max(np.hypot(x, y))) if x.size else 0.0

    @property
    def header_span(self) -> float:
        """Vertical distance between the distributor and the collector [m]."""
        if not self.distributors or not self.collectors:
            return 0.0
        z_bottom = min(float(np.min(run.points[:, 2])) for run in self.distributors)
        z_top = max(float(np.max(run.points[:, 2])) for run in self.collectors)
        return abs(z_top - z_bottom)

    def header_rule(self) -> str:
        """Verdict of the 3 m header rule on the span between the two headers."""
        span = self.header_span
        if span > HEADER_LIMIT:
            return f"split into parallel modules above {HEADER_LIMIT:.0f} m"
        if span > HEADER_SAFE:
            return (f"check the flow distribution between {HEADER_SAFE:.0f} and "
                    f"{HEADER_LIMIT:.0f} m")
        return f"no care needed below {HEADER_SAFE:.0f} m"

    # -------------------------------------------------------------------- flow
    def paths(self) -> np.ndarray:
        """Path length of every branch [m], in branch order."""
        return np.asarray([branch.path_length for branch in self.branches], dtype=float)

    def path_spread(self) -> float:
        """``max(path) / min(path)`` over the branches [-]: 1.0 is a balanced network."""
        paths = self.paths()
        if paths.size == 0 or float(np.min(paths)) <= 0.0:
            return 1.0
        return float(np.max(paths) / np.min(paths))

    def split(self) -> np.ndarray:
        """Share of the total mass flow per branch [-]: sums to 1 by construction.

        ``equal`` is the design target (every riser sees the same flow); ``path``
        follows the branch path, the coarse distribution rule a header is sized
        against, in which the branch that travels furthest carries proportionally
        more.  The result is what ``FluidLoop(split=...)`` wants.
        """
        count = len(self.branches)
        if count == 0:
            return np.empty(0)
        if self.config.split_mode == SPLIT_PATH:
            weights = self.paths()
            total = float(np.sum(weights))
            if total > 0.0:
                return weights / total
        return np.full(count, 1.0 / count)

    # ------------------------------------------------------------------ output
    def summarize(self) -> dict:
        """The numbers of :meth:`summary` as a dictionary (GUI and reports)."""
        split = self.split()
        paths = self.paths()
        return {
            "n_risers": self.n_risers,
            "n_headers": len(self.headers),
            "total_length": self.total_length,
            "riser_length": self.riser_length,
            "total_area": self.total_area,
            "riser_area": self.riser_area,
            "bed_volume": self.bed_volume,
            "specific_area": self.specific_area,
            "bundle_width": self.bundle_width,
            "bundle_height": self.bundle_height,
            "bundle_radius": self.bundle_radius,
            "header_span": self.header_span,
            "header_rule": self.header_rule(),
            "path_min": float(np.min(paths)) if paths.size else 0.0,
            "path_max": float(np.max(paths)) if paths.size else 0.0,
            "path_spread": self.path_spread(),
            "split_min": float(np.min(split)) if split.size else 0.0,
            "split_max": float(np.max(split)) if split.size else 0.0,
        }

    def summary(self) -> str:
        """Design summary: tubes, pitches, bundle, lengths, areas, headers, split."""
        config = self.config
        data = self.summarize()
        lines = [
            f"network: {config.layout} layout, {config.collection}, "
            f"{self.n_risers} risers in {len(self.distributors)} distributor and "
            f"{len(self.collectors)} collector runs",
            f"pitch: {config.pitch_h * 1000:.0f} mm horizontal, "
            f"{config.pitch_v * 1000:.0f} mm vertical "
            f"({config.pitch_h / config.diameter:.2f} d / "
            f"{config.pitch_v / config.diameter:.2f} d), d = "
            f"{config.diameter * 1000:.1f} mm",
            f"bundle: {data['bundle_width']:.2f} x {data['bundle_height']:.2f} m plan "
            f"inside r = {data['bundle_radius']:.2f} m of a {config.radius:.2f} m "
            f"vessel, risers {self.riser_length / max(self.n_risers, 1):.2f} m long",
            f"tube surface: risers {self.riser_area:.1f} m2 over "
            f"{self.riser_length:.0f} m, whole network {self.total_area:.1f} m2 over "
            f"{self.total_length:.0f} m",
            f"specific area: {self.specific_area:.2f} m2/m3 of active band "
            f"({self.bed_volume:.1f} m3)",
            f"headers: distributor {config.z_bottom:.2f} m, collector "
            f"{config.z_top:.2f} m, span {data['header_span']:.2f} m - the 3 m rule: "
            f"{data['header_rule']}",
            f"flow: {len(self.branches)} branches, split {config.split_mode} "
            f"({100 * data['split_min']:.4f}%-{100 * data['split_max']:.4f}%), path "
            f"{data['path_min']:.2f}-{data['path_max']:.2f} m "
            f"(spread {data['path_spread']:.3f}x)",
        ]
        lines.extend(f"note: {note}" for note in self.notes)
        return "\n".join(lines)

    # ------------------------------------------------------------- validation
    def validate(self) -> list[str]:
        """Problems with the *built* network ([] when it is sound).

        The checks the geometry has to pass after the build: every riser runs from a
        distributor to a collector, every header - every ring included - carries at
        least one riser and is reached from the inlet duct, nothing pokes out of the
        vessel.  A ring with no riser or with no link to the duct is an error, not a
        warning.
        """
        problems: list[str] = []
        if not self.risers:
            problems.append("the network has no riser: the layout places no tube "
                            "inside the vessel")
            return problems

        tapped: dict[int, int] = {}
        for branch in self.branches:
            tapped[id(branch.distributor)] = tapped.get(id(branch.distributor), 0) + 1
            tapped[id(branch.collector)] = tapped.get(id(branch.collector), 0) + 1
        for run in self.headers:
            if tapped.get(id(run), 0) == 0:
                problems.append(
                    f"the header {run.name!r} carries no riser: every ring and every "
                    f"ladder must be fed by and must collect at least one tube")

        for index, branch in enumerate(self.branches):
            if not _touches_point(branch.distributor, branch.riser.points[0]):
                problems.append(
                    f"riser {index} does not start on the distributor "
                    f"{branch.distributor.name!r}: every riser must run from the "
                    f"bottom header to the top one")
            if not _touches_point(branch.collector, branch.riser.points[-1]):
                problems.append(
                    f"riser {index} does not reach the collector "
                    f"{branch.collector.name!r}: every riser must run from the bottom "
                    f"header to the top one")

        plan_x, plan_y = self.plan_xy()
        for index, radius in enumerate(np.hypot(plan_x, plan_y)):
            if radius > self.config.radius + 1e-9:
                problems.append(
                    f"riser {index} sits at r = {radius:.3f} m, outside the vessel "
                    f"wall ({self.config.radius:.3f} m)")

        roof = self.config.roof_z
        for run in self.headers:
            high = float(np.max(run.points[:, 2])) + 0.5 * run.diameter
            if high > roof + 1e-9:
                problems.append(
                    f"the header {run.name!r} reaches z = {high:.3f} m, above the "
                    f"vessel roof ({roof:.2f} m)")

        for run in _unconnected(self.runs, self.inlet):
            problems.append(
                f"the run {run.name!r} is not connected to the inlet duct: the "
                f"headers of a network must be linked to each other and to the duct")
        return problems

    # ----------------------------------------------------------- voxelisation
    def voxelize(self, mesh: Mesh3D | None = None) -> tuple[np.ndarray, np.ndarray,
                                                            np.ndarray]:
        """Voxelise the network: flat cell indices, length and wetted area per cell.

        The same interface as :func:`src.core.pipes.rasterize_pipe` - flat cell indices
        plus the length inside each cell - with the wetted area added, summed over
        every run of the network.  ``mesh`` re-rasterises the centrelines onto another
        grid; without it the runs keep the cells of the mesh the network was built on.

        The invariant is the one of :mod:`src.core.pipes`: ``sum(area) = pi d L`` to
        machine precision, per run and for the network as a whole.
        """
        lengths: dict[int, float] = {}
        areas: dict[int, float] = {}
        for run in self.runs:
            placed = (run if mesh is None else
                      rasterize_pipe(mesh, run.points, run.diameter, name=run.name))
            perimeter = placed.perimeter
            for cell, piece in zip(placed.cells.tolist(), placed.length.tolist(),
                                   strict=True):
                lengths[cell] = lengths.get(cell, 0.0) + piece
                areas[cell] = areas.get(cell, 0.0) + perimeter * piece
        order = sorted(lengths)
        return (np.asarray(order, dtype=np.int64),
                np.asarray([lengths[cell] for cell in order], dtype=float),
                np.asarray([areas[cell] for cell in order], dtype=float))


# ----------------------------------------------------------------- connectivity
def _point_on_polyline(point: np.ndarray, points: np.ndarray, tol: float) -> bool:
    """Whether ``point`` lies within ``tol`` of the polyline ``points``."""
    start = points[:-1]
    step = points[1:] - start
    length = np.einsum("ij,ij->i", step, step)
    safe = np.where(length > 0.0, length, 1.0)
    fraction = np.clip(-np.einsum("ij,ij->i", start - point, step) / safe, 0.0, 1.0)
    projected = start + fraction[:, None] * step
    return bool(np.min(np.linalg.norm(projected - point, axis=1)) <= tol)


def _touches_point(run: PipeRun, point: np.ndarray) -> bool:
    """Whether a point sits on the centreline of ``run`` (within half its diameter)."""
    return _point_on_polyline(np.asarray(point, dtype=float), run.points,
                              0.5 * run.diameter + 1e-9)


def _touches(first: PipeRun, second: PipeRun) -> bool:
    """Whether two runs touch: their centrelines come within the sum of their radii."""
    tol = 0.5 * (first.diameter + second.diameter) + 1e-9
    return (_point_on_polyline(first.points[0], second.points, tol)
            or _point_on_polyline(first.points[-1], second.points, tol)
            or _point_on_polyline(second.points[0], first.points, tol)
            or _point_on_polyline(second.points[-1], first.points, tol))


def _unconnected(runs: list[PipeRun], start: PipeRun) -> list[PipeRun]:
    """Runs the flow cannot reach from ``start`` through touching runs."""
    bounds = {id(run): (np.min(run.points, axis=0), np.max(run.points, axis=0),
                        0.5 * run.diameter) for run in runs}
    seen = {id(start)}
    queue = [start]
    while queue:
        current = queue.pop()
        low, high, radius = bounds[id(current)]
        for other in runs:
            if id(other) in seen:
                continue
            other_low, other_high, other_radius = bounds[id(other)]
            tol = radius + other_radius + 1e-9
            if not np.all(other_low - tol <= high) or not np.all(low - tol <= other_high):
                continue                      # bounding boxes apart: cannot touch
            if _touches(current, other):
                seen.add(id(other))
                queue.append(other)
    return [run for run in runs if id(run) not in seen]


# ---------------------------------------------------------------------- builder
def _wall_point(config: PipeNetworkConfig, center: tuple[float, float],
                azimuth_deg: float, elevation: float,
                outside: float = 0.0) -> tuple[float, float, float]:
    """A point on the vessel wall at ``azimuth_deg``, ``outside`` metres out of it."""
    theta = float(np.deg2rad(azimuth_deg))
    radius = config.radius + outside
    return (center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta),
            elevation)


def build_pipe_network(mesh: Mesh3D, config: PipeNetworkConfig,
                       center: tuple[float, float] | None = None) -> PipeNetwork:
    """Rasterise the configured network on ``mesh``; the centre defaults to the box.

    Raises :class:`ValueError` when the configuration is not usable, so a network that
    breaks the design rules of the module docstring is never silently built: the
    messages of :meth:`PipeNetworkConfig.validate` name the parameter to change.
    """
    errors = [problem for problem in config.validate()
              if not problem.startswith(WARNING)]
    if errors:
        raise ValueError("the pipe network configuration is not usable: "
                         + "; ".join(errors))

    cx, cy = ((0.5 * mesh.Lx, 0.5 * mesh.Ly) if center is None
              else (float(center[0]), float(center[1])))
    diameter, duct = config.diameter, config.duct_d
    z_bottom, z_top = config.z_bottom, config.z_top
    central = config.collection == COLLECTION_CENTRAL
    balanced = config.collection in BALANCED_COLLECTIONS
    plan = _plan(config)
    if not plan.points:
        raise ValueError("the layout places no riser inside the vessel: reduce the "
                         "pitches or enlarge the vessel")
    rings = plan.family == "rings"
    levels = _levels(config, plan.n_groups)

    # ------------------------------------------------------------------- risers
    risers: list[PipeRun] = []
    for index, (x, y) in enumerate(plan.points):
        level = levels[plan.group[index]]
        risers.append(rasterize_pipe(mesh, [(cx + x, cy + y, z_bottom + level),
                                            (cx + x, cy + y, z_top + level)],
                                     diameter, name=f"riser_{index}"))

    def local(point_index: int, elevation: float) -> tuple[float, float, float]:
        x, y = plan.points[point_index]
        return (cx + x, cy + y, elevation)

    # ------------------------------------------------------------------ headers
    distributors: list[PipeRun] = []
    collectors: list[PipeRun] = []
    connectors: list[PipeRun] = []
    entry: dict[tuple[str, int], np.ndarray] = {}
    leaving: dict[tuple[str, int], np.ndarray] = {}

    if rings:
        for group in range(plan.n_groups):
            taps = [index for index in range(len(plan.points))
                    if plan.group[index] == group]
            for side, runs, z in (("bottom", distributors, z_bottom),
                                  ("top", collectors, z_top)):
                points = [local(index, z + levels[group]) for index in taps]
                points.append(points[0])                  # close the ring main
                runs.append(rasterize_pipe(mesh, points, duct,
                                           name=f"{side}_ring_{group}"))
                entry[(side, group)] = np.asarray(points[0], dtype=float)
                # an even tap count puts the exit on a tap, diametrically opposite
                leaving[(side, group)] = np.asarray(points[len(taps) // 2], dtype=float)
        # the chain runs from the outermost ring inwards, where the nozzles are
        dist_order = list(range(plan.n_groups - 1, -1, -1))
    else:
        bottom = [local(index, z_bottom) for index in range(len(plan.points))]
        top = [local(index, z_top) for index in range(len(plan.points))]
        distributors.append(rasterize_pipe(mesh, bottom, duct, name="distributor"))
        collectors.append(rasterize_pipe(mesh, top, duct, name="collector"))
        for index in range(len(plan.points)):
            entry[("bottom", index)] = np.asarray(bottom[index], dtype=float)
            entry[("top", index)] = np.asarray(top[index], dtype=float)
            leaving[("bottom", index)] = entry[("bottom", index)]
            leaving[("top", index)] = entry[("top", index)]
        dist_order = list(range(len(plan.points)))

    half = ([0.5 * plan.perimeter[group] for group in range(plan.n_groups)] if rings
            else [0.0] * len(plan.points))

    def chain(order: list[int], side: str, name: str) -> tuple[dict[int, float], float]:
        """Cumulative arc at each group's entry along ``order``, and the total.

        Consecutive groups are joined by a run of their own only for the ring family,
        where the header of a group is a closed ring; the ladder header is one run and
        its segments are the steps themselves.
        """
        offsets: dict[int, float] = {}
        position = 0.0
        for step, group in enumerate(order):
            offsets[group] = position
            position += half[group]
            if step + 1 == len(order):
                continue
            following = order[step + 1]
            jumper = float(np.linalg.norm(leaving[(side, group)]
                                          - entry[(side, following)]))
            position += jumper
            if rings:
                connectors.append(rasterize_pipe(
                    mesh, [leaving[(side, group)], entry[(side, following)]], duct,
                    name=f"{name}_jumper_{group}_{following}"))
        return offsets, position

    # a balancing collection walks the collector in the same order as the distributor,
    # so the branch fed first is the branch returned last and every path adds up the same
    coll_order = dist_order if balanced else list(reversed(dist_order))
    off_dist, _ = chain(dist_order, "bottom", "distributor")
    off_coll, total_coll = chain(coll_order, "top", "collector")

    # -------------------------------------------------------------------- ducts
    feed_vertex = entry[("bottom", dist_order[0])]
    discharge_vertex = leaving[("top", coll_order[-1])]
    inlet_points: list[tuple[float, float, float]] = [
        _wall_point(config, (cx, cy), config.azimuth_in, config.inlet_elevation,
                    WALL_STUB),
        _wall_point(config, (cx, cy), config.azimuth_in, config.inlet_elevation),
    ]
    if central:
        inlet_points.append((cx, cy, config.inlet_elevation))
        inlet_points.append((cx, cy, float(feed_vertex[2])))
    inlet_points.append((float(feed_vertex[0]), float(feed_vertex[1]),
                         float(feed_vertex[2])))
    inlet = rasterize_pipe(mesh, inlet_points, duct, name="inlet_duct")

    outlet_points: list[tuple[float, float, float]] = [
        (float(discharge_vertex[0]), float(discharge_vertex[1]),
         float(discharge_vertex[2])),
    ]
    if central:
        outlet_points.append((cx, cy, float(discharge_vertex[2])))
        outlet_points.append((cx, cy, config.outlet_elevation))
    outlet_points.append(_wall_point(config, (cx, cy), config.outlet_azimuth,
                                     config.outlet_elevation))
    outlet_points.append(_wall_point(config, (cx, cy), config.outlet_azimuth,
                                     config.outlet_elevation, WALL_STUB))
    outlet = rasterize_pipe(mesh, outlet_points, duct, name="outlet_duct")

    # ----------------------------------------------------------------- branches
    link_in, link_out = inlet.total_length, outlet.total_length
    branches: list[Branch] = []
    for index in range(len(plan.points)):
        group = plan.group[index]
        if rings:
            # a ring fed at one point divides its flow both ways round: the branch
            # takes the shorter arc in and the rest of its half out
            turn = min(plan.forward[index], plan.perimeter[group] - plan.forward[index])
            onward = half[group] - turn
            distributor_run, collector_run = distributors[group], collectors[group]
        else:
            turn, onward = 0.0, 0.0
            distributor_run, collector_run = distributors[0], collectors[0]
        branches.append(Branch(
            index=index, riser=risers[index], distributor=distributor_run,
            collector=collector_run,
            feed_length=link_in + off_dist[group] + turn,
            return_length=(onward + (total_coll - off_coll[group] - half[group])
                           + link_out)))

    # -------------------------------------------------------------------- notes
    network = PipeNetwork(config=config, center=(cx, cy), risers=risers,
                          distributors=distributors, collectors=collectors,
                          connectors=connectors, inlet=inlet, outlet=outlet,
                          branches=branches)
    span = network.header_span
    if span > HEADER_LIMIT:
        network.notes.append(
            f"the two headers are {span:.1f} m apart: above {HEADER_LIMIT:.0f} m the "
            f"bundle must be split into identical parallel modules to keep the flow "
            f"distribution uniform")
    elif span > HEADER_SAFE:
        network.notes.append(
            f"the two headers are {span:.1f} m apart: check the flow distribution "
            f"between the risers")
    band = config.z_top - config.z_bottom
    if band > MODULE_HEIGHT_LIMIT:
        network.notes.append(
            f"the risers are {band:.1f} m long: the published practice keeps a module "
            f"below {MODULE_HEIGHT_LIMIT:.0f} m of bed, so consider splitting the "
            f"height into modules")
    if config.collection == COLLECTION_DIRECT and network.path_spread() > 1.0 + 1e-9:
        paths = network.paths()
        network.notes.append(
            f"direct return: the branches run {float(np.min(paths)):.2f}-"
            f"{float(np.max(paths)):.2f} m "
            f"({100.0 * (network.path_spread() - 1.0):.1f}% spread), so the shortest "
            f"branch takes more flow; the {COLLECTION_REVERSE} collection equalises "
            f"the paths")
    if config.radius + WALL_STUB > 0.5 * min(mesh.Lx, mesh.Ly):
        network.notes.append(
            f"the vessel and its nozzle stubs do not fit inside the domain "
            f"({mesh.Lx:.2f} x {mesh.Ly:.2f} m): what falls outside is dropped from "
            f"the voxelisation")
    return network
