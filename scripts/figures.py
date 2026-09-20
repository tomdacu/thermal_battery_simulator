"""Figures for the README: geometry, fields, mesh, and the gas-loop schema.

Run it from the repository root::

    python scripts/figures.py

Everything is generated offscreen and deterministically, so the pictures in the
documentation are reproducible from the code that produced them instead of being
one-off screenshots.  Output goes to ``docs/figures/``.
"""
from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pyvista as pv                     # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.analysis.balance import compute_balance        # noqa: E402
from src.core.geometry import create_small_test_geometry  # noqa: E402
from src.core.mesh import Mesh3D                        # noqa: E402
from src.core.pipes import rasterize_pipe, staggered_bank  # noqa: E402
from src.core.refinement import Band, GridSpec          # noqa: E402
from src.solver.fluid import Fluid, FluidLoop, pipe_h   # noqa: E402
from src.solver.steady import SolverConfig, SteadyStateSolver  # noqa: E402

OUT = pathlib.Path("docs/figures")
INK = "#1c1c1c"
HOT = "#c73a1b"
COLD = "#1f6fb4"
pv.OFF_SCREEN = True


def _save(fig, name: str, dpi: int = 190) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name}")


# ---------------------------------------------------------------------------
def battery_with_pipes() -> None:
    """Annular riser bundle: every manifold ring carries risers and is connected.

    Two concentric rows of risers so the bundle fills the annulus, each row served by
    its own manifold ring, the rings joined by radial links, and both ducts entering
    and leaving from the SIDE: a real vessel is not axisymmetric and nothing comes out
    of the roof.
    """
    radius, height, base = 2.0, 4.0, 0.4
    z0, z1 = base + 0.25, base + height - 0.55
    rows = ((1.55, 18, 0.0), (1.05, 12, np.pi / 12))

    plot = pv.Plotter(off_screen=True, window_size=(1500, 1000), border=False)
    plot.set_background("white")
    shell = pv.Cylinder(center=(0, 0, base + height / 2), direction=(0, 0, 1),
                        radius=radius, height=height, resolution=64, capping=False)
    plot.add_mesh(shell, color="#e8dcc0", opacity=0.18, smooth_shading=True)
    plot.add_mesh(shell.extract_feature_edges(), color="#8a7a5c", line_width=2)
    plot.add_mesh(pv.Cylinder(center=(0, 0, base + 0.02), direction=(0, 0, 1),
                              radius=radius, height=0.04, resolution=64),
                  color="#d9c9a8", opacity=0.35)
    plot.add_mesh(pv.Cylinder(center=(0, 0, base + height), direction=(0, 0, 1),
                              radius=radius, height=0.04, resolution=64),
                  color="#d9c9a8", opacity=0.15)

    def ring(ring_radius: float, z: float):
        angles = np.linspace(0.0, 2.0 * np.pi, 129)
        return pv.lines_from_points(np.column_stack(
            [ring_radius * np.cos(angles), ring_radius * np.sin(angles),
             np.full(angles.shape, z)]))

    # risers: each row sits on its own manifold ring, so no ring is decorative
    for ring_radius, count, phase in rows:
        for index in range(count):
            angle = phase + 2.0 * np.pi * index / count
            x, y = ring_radius * np.cos(angle), ring_radius * np.sin(angle)
            plot.add_mesh(pv.Line((x, y, z0), (x, y, z1)).tube(radius=0.033, n_sides=10),
                          color=COLD, smooth_shading=True)
    for ring_radius, _count, _phase in rows:
        for z, colour in ((z0, COLD), (z1, HOT)):
            plot.add_mesh(ring(ring_radius, z).tube(radius=0.05, n_sides=10),
                          color=colour, smooth_shading=True)
    # the two rows are joined: a manifold that is not connected starves the inner risers
    for index in range(6):
        angle = 2.0 * np.pi * index / 6 + 0.2
        for z, colour in ((z0, COLD), (z1, HOT)):
            a = (rows[0][0] * np.cos(angle), rows[0][0] * np.sin(angle), z)
            b = (rows[1][0] * np.cos(angle), rows[1][0] * np.sin(angle), z)
            plot.add_mesh(pv.Line(a, b).tube(radius=0.045, n_sides=10), color=colour,
                          smooth_shading=True)

    # ducts: in from the side at the bottom, out from the side at the top
    inlet = pv.Line((-radius - 1.0, 0.0, z0), (-rows[0][0], 0.0, z0))
    plot.add_mesh(inlet.tube(radius=0.105, n_sides=12), color=COLD, smooth_shading=True)
    outlet = pv.Line((rows[0][0], 0.0, z1), (radius + 1.1, 0.0, z1))
    plot.add_mesh(outlet.tube(radius=0.105, n_sides=12), color=HOT, smooth_shading=True)
    for x_sign, colour in ((-1.0, COLD), (1.0, HOT)):
        flange = pv.Cylinder(center=(x_sign * (radius + 0.1), 0.0, z0 if x_sign < 0 else z1),
                             direction=(1, 0, 0), radius=0.17, height=0.16, resolution=28)
        plot.add_mesh(flange, color="#b9b9b9", smooth_shading=True)

    plot.camera_position = [(8.0, -7.6, 5.0), (0, 0, base + height / 2), (0, 0, 1)]
    plot.add_text("two concentric rows of risers, each on its own manifold ring\n"
                  "joined by radial links - gas in from the side at the bottom,\n"
                  "out from the side at the top (blue cold, red hot)",
                  position="upper_left", font_size=13, color=INK)
    plot.screenshot(str(OUT / "geometry_buried_pipes.png"))
    plot.close()


def temperature_field() -> None:
    """Steady field of the small test battery, sliced through the middle."""
    from src.viz import scene
    mesh = Mesh3D(6.0, 6.0, 5.6, spacing=0.2)
    geometry = create_small_test_geometry()
    geometry.heaters.power_total = 8.0
    geometry.apply_to_mesh(mesh)
    SteadyStateSolver(mesh, SolverConfig(method="cg", preconditioner="amg_rs",
                                         tolerance=1e-8)).solve()
    balance = compute_balance(mesh)
    plot = pv.Plotter(off_screen=True, window_size=(1500, 1000), border=False)
    plot.set_background("white")
    scene.add_field(plot, mesh, "Temperature", axis="y", fraction=0.5)
    plot.camera_position = [(4.0, -9.5, 4.2), (3.0, 3.0, 2.8), (0, 0, 1)]
    plot.add_text(f"storage mean {balance.t_mean_storage - 273.15:.0f} C, "
                  f"envelope loss {balance.q_battery / 1000:.1f} kW",
                  position="upper_left", font_size=13, color=INK)
    plot.screenshot(str(OUT / "temperature_slice.png"))
    plot.close()


def graded_mesh() -> None:
    """A graded grid in cross-section: fine where the gradients are."""
    spec = GridSpec(x=(Band(0, 0.6, 0.03), Band(0, 2.4, 0.25)),
                    y=(Band(0, 0.6, 0.03), Band(0, 2.4, 0.25)),
                    z=(Band(0, 2.4, 0.25),), growth=1.3)
    edges = spec.edges(2.4, 2.4, 2.4)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for x in edges[0]:
        for y in edges[1]:
            pass
    for x0, x1 in zip(edges[0][:-1], edges[0][1:], strict=True):
        for y0, y1 in zip(edges[1][:-1], edges[1][1:], strict=True):
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       edgecolor=INK, linewidth=0.35))
    ax.axvline(0.6, color=HOT, linewidth=1.6)
    ax.axhline(0.6, color=HOT, linewidth=1.6)
    ax.text(0.35, 2.25, "fine band\n(cells 30 mm)", color=HOT, fontsize=10,
            ha="center")
    ax.text(1.55, 2.25, "linear ramp, ratio <= growth", color=INK, fontsize=10,
            ha="center")
    summary = spec.describe(edges)
    ax.set_title(f"graded grid: {summary['cells_axis'][0]} x {summary['cells_axis'][1]} "
                 f"cells, size {summary['min_size'] * 1000:.0f}-"
                 f"{summary['max_size'] * 1000:.0f} mm", fontsize=11)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    _save(fig, "graded_mesh.png")


def adaptive_concept() -> None:
    """The 2:1 octree: a coarse cell touches at most four half-size faces."""
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    big = 1.0
    ax.add_patch(plt.Rectangle((0, 0), 2 * big, 2 * big, fill=False, edgecolor=INK,
                               linewidth=1.6))
    for i, j in ((0, 0), (1, 0), (0, 1), (1, 1)):
        ax.add_patch(plt.Rectangle((i * big, j * big), big, big, fill=False,
                                   edgecolor=INK, linewidth=1.0))
    fine = 0.5
    for i in range(4):
        for j in range(4):
            fill = (i >= 2 and j >= 2)
            ax.add_patch(plt.Rectangle((2 * big + i * fine, j * fine), fine, fine,
                                       facecolor="#f3e2c7" if fill else "none",
                                       edgecolor=HOT if fill else "#9a9a9a",
                                       linewidth=1.0 if fill else 0.6))
    ax.text(1.0, 2.1, "coarse cells", ha="center", fontsize=11, color=INK)
    ax.text(2.9, 2.1, "one level finer (2:1)", ha="center", fontsize=11, color=HOT)
    ax.annotate("", xy=(2.0, 1.25), xytext=(2.0, 0.75),
                arrowprops=dict(arrowstyle="<->", color=COLD, lw=1.6))
    ax.text(2.05, 2.35, "a face is shared by at most\nfour finer faces; the hanging "
                        "nodes are\neliminated so the operator stays symmetric",
            fontsize=10, color=COLD)
    ax.set_xlim(-0.15, 4.4)
    ax.set_ylim(-0.15, 2.75)
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, "adaptive_mesh_concept.png")


def gas_loop_schema() -> None:
    """The closed gas loop: cold in at the bottom, hot collected at the top."""
    fig, ax = plt.subplots(figsize=(7.6, 8.8))
    # vessel
    ax.add_patch(FancyBboxPatch((1.0, 1.0), 4.0, 6.0, boxstyle="round,pad=0.10",
                                facecolor="#efe3c8", edgecolor=INK, linewidth=1.5))
    ax.text(3.0, 7.25, "insulated vessel, sand bed", ha="center", fontsize=11.5,
            color=INK)

    # the pipe bundle and the two manifolds
    for x in np.linspace(1.45, 4.55, 6):
        ax.plot([x, x], [1.95, 5.95], color=COLD, linewidth=2.2, zorder=3)
    ax.plot([1.45, 4.55], [5.95, 5.95], color=HOT, linewidth=3.4, zorder=4)
    ax.plot([1.45, 4.55], [1.95, 1.95], color=COLD, linewidth=3.4, zorder=4)
    ax.text(3.0, 6.18, "hot collecting manifold", ha="center", fontsize=10.5, color=HOT)
    ax.text(3.0, 1.62, "cold distributing manifold", ha="center", fontsize=10.5,
            color=COLD)
    ax.text(5.15, 3.9, "gas risers\nburied in the bed", fontsize=10.5, color=COLD,
            va="center")

    # out of the vessel, through the exchanger, fan and resistors, and back
    ax.plot([3.0, 3.0], [5.95, 8.15], color=HOT, linewidth=3.4, zorder=5)
    ax.add_patch(FancyBboxPatch((2.05, 8.15), 1.9, 0.60, boxstyle="round,pad=0.08",
                                facecolor="#d7e6f5", edgecolor=COLD, linewidth=1.4))
    ax.text(3.0, 8.45, "heat exchanger", ha="center", fontsize=10.5, color=COLD)
    ax.text(3.0, 9.02, "hot out to the user", ha="center", fontsize=10.5, color=HOT)

    ax.plot([3.95, 7.30], [8.45, 8.45], color=COLD, linewidth=3.4)
    ax.plot([7.30, 7.30], [8.45, 1.00], color=COLD, linewidth=3.4)
    ax.plot([7.30, 3.0], [1.00, 1.00], color=COLD, linewidth=3.4)
    ax.plot([3.0, 3.0], [1.00, 1.95], color=COLD, linewidth=3.4)
    ax.add_patch(plt.Circle((7.30, 5.60), 0.30, facecolor="#e9e9e9", edgecolor=INK,
                            zorder=6))
    ax.text(7.30, 5.60, "fan", ha="center", va="center", fontsize=9.5, color=INK,
            zorder=7)
    ax.add_patch(FancyBboxPatch((7.75, 3.05), 1.35, 0.80, boxstyle="round,pad=0.08",
                                facecolor="#f7e0d8", edgecolor=HOT, linewidth=1.4))
    ax.text(8.42, 3.45, "electric\nresistors", fontsize=9.5, color=HOT, va="center",
            ha="left")
    ax.plot([7.30, 7.30], [3.85, 3.05], color=COLD, linewidth=3.4, zorder=4)

    ax.text(0.80, 1.95, "cold gas in", ha="right", fontsize=10.5, color=COLD)
    ax.annotate("", xy=(1.20, 5.60), xytext=(1.20, 2.40),
                arrowprops=dict(arrowstyle="->", color=COLD, lw=2.0))
    ax.text(6.05, 6.75, "charging:  the resistors heat the gas,\n"
                        "                    the gas heats the bed\n\n"
                        "discharging:  the gas takes the heat\n"
                        "                          from the bed to the exchanger",
            fontsize=10.0, color=INK, va="top")

    ax.set_xlim(0.1, 9.3)
    ax.set_ylim(0.5, 9.4)
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, "closed_gas_loop.png")


def design_curves() -> None:
    """Effectiveness and blower share against the flow: the storage regimes."""
    fluid = Fluid().at(600.0)
    area = np.pi * 0.05 * 4.0 * 24          # 24 risers of 50 mm and 4 m
    flows = np.logspace(-2, 0.5, 60)
    ntu = np.array([500.0 * area / (m * fluid.cp) for m in flows])
    effectiveness = 1.0 - np.exp(-ntu)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.semilogx(flows, effectiveness * 100, color=HOT, linewidth=2.2,
                label="effectiveness $1-e^{-NTU}$")
    ax.axhline(95.0, color="#9a9a9a", linestyle=":", linewidth=1.2)
    ax.axvline(float(np.interp(0.95, effectiveness, flows)), color=COLD,
               linestyle="--", linewidth=1.4)
    ax.text(float(np.interp(0.95, effectiveness, flows)) * 1.1, 55.0,
            "a storage works here:\nthe heat is extracted,\nthe gas leaves hot",
            fontsize=9.5, color=COLD)
    ax.text(0.012, 20.0, "area-limited:\nthe gas leaves cold", fontsize=9.5, color=INK)
    ax.set_xlabel("mass flow [kg/s]")
    ax.set_ylabel("effectiveness [%]")
    ax.set_title("what the flow buys: the two regimes of a storage discharge",
                 fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "design_effectiveness.png")

    # blower share of the exchanged power for the same network
    diameters = (0.05, 0.08, 0.12)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for diameter in diameters:
        from src.solver.fluid import pressure_drop
        delta_p = np.array([pressure_drop(m, diameter, 4.0, fluid, 4.5e-5, 12.0)
                            for m in flows])
        fan = flows / fluid.rho * delta_p / 0.7
        ax.loglog(flows, 100 * fan / (500.0 * area / np.maximum(ntu, 1e-9) * 0.0 + 60_000.0),
                  linewidth=2.0, label=f"pipe d = {diameter * 1000:.0f} mm")
    ax.set_xlabel("mass flow [kg/s]")
    ax.set_ylabel("blower power / heat delivered [%]")
    ax.set_title("the price of the flow: blower work (60 kW delivered, 24 risers)",
                 fontsize=11)
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    _save(fig, "design_blower.png")


def main() -> None:
    print("writing figures to", OUT)
    battery_with_pipes()
    temperature_field()
    graded_mesh()
    adaptive_concept()
    gas_loop_schema()
    design_curves()
    print("done")


if __name__ == "__main__":
    main()
