"""Create the figures used in the Thermal Battery Simulator README.

The three PyVista scenes use the same geometry, mesh, and field conventions as the
application.  The remaining figures are deterministic Matplotlib diagrams and
representative cycle data, so running this file recreates every image in one step:

    QT_QPA_PLATFORM=offscreen python scripts/make_figures.py

The script writes only to ``docs/figures``.
"""
from __future__ import annotations

import os
import sys

# These variables must be set before importing PyVista/VTK.  This keeps rendering
# usable on a build machine without a display server.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from src.core.geometry import (BatteryGeometry, CylinderGeometry, HeaterConfig,
                               TubeConfig, TubePattern)
from src.core.mesh import MaterialID, Mesh3D
from src.core.refinement import Band, GridSpec
from src.viz.scene import add_geometry_preview, to_image_data


OUTPUT = ROOT / "docs" / "figures"
WIDTH_PX = 1600
HEIGHT_PX = 900
DPI = 100

INK = "#172033"
MUTED = "#5b6577"
GRID = "#d7dde7"
BLUE = "#1479a6"
TEAL = "#008b8b"
ORANGE = "#e4902e"
RED = "#c94c4c"
GREEN = "#318c64"
SAND = "#e4bd63"
INSULATION = "#d79b51"
STEEL = "#8d96a6"


def output_path(name: str) -> Path:
    """Return an output path and create the permitted output directory."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    return OUTPUT / name


def configure_matplotlib() -> None:
    """Apply one readable, white-background style to every static figure."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "axes.titlesize": 23,
        "axes.titleweight": "bold",
        "axes.labelsize": 17,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#a9b2c2",
        "axes.linewidth": 1.1,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "legend.fontsize": 14,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#cbd3df",
    })


def battery_geometry(tubes_active: bool = True) -> BatteryGeometry:
    """Create the representative battery used by all three 3-D scenes."""
    return BatteryGeometry(
        cylinder=CylinderGeometry(
            center_x=4.0,
            center_y=4.0,
            base_z=0.4,
            height=5.0,
            r_storage=2.2,
            insulation_thickness=0.35,
            shell_thickness=0.12,
            insulation_slab_bottom=0.25,
            insulation_slab_top=0.25,
            roof_angle_deg=16.0,
            foundation_margin=0.6,
        ),
        heaters=HeaterConfig(power_total=250.0),
        tubes=TubeConfig(
            n_tubes=12,
            diameter=0.14,
            active=tubes_active,
            pattern=TubePattern.RADIAL_ARRAY,
            n_rings=3,
        ),
    )


def set_representative_temperature(mesh: Mesh3D, battery: BatteryGeometry) -> None:
    """Populate a smooth, plausible hot-storage field in Kelvin for the slice view."""
    cyl = battery.cylinder
    radial = np.sqrt((mesh.X - cyl.center_x) ** 2 + (mesh.Y - cyl.center_y) ** 2)
    z_fraction = np.clip((mesh.Z - cyl.z_storage_start) / cyl.height, 0.0, 1.0)
    vertical = 0.82 + 0.18 * np.cos(2.0 * np.pi * (z_fraction - 0.5))
    radial_core = np.exp(-0.60 * (radial / cyl.r_storage) ** 2)
    storage_region = (
        (radial <= cyl.r_storage)
        & (mesh.Z >= cyl.z_storage_start)
        & (mesh.Z <= cyl.z_storage_end)
    )

    # The surrounding layers follow the storage temperature with a shorter radial
    # length scale.  Air outside the vessel remains close to the ambient value.
    distance_from_storage = np.maximum(radial - cyl.r_storage, 0.0)
    outside = 235.0 * np.exp(-distance_from_storage / 0.45) * (0.84 + 0.16 * vertical)
    storage = 620.0 * radial_core * vertical
    temperature_c = np.where(storage_region, 25.0 + storage, 25.0 + outside)
    temperature_c = np.where(mesh.material_id == int(MaterialID.AIR),
                             np.minimum(temperature_c, 85.0), temperature_c)
    mesh.T[:] = temperature_c + 273.15


def new_plotter() -> pv.Plotter:
    """Return a consistently sized off-screen PyVista plotter."""
    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=(WIDTH_PX, HEIGHT_PX))
    plotter.set_background("white")
    return plotter


def save_plotter(plotter: pv.Plotter, path: Path) -> None:
    """Render a PyVista plotter to an opaque PNG and close it."""
    plotter.render()
    plotter.screenshot(str(path), transparent_background=False)
    plotter.close()


def add_3d_bounds(plotter: pv.Plotter) -> None:
    """Add English axis labels that remain legible on a white background."""
    plotter.add_axes(line_width=2, labels_off=False)
    plotter.show_bounds(
        location="outer",
        all_edges=True,
        ticks="outside",
        xtitle="x [m]",
        ytitle="y [m]",
        ztitle="z [m]",
        color=INK,
        font_size=12,
    )


def render_temperature_slice() -> Path:
    """Render the temperature field on a vertical centre slice."""
    battery = battery_geometry(tubes_active=True)
    mesh = Mesh3D(Lx=8.0, Ly=8.0, Lz=7.5, spacing=0.16)
    battery.apply_to_mesh(mesh)
    set_representative_temperature(mesh, battery)

    grid = to_image_data(mesh, "Temperature")
    slice_grid = grid.slice(
        normal=(0.0, 1.0, 0.0),
        origin=(battery.cylinder.center_x, battery.cylinder.center_y, 3.3),
    )
    path = output_path("temperature_slice.png")
    plotter = new_plotter()
    plotter.add_mesh(
        slice_grid,
        scalars="Temperature",
        cmap="coolwarm",
        clim=[25.0, 650.0],
        show_edges=False,
        smooth_shading=True,
        scalar_bar_args={
            "title": "Temperature [deg C]",
            "n_labels": 6,
            "fmt": "%.0f",
            "position_x": 0.88,
            "position_y": 0.16,
            "width": 0.08,
            "height": 0.68,
            "title_font_size": 16,
            "label_font_size": 13,
        },
    )
    plotter.add_mesh(slice_grid.outline(), color=INK, line_width=2)
    plotter.add_text(
        "Temperature field on a vertical centre slice",
        position="upper_left",
        font_size=20,
        color=INK,
    )
    plotter.add_text(
        "Representative hot-storage state",
        position="lower_left",
        font_size=14,
        color=MUTED,
    )
    add_3d_bounds(plotter)
    plotter.view_vector((0.0, -1.0, 0.0), viewup=(0.0, 0.0, 1.0))
    plotter.camera.zoom(1.08)
    save_plotter(plotter, path)
    return path


def render_geometry_with_pipes() -> Path:
    """Render a cutaway of the vessel, insulation, shell, and buried pipes."""
    battery = battery_geometry(tubes_active=True)
    frame_mesh = Mesh3D(Lx=8.0, Ly=8.0, Lz=7.5, spacing=0.35)
    path = output_path("geometry_buried_pipes.png")
    plotter = new_plotter()
    add_geometry_preview(
        plotter,
        battery,
        mesh=frame_mesh,
        clip=("y", battery.cylinder.center_y),
        opacity_scale=1.0,
    )
    plotter.add_legend(
        labels=[
            ("Sand storage", SAND),
            ("Insulation", INSULATION),
            ("Steel shell", STEEL),
            ("Buried heat exchanger pipes", BLUE),
        ],
        bcolor="white",
        border=True,
        size=(0.28, 0.22),
        loc="upper left",
    )
    plotter.add_text(
        "Battery geometry with buried heat exchanger pipes",
        position="upper_right",
        font_size=19,
        color=INK,
    )
    plotter.add_text(
        "Cutaway view through the vessel centre",
        position="lower_left",
        font_size=14,
        color=MUTED,
    )
    add_3d_bounds(plotter)
    plotter.view_isometric()
    plotter.camera.zoom(1.12)
    save_plotter(plotter, path)
    return path


def graded_mesh_spec() -> GridSpec:
    """Return a compact refinement request with fine storage/interface bands."""
    return GridSpec(
        x=(Band(0.0, 8.0, 0.45), Band(1.45, 6.55, 0.18)),
        y=(Band(0.0, 8.0, 0.45), Band(1.45, 6.55, 0.18)),
        z=(Band(0.0, 7.5, 0.45), Band(0.55, 5.95, 0.18)),
        growth=1.25,
        max_cells=250_000,
    )


def render_graded_mesh() -> Path:
    """Render the realised graded Cartesian mesh and its local cell size."""
    mesh = Mesh3D(Lx=8.0, Ly=8.0, Lz=7.5, grid=graded_mesh_spec())
    grid = pv.RectilinearGrid(mesh.edges_x, mesh.edges_y, mesh.edges_z)
    local_size_mm = np.broadcast_to(mesh.dx[:, None, None] * 1000.0, mesh.T.shape)
    grid.cell_data["Cell size [mm]"] = local_size_mm.ravel(order="F")
    cut = grid.clip(normal=(1.0, 0.0, 0.0), origin=(4.0, 4.0, 3.75), invert=False)

    summary = mesh.grid_summary()
    path = output_path("graded_mesh.png")
    plotter = new_plotter()
    plotter.add_mesh(
        cut,
        scalars="Cell size [mm]",
        cmap="viridis",
        clim=[float(local_size_mm.min()), float(local_size_mm.max())],
        show_edges=True,
        edge_color="#536174",
        line_width=0.45,
        scalar_bar_args={
            "title": "Cell size [mm]",
            "n_labels": 5,
            "fmt": "%.0f",
            "position_x": 0.88,
            "position_y": 0.16,
            "width": 0.08,
            "height": 0.68,
            "title_font_size": 16,
            "label_font_size": 13,
        },
    )
    plotter.add_mesh(
        pv.Box(bounds=(0.0, mesh.Lx, 0.0, mesh.Ly, 0.0, mesh.Lz)),
        style="wireframe",
        color=INK,
        line_width=2,
    )
    plotter.add_text(
        "Graded Cartesian mesh",
        position="upper_left",
        font_size=21,
        color=INK,
    )
    plotter.add_text(
        f"{summary['cells']:,} cells | {summary['min_size'] * 1000:.0f} to "
        f"{summary['max_size'] * 1000:.0f} mm | neighbour ratio <= "
        f"{summary['worst_ratio']:.2f}",
        position=(22, 790),
        font_size=13,
        color=MUTED,
    )
    add_3d_bounds(plotter)
    plotter.view_isometric()
    plotter.camera.zoom(1.06)
    save_plotter(plotter, path)
    return path


def smoothstep(x: np.ndarray) -> np.ndarray:
    """Smoothly clamp a transition to the interval [0, 1]."""
    clipped = np.clip(x, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def phase_gate(t: np.ndarray, start: float, stop: float, ramp: float = 0.45) -> np.ndarray:
    """Return a smooth rectangular operating phase."""
    rise = smoothstep((t - start) / ramp)
    fall = smoothstep((t - (stop - ramp)) / ramp)
    return rise * (1.0 - fall)


def cycle_data() -> dict[str, np.ndarray]:
    """Create a representative 24-hour charge, hold, and discharge cycle."""
    time_h = np.linspace(0.0, 24.0, 241)
    charge_progress = smoothstep((time_h - 0.5) / 9.5)
    discharge_progress = smoothstep((time_h - 12.0) / 8.0)
    storage_temperature = (
        25.0 + 575.0 * charge_progress - 385.0 * discharge_progress
        + 2.5 * np.sin(2.0 * np.pi * time_h / 24.0)
    )
    charge = 1250.0 * phase_gate(time_h, 0.0, 10.0)
    discharge = 950.0 * phase_gate(time_h, 12.0, 20.0)
    losses = 34.0 + 0.000105 * np.maximum(storage_temperature - 25.0, 0.0) ** 2
    side = losses * (0.48 + 0.03 * np.sin(np.pi * time_h / 12.0))
    top = losses * (0.30 + 0.02 * np.cos(np.pi * time_h / 12.0))
    bottom = losses - side - top
    return {
        "time": time_h,
        "temperature": storage_temperature,
        "charge": charge,
        "discharge": discharge,
        "losses": losses,
        "loss_side": side,
        "loss_top": top,
        "loss_bottom": bottom,
    }


def save_matplotlib(fig: plt.Figure, path: Path) -> None:
    """Save a Matplotlib figure at exactly 1600 px wide."""
    fig.savefig(path, dpi=DPI, facecolor="white", edgecolor="none")
    plt.close(fig)


def mark_cycle_phases(ax: plt.Axes, show_labels: bool = True) -> None:
    """Shade the charge and discharge windows in a time-series plot."""
    ax.axvspan(0.0, 10.0, color="#e7f1f5", alpha=0.8, zorder=0)
    ax.axvspan(12.0, 20.0, color="#f9eee1", alpha=0.8, zorder=0)
    if not show_labels:
        return
    ax.text(5.0, 0.97, "Charge", transform=ax.get_xaxis_transform(),
            ha="center", va="top", color=BLUE, fontsize=14, weight="bold")
    ax.text(16.0, 0.97, "Discharge", transform=ax.get_xaxis_transform(),
            ha="center", va="top", color=ORANGE, fontsize=14, weight="bold")


def format_cycle_axis(ax: plt.Axes) -> None:
    """Apply common time-axis formatting."""
    ax.set_xlim(0.0, 24.0)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.set_xlabel("Time [h]")
    ax.grid(True, color=GRID, linewidth=0.9, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_cycle_temperature() -> Path:
    """Render mean storage temperature through the operating cycle."""
    data = cycle_data()
    fig, ax = plt.subplots(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)
    mark_cycle_phases(ax)
    ax.plot(data["time"], data["temperature"], color=RED, linewidth=3.4,
            label="Mean storage temperature")
    ax.axhline(25.0, color=MUTED, linewidth=1.5, linestyle="--",
               label="Ambient reference")
    ax.set_ylim(0.0, 660.0)
    ax.set_ylabel("Mean storage temperature [deg C]")
    ax.set_title("Mean storage temperature over one charge-discharge cycle", pad=18)
    ax.legend(loc="lower right")
    format_cycle_axis(ax)
    fig.tight_layout(pad=2.0)
    path = output_path("cycle_temperature.png")
    save_matplotlib(fig, path)
    return path


def render_cycle_power() -> Path:
    """Render charge input, discharge output, and total thermal losses."""
    data = cycle_data()
    fig, ax = plt.subplots(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)
    mark_cycle_phases(ax)
    ax.plot(data["time"], data["charge"], color=BLUE, linewidth=3.0,
            label="Electrical charge input")
    ax.plot(data["time"], data["discharge"], color=GREEN, linewidth=3.0,
            label="Useful discharge output")
    ax.plot(data["time"], data["losses"], color=RED, linewidth=2.8,
            linestyle="--", label="Thermal losses")
    ax.fill_between(data["time"], data["losses"], color=RED, alpha=0.10)
    ax.set_ylim(0.0, 1400.0)
    ax.set_ylabel("Power [kW]")
    ax.set_title("Power in, power out, and thermal losses", pad=18)
    ax.legend(loc="upper right")
    format_cycle_axis(ax)
    fig.tight_layout(pad=2.0)
    path = output_path("cycle_power.png")
    save_matplotlib(fig, path)
    return path


def render_cycle_losses() -> Path:
    """Render the top, side, and bottom components of the loss rate."""
    data = cycle_data()
    fig, ax = plt.subplots(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)
    mark_cycle_phases(ax, show_labels=False)
    ax.stackplot(
        data["time"],
        data["loss_side"],
        data["loss_top"],
        data["loss_bottom"],
        colors=[RED, ORANGE, "#7b8da6"],
        alpha=0.85,
        labels=["Side envelope", "Top envelope", "Bottom / ground"],
    )
    ax.plot(data["time"], data["losses"], color=INK, linewidth=2.4,
            label="Total losses")
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("Loss rate [kW]")
    ax.set_title("Thermal losses through the storage envelope", pad=18)
    ax.legend(loc="upper right", ncol=2)
    format_cycle_axis(ax)
    fig.tight_layout(pad=2.0)
    path = output_path("cycle_losses.png")
    save_matplotlib(fig, path)
    return path


def draw_box(ax: plt.Axes, x: float, y: float, width: float, height: float,
             title: str, detail: str, face: str, edge: str = INK) -> None:
    """Draw a labelled rounded box for a technical diagram."""
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        facecolor=face, edgecolor=edge, linewidth=2.0,
    ))
    ax.text(x + width / 2.0, y + height * 0.62, title, ha="center", va="center",
            fontsize=16, weight="bold", color=INK)
    ax.text(x + width / 2.0, y + height * 0.30, detail, ha="center", va="center",
            fontsize=12.5, color=MUTED)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
          color: str = BLUE, linewidth: float = 3.0, connectionstyle: str = "arc3") -> None:
    """Draw a clear directional arrow."""
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=linewidth,
        color=color,
        connectionstyle=connectionstyle,
    ))


def render_gas_loop() -> Path:
    """Draw the closed gas loop through the buried pipes."""
    fig, ax = plt.subplots(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)
    ax.set_xlim(0.0, 16.0)
    ax.set_ylim(0.0, 9.0)
    ax.axis("off")
    ax.set_title("Closed gas loop through buried heat exchanger pipes", pad=22)

    # Storage vessel and the vertical buried-pipe bundle.
    ax.add_patch(FancyBboxPatch(
        (5.0, 1.0), 6.0, 7.0,
        boxstyle="round,pad=0.12,rounding_size=0.22",
        facecolor="#f7f9fb", edgecolor=INK, linewidth=2.5,
    ))
    ax.add_patch(Rectangle((5.55, 1.55), 4.9, 5.9, facecolor="#e6c477",
                           edgecolor="#b3832c", linewidth=1.8))
    ax.text(8.0, 7.72, "Thermal storage vessel", ha="center", va="center",
            fontsize=17, weight="bold")
    pipe_x = np.linspace(6.25, 9.75, 7)
    for index, x in enumerate(pipe_x):
        ax.plot([x, x], [1.85, 7.15], color=BLUE, linewidth=5.5,
                solid_capstyle="round")
        if index % 2 == 0:
            arrow(ax, (x, 2.25), (x, 6.65), color=BLUE, linewidth=1.8)
    ax.plot([6.15, 9.85], [7.15, 7.15], color=BLUE, linewidth=5.5,
            solid_capstyle="round")
    ax.plot([6.15, 9.85], [1.85, 1.85], color=BLUE, linewidth=5.5,
            solid_capstyle="round")
    ax.text(8.0, 4.25, "Sand / rock bed", ha="center", va="center", fontsize=16,
            weight="bold", color="#6d4e13")
    ax.text(8.0, 3.75, "heat crosses the pipe walls", ha="center", va="center",
            fontsize=13, color="#6d4e13")

    draw_box(ax, 0.75, 1.05, 2.2, 1.15, "Blower", "circulates gas", "#e8f1f5", BLUE)
    draw_box(ax, 3.25, 1.05, 2.2, 1.15, "Electric heater", "charge mode", "#fff0dd", ORANGE)
    draw_box(ax, 12.15, 6.45, 2.75, 1.15, "Heat exchanger", "discharge mode", "#e6f2ea", GREEN)

    # The arrows form one continuous closed circuit.
    arrow(ax, (2.95, 1.62), (3.22, 1.62), color=BLUE)
    arrow(ax, (5.48, 1.62), (6.05, 1.85), color=ORANGE,
          connectionstyle="angle3,angleA=0,angleB=90")
    arrow(ax, (9.95, 7.15), (12.08, 7.02), color=BLUE,
          connectionstyle="angle3,angleA=90,angleB=0")
    arrow(ax, (13.55, 6.4), (13.55, 0.55), color=GREEN)
    arrow(ax, (13.55, 0.55), (0.55, 0.55), color=BLUE)
    arrow(ax, (0.55, 0.55), (0.55, 1.0), color=BLUE)
    ax.text(8.0, 0.64, "closed-loop return gas", ha="center", va="bottom",
            fontsize=12.5, color=MUTED)
    ax.text(11.7, 7.78, "hot gas during charge", ha="right", va="bottom",
            fontsize=13, color=ORANGE, weight="bold")
    ax.text(12.8, 2.55, "heat delivered to the load", ha="center", va="bottom",
            fontsize=13, color=GREEN, rotation=90)
    ax.text(8.0, 0.30, "The same closed loop can charge the bed or extract heat from it.",
            ha="center", va="bottom", fontsize=14, color=MUTED)
    fig.tight_layout(pad=1.2)
    path = output_path("closed_gas_loop.png")
    save_matplotlib(fig, path)
    return path


def render_efficiency_decomposition() -> Path:
    """Draw a round-trip efficiency decomposition as an energy path."""
    fig, ax = plt.subplots(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)
    ax.set_xlim(0.0, 16.0)
    ax.set_ylim(0.0, 9.0)
    ax.axis("off")
    ax.set_title("Round-trip efficiency decomposition", pad=22)

    stages = [
        ("Electrical\ninput", "100%", "#e8f1f5", BLUE, "reference energy"),
        ("Charge\nconversion", "eta = 92%", "#fff0dd", ORANGE, "resistive and power electronics"),
        ("Storage\nretention", "eta = 96%", "#f4e9d5", "#b3832c", "envelope and standby losses"),
        ("Discharge /\nheat transfer", "eta = 94%", "#e6f2ea", GREEN, "pipes and heat exchanger"),
        ("Auxiliaries", "eta = 99%", "#ececf7", "#6969a6", "blower and controls"),
        ("Useful\noutput", "82%", "#e1f1e9", "#247550", "delivered energy"),
    ]
    positions = [0.25, 2.85, 5.45, 8.05, 10.65, 13.25]
    width = 2.25
    for index, (title, value, face, edge, detail) in enumerate(stages):
        draw_box(ax, positions[index], 4.0, width, 2.15, title, value, face, edge)
        ax.text(positions[index] + width / 2.0, 3.45, detail, ha="center", va="top",
                fontsize=11.5, color=MUTED, wrap=True)
        if index < len(stages) - 1:
            arrow(ax, (positions[index] + width + 0.08, 5.08),
                  (positions[index + 1] - 0.10, 5.08), color=TEAL, linewidth=2.5)

    ax.text(8.0, 1.95,
            r"eta_rt = eta_charge x eta_storage x eta_discharge x eta_aux",
            ha="center", va="center", fontsize=21, weight="bold", color=INK)
    ax.text(8.0, 1.25, "0.92 x 0.96 x 0.94 x 0.99 = 0.82  (illustrative energy basis)",
            ha="center", va="center", fontsize=15, color=MUTED)
    ax.text(8.0, 7.7,
            "Each factor removes one physically distinct part of the energy path.",
            ha="center", va="center", fontsize=16, color=MUTED)
    fig.tight_layout(pad=1.2)
    path = output_path("round_trip_efficiency.png")
    save_matplotlib(fig, path)
    return path


def draw_mesh_panel(ax: plt.Axes, left: float, graded: bool) -> None:
    """Draw a uniform or graded conceptual mesh panel."""
    bottom = 1.35
    width = 5.35
    height = 5.85
    ax.add_patch(Rectangle((left, bottom), width, height, facecolor="#fbfcfd",
                           edgecolor=INK, linewidth=2.0))

    if graded:
        left_band = np.linspace(left, left + 1.40, 5, endpoint=False)
        fine_band = np.linspace(left + 1.40, left + 3.95, 25, endpoint=False)
        right_band = np.linspace(left + 3.95, left + width, 6)
        x_lines = np.unique(np.r_[left_band, fine_band, right_band, left + width])
        y_lines = np.unique(np.r_[
            np.linspace(bottom, bottom + 1.25, 5, endpoint=False),
            np.linspace(bottom + 1.25, bottom + 4.65, 25, endpoint=False),
            np.linspace(bottom + 4.65, bottom + height, 6),
        ])
    else:
        x_lines = np.linspace(left, left + width, 13)
        y_lines = np.linspace(bottom, bottom + height, 13)

    for x in x_lines:
        ax.plot([x, x], [bottom, bottom + height], color="#a9b5c5", linewidth=0.7)
    for y in y_lines:
        ax.plot([left, left + width], [y, y], color="#a9b5c5", linewidth=0.7)

    # The centre marks the storage region; the blue lines represent pipes or heaters.
    storage = Circle((left + width / 2.0, bottom + height / 2.0), 1.18,
                     facecolor="#e4bd63", edgecolor="#b3832c", linewidth=2.0,
                     alpha=0.78)
    ax.add_patch(storage)
    for x in np.linspace(left + width / 2.0 - 0.65, left + width / 2.0 + 0.65, 5):
        ax.plot([x, x], [bottom + 2.55, bottom + 4.45], color=BLUE, linewidth=2.8)
    ax.text(left + width / 2.0, bottom + 0.30, "far field", ha="center", va="center",
            fontsize=13, color=MUTED)
    ax.text(left + width / 2.0, bottom + height / 2.0, "storage", ha="center",
            va="center", fontsize=15, weight="bold", color="#6d4e13")


def render_adaptive_mesh_concept() -> Path:
    """Draw the adaptive-mesh concept used by the graded Cartesian grid."""
    fig, ax = plt.subplots(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)
    ax.set_xlim(0.0, 16.0)
    ax.set_ylim(0.0, 9.0)
    ax.axis("off")
    ax.set_title("Adaptive-mesh concept: resolve gradients where they occur", pad=22)

    ax.text(3.45, 7.75, "Uniform mesh", ha="center", va="center", fontsize=19,
            weight="bold", color=INK)
    ax.text(12.55, 7.75, "Graded mesh", ha="center", va="center", fontsize=19,
            weight="bold", color=INK)
    draw_mesh_panel(ax, 0.75, graded=False)
    draw_mesh_panel(ax, 9.85, graded=True)

    arrow(ax, (6.45, 4.25), (9.15, 4.25), color=TEAL, linewidth=3.0)
    ax.text(7.8, 4.75, "place cells by\nphysical targets", ha="center", va="center",
            fontsize=14, color=TEAL, weight="bold")
    ax.text(3.45, 0.70, "same cell size everywhere", ha="center", va="center",
            fontsize=14, color=MUTED)
    ax.text(12.55, 0.70, "fine near pipes and interfaces; coarse in smooth regions",
            ha="center", va="center", fontsize=14, color=MUTED)
    ax.annotate("temperature gradients", xy=(11.95, 5.75), xytext=(14.45, 7.0),
                ha="center", va="center", fontsize=13, color=RED,
                arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 2.0})
    ax.annotate("linear transition\nwith bounded growth", xy=(14.95, 3.2),
                xytext=(13.6, 2.5), ha="center", va="center", fontsize=12.5,
                color=BLUE, arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 2.0})
    fig.tight_layout(pad=1.2)
    path = output_path("adaptive_mesh_concept.png")
    save_matplotlib(fig, path)
    return path


def main() -> None:
    """Generate all README figures and print the produced-file manifest."""
    configure_matplotlib()
    produced = [
        (render_temperature_slice(), "PyVista temperature field on a vertical slice."),
        (render_geometry_with_pipes(), "PyVista cutaway of the storage geometry and buried pipes."),
        (render_graded_mesh(), "PyVista view of the realised graded Cartesian mesh."),
        (render_cycle_temperature(), "Mean storage temperature through a representative cycle."),
        (render_cycle_power(), "Charge input, discharge output, and total losses."),
        (render_cycle_losses(), "Top, side, and bottom components of thermal losses."),
        (render_gas_loop(), "Closed gas loop through the buried heat exchanger pipes."),
        (render_efficiency_decomposition(), "Round-trip efficiency factors along the energy path."),
        (render_adaptive_mesh_concept(), "Conceptual comparison of uniform and graded meshes."),
    ]
    print("Produced files:")
    for path, description in produced:
        print(f"- {path.relative_to(ROOT)}: {description}")


if __name__ == "__main__":
    main()
