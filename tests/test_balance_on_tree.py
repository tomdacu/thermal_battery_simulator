"""The balance and the flux report on both meshes: one identity, two face lists.

What this module pins, in the order the claims matter:

* **the same numbers on both roads** - ``create_small_test_geometry()`` painted on a
  uniform tree and on the matching ``Mesh3D`` (the ``paint_pair`` fixture), solved by the
  same :class:`~src.solver.steady.SteadyStateSolver`: every field of
  :class:`~src.analysis.balance.Balance` agrees, field by field, on a model that already
  exercises an excluded air box, its film and the ground face;
* **the balance closes** - ``p_input - p_extracted - q_domain`` at the round-off of the
  powers, on a uniform pair, on a refined tree and with the source deposited in cells the
  problem dropped;
* **the loss is the film the assembly applied** - the report's ``q_domain`` is the sum of
  the per-face film rates, the conduction into the pinned leaves and the environment
  entry of :meth:`AdaptiveMesh.balance`, so the two implementations cannot drift;
* **the report reads the assembly's faces** - ``fluxes.face_fluxes`` is the mesh's own
  face list, and accumulating it around a cell reproduces the operator the assembly built
  (``AdaptiveMesh.matrix`` on a tree, the structured face coefficients on ``Mesh3D``) for
  an arbitrary *perturbed* field, not just for the solution;
* **nothing is counted twice** - with the air dropped, the conduction into it is gone
  from the envelope integral and the film is the whole loss, and a source sitting in an
  excluded cell is not reported as input (the row is the identity: the solver never sees
  it).

Tolerances are declared where a comparison needs one.  The two assemblies run the same
algebra over the same cells and sum it in a different order: the flux integrals agree to
the round-off of the terms (``TOLERANCE``, 1e-9 relative - generous, the measurement is
three orders below it), and ``imbalance`` is a *residual* of the identity, so it is
bounded against the power that drives the model (``RESIDUAL``, 1e-12 relative) rather
than compared field by field.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from src.analysis import fluxes
from src.analysis.balance import Balance, compute_balance
from src.core.adaptive_mesh import AdaptiveMesh, RefinementBand
from src.core.grid import GridIndex
from src.core.geometry import create_small_test_geometry
from src.core.mesh import Mesh3D
from src.solver.matrix import build_steady_matrix, face_conductance, face_coefficients
from src.solver.steady import SolverConfig, SteadyStateSolver

BOX = 1.0                        # the box of the one-dimensional cases
FINEST = 16                      # 16 leaves a side: 4096 cells of 1/16 m
H = BOX / FINEST
ALL_WALLS = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
#: the two roads sum the same faces in a different order: the fields agree at the
#: round-off of the terms (measured three orders below this bound)
TOLERANCE = 1e-9
#: the residual of the identity, as a fraction of the power that drives the model
RESIDUAL = 1e-12


# ------------------------------------------------------------------- helpers
def uniform_pair(box: float = BOX, spacing: float = H) -> tuple[AdaptiveMesh, Mesh3D]:
    """A uniform tree and the ``Mesh3D`` of the *same* cells, both empty."""
    side = int(round(box / spacing))
    return AdaptiveMesh.uniform(side, spacing, 0), Mesh3D(box, box, box, spacing=spacing)


def air_above(mesh) -> np.ndarray:
    """The upper half of the box, as a mask of either mesh's own shape."""
    return mesh.centres()[:, 2] > 0.5 if hasattr(mesh, "centres") else mesh.Z > 0.5


def excluded_slab() -> tuple[AdaptiveMesh, Mesh3D]:
    """A hot half-box under a dropped air box, on both meshes.

    The upper half is excluded from the problem (the air of an environment model): its
    interface carries the outside film and its cells are pinned at the ambient.  The
    source is deposited in *every* cell - a power profile does not know about the mask -
    and the walls of the box are adiabatic, so the film on the interface is the only
    exit and the identity has nothing else to balance.
    """
    density, film, t_ambient, emissivity = 5000.0, 10.0, 293.15, 0.8
    tree, structured = uniform_pair()
    for mesh in (tree, structured):
        mesh.k[:] = 1.0
        mesh.rho[:] = 1000.0
        mesh.cp[:] = 1000.0
        mesh.excluded = air_above(mesh)
        mesh.Q_source[:] = density
        mesh.h_out = film
        mesh.h_out_conv = film
        mesh.environment_emissivity = emissivity
        mesh.t_ambient = t_ambient
        for face in ALL_WALLS:
            mesh.set_adiabatic(face)
    return tree, structured


def conduction_pair() -> tuple[AdaptiveMesh, Mesh3D]:
    """A plain box with a source and a fixed ground on both meshes: conduction only.

    The face report is compared with the operator the assembly built, and a film would sit
    on that operator's diagonal too: the walls stay adiabatic and the ground is a
    Dirichlet *row* (an identity row, not a diagonal term).
    """
    tree, structured = uniform_pair()
    for mesh in (tree, structured):
        mesh.k[:] = 0.7
        mesh.rho[:] = 1000.0
        mesh.cp[:] = 1000.0
        mesh.Q_source[:] = 1000.0
        for face in ALL_WALLS:
            mesh.set_adiabatic(face)
        mesh.set_fixed_temperature_bc("z_min", 300.0)
    return tree, structured


def solve(mesh, method: str = "direct"):
    """Solve either mesh with the same driver and return the field it left behind."""
    result = SteadyStateSolver(mesh, SolverConfig(method=method)).solve()
    assert result.converged
    return result


def assert_the_same_balance(adaptive: Balance, structured: Balance) -> None:
    """Every field of ``Balance``, from the two roads, at the declared tolerances.

    ``imbalance`` is the residual of the identity rather than a physical power: the two
    residuals have no reason to be equal to each other (they are round-off), so each is
    bounded against the power that drives the model.
    """
    left, right = asdict(adaptive), asdict(structured)
    for name, value in left.items():
        if name == "notes":
            continue
        if name == "imbalance":
            assert abs(value) <= RESIDUAL * scale_of(adaptive), name
            assert abs(right[name]) <= RESIDUAL * scale_of(structured), name
            continue
        if value is None:
            # no efficiency to report (a model with no input): both roads say None
            assert right[name] is None, name
            continue
        if not np.isfinite(value):
            # a mean over a material the model does not carry: NaN on both roads
            assert not np.isfinite(right[name]), name
            continue
        assert value == pytest.approx(right[name], rel=TOLERANCE, abs=TOLERANCE), name


def scale_of(balance: Balance) -> float:
    """The largest rate a balance carries: what its residual has to be read against.

    Not the total loss alone: a model whose Neumann face and ground face cancel carries a
    total loss of zero between two 100 W terms, and a residual is round-off *of those*.
    """
    return max(abs(balance.p_input), abs(balance.p_extracted), abs(balance.q_domain),
               abs(balance.q_domain_top), abs(balance.q_domain_bottom),
               abs(balance.q_domain_side), abs(balance.q_battery), 1.0)


def assert_closes(balance: Balance, mesh) -> None:
    """The identity ``p_input - p_extracted - q_domain`` at the round-off of the powers.

    The residual is printed with the model when it does not fit: a balance that lies is
    the one failure that makes every other number untrustworthy.
    """
    residual = abs(balance.imbalance)
    scale = scale_of(balance)
    assert residual <= RESIDUAL * scale, (
        f"{getattr(mesh, 'summary', lambda: type(mesh).__name__)()} residual {residual:.3e} W "
        f"of {scale:.3e} W, {residual / scale:.2e} relative")


# --------------------------------------------------- the same numbers, two roads
def test_the_balance_is_the_same_field_by_field_on_both_meshes(paint_pair):
    """One battery model painted twice, solved twice: the same ``Balance``.

    The fixture model is not a toy: the air around the vessel is dropped, its interface
    carries the outside film and the ground is a Dirichlet face, so the comparison covers
    the environment entry, the films and the pinned rows - the three places where the two
    reports could have drifted apart.
    """
    tree, structured = paint_pair()
    solve(tree)
    solve(structured)
    balance = compute_balance(tree)
    reference = compute_balance(structured)

    assert balance.q_domain == pytest.approx(reference.q_domain, rel=TOLERANCE)
    assert balance.q_battery == pytest.approx(reference.q_battery, rel=TOLERANCE)
    assert_the_same_balance(balance, reference)
    assert_closes(balance, tree)
    assert_closes(reference, structured)


def test_the_air_dropped_from_the_problem_is_not_counted_twice():
    """Excluded cells, their film, and a source they carry: the identity still closes.

    The film is the only exit and the whole of it is the loss (``q_domain`` = the
    environment entry, ``q_battery`` the same), the envelope integral is *zero* - the
    assembly zeroes the conductance into an excluded cell, so reporting it would be
    reporting a flux that never flows - and the source deposited in the excluded half is
    not input: those rows are the identity.
    """
    tree, structured = excluded_slab()
    solve(tree)
    solve(structured)
    balance = compute_balance(tree)

    active_volume = float(fluxes.as_flat(tree.V)[~np.asarray(tree.excluded)].sum())
    assert balance.p_input == pytest.approx(5000.0 * active_volume, rel=1e-12)
    assert balance.p_input < 5000.0 * float(fluxes.as_flat(tree.V).sum())
    assert balance.q_domain == pytest.approx(balance.p_input, rel=RESIDUAL)
    assert balance.q_battery == pytest.approx(balance.q_domain, rel=RESIDUAL)
    assert fluxes.envelope_fluxes(tree)["total"] == pytest.approx(0.0, abs=1e-9)
    assert fluxes.envelope_fluxes(structured)["total"] == pytest.approx(0.0, abs=1e-9)
    # the film the report integrates is the one the assembly wrote back to the mesh
    assert fluxes.environment_flux(tree) == pytest.approx(tree.balance().q_environment,
                                                          rel=1e-12)
    # and the conduction the assembly zeroed into the dropped air is zero in the report
    # too, on either road: a flux the operator does not carry cannot be reported
    index = GridIndex.from_mesh(structured)
    i, j, _axis, _area, _distance = index.face_rows()
    dropped = fluxes.as_flat(structured.excluded)[i] | fluxes.as_flat(structured.excluded)[j]
    assert np.allclose(fluxes.face_fluxes(structured, index=index)[dropped], 0.0, atol=1e-12)
    tree_i, tree_j, *_rest = tree.face_rows()
    dropped = tree.excluded[tree_i] | tree.excluded[tree_j]
    assert np.allclose(fluxes.face_fluxes(tree)[dropped], 0.0, atol=1e-12)
    assert_closes(balance, tree)
    assert_closes(compute_balance(structured), structured)
    assert_the_same_balance(balance, compute_balance(structured))


def test_the_loss_is_the_film_rates_the_assembly_applied(paint_pair):
    """``q_domain`` is the mesh's own balance entry, not a second opinion about it.

    :meth:`AdaptiveMesh.balance` is the identity written from the inside (the per-leaf
    closure), :func:`compute_balance` the same thing written from the boundary: the fixed
    leaves plus the films plus the environment must be the same number, or one of the two
    is wrong.  Checked on a model whose only films are the environment one and a ground
    face, and again on the painted vessel, whose box faces carry convective films too.
    """
    tree, _structured = excluded_slab()
    solve(tree)
    report = compute_balance(tree)
    inside = tree.balance()
    assert report.q_domain == pytest.approx(
        inside.q_fixed + sum(inside.q_faces.values()) + inside.q_environment, rel=1e-12)
    assert inside.closure < 1e-12

    painted, _reference = paint_pair()
    solve(painted)
    report = compute_balance(painted)
    inside = painted.balance()
    assert report.q_domain == pytest.approx(
        inside.q_fixed + sum(inside.q_faces.values()) + inside.q_environment, rel=1e-12)
    assert inside.closure < 1e-12


def test_a_convective_box_face_closes_on_a_tree():
    """Six convective faces and a source: the film the assembly applied is the loss.

    The tree's box-face report is the half-cell film of its wall leaves
    (``h_eff = 2kh/(2k + h d)``), the law the assembly puts on the diagonal; the
    structured mesh reaches the same number from its own slides, so the two balances
    agree field by field and each closes on its own.
    """
    tree, structured = uniform_pair()
    for mesh in (tree, structured):
        mesh.k[:] = 1.0
        mesh.rho[:] = 1000.0
        mesh.cp[:] = 1000.0
        mesh.Q_source[:] = 500.0
        for face in ALL_WALLS:
            mesh.set_convection_bc(face, 8.0, 293.15)

    solve(tree)
    solve(structured)
    balance = compute_balance(tree)
    assert balance.q_domain > 0.0
    assert balance.q_domain == pytest.approx(
        balance.q_domain_top + balance.q_domain_bottom + balance.q_domain_side, rel=1e-12)
    assert_closes(balance, tree)
    assert_closes(compute_balance(structured), structured)
    assert_the_same_balance(balance, compute_balance(structured))


def test_an_imposed_flux_face_enters_the_identity_on_a_tree():
    """A Neumann face on a tree: the imposed flux is read back the way it was applied.

    No source and one ground face: the 100 W/m^2 entering the top leaves the domain
    through the bottom, so the total loss is zero and its two halves are the two terms -
    which is only true if the report adds ``-q A`` over exactly the wall leaves the
    assembly added ``q/d`` to.
    """
    tree, structured = uniform_pair()
    for mesh in (tree, structured):
        mesh.k[:] = 1.0
        mesh.rho[:] = 1000.0
        mesh.cp[:] = 1000.0
        for face in ALL_WALLS:
            mesh.set_adiabatic(face)
        mesh.set_heat_flux_bc("z_max", 100.0)
        mesh.set_fixed_temperature_bc("z_min", 300.0)

    solve(tree)
    solve(structured)
    balance = compute_balance(tree)
    assert balance.q_domain_top == pytest.approx(-100.0, rel=1e-9)
    assert balance.q_domain_bottom == pytest.approx(100.0, rel=1e-9)
    assert balance.q_domain == pytest.approx(0.0, abs=1e-9)
    assert_closes(balance, tree)
    assert_the_same_balance(balance, compute_balance(structured))


def test_the_tube_film_is_extracted_on_a_tree():
    """The fluid side on a tree: ``bc_h / V^(1/3)`` on the leaves away from the box faces.

    The mask is the one the assembly applies (:func:`is_interior_tube`), the coefficient
    the one it puts on the diagonal: the extraction enters ``p_extracted`` and the
    identity closes on both meshes with it.
    """
    tree, structured = uniform_pair()
    for mesh in (tree, structured):
        mesh.k[:] = 1.0
        mesh.rho[:] = 1000.0
        mesh.cp[:] = 1000.0
        mesh.Q_source[:] = 2000.0
        for face in ALL_WALLS:
            mesh.set_adiabatic(face)
        mesh.set_fixed_temperature_bc("z_min", 300.0)
        mesh.set_internal_convection(inner_cells(mesh), 20.0, 300.0)

    solve(tree)
    solve(structured)
    balance = compute_balance(tree)
    assert balance.p_extracted > 0.0
    assert fluxes.tube_flux(tree) == pytest.approx(fluxes.tube_flux(structured), rel=1e-12)
    assert_closes(balance, tree)
    assert_closes(compute_balance(structured), structured)
    assert_the_same_balance(balance, compute_balance(structured))


def inner_cells(mesh) -> np.ndarray:
    """A block of cells around the centre of the box: tube cells, none on a box face."""
    if hasattr(mesh, "centres"):
        centres = mesh.centres()
        axes = (centres[:, 0], centres[:, 1], centres[:, 2])
    else:
        axes = (mesh.X, mesh.Y, mesh.Z)
    return np.logical_and.reduce([np.abs(axis - 0.5) < 0.2 for axis in axes])


# ------------------------------------------------------------------ a refined tree
def test_a_refined_tree_closes_its_balance_like_the_uniform_one():
    """Hanging nodes do not touch the identity: the same model, a coarser far field.

    The tree is refined to 1/2 m inside the vessel and left at 1 m outside, so its face
    list carries coarse/fine pairs and hanging nodes, and the painted model is the one the
    uniform tree carries at the same vessel resolution.  Only the
    dropped air is meshed coarsely, and the air is what the film *replaces*: the loss is
    the same number on both, and the identity closes on both at the round-off of the
    powers.
    """
    battery = create_small_test_geometry()
    refined = AdaptiveMesh.from_bands(
        32, 0.25, [RefinementBand((1.0, 1.0, 0.0), (7.0, 7.0, 5.5), 0.5)], base_level=2)
    battery.apply_to_mesh(refined)
    uniform = AdaptiveMesh.uniform(16, 0.5, 0)
    battery.apply_to_mesh(uniform)
    assert len(refined.level_histogram()) > 1, "the tree must have leaves of two sizes"
    assert refined.sizes.min() == uniform.sizes.min()

    solve(refined)
    solve(uniform)
    balance = compute_balance(refined)
    reference = compute_balance(uniform)
    assert_closes(balance, refined)
    assert_closes(reference, uniform)
    # the loss lives on the vessel's own faces, and the two trees mesh the vessel
    # identically: coarsening the dropped air moves none of them
    for name in ("p_input", "p_extracted", "q_domain", "q_domain_top", "q_domain_bottom",
                 "q_domain_side", "q_battery", "q_battery_top", "q_battery_bottom",
                 "q_battery_side", "t_max", "t_min", "t_mean_storage", "eta_energy"):
        assert (getattr(balance, name)
                == pytest.approx(getattr(reference, name), rel=TOLERANCE)), name
    # the stored energy is a volume integral over the *whole* box - the concrete under the
    # vessel and the far field included - and there the refinement did change the cells:
    # it agrees within the resolution of the coarse part, not to the last digit
    assert balance.e_stored == pytest.approx(reference.e_stored, rel=1e-3)


# ---------------------------------------------- the report reads the assembly
def test_the_face_report_is_the_assembly_faces_on_a_tree():
    """The tree's report: ``faces()``, the shared rule, and the operator they assemble.

    The field is *perturbed* (the claim must not hold only for the solution), and two
    identities are checked: every entry is ``g (T_i - T_j)`` with ``g`` recomputed by
    :func:`src.solver.matrix.face_conductance` from the raw ``faces()`` rows, and the
    rates accumulated around a cell are the conduction the assembled operator applies
    there - so the report cannot be reading a face list of its own.
    """
    tree, _structured = conduction_pair()
    rng = np.random.default_rng(20260921)
    perturbed = tree.T + rng.normal(0.0, 25.0, tree.n_cells)

    rates = fluxes.face_fluxes(tree, perturbed)
    i, j, _axis, area, distance, g = tree.face_rows()
    assert rates.shape == (len(tree.faces()),)

    size = tree.sizes
    expected = face_conductance(
        tree.k[i], tree.k[j], area, distance, size_a=size[i], size_b=size[j],
        material_a=tree.material_id[i], material_b=tree.material_id[j],
        h_contact=tree.h_contact, excluded_b=tree.excluded[i] | tree.excluded[j])
    assert np.allclose(g, expected, rtol=1e-12)
    assert np.allclose(rates, expected * (perturbed[i] - perturbed[j]), rtol=1e-12)

    net = np.zeros(tree.n_cells)
    np.add.at(net, i, rates)
    np.add.at(net, j, -rates)
    assembled = np.asarray(tree.matrix() @ perturbed) * tree.V
    assert np.abs(net - assembled).max() < 1e-9 * max(np.abs(assembled).max(), 1.0)


def test_the_face_report_is_the_assembly_faces_on_a_structured_mesh():
    """The same claim on ``Mesh3D``: the ``GridIndex`` list and the assembled operator.

    The accumulation of the report's rates around a cell must be the conduction of
    ``build_steady_matrix`` - the operator the solver actually factorises - which is only
    true if the report lists each interior interface once, with the coefficient the
    structured assembly put in it.
    """
    _tree, structured = conduction_pair()
    index = GridIndex.from_mesh(structured)
    rng = np.random.default_rng(20260921)
    perturbed = np.asarray(structured.T) + rng.normal(0.0, 25.0, structured.T.shape)

    rates = fluxes.face_fluxes(structured, perturbed, index)
    i, j, axis, _area, _distance = index.face_rows()
    field = fluxes.as_flat(perturbed)
    coefficient = face_coefficients(structured, index)
    expected = (coefficient[2 * axis + 1, i] * index.volume[i]
                * (field[i] - field[j]))
    assert rates.shape == (i.size,)
    assert np.abs(rates - expected).max() < 1e-12 * max(np.abs(expected).max(), 1.0)

    net = np.zeros(index.flat.size)
    np.add.at(net, i, rates)
    np.add.at(net, j, -rates)
    operator, _rhs = build_steady_matrix(structured, index=index,
                                         enforce_dirichlet=False)
    assembled = np.asarray(operator @ field) * index.volume
    assert np.abs(net - assembled).max() < 1e-9 * max(np.abs(assembled).max(), 1.0)
