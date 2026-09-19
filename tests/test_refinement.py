"""Mesh refinement tests: the graded grid must be predictable and exact."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.refinement import (
    Band,
    describe,
    edges_to_sizes,
    graded_edges,
    partition,
    size_at,
    worst_ratio,
)


def test_uniform_request_gives_an_exactly_uniform_grid():
    edges = graded_edges(3.0, [Band(0.0, 3.0, 0.1)], growth=1.3)
    assert edges.size - 1 == 30
    assert np.allclose(np.diff(edges), 0.1, atol=1e-12)
    assert edges[-1] == pytest.approx(3.0, abs=1e-12)


def test_grid_starts_at_zero_and_ends_exactly_at_the_domain():
    for length in (0.37, 1.0, 5.6, 12.345):
        edges = graded_edges(length, [Band(0.0, length, length / 13)], growth=1.3)
        assert edges[0] == 0.0
        assert edges[-1] == pytest.approx(length, abs=1e-9)


def test_band_boundaries_land_on_grid_lines():
    """Interfaces must be grid lines, otherwise a mask splits a cell."""
    boundaries = (0.3, 0.5, 4.5, 4.7)
    edges = graded_edges(5.6, [Band(0.0, 0.3, 0.05), Band(0.3, 0.5, 0.05),
                               Band(0.5, 4.5, 0.15), Band(4.5, 4.7, 0.05),
                               Band(4.7, 5.6, 0.35)], growth=1.3)
    for boundary in boundaries:
        assert np.any(np.abs(edges - boundary) < 1e-9), boundary


def test_refined_region_gets_the_requested_size():
    """Inside a wide refined band the cells are the requested size.

    The ramp towards the fine size costs cells, so the band must be wider than the
    ramp for its core to reach the target - that is normal graded-mesh behaviour.
    """
    edges = graded_edges(6.0, [Band(0.0, 6.0, 0.2), Band(1.5, 4.5, 0.05)], growth=1.25)
    core = edges_to_sizes(edges)[(edges[:-1] >= 2.5) & (edges[1:] <= 3.5)]
    assert core.min() == pytest.approx(0.05, rel=1e-6)
    assert core.max() == pytest.approx(0.05, rel=0.2)
    ramp = edges_to_sizes(edges)[(edges[:-1] >= 1.5) & (edges[1:] <= 2.0)]
    assert ramp.max() < 0.2                          # but never coarser than the bulk


def test_growth_between_neighbours_stays_bounded():
    """The requested growth is honoured up to the remainder absorbed at a band end."""
    edges = graded_edges(5.6, [Band(0.0, 0.4, 0.04), Band(0.4, 5.6, 0.4)], growth=1.25)
    assert worst_ratio(edges) <= 1.25 * 1.15
    sizes = edges_to_sizes(edges)
    assert sizes.min() == pytest.approx(0.04, rel=0.05)
    assert sizes.max() <= 0.4 * 1.05                 # never coarser than requested


def test_no_sliver_cells():
    for spec in ([Band(0.0, 0.333, 0.05), Band(0.333, 1.777, 0.31)],
                 [Band(0.0, 1.0, 0.4), Band(1.0, 1.02, 0.05)]):
        length = spec[-1].end
        edges = graded_edges(length, spec, growth=1.3)
        sizes = edges_to_sizes(edges)
        assert sizes.min() > 0.2 * min(band.target for band in spec)


def test_cell_budget_is_respected_by_scaling_the_targets():
    """A million cells must not be produced silently: the budget scales the targets."""
    fine = graded_edges(5.6, [Band(0.0, 5.6, 0.002)], growth=1.3)
    assert fine.size - 1 > 2000
    capped = graded_edges(5.6, [Band(0.0, 5.6, 0.002)], growth=1.3, max_cells=500)
    assert capped.size - 1 <= 500
    assert describe(capped)["cells"] == capped.size - 1


def test_partition_keeps_the_finest_target_in_the_overlap():
    bands = partition(10.0, [Band(0.0, 10.0, 0.5), Band(2.0, 4.0, 0.1),
                             Band(7.0, 9.0, 0.2)])
    starts = [(band.start, band.end, band.target) for band in bands]
    assert (2.0, 4.0, 0.1) in starts
    assert (7.0, 9.0, 0.2) in starts
    for _, end, _ in starts:                        # contiguous, no gap, no overlap
        assert any(abs(other_start - end) < 1e-12 for other_start, _, _ in starts) \
            or end == pytest.approx(10.0)


def test_size_lookup_is_used_to_size_element_masks():
    edges = graded_edges(5.6, [Band(0.0, 5.6, 0.2), Band(2.0, 3.0, 0.05)], growth=1.3)
    assert size_at(edges, 2.5) == pytest.approx(0.05, rel=0.2)
    assert size_at(edges, 0.1) > 0.1


def test_degenerate_requests_are_rejected_or_degraded_gracefully():
    with pytest.raises(ValueError):
        graded_edges(0.0, [Band(0.0, 1.0, 0.1)])
    with pytest.raises(ValueError):
        graded_edges(1.0, [])
    coarse = graded_edges(1.0, [Band(0.0, 1.0, 2.0)], growth=1.3)
    assert coarse.size == 2 and coarse[-1] == pytest.approx(1.0)
