# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for the ribbon sweep's bend handling.

The pipeline-level checks live in tests/test_watertight.py; these pin down the
two functions that decide how the cross-section behaves in a bend, without a
DEM, a GPX file or an exporter in the way.

The rule both of them serve: a ribbon of half-width w can only follow a turn
whose radius stays above w. Below that its inner edge runs backwards and the
wall quads fold inside out — a closed, watertight, unprintable mesh.
"""

import math

import numpy as np
import pytest
from src.core.mesh_generator import (
    JOIN_RADIUS_FACTOR,
    JOIN_ROUNDING_TURN_DEG,
    MAX_MITRE_STRETCH,
    MIN_HALF_WIDTH_MM,
    WIDTH_SLOPE_LIMIT,
    _resample_path,
    _round_sharp_joins,
    _subdivide_path,
    _sweep_offsets,
    sample_z_from_terrain_mesh,
)

HALF_WIDTH_MM = 0.75  # half of the 1.5mm ribbon the tests elsewhere print


def _straight(step_mm: float = 2.0, count: int = 6) -> list[tuple[float, float]]:
    return [(step_mm * i, 0.0) for i in range(count)]


def _corner(turn_deg: float, arm_mm: float = 8.0) -> list[tuple[float, float]]:
    """Two arms of `arm_mm` meeting at the origin, deflecting by `turn_deg`."""
    turn = math.radians(turn_deg)
    return [(-arm_mm, 0.0), (0.0, 0.0), (arm_mm * math.cos(turn), arm_mm * math.sin(turn))]


def _arc(radius_mm: float, points: int = 16, sweep_deg: float = 180.0) -> list[tuple[float, float]]:
    """A circular arc — a switchback, once the radius drops under the ribbon."""
    return [
        (
            radius_mm * math.cos(math.radians(sweep_deg) * i / (points - 1)),
            radius_mm * math.sin(math.radians(sweep_deg) * i / (points - 1)),
        )
        for i in range(points)
    ]


def _turns_deg(path: list[tuple[float, float]]) -> list[float]:
    """Deflection at each interior point of a polyline, in degrees."""
    points = np.asarray(path)
    steps = np.diff(points, axis=0)
    lengths = np.hypot(steps[:, 0], steps[:, 1])
    units = steps / lengths[:, None]
    dots = np.clip(np.einsum("ij,ij->i", units[:-1], units[1:]), -1.0, 1.0)
    return list(np.degrees(np.arccos(dots)))


def _inversion_slack_mm(
    path: list[tuple[float, float]], offsets: list[tuple[float, float]]
) -> float:
    """Smallest forward advance of a ribbon edge over the whole sweep.

    Between two cross-sections each edge advances by
    `L - (w_i*|n_i.d| + w_j*|n_j.d|)`. Negative anywhere means that quad has
    folded inside out.
    """
    points = np.asarray(path)
    vectors = np.asarray(offsets)
    steps = np.diff(points, axis=0)
    lengths = np.hypot(steps[:, 0], steps[:, 1])
    units = steps / lengths[:, None]
    lead = np.abs(np.einsum("ij,ij->i", vectors[:-1], units))
    trail = np.abs(np.einsum("ij,ij->i", vectors[1:], units))
    return float(np.min(lengths - lead - trail))


# --------------------------------------------------------------------------
# _round_sharp_joins
# --------------------------------------------------------------------------


def test_gentle_path_is_left_alone():
    """Nothing to round: a straight run comes back untouched."""
    path = _straight()
    assert _round_sharp_joins(path, HALF_WIDTH_MM * JOIN_RADIUS_FACTOR, HALF_WIDTH_MM) == path


def test_short_paths_survive():
    """One or two points cannot have a corner — and must not raise."""
    assert _round_sharp_joins([], HALF_WIDTH_MM, HALF_WIDTH_MM) == []
    assert _round_sharp_joins([(1.0, 2.0)], HALF_WIDTH_MM, HALF_WIDTH_MM) == [(1.0, 2.0)]
    assert _round_sharp_joins([(0.0, 0.0), (1.0, 0.0)], HALF_WIDTH_MM, HALF_WIDTH_MM) == [
        (0.0, 0.0),
        (1.0, 0.0),
    ]


@pytest.mark.parametrize("turn_deg", [45.0, 90.0, 135.0, 179.0])
def test_a_sharp_corner_becomes_an_arc(turn_deg):
    """The whole turn is spread over several steps instead of one.

    One cross-section per point means an unrounded corner asks the section to
    rotate by the entire turn in a single step: the strip between its
    neighbours twists, and near a reversal it twists through itself.
    """
    rounded = _round_sharp_joins(
        _corner(turn_deg), HALF_WIDTH_MM * JOIN_RADIUS_FACTOR, HALF_WIDTH_MM
    )

    assert len(rounded) > 3, "corner was not subdivided"
    assert rounded[0] == (-8.0, 0.0)
    assert rounded[-1] == pytest.approx(_corner(turn_deg)[-1])
    assert max(_turns_deg(rounded)) < turn_deg, "no step may turn as far as the raw corner"


def test_an_exact_reversal_gets_a_tip_to_walk_around():
    """An out-and-back doubling over itself is the case with no tangent arc.

    The two arms are collinear, so no circle is tangent to both. Without a
    substitute the cross-section flips end for end between two neighbouring
    rings and the tube passes through itself; the sweep loops the path around a
    hundredth-millimetre tip instead.
    """
    path = [(-8.0, 0.0), (0.0, 0.0), (-8.0, 0.0)]
    rounded = _round_sharp_joins(path, HALF_WIDTH_MM * JOIN_RADIUS_FACTOR, HALF_WIDTH_MM)

    assert len(rounded) > 3
    assert max(_turns_deg(rounded)) < 180.0
    tip = np.asarray(rounded[1:-1])
    # The detour stays far below what a 0.4mm nozzle renders.
    assert float(np.abs(tip[:, 1]).max()) < 0.2


def test_rounding_only_shortens_the_path():
    """The arc cuts the corner, so the polyline can never grow longer."""
    path = _corner(120.0)
    rounded = _round_sharp_joins(path, HALF_WIDTH_MM * JOIN_RADIUS_FACTOR, HALF_WIDTH_MM)

    def length(points):
        steps = np.diff(np.asarray(points), axis=0)
        return float(np.hypot(steps[:, 0], steps[:, 1]).sum())

    assert length(rounded) <= length(path) + 1e-9


# --------------------------------------------------------------------------
# _sweep_offsets
# --------------------------------------------------------------------------


def test_a_straight_run_keeps_the_full_width():
    offsets = _sweep_offsets(_straight(), HALF_WIDTH_MM)

    assert len(offsets) == len(_straight())
    for offset in offsets:
        assert offset == pytest.approx((0.0, HALF_WIDTH_MM))


def test_a_gentle_bend_is_mitred():
    """The cross-section grows by 1/cos(turn/2) so the ribbon keeps its width."""
    path = _corner(20.0, arm_mm=20.0)
    offsets = _sweep_offsets(path, HALF_WIDTH_MM)

    corner_width = float(np.hypot(*offsets[1]))
    assert corner_width == pytest.approx(HALF_WIDTH_MM / math.cos(math.radians(10.0)), rel=1e-9)


def test_the_mitre_is_capped():
    """Past the mitre limit a hairpin grows a stub, not a spike."""
    offsets = _sweep_offsets(_corner(175.0), HALF_WIDTH_MM)
    widths = np.hypot(np.asarray(offsets)[:, 0], np.asarray(offsets)[:, 1])

    assert widths.max() <= HALF_WIDTH_MM * MAX_MITRE_STRETCH + 1e-9


@pytest.mark.parametrize("radius_mm", [4.0, 1.5, 0.75, 0.4, 0.1, 0.02])
def test_no_offset_can_invert_its_quad(radius_mm):
    """The clamp holds at every radius, including far below the half-width.

    This is the failure the watertight checks cannot see: the mesh stays a
    closed manifold while its walls face inwards.
    """
    path = _round_sharp_joins(_arc(radius_mm), HALF_WIDTH_MM * JOIN_RADIUS_FACTOR, HALF_WIDTH_MM)
    offsets = _sweep_offsets(path, HALF_WIDTH_MM)

    # Once the radius drops under MIN_HALF_WIDTH_MM the floor stops the clamp,
    # so the advance bottoms out at zero — a degenerate step, still not an
    # inverted one, and 0.02mm is a twentieth of what a nozzle can render.
    assert _inversion_slack_mm(path, offsets) >= -1e-9


@pytest.mark.parametrize("radius_mm", [4.0, 0.75, 0.1])
def test_the_width_changes_gradually(radius_mm):
    """A width that jumps ring to ring zigzags the edge into an accordion."""
    path = _round_sharp_joins(_arc(radius_mm), HALF_WIDTH_MM * JOIN_RADIUS_FACTOR, HALF_WIDTH_MM)
    offsets = np.asarray(_sweep_offsets(path, HALF_WIDTH_MM))
    widths = np.hypot(offsets[:, 0], offsets[:, 1])
    steps = np.diff(np.asarray(path), axis=0)
    lengths = np.hypot(steps[:, 0], steps[:, 1])

    assert np.all(np.abs(np.diff(widths)) <= WIDTH_SLOPE_LIMIT * lengths + 1e-9)


def test_a_pinched_ribbon_never_collapses_onto_its_centre_line():
    """Zero width would weld the two sides of the ring into a non-manifold pinch."""
    path = _round_sharp_joins(_arc(0.01), HALF_WIDTH_MM * JOIN_RADIUS_FACTOR, HALF_WIDTH_MM)
    offsets = np.asarray(_sweep_offsets(path, HALF_WIDTH_MM))

    assert np.hypot(offsets[:, 0], offsets[:, 1]).min() >= MIN_HALF_WIDTH_MM - 1e-12


def test_rounding_keeps_the_ribbon_at_full_width_through_a_right_angle():
    """A rounded corner is swept at (near) the requested width, not a stub.

    Before the arc, a 90 degree corner was one stretched cross-section 1.41x
    the half-width; sharper than 60 degrees it hit the mitre cap and printed a
    blob twice the ribbon's width.
    """
    path = _round_sharp_joins(_corner(90.0), HALF_WIDTH_MM * JOIN_RADIUS_FACTOR, HALF_WIDTH_MM)
    offsets = np.asarray(_sweep_offsets(path, HALF_WIDTH_MM))
    widths = np.hypot(offsets[:, 0], offsets[:, 1])

    stretch_ceiling = 1.0 / math.cos(math.radians(JOIN_ROUNDING_TURN_DEG) / 2)
    assert widths.max() <= HALF_WIDTH_MM * stretch_ceiling + 1e-9
    assert widths.min() > HALF_WIDTH_MM * 0.5, "the corner must not pinch away"


def test_short_paths_have_no_cross_sections():
    """Fewer than two points is not a tube; the caller reports that failure."""
    assert _sweep_offsets([], HALF_WIDTH_MM) == []
    assert _sweep_offsets([(0.0, 0.0)], HALF_WIDTH_MM) == []


# --------------------------------------------------------------------------
# _resample_path
# --------------------------------------------------------------------------


def _jittery_line(points: int = 400, span_mm: float = 40.0, jitter_mm: float = 0.05):
    """A straight run sampled far finer than the ribbon, with GPS-scale noise.

    This is what a real GPX looks like in model space: simplified to ten metres
    on the ground, which is hundredths of a millimetre here.
    """
    rng = np.random.default_rng(7)
    xs = np.linspace(0.0, span_mm, points)
    ys = rng.normal(0.0, jitter_mm, points)
    return [(float(x), float(y)) for x, y in zip(xs, ys, strict=True)]


def test_thinning_keeps_the_path_inside_the_ribbon():
    """Every dropped point stays within the tolerance of what is left.

    The ribbon drawn on the thinned path still has to cover the trace it came
    from, which is why the tolerance is a half-width and not more.
    """
    path = _jittery_line()
    thinned = _resample_path(path, HALF_WIDTH_MM * 0.5, HALF_WIDTH_MM)

    assert thinned[0] == path[0]
    assert thinned[-1] == path[-1]
    kept = np.asarray(thinned)
    for point in np.asarray(path):
        segments = kept[1:] - kept[:-1]
        lengths = np.hypot(segments[:, 0], segments[:, 1])
        offsets = point - kept[:-1]
        along = np.clip(np.einsum("ij,ij->i", offsets, segments) / lengths**2, 0.0, 1.0)
        closest = kept[:-1] + along[:, None] * segments
        assert float(np.hypot(*(point - closest).T.min(axis=1))) <= HALF_WIDTH_MM + 1e-9


def test_thinning_leaves_a_path_the_sweep_can_carry_at_full_width():
    """The point of thinning: no clamp, so the ribbon keeps its width.

    Swept raw, a jittery trace forces the curvature clamp on at nearly every
    ring — the ribbon collapses to a knife edge along its whole length.
    """
    path = _jittery_line()
    raw = np.asarray(_sweep_offsets(path, HALF_WIDTH_MM))
    thinned = _resample_path(path, HALF_WIDTH_MM * 0.5, HALF_WIDTH_MM)
    swept = np.asarray(_sweep_offsets(thinned, HALF_WIDTH_MM))

    raw_widths = np.hypot(raw[:, 0], raw[:, 1])
    widths = np.hypot(swept[:, 0], swept[:, 1])
    assert np.median(raw_widths) < HALF_WIDTH_MM * 0.5, "fixture is not actually jittery"
    assert np.median(widths) > HALF_WIDTH_MM * 0.9


def test_thinning_keeps_a_corner():
    """Douglas-Peucker first, so a real turn is never smoothed away."""
    path = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (10.0, 5.0), (10.0, 10.0)]
    thinned = _resample_path(path, HALF_WIDTH_MM, HALF_WIDTH_MM)

    assert (10.0, 0.0) in thinned


def test_thinning_a_short_path_changes_nothing():
    """Two points carry no detail to drop."""
    assert _resample_path([(0.0, 0.0), (1.0, 0.0)], 1.0, 1.0) == [(0.0, 0.0), (1.0, 0.0)]


# --------------------------------------------------------------------------
# _subdivide_path
# --------------------------------------------------------------------------


def test_subdivision_keeps_the_original_points():
    path = [(0.0, 0.0), (10.0, 0.0)]
    dense = _subdivide_path(path, 3.0)

    assert dense[0] == path[0]
    assert dense[-1] == path[-1]
    assert len(dense) == 5  # ceil(10/3) = 4 pieces
    steps = np.diff(np.asarray(dense), axis=0)
    assert np.hypot(steps[:, 0], steps[:, 1]).max() <= 3.0 + 1e-9


def test_subdivision_leaves_short_segments_alone():
    path = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.5)]
    assert _subdivide_path(path, 3.0) == path


# --------------------------------------------------------------------------
# sample_z_from_terrain_mesh
# --------------------------------------------------------------------------


def _terrain_grid(rows: int, cols: int, width_mm: float, height_mm: float, heights):
    """Grid vertices in the row-major order the terrain mesh stores them in."""
    return np.array(
        [
            [
                col / (cols - 1) * width_mm,
                row / (rows - 1) * height_mm,
                heights(col / (cols - 1) * width_mm, row / (rows - 1) * height_mm),
            ]
            for row in range(rows)
            for col in range(cols)
        ]
    )


def test_terrain_sampling_reproduces_a_tilted_plane():
    """Both triangles of every cell must agree with the surface they came from.

    The far triangle of each cell used to be interpolated with its barycentric
    weights rotated between the corners, so the sampled height jumped across
    the cell diagonal. The track rides on this surface: it crinkled with it,
    and the wall quads either side of a jump folded over each other.
    """
    rows, cols, width_mm, height_mm = 5, 7, 60.0, 40.0

    def plane(x_mm: float, y_mm: float) -> float:
        return 2.0 + 0.3 * x_mm - 0.2 * y_mm

    vertices = _terrain_grid(rows, cols, width_mm, height_mm, plane)
    # The far edge is excluded: the sampler clamps the query a thousandth of a
    # cell short of it to keep the corner indices in range, which costs a few
    # microns there and nothing anywhere else.
    for x_mm in np.linspace(0.0, width_mm, 37, endpoint=False):
        for y_mm in np.linspace(0.0, height_mm, 29, endpoint=False):
            sampled = sample_z_from_terrain_mesh(
                float(x_mm), float(y_mm), vertices, rows, cols, width_mm, height_mm
            )
            assert sampled == pytest.approx(plane(float(x_mm), float(y_mm)), abs=1e-9)

    corner = sample_z_from_terrain_mesh(
        width_mm, height_mm, vertices, rows, cols, width_mm, height_mm
    )
    assert corner == pytest.approx(plane(width_mm, height_mm), abs=0.01)


def test_terrain_sampling_is_continuous_across_the_cell_diagonal():
    """A step across the diagonal is what tears the ribbon riding over it."""
    rows, cols, width_mm, height_mm = 4, 4, 30.0, 30.0

    def bumpy(x_mm: float, y_mm: float) -> float:
        return math.sin(x_mm / 3.0) * math.cos(y_mm / 4.0) * 5.0

    vertices = _terrain_grid(rows, cols, width_mm, height_mm, bumpy)
    cell_x, cell_y = width_mm / (cols - 1), height_mm / (rows - 1)

    # Walk the diagonal of one cell and sample either side of it.
    for fraction in np.linspace(0.05, 0.95, 19):
        x_mm = float(fraction * cell_x)
        y_mm = float((1.0 - fraction) * cell_y)
        near = sample_z_from_terrain_mesh(
            x_mm - 1e-6, y_mm - 1e-6, vertices, rows, cols, width_mm, height_mm
        )
        far = sample_z_from_terrain_mesh(
            x_mm + 1e-6, y_mm + 1e-6, vertices, rows, cols, width_mm, height_mm
        )
        assert near == pytest.approx(far, abs=1e-5)
