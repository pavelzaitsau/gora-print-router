# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for src.core.footprint.Footprint."""

from __future__ import annotations

import math

import pytest
from src.core.footprint import Footprint

# ---------------------------------------------------------------------------
# Rectangle
# ---------------------------------------------------------------------------


class TestRectangle:
    def test_dimensions(self):
        f = Footprint.rectangle(90, 70)
        assert f.shape == "rectangle"
        assert f.width_mm == 90
        assert f.height_mm == 70
        assert f.bbox == (0.0, 0.0, 90.0, 70.0)
        assert math.isclose(f.area_mm2, 90 * 70)

    def test_contains(self):
        f = Footprint.rectangle(90, 70)
        assert f.contains(45, 35)  # centre
        assert f.contains(0, 0)  # corner on boundary
        assert f.contains(90, 70)  # opposite corner
        assert not f.contains(-1, 35)  # outside left
        assert not f.contains(91, 35)  # outside right
        assert not f.contains(45, 71)  # outside top

    def test_perimeter_points(self):
        f = Footprint.rectangle(10, 20)
        pts = f.perimeter_points()
        assert len(pts) == 4
        # Should contain the four corners (order may vary by CCW orientation).
        assert set(pts) == {(0.0, 0.0), (10.0, 0.0), (10.0, 20.0), (0.0, 20.0)}

    def test_signed_distance(self):
        f = Footprint.rectangle(10, 10)
        assert math.isclose(f.signed_distance(5, 5), 5.0)  # interior, distance to nearest edge
        assert math.isclose(f.signed_distance(0, 5), 0.0, abs_tol=1e-9)
        assert math.isclose(f.signed_distance(-3, 5), -3.0)


# ---------------------------------------------------------------------------
# Hexagon
# ---------------------------------------------------------------------------


class TestHexagonRegularity:
    """
    A *regular* hexagon (all 6 edges equal) requires a bbox aspect of
    sqrt(3)/2 for flat-top and 2/sqrt(3) for pointy-top. Footprint.hexagon
    itself just stretches into whatever bbox the caller gives it; the
    router layer is responsible for deriving the right y from x. These
    tests pin the math callers depend on.
    """

    def test_flat_regular_when_y_derived_from_x(self):
        # y = x * sqrt(3)/2 must yield equal edge lengths.
        x = 90.0
        y = x * math.sqrt(3.0) / 2.0
        f = Footprint.hexagon(x, y, "flat")
        pts = f.perimeter_points()
        assert len(pts) == 6
        edges = [
            math.hypot(pts[(i + 1) % 6][0] - pts[i][0], pts[(i + 1) % 6][1] - pts[i][1])
            for i in range(6)
        ]
        assert max(edges) - min(edges) < 1e-6

    def test_pointy_regular_when_y_derived_from_x(self):
        x = 90.0
        y = x * 2.0 / math.sqrt(3.0)
        f = Footprint.hexagon(x, y, "pointy")
        pts = f.perimeter_points()
        assert len(pts) == 6
        edges = [
            math.hypot(pts[(i + 1) % 6][0] - pts[i][0], pts[(i + 1) % 6][1] - pts[i][1])
            for i in range(6)
        ]
        assert max(edges) - min(edges) < 1e-6

    def test_diagonal_equals_model_size_x_pointy(self):
        # Public CLI contract: for hexagon, --model-size-x is the LONG
        # diagonal (vertex-to-opposite-vertex through centre = 2·edge).
        # For pointy-top with diag d, bbox = (d·sqrt(3)/2, d).
        diag = 90.0
        f = Footprint.hexagon(diag * math.sqrt(3.0) / 2.0, diag, "pointy")
        pts = f.perimeter_points()
        # Three long diagonals through centre.
        diagonals = [
            math.hypot(pts[i][0] - pts[i + 3][0], pts[i][1] - pts[i + 3][1]) for i in range(3)
        ]
        assert all(math.isclose(d, diag, rel_tol=1e-9) for d in diagonals)

    def test_diagonal_equals_model_size_x_flat(self):
        diag = 90.0
        f = Footprint.hexagon(diag, diag * math.sqrt(3.0) / 2.0, "flat")
        pts = f.perimeter_points()
        diagonals = [
            math.hypot(pts[i][0] - pts[i + 3][0], pts[i][1] - pts[i + 3][1]) for i in range(3)
        ]
        assert all(math.isclose(d, diag, rel_tol=1e-9) for d in diagonals)

    def test_bbox_corner_inset_pointy_hex(self):
        # Router uses signed_distance at bbox corners to inflate the border
        # so the track stays inside the inscribed polygon. For a pointy-top
        # hex with bbox (W, H=2s), the (0, 0) bbox corner sits W/4 outside
        # the hex polygon. signed_distance returns negative outside.
        diag = 90.0
        W = diag * math.sqrt(3.0) / 2.0
        H = diag
        f = Footprint.hexagon(W, H, "pointy")
        # Allow small numeric tolerance because signed_distance is from the
        # nearest segment which is a straight chord, not the analytic edge.
        d = f.signed_distance(0.0, 0.0)
        expected = -W / 4.0
        assert abs(d - expected) < 1e-6, f"got {d}, want {expected}"

    def test_bbox_corner_inset_flat_hex(self):
        # Flat-top symmetry: corner inset is H/4.
        diag = 90.0
        W = diag
        H = diag * math.sqrt(3.0) / 2.0
        f = Footprint.hexagon(W, H, "flat")
        d = f.signed_distance(0.0, 0.0)
        expected = -H / 4.0
        assert abs(d - expected) < 1e-6

    def test_square_bbox_makes_irregular_hex(self):
        # Sanity check that y == x is NOT regular (this was the bug the
        # router fix addressed).
        f = Footprint.hexagon(90.0, 90.0, "pointy")
        pts = f.perimeter_points()
        edges = [
            math.hypot(pts[(i + 1) % 6][0] - pts[i][0], pts[(i + 1) % 6][1] - pts[i][1])
            for i in range(6)
        ]
        # Edges along the long axis vs short axis differ substantially.
        assert max(edges) - min(edges) > 1.0


class TestHexagon:
    def test_flat_six_vertices(self):
        f = Footprint.hexagon(100, 100, "flat")
        assert f.shape == "hexagon"
        assert f.meta["orientation"] == "flat"
        assert len(f.perimeter_points()) == 6

    def test_pointy_six_vertices(self):
        f = Footprint.hexagon(100, 100, "pointy")
        assert f.meta["orientation"] == "pointy"
        assert len(f.perimeter_points()) == 6

    def test_bbox_matches_input(self):
        f = Footprint.hexagon(120, 80, "flat")
        min_x, min_y, max_x, max_y = f.bbox
        assert math.isclose(min_x, 0.0, abs_tol=1e-9)
        assert math.isclose(min_y, 0.0, abs_tol=1e-9)
        assert math.isclose(max_x, 120.0)
        assert math.isclose(max_y, 80.0)

    def test_centre_inside(self):
        f = Footprint.hexagon(100, 100, "flat")
        assert f.contains(50, 50)

    def test_corners_outside(self):
        # Bbox corners are outside a hexagon inscribed in the bbox.
        f = Footprint.hexagon(100, 100, "flat")
        assert not f.contains(0.1, 0.1)
        assert not f.contains(99.9, 99.9)

    def test_area_less_than_bbox(self):
        f = Footprint.hexagon(100, 100, "flat")
        assert f.area_mm2 < 100 * 100
        # Regular hex inscribed in unit square: area = 3*sqrt(3)/8 * (long-axis)^2
        # For flat-top with bbox 100x100 the hex isn't regular (stretched),
        # but area = 0.75 of bbox (verifiable analytically: 1 - 4*triangle area).
        # Triangles cut: 4 right triangles of legs 0.25*w and 0.5*h.
        # Total cut = 4 * 0.5 * 0.25*100 * 0.5*100 = 2500.
        assert math.isclose(f.area_mm2, 100 * 100 - 2500.0)

    def test_invalid_orientation(self):
        with pytest.raises(ValueError, match="orientation must be one of"):
            Footprint.hexagon(100, 100, "diagonal")

    def test_pick_orientation(self):
        assert Footprint.pick_hex_orientation(120, 80) == "flat"
        assert Footprint.pick_hex_orientation(80, 120) == "pointy"
        assert Footprint.pick_hex_orientation(100, 100) == "flat"  # tie -> flat
        assert Footprint.pick_hex_orientation(0, 0) == "flat"  # degenerate


# ---------------------------------------------------------------------------
# Circle / oval
# ---------------------------------------------------------------------------


class TestCircle:
    def test_circle_dimensions(self):
        f = Footprint.circle(100, 100, segments=128)
        assert f.shape == "circle"
        assert math.isclose(f.width_mm, 100.0)
        assert math.isclose(f.height_mm, 100.0)
        # Inscribed polygon area is slightly less than the true pi*r^2.
        true_area = math.pi * 50 * 50
        assert f.area_mm2 < true_area
        assert f.area_mm2 > 0.99 * true_area  # 128 segs is plenty close

    def test_oval(self):
        f = Footprint.circle(120, 60, segments=128)
        min_x, min_y, max_x, max_y = f.bbox
        assert math.isclose(max_x - min_x, 120.0, abs_tol=1e-6)
        assert math.isclose(max_y - min_y, 60.0, abs_tol=1e-6)

    def test_centre_inside(self):
        f = Footprint.circle(100, 100)
        assert f.contains(50, 50)

    def test_corner_outside(self):
        f = Footprint.circle(100, 100)
        assert not f.contains(1, 1)
        assert not f.contains(99, 99)

    def test_min_segments(self):
        with pytest.raises(ValueError, match="segments must be >= 12"):
            Footprint.circle(100, 100, segments=6)

    def test_segments_stored(self):
        f = Footprint.circle(50, 50, segments=64)
        assert f.meta["segments"] == 64
        # Polygon should have exactly `segments` unique vertices.
        assert len(f.perimeter_points()) == 64


# ---------------------------------------------------------------------------
# Outset (Minkowski)
# ---------------------------------------------------------------------------


class TestOutset:
    def test_rectangle_outset(self):
        f = Footprint.rectangle(100, 80).outset(10)
        assert f.shape == "rectangle"
        # Mitre buffer of a rectangle by d = larger rectangle (w+2d, h+2d).
        assert math.isclose(f.width_mm, 120.0, abs_tol=1e-6)
        assert math.isclose(f.height_mm, 100.0, abs_tol=1e-6)
        # Origin still bottom-left.
        assert math.isclose(f.bbox[0], 0.0, abs_tol=1e-6)
        assert math.isclose(f.bbox[1], 0.0, abs_tol=1e-6)

    def test_hexagon_outset_stays_hexagon(self):
        f = Footprint.hexagon(100, 100, "flat").outset(5)
        # 6 vertices preserved (mitre join, regular hex).
        assert len(f.perimeter_points()) == 6
        # bbox grew.
        assert f.width_mm > 100
        assert f.height_mm > 100

    def test_circle_outset_grows(self):
        f0 = Footprint.circle(100, 100, segments=128)
        f1 = f0.outset(10)
        # Round buffer grows roughly to diameter 120.
        assert f1.width_mm > 110
        assert f1.height_mm > 110

    def test_outset_zero_is_identity(self):
        f = Footprint.rectangle(50, 50)
        assert f.outset(0).polygon.equals(f.polygon)


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


class TestBuild:
    def test_build_rectangle(self):
        f = Footprint.build("rectangle", 90, 70)
        assert f.shape == "rectangle"

    def test_build_hexagon(self):
        f = Footprint.build("hexagon", 90, 70, hex_orientation="pointy")
        assert f.shape == "hexagon"
        assert f.meta["orientation"] == "pointy"

    def test_build_circle(self):
        f = Footprint.build("circle", 90, 90, circle_segments=64)
        assert f.shape == "circle"
        assert f.meta["segments"] == 64

    def test_build_invalid(self):
        with pytest.raises(ValueError, match="shape must be one of"):
            Footprint.build("triangle", 90, 70)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize("shape", ["rectangle", "hexagon", "circle"])
    def test_non_positive_width(self, shape):
        with pytest.raises(ValueError, match="width_mm must be > 0"):
            Footprint.build(shape, 0, 50)

    @pytest.mark.parametrize("shape", ["rectangle", "hexagon", "circle"])
    def test_non_positive_height(self, shape):
        with pytest.raises(ValueError, match="height_mm must be > 0"):
            Footprint.build(shape, 50, -1)
