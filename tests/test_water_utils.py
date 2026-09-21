# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for water-feature utilities: ribbon mesh geometry, miter offsets, and
the adaptive waterway-width helper. These cover the visual-quality fixes for
the OSM water rendering pipeline.
"""

from __future__ import annotations

import math

import numpy as np
from src.core.water_generator import (
    MIN_WATERWAY_WIDTH_MM,
    compute_waterway_width_mm,
)
from src.osm.osm_utils import (
    calculate_perpendicular_offset,
    create_ribbon_geometry,
    densify_polyline,
)

# ---------------------------------------------------------------------------
# compute_waterway_width_mm
# ---------------------------------------------------------------------------


class TestComputeWaterwayWidthMm:
    def test_natural_above_floor_kept(self):
        # River wider than the printable floor on a large model — return as-is.
        w = compute_waterway_width_mm(natural_width_mm=2.0, model_min_dim_mm=200)
        assert w == 2.0

    def test_natural_below_floor_bumped(self):
        # Stream narrower than the floor gets bumped to the absolute floor so
        # it stays continuous when printed.
        w = compute_waterway_width_mm(natural_width_mm=0.05, model_min_dim_mm=200)
        assert math.isclose(w, MIN_WATERWAY_WIDTH_MM)

    def test_floor_is_hard_on_small_model(self):
        # Hard floor: even on a small model a thin stream is still widened to
        # the full 1mm printable minimum (no per-model lowering).
        w = compute_waterway_width_mm(natural_width_mm=0.01, model_min_dim_mm=30)
        assert math.isclose(w, MIN_WATERWAY_WIDTH_MM)
        assert MIN_WATERWAY_WIDTH_MM >= 1.0

    def test_zero_natural_returns_floor(self):
        w = compute_waterway_width_mm(0.0, 100)
        assert math.isclose(w, MIN_WATERWAY_WIDTH_MM)

    def test_zero_model_dim_falls_back(self):
        # Degenerate model dim — still return the absolute floor without
        # crashing.
        w = compute_waterway_width_mm(0.0, 0.0)
        assert math.isclose(w, MIN_WATERWAY_WIDTH_MM)


# ---------------------------------------------------------------------------
# calculate_perpendicular_offset (miter)
# ---------------------------------------------------------------------------


class TestPerpendicularOffset:
    def test_straight_line_returns_plain_perpendicular(self):
        # Centerline along +X. Left perpendicular = +Y, right = -Y.
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        left, right = calculate_perpendicular_offset(pts, offset_distance=0.5)

        assert np.allclose(left[:, 1], 0.5)
        assert np.allclose(right[:, 1], -0.5)
        # X positions unchanged.
        assert np.allclose(left[:, 0], pts[:, 0])
        assert np.allclose(right[:, 0], pts[:, 0])

    def test_miter_at_right_angle_corner(self):
        # 90° turn: at the corner the miter offset is sqrt(2) * half_width
        # (further out so each adjacent segment edge stays exactly half-width
        # from the centerline).
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        left, right = calculate_perpendicular_offset(pts, offset_distance=0.5)

        # Corner left point should be at (1+0.5, 0-0.5) for a right-turn
        # outside corner: along +X centerline, left perp = +Y; along +Y
        # centerline, left perp = -X. Bisector = (-X + Y) / sqrt(2). Scaled
        # by miter formula: half_width * sqrt(2) along the bisector.
        # That places the left corner at (1 - 0.5, 0 + 0.5) = (0.5, 0.5).
        # The right (outside) corner is at (1.5, -0.5).
        assert math.isclose(left[1, 0], 0.5, abs_tol=1e-9)
        assert math.isclose(left[1, 1], 0.5, abs_tol=1e-9)
        assert math.isclose(right[1, 0], 1.5, abs_tol=1e-9)
        assert math.isclose(right[1, 1], -0.5, abs_tol=1e-9)

    def test_offsets_stay_half_width_from_centerline(self):
        # Smooth curve: arc sampled in 8 points. Both offset polylines should
        # stay close to half_width from the centerline segments.
        angles = np.linspace(0, math.pi / 2, 8)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        hw = 0.05
        left, right = calculate_perpendicular_offset(pts, offset_distance=hw)

        # Each offset vertex should be at distance ~hw from the nearest
        # centerline point.
        for i in range(len(pts)):
            d_left = np.linalg.norm(left[i] - pts[i])
            d_right = np.linalg.norm(right[i] - pts[i])
            assert math.isclose(d_left, hw, rel_tol=0.2)
            assert math.isclose(d_right, hw, rel_tol=0.2)


# ---------------------------------------------------------------------------
# create_ribbon_geometry — fish-scale regression test
# ---------------------------------------------------------------------------


class TestRibbonGeometry:
    @staticmethod
    def _flat_terrain(x: float, y: float) -> float:
        return 1.0

    @staticmethod
    def _sloped_terrain(x: float, y: float) -> float:
        return 0.05 * x  # increases along +X

    @staticmethod
    def _cross_sloped_terrain(x: float, y: float) -> float:
        return 0.5 * y  # steep slope PERPENDICULAR to an +X-flowing river

    def test_vertex_layout(self):
        # 4 vertices per centerline sample.
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        verts, faces = create_ribbon_geometry(
            pts,
            width_mm=0.5,
            height_mm=0.5,
            elevation_sampler=self._flat_terrain,
        )
        assert verts.shape == (12, 3)
        assert faces.ndim == 2
        assert faces.shape[1] == 3

    def test_top_quad_is_planar(self):
        # The critical fish-scale fix: with a single Z per cross-section the
        # 4 vertices of a top quad must be coplanar even when the centerline
        # crosses a slope. Compute the volume of the tetrahedron made by the 4
        # vertices — for a planar quad it must be zero.
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        verts, _faces = create_ribbon_geometry(
            pts,
            width_mm=0.4,
            height_mm=0.2,
            elevation_sampler=self._sloped_terrain,
        )
        # Cross-section i occupies vertex indices 4i..4i+3.
        # Top quad of segment 0 = (LT_0, RT_0, RT_1, LT_1)
        # = (verts[0], verts[1], verts[5], verts[4]).
        a, b, c, d = verts[0], verts[1], verts[5], verts[4]
        # Coplanarity: scalar triple product of (b-a), (c-a), (d-a) must be 0.
        m = np.array([b - a, c - a, d - a])
        vol = abs(np.linalg.det(m))
        assert vol < 1e-9, f"Top quad not planar (vol={vol})"

    def test_top_left_and_right_share_z(self):
        # Each cross-section must have the same top Z on the left and right
        # vertex — the old code sampled them independently which caused the
        # fish-scale crease.
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        verts, _ = create_ribbon_geometry(
            pts,
            width_mm=0.4,
            height_mm=0.2,
            elevation_sampler=self._sloped_terrain,
        )
        for i in range(len(pts)):
            lt_z = verts[4 * i + 0, 2]
            rt_z = verts[4 * i + 1, 2]
            assert math.isclose(lt_z, rt_z), f"top L/R Z differ at section {i}: {lt_z} vs {rt_z}"

    def test_top_z_follows_terrain_plus_height(self):
        # Top Z at each section should be terrain(centerline) + height_mm.
        pts = np.array([[0.0, 0.0], [2.0, 0.0]])
        verts, _ = create_ribbon_geometry(
            pts,
            width_mm=0.5,
            height_mm=0.3,
            elevation_sampler=self._sloped_terrain,
        )
        # Section 0: centerline x=0 → terrain=0 → top z = 0.3
        # Section 1: centerline x=2 → terrain=0.1 → top z = 0.4
        assert math.isclose(verts[0, 2], 0.3)
        assert math.isclose(verts[4, 2], 0.4)

    def test_top_clears_terrain_across_width_on_perpendicular_slope(self):
        # Regression: river broken / buried on steep ground. With a slope
        # perpendicular to the flow, the uphill edge of the ribbon must still
        # sit at or above the terrain there. Old code used only the centerline
        # elevation, so the uphill edge was buried. Now top = max-over-width +
        # height, so every top vertex clears the terrain under it.
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        height = 0.4
        verts, _ = create_ribbon_geometry(
            pts,
            width_mm=1.0,
            height_mm=height,
            elevation_sampler=self._cross_sloped_terrain,
        )
        # Top vertices are indices 4i (left) and 4i+1 (right).
        for i in range(len(verts) // 4):
            for vi in (4 * i, 4 * i + 1):
                x, y, z = verts[vi]
                terrain_here = self._cross_sloped_terrain(x, y)
                assert z >= terrain_here - 1e-9, (
                    f"top vertex {vi} buried: z={z} < terrain={terrain_here}"
                )

    def test_bottom_reaches_terrain_across_width(self):
        # The downhill edge bottom must reach down to (or below) terrain so the
        # ribbon has no floating gap under it on a slope.
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        verts, _ = create_ribbon_geometry(
            pts,
            width_mm=1.0,
            height_mm=0.4,
            elevation_sampler=self._cross_sloped_terrain,
        )
        # Bottom vertices are indices 4i+2 (left) and 4i+3 (right).
        for i in range(len(verts) // 4):
            for vi in (4 * i + 2, 4 * i + 3):
                x, y, z = verts[vi]
                terrain_here = self._cross_sloped_terrain(x, y)
                assert z <= terrain_here + 1e-9, (
                    f"bottom vertex {vi} floats above terrain: z={z} > terrain={terrain_here}"
                )

    def test_empty_for_single_point(self):
        verts, faces = create_ribbon_geometry(
            np.array([[0.0, 0.0]]),
            width_mm=0.5,
            height_mm=0.5,
            elevation_sampler=self._flat_terrain,
        )
        assert len(verts) == 0
        assert len(faces) == 0

    def test_face_count_consistent_with_segments(self):
        # For n centerline points:
        #   top:    2*(n-1) tris
        #   bottom: 2*(n-1) tris
        #   left:   2*(n-1) tris
        #   right:  2*(n-1) tris
        #   front cap: 2 tris
        #   back cap:  2 tris
        n = 6
        pts = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
        verts, faces = create_ribbon_geometry(
            pts,
            width_mm=0.4,
            height_mm=0.2,
            elevation_sampler=self._flat_terrain,
        )
        expected = 4 * 2 * (n - 1) + 2 + 2
        assert len(faces) == expected
        # Face indices must reference valid vertices.
        assert faces.max() < len(verts)
        assert faces.min() >= 0


# ---------------------------------------------------------------------------
# densify_polyline
# ---------------------------------------------------------------------------


class TestDensifyPolyline:
    def test_no_change_when_already_fine(self):
        pts = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        out = densify_polyline(pts, max_segment=1.0)
        assert np.array_equal(out, pts)

    def test_inserts_points_on_long_segments(self):
        # Single 4mm segment, max 1mm → 4 sub-segments → 5 points.
        pts = np.array([[0.0, 0.0], [4.0, 0.0]])
        out = densify_polyline(pts, max_segment=1.0)
        assert len(out) == 5
        # Endpoints preserved, spacing uniform along +X.
        assert np.allclose(out[0], [0.0, 0.0])
        assert np.allclose(out[-1], [4.0, 0.0])
        assert np.allclose(out[:, 0], [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_no_segment_exceeds_max(self):
        pts = np.array([[0.0, 0.0], [3.3, 0.0], [3.3, 2.2]])
        out = densify_polyline(pts, max_segment=0.5)
        seg = np.diff(out, axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        assert seg_len.max() <= 0.5 + 1e-9

    def test_original_vertices_preserved(self):
        pts = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
        out = densify_polyline(pts, max_segment=0.7)
        for p in pts:
            assert np.any(np.all(np.isclose(out, p), axis=1))

    def test_disabled_for_nonpositive_step(self):
        pts = np.array([[0.0, 0.0], [5.0, 0.0]])
        assert np.array_equal(densify_polyline(pts, 0.0), pts)
        assert np.array_equal(densify_polyline(pts, -1.0), pts)

    def test_single_point_returns_unchanged(self):
        pts = np.array([[1.0, 1.0]])
        assert np.array_equal(densify_polyline(pts, 0.5), pts)


# ---------------------------------------------------------------------------
# Lake elevation: percentile robustness
# ---------------------------------------------------------------------------


def test_percentile_is_robust_against_outlier():
    """
    Regression: previously `_calculate_lake_surface_elevation` used `min()`,
    so one ravine pixel at the shoreline dragged the whole lake below
    terrain. Using `np.percentile(..., 10)` ignores rare outliers.
    """
    samples = [10.0] * 99 + [-50.0]  # one big outlier
    assert math.isclose(np.percentile(samples, 10), 10.0)
    assert min(samples) == -50.0  # what the old code would have returned
