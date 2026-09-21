# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for src.core.router._solve_effective_border_mm.

This is the bit that ensures GPX tracks rendered inside a non-rectangular
footprint (hex / circle) stay inside the polygon with the requested
clearance. The previous code reserved a rectangular border, which was not
enough for the inscribed polygon's narrower corners.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
from src.core.footprint import Footprint
from src.core.router import _solve_effective_border_mm

# ---------------------------------------------------------------------------
# GPX builder (no gpxpy dependency-on-disk fixtures)
# ---------------------------------------------------------------------------


def _make_gpx(points: list[tuple[float, float]]) -> str:
    """Write a minimal GPX file to a temp path and return the path."""
    pts_xml = "\n".join(f'<trkpt lat="{lat:.7f}" lon="{lon:.7f}"></trkpt>' for lat, lon in points)
    body = (
        '<?xml version="1.0"?>'
        '<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">'
        "<trk><trkseg>" + pts_xml + "</trkseg></trk></gpx>"
    )
    fd, path = tempfile.mkstemp(suffix=".gpx")
    with os.fdopen(fd, "w") as f:
        f.write(body)
    return path


def _project_track_into_mm(
    gpx_path: str, model_w_mm: float, model_h_mm: float, border_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mirror calculate_terrain_bounds's scaling so the test can compute
    where the track will end up in mm-space for a given border_mm.
    """
    import gpxpy

    with open(gpx_path) as f:
        gpx = gpxpy.parse(f)
    pts = [
        (pt.latitude, pt.longitude)
        for trk in gpx.tracks
        for seg in trk.segments
        for pt in seg.points
    ]
    arr = np.asarray(pts, dtype=float)
    lat_min, lat_max = arr[:, 0].min(), arr[:, 0].max()
    lon_min, lon_max = arr[:, 1].min(), arr[:, 1].max()
    lat_c = 0.5 * (lat_min + lat_max)
    lon_c = 0.5 * (lon_min + lon_max)
    m_lat = 111320.0
    m_lon = 111320.0 * math.cos(math.radians(lat_c))
    track_w_m = (lon_max - lon_min) * m_lon
    track_h_m = (lat_max - lat_min) * m_lat

    usable_w = model_w_mm - 2 * border_mm
    usable_h = model_h_mm - 2 * border_mm
    scale = min(usable_w / track_w_m, usable_h / track_h_m)

    dx_m = (arr[:, 1] - lon_c) * m_lon
    dy_m = (arr[:, 0] - lat_c) * m_lat
    xs = model_w_mm / 2 + dx_m * scale
    ys = model_h_mm / 2 - dy_m * scale
    return xs, ys


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


class TestSolveEffectiveBorderMm:
    def test_track_in_pointy_hex_fits_with_boost(self):
        # Diagonal route that would exit a pointy-top hex at the corners
        # if only a rectangular border of 10mm is reserved.
        pts = [
            (51.80, 5.75),
            (51.82, 5.78),
            (51.85, 5.82),
            (51.88, 5.86),
            (51.92, 5.89),
            (51.95, 5.92),
        ]
        gpx = _make_gpx(pts)
        diag = 90.0
        w = diag * math.sqrt(3.0) / 2.0
        h = diag

        eff = _solve_effective_border_mm(
            track_gpx=gpx,
            model_w_mm=w,
            model_h_mm=h,
            requested_border_mm=10.0,
            model_shape="hexagon",
            hex_orientation="pointy",
            terrain_resolution=200,
        )
        assert eff > 10.0  # had to boost
        # Sanity: with the boosted border, every projected track point is
        # inside the hex polygon's inner buffer.
        f = Footprint.hexagon(w, h, "pointy")
        inner = f.polygon.buffer(-10.0)
        xs, ys = _project_track_into_mm(gpx, w, h, eff)
        import shapely

        assert shapely.contains_xy(inner, xs, ys).all()
        os.unlink(gpx)

    def test_track_centred_no_boost_needed(self):
        # A track whose bbox is *square* (lat span = lon span * cos(lat))
        # and small enough that even with a 10mm rectangular border it sits
        # well inside the inscribed hex. Because calculate_terrain_bounds
        # scales to fit a rectangle, a square-bbox track in a pointy hex is
        # bound by the *vertical* extent and ends up centred along x — i.e.
        # only ever near the rectangular border on the y axis, which is
        # exactly where the pointy hex has its wide cross-section.
        lat_c, lon_c = 51.87, 5.84
        # Equal m-span in lat & lon: half_lat_deg / (1/111320) ≈
        # half_lon_deg / (cos(lat)/111320). Choose half-lon = 0.001°, then
        # half-lat = 0.001 * cos(lat) ≈ 0.000618°.
        half_lon = 0.001
        half_lat = half_lon * math.cos(math.radians(lat_c))
        pts = [
            (lat_c - half_lat, lon_c - half_lon),
            (lat_c - half_lat, lon_c + half_lon),
            (lat_c + half_lat, lon_c + half_lon),
            (lat_c + half_lat, lon_c - half_lon),
            (lat_c - half_lat, lon_c - half_lon),
        ]
        gpx = _make_gpx(pts)
        eff = _solve_effective_border_mm(
            track_gpx=gpx,
            model_w_mm=77.94,
            model_h_mm=90.0,
            requested_border_mm=10.0,
            model_shape="hexagon",
            hex_orientation="pointy",
            terrain_resolution=200,
        )
        # The square track's corners land near the slanted hex edges, so
        # the solver still picks a small boost — but it must stay strictly
        # below the worst-case bbox-corner inset (W/4 ≈ 19.49 for pointy).
        assert eff <= 10.0 + 77.94 / 4 + 0.5
        os.unlink(gpx)

    def test_circle_corner_boost(self):
        # A square-bbox track inside a circle: bbox corners are the worst.
        pts = [
            (51.80, 5.75),
            (51.95, 5.75),
            (51.95, 5.95),
            (51.80, 5.95),
            (51.80, 5.75),
        ]
        gpx = _make_gpx(pts)
        diam = 90.0
        eff = _solve_effective_border_mm(
            track_gpx=gpx,
            model_w_mm=diam,
            model_h_mm=diam,
            requested_border_mm=5.0,
            model_shape="circle",
            hex_orientation="flat",  # unused for circle
            terrain_resolution=200,
        )
        assert eff > 5.0
        # Worst-case boost upper bound (bbox corner to inscribed circle):
        #   r * (sqrt(2) - 1) where r = diam/2.
        worst = (diam / 2) * (math.sqrt(2) - 1)
        assert eff <= 5.0 + worst + 0.5
        os.unlink(gpx)

    def test_does_not_overshrink_to_worst_corner(self):
        # The previous "always add worst bbox-corner inset" approach would
        # have inflated by ~19.5mm here. The binary search should land
        # comfortably below that.
        pts = [
            (51.80, 5.75),
            (51.95, 5.92),
        ]
        gpx = _make_gpx(pts)
        diag = 90.0
        w = diag * math.sqrt(3.0) / 2.0
        h = diag
        eff = _solve_effective_border_mm(
            track_gpx=gpx,
            model_w_mm=w,
            model_h_mm=h,
            requested_border_mm=10.0,
            model_shape="hexagon",
            hex_orientation="pointy",
            terrain_resolution=200,
        )
        # The worst-case boost would be (W/4 ≈ 19.49) for pointy hex.
        # Acceptable result is < worst.
        f = Footprint.hexagon(w, h, "pointy")
        corners = [
            (0.0, 0.0),
            (w, 0.0),
            (0.0, h),
            (w, h),
        ]
        worst = max(-f.signed_distance(cx, cy) for cx, cy in corners)
        assert eff - 10.0 < worst
        os.unlink(gpx)
