# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Watertightness of the printable body: no holes, no flipped faces, no voids.

Three layers:

1. Self-checks on a hand-built cube — a watertightness assert that cannot fail
   is worthless, so every failure mode is provoked deliberately first.
2. Unit checks on `_generate_footprint_terrain_mesh` (hexagon/circle/oval),
   fed a synthetic elevation grid — no DEM tiles, no network.
3. Integration checks that run `generate_terrain_stl` end to end over a
   synthetic GeoTIFF plus a synthetic GPX and re-parse the exported OBJ. An
   opt-in variant runs the same checks over real FABDEM tiles and a real GPX
   when an external data library is present.

The real-data variant is marked `slow` and skips wherever the tiles are
absent, which is always in CI — so anything it alone would catch is untested
there. The two properties real tiles bring that a plain synthetic grid does
not, nodata voids and a tile seam under the model box, are therefore covered
by synthetic fixtures of their own (`synthetic_gappy_dem`,
`synthetic_tile_pair`) that run everywhere.

Every exported group must be a closed solid on its own. The track is sunk into
the terrain and the slicer unions the two, but a track that is not itself a
closed manifold leaves the union at the slicer's mercy — so it gets the same
treatment as the terrain body, over a set of GPX shapes chosen to hit the
corner cases of the sweep (straight, zigzag, hairpin, retrace, self-crossing).
The water layer is checked per shell (each bay or lake is its own body) here
and unit-tested in [test_water_watertight.py].
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np
import pytest
import rasterio
import shapely
from rasterio.transform import from_origin
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
from src.core.footprint import Footprint
from src.core.mesh_generator import (
    TRACK_EMBEDDING_DEPTH_MM,
    _generate_footprint_terrain_mesh,
    generate_terrain_stl,
)

from tests.mesh_topology import (
    FloatArray,
    Triangles,
    analyze,
    assert_no_folds,
    assert_watertight,
    assert_watertight_shells,
    find_folds,
    parse_obj_groups,
    parse_obj_groups_raw,
    split_shells,
    triangles_of,
    unit_cube_triangles,
)

# Synthetic test site: a ~11 x 8 km box in the Sudetes, chosen only because the
# real-data test below uses a track from the same region.
LAT_BOTTOM, LAT_TOP = 50.80, 50.90
LON_LEFT, LON_RIGHT = 16.65, 16.77

# The real-data tests read from a library outside the repository, named by
# GORA_ROUTER_TEST_DATA. That directory holds `FABDEM_V1.2/` with its
# `bounds.csv` index, and `gpx/`. The default cannot exist, so leaving the
# variable unset skips those tests -- which is what CI does, and why every
# failure they alone would catch has a synthetic counterpart above.
REAL_DATA_ENV = "GORA_ROUTER_TEST_DATA"
REAL_DATA_DIR = Path(os.environ.get(REAL_DATA_ENV, "/nonexistent/gora-router-test-data"))
REAL_GPX_DIR = REAL_DATA_DIR / "gpx"
REAL_FABDEM_DIR = REAL_DATA_DIR / "FABDEM_V1.2"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


# One shared grid for every synthetic tile below: ~55 m pixels, coarser than
# FABDEM's 1" but fine enough to mesh, with a margin so the model box never
# reaches a tile edge.
PIXEL_DEG = 0.0005
MARGIN_PX = 10
GRID_ROWS = int((LAT_TOP - LAT_BOTTOM) / PIXEL_DEG) + 2 * MARGIN_PX
GRID_COLS = int((LON_RIGHT - LON_LEFT) / PIXEL_DEG) + 2 * MARGIN_PX
GRID_LON_ORIGIN = LON_LEFT - MARGIN_PX * PIXEL_DEG
GRID_LAT_ORIGIN = LAT_TOP + MARGIN_PX * PIXEL_DEG


def _write_dem(
    path: Path,
    elevation: FloatArray,
    *,
    lon_origin: float = GRID_LON_ORIGIN,
    lat_origin: float = GRID_LAT_ORIGIN,
) -> Path:
    """Write one float32 GeoTIFF tile in EPSG:4326 with FABDEM's nodata value."""
    height, width = elevation.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(lon_origin, lat_origin, PIXEL_DEG, PIXEL_DEG),
        nodata=-9999.0,
    ) as dst:
        dst.write(elevation.astype("float32"), 1)
    return path


def _synthetic_elevation(rows: int, cols: int) -> FloatArray:
    """A smooth hill with ridges — non-flat everywhere, so no accidental welds."""
    y, x = np.mgrid[0:rows, 0:cols]
    hill = 400.0 * np.exp(
        -(((x - cols / 2) / (cols / 4)) ** 2 + ((y - rows / 2) / (rows / 4)) ** 2)
    )
    ridges = 25.0 * np.sin(x / 6.0) * np.cos(y / 5.0)
    return np.asarray(300.0 + hill + ridges, dtype=np.float64)


@pytest.fixture(scope="module")
def synthetic_dem(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A single-tile GeoTIFF covering the test bounds with a bit of margin."""
    return _write_dem(
        tmp_path_factory.mktemp("dem") / "synthetic_fabdem.tif",
        _synthetic_elevation(GRID_ROWS, GRID_COLS),
    )


@pytest.fixture(scope="module")
def synthetic_gappy_dem(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """`synthetic_dem` with nodata voids punched into it.

    Real FABDEM tiles carry voids — water bodies, radar shadow, tile edges —
    and they land in the middle of the meshed area, not politely outside it.
    Without a fixture that has them, the nodata fill path only ever ran on the
    developer's machine, since the real-data test below skips in CI.
    """
    elevation = _synthetic_elevation(GRID_ROWS, GRID_COLS)
    y, x = np.mgrid[0:GRID_ROWS, 0:GRID_COLS]
    # A round void over the summit, a diagonal scar and a rectangular block:
    # an interior hole, an edge-touching gash and a straight-sided patch are
    # the three shapes a fill has to close.
    void = ((x - GRID_COLS * 0.5) ** 2 + (y - GRID_ROWS * 0.5) ** 2) < (
        min(GRID_ROWS, GRID_COLS) * 0.12
    ) ** 2
    void |= np.abs((x - GRID_COLS * 0.2) - (y - GRID_ROWS * 0.8)) < 3
    void[
        int(GRID_ROWS * 0.05) : int(GRID_ROWS * 0.12),
        int(GRID_COLS * 0.6) : int(GRID_COLS * 0.9),
    ] = True
    elevation[void] = -9999.0

    return _write_dem(tmp_path_factory.mktemp("gappy") / "synthetic_gappy.tif", elevation)


@pytest.fixture(scope="module")
def synthetic_tile_pair(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    """The same terrain split across two adjacent tiles, west and east.

    Exercises the multi-tile merge: the model box straddles the seam, so a
    half-pixel misalignment or a gap at the join shows up as a hole in the
    terrain body. One tile can never catch that.
    """
    elevation = _synthetic_elevation(GRID_ROWS, GRID_COLS)
    split = GRID_COLS // 2
    directory = tmp_path_factory.mktemp("tiles")
    west = _write_dem(directory / "west.tif", elevation[:, :split])
    east = _write_dem(
        directory / "east.tif",
        elevation[:, split:],
        lon_origin=GRID_LON_ORIGIN + split * PIXEL_DEG,
    )
    return [str(west), str(east)]


def _coastal_elevation(rows: int, cols: int) -> FloatArray:
    """Sea in the west, land climbing east — gives the water layer something to do."""
    y, x = np.mgrid[0:rows, 0:cols]
    return np.asarray(-50.0 + 900.0 * (x / cols) ** 1.5 + 20.0 * np.sin(y / 9.0), dtype=np.float64)


@pytest.fixture(scope="module")
def synthetic_coast_dem(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Same grid as `synthetic_dem`, but a third of it lies below sea level."""
    return _write_dem(
        tmp_path_factory.mktemp("coast") / "synthetic_coast.tif",
        _coastal_elevation(GRID_ROWS, GRID_COLS),
    )


def _write_gpx(path: Path, points: list[tuple[float, float]]) -> Path:
    """Write lat/lon pairs as a single-segment GPX track."""
    body = "\n".join(
        f'<trkpt lat="{lat:.6f}" lon="{lon:.6f}"><ele>500</ele></trkpt>' for lat, lon in points
    )
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" creator="tests">\n<trk><trkseg>\n'
        + body
        + "\n</trkseg></trk>\n</gpx>\n",
        encoding="utf-8",
    )
    return path


def _wiggle_points() -> list[tuple[float, float]]:
    """A gently curving diagonal — the ordinary case."""
    return [
        (
            LAT_BOTTOM + 0.02 + 0.06 * (i / 119),
            LON_LEFT + 0.02 + 0.07 * (i / 119) + 0.01 * math.sin(6 * i / 119),
        )
        for i in range(120)
    ]


def _straight_points() -> list[tuple[float, float]]:
    """No turns at all — every join is collinear.

    This is what used to collapse: the old builder emitted a ring on each side
    of every join, and with no turn the two coincided, so the strip between
    them was zero-area triangles.
    """
    return [(LAT_BOTTOM + 0.03, LON_LEFT + 0.02 + 0.08 * (i / 39)) for i in range(40)]


def _zigzag_points() -> list[tuple[float, float]]:
    """Alternating ~90 degree corners — every join past the old fillet threshold."""
    points = []
    for i in range(24):
        lat = LAT_BOTTOM + 0.03 + 0.02 * (i % 2)
        lon = LON_LEFT + 0.02 + 0.003 * i
        points.append((lat, lon))
    return points


def _retrace_points() -> list[tuple[float, float]]:
    """Out and back over the *identical* coordinates — an out-and-back trail.

    The second pass rebuilds rings on top of the first, so without the revisit
    nudge in the sweep the two stretches of tube weld into one non-manifold
    pinch. Real GPX libraries are full of these.
    """
    out = [(LAT_BOTTOM + 0.03, LON_LEFT + 0.02 + 0.05 * (i / 19)) for i in range(20)]
    return out + list(reversed(out))


def _hairpin_points() -> list[tuple[float, float]]:
    """Out and back along the same line — the mitre direction is undefined there."""
    out = [(LAT_BOTTOM + 0.03, LON_LEFT + 0.02 + 0.05 * (i / 19)) for i in range(20)]
    back = [(LAT_BOTTOM + 0.0305, lon) for _, lon in reversed(out)]
    return out + back


def _self_crossing_points() -> list[tuple[float, float]]:
    """A figure eight: the ribbon overlaps itself where the loops meet.

    Self-intersection is legal — the surface stays a closed manifold and the
    slicer resolves the overlap — but it must not tear the mesh open. The loop
    drifts slightly so the two passes are ~0.5mm apart in model space: a track
    that returns to the *exact* same sample points welds two surface patches
    into one vertex, which no sweep can avoid and no real GPX produces.
    """
    points = []
    for i in range(80):
        t = 2 * math.pi * i / 79
        drift = 0.0008 * i / 79
        points.append(
            (
                LAT_BOTTOM + 0.05 + 0.02 * math.sin(2 * t) + drift,
                LON_LEFT + 0.06 + 0.03 * math.sin(t),
            )
        )
    return points


# The rectangle every track test below prints on, and the scale that follows
# from it: 80mm across 0.12 degrees of longitude is ~667 mm per degree, so half
# of a 1.5mm ribbon covers ~90m on the ground. Any switchback tighter than that
# turns faster than the ribbon is wide — which is every real one.
TRACK_MODEL_W_MM, TRACK_MODEL_H_MM = 80.0, 60.0
MM_PER_DEG_LON = TRACK_MODEL_W_MM / (LON_RIGHT - LON_LEFT)
MM_PER_DEG_LAT = TRACK_MODEL_H_MM / (LAT_TOP - LAT_BOTTOM)


def _model_xy_to_latlon(x_mm: float, y_mm: float) -> tuple[float, float]:
    return (LAT_TOP - y_mm / MM_PER_DEG_LAT, LON_LEFT + x_mm / MM_PER_DEG_LON)


def _switchback_points(
    radius_mm: float, arm_mm: float = 15.0, arc_points: int = 13
) -> list[tuple[float, float]]:
    """Two straight arms joined by a U-turn of exactly `radius_mm` in model space.

    Sweeping a ribbon around a radius smaller than its own half-width used to
    turn the inner edge inside out at the tip.
    """
    centre_x, centre_y = TRACK_MODEL_W_MM / 2, TRACK_MODEL_H_MM / 2
    points = [
        _model_xy_to_latlon(centre_x - arm_mm + arm_mm * i / 10, centre_y - radius_mm)
        for i in range(10)
    ]
    points += [
        _model_xy_to_latlon(
            centre_x + radius_mm * math.cos(-math.pi / 2 + math.pi * i / (arc_points - 1)),
            centre_y + radius_mm * math.sin(-math.pi / 2 + math.pi * i / (arc_points - 1)),
        )
        for i in range(arc_points)
    ]
    points += [
        _model_xy_to_latlon(centre_x - arm_mm * i / 10, centre_y + radius_mm) for i in range(1, 11)
    ]
    return points


def _stacked_switchbacks(
    count: int = 12, radius_mm: float = 0.2, pitch_mm: float = 2.0, arm_mm: float = 20.0
) -> list[tuple[float, float]]:
    """`count` U-turns stacked `pitch_mm` apart, alternating direction.

    One switchback of any radius sweeps cleanly; a staircase of them is what a
    real mountain route is, and the arms between the turns are short enough
    that consecutive turns share cross-sections.
    """
    points: list[tuple[float, float]] = []
    x0 = TRACK_MODEL_W_MM / 2 - arm_mm / 2
    y0 = TRACK_MODEL_H_MM / 2 - count * pitch_mm / 2
    for i in range(count):
        y = y0 + i * pitch_mm
        rightwards = i % 2 == 0
        xs = [x0 + arm_mm * j / 8 for j in range(9)]
        if not rightwards:
            xs = list(reversed(xs))
        points += [_model_xy_to_latlon(x, y) for x in xs]
        centre_x, centre_y = xs[-1], y + pitch_mm / 2
        sign = 1.0 if rightwards else -1.0
        for k in range(9):
            angle = -math.pi / 2 + math.pi * k / 8
            points.append(
                _model_xy_to_latlon(
                    centre_x + sign * radius_mm * math.cos(angle),
                    centre_y + radius_mm * math.sin(angle),
                )
            )
    return points


TRACK_SHAPES = {
    "wiggle": _wiggle_points,
    "straight": _straight_points,
    "zigzag": _zigzag_points,
    "hairpin": _hairpin_points,
    "retrace": _retrace_points,
    "self-crossing": _self_crossing_points,
}


@pytest.fixture(scope="module")
def synthetic_gpx(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A wiggly diagonal track well inside the test bounds."""
    return _write_gpx(tmp_path_factory.mktemp("gpx") / "synthetic_track.gpx", _wiggle_points())


def _generate(
    dem: Path | None,
    gpx: Path | None,
    out: Path,
    footprint: Footprint,
    *,
    map_tif: str | list[str] | None = None,
    lat_bottom: float = LAT_BOTTOM,
    lat_top: float = LAT_TOP,
    lon_left: float = LON_LEFT,
    lon_right: float = LON_RIGHT,
    terrain_resolution: int = 60,
    water_detail: int = 0,
) -> dict[str, Triangles]:
    """Run the full pipeline offline and return the exported OBJ by group."""
    tiles = map_tif if map_tif is not None else str(dem)
    generate_terrain_stl(
        lat_top=lat_top,
        lon_left=lon_left,
        lat_bottom=lat_bottom,
        lon_right=lon_right,
        map_tif=tiles,
        output=str(out),
        track_gpx=None if gpx is None else str(gpx),
        model_width_mm=footprint.width_mm,
        model_height_mm=footprint.height_mm,
        vertical_exaggeration=1.5,
        model_z_height_mm=0,
        base_thickness_mm=3.0,
        track_width_mm=1.5,
        track_height_mm=1.0,
        include_water=water_detail > 0,
        terrain_resolution=terrain_resolution,
        terrain_upsample=1,
        smoothing_iterations=1,
        osm_water_detail=water_detail,  # 0 keeps Overpass out of the test
        footprint=footprint,
    )
    return parse_obj_groups(out)


# --------------------------------------------------------------------------
# 1. The checker itself must be able to fail
# --------------------------------------------------------------------------


def test_closed_cube_is_watertight():
    topo = assert_watertight(unit_cube_triangles(), "cube")
    assert topo.euler_characteristic == 2
    assert topo.volume_mm3 == pytest.approx(1.0)


def test_cube_with_missing_face_reports_a_hole():
    topo = analyze(unit_cube_triangles(missing=[2]))
    assert not topo.is_watertight
    assert topo.boundary_edges == 4  # the square rim left behind
    assert topo.euler_characteristic == 1


def test_cube_with_one_flipped_face_reports_bad_winding():
    tris = unit_cube_triangles()
    tris[0] = tris[0][::-1]
    topo = analyze(tris)
    assert topo.boundary_edges == 0  # still closed...
    assert topo.flipped_edges == 3  # ...but the winding disagrees on 3 edges
    assert not topo.is_watertight


def test_duplicated_face_reports_non_manifold_edge():
    tris = unit_cube_triangles()
    topo = analyze(np.concatenate([tris, tris[:1]]))
    assert topo.nonmanifold_edges == 3
    assert not topo.is_watertight


def test_inverted_cube_reports_negative_volume():
    topo = analyze(unit_cube_triangles()[:, ::-1])
    assert topo.volume_mm3 == pytest.approx(-1.0)
    with pytest.raises(AssertionError, match="inward-facing normals"):
        assert_watertight(unit_cube_triangles()[:, ::-1], "inverted cube")


def test_a_closed_solid_has_no_folds():
    """Right angles are not folds: a cube's edges must not trip the check."""
    folds = find_folds(unit_cube_triangles())
    assert folds.count == 0, folds.describe("cube")
    assert folds.worst_dot == pytest.approx(0.0, abs=1e-12)


def _folded_strip(scale_mm: float) -> Triangles:
    """Two quads sharing an edge, the second doubled back over the first.

    Nothing here is open, non-manifold or wrongly wound; only the two faces
    across the shared edge point at each other.
    """
    quad = [
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)],
        [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
        [(0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.9, 0.05)],
        [(0.0, 1.0, 0.0), (1.0, 0.9, 0.05), (0.0, 0.9, 0.05)],
    ]
    return np.array(quad, dtype=np.float64) * scale_mm


def test_a_folded_strip_is_reported():
    """The fold check has to fire, or it is worse than no check at all.

    This is the blind spot that let a switchback tighter than the ribbon ship
    as "watertight".
    """
    folded = _folded_strip(1.0)
    folds = find_folds(folded)

    assert folds.count == 1, folds.describe("folded strip")
    assert folds.worst_dot < -0.5
    with pytest.raises(AssertionError, match="folds back on itself"):
        assert_no_folds(folded, "folded strip")


def test_a_sub_resolution_fold_is_reported_but_not_failed():
    """A fold on faces smaller than one extrusion is noise, not a defect.

    A hairpin tighter than a tenth of the ribbon pinches down to a needle; what
    happens on faces that small cannot reach the print.
    """
    tiny = _folded_strip(0.04)
    folds = find_folds(tiny)

    assert folds.count == 0
    assert folds.sub_resolution_count == 1
    assert_no_folds(tiny, "tiny fold")


# --------------------------------------------------------------------------
# 2. Footprint terrain mesh (hexagon / circle / oval), no DEM needed
# --------------------------------------------------------------------------


FOOTPRINTS = {
    "hexagon-flat": Footprint.hexagon(80.0, 80.0 / 1.1547, "flat"),
    "hexagon-pointy": Footprint.hexagon(80.0, 80.0 * 1.1547, "pointy"),
    "circle": Footprint.circle(60.0, 60.0),
    "oval": Footprint.circle(60.0, 40.0, segments=64),
}


@pytest.fixture(params=sorted(FOOTPRINTS))
def footprint_case(request: pytest.FixtureRequest) -> tuple[str, Footprint]:
    return request.param, FOOTPRINTS[request.param]


def _footprint_mesh(footprint: Footprint, rows: int = 40, cols: int = 46):
    return _generate_footprint_terrain_mesh(
        footprint=footprint,
        processed_elevation=_synthetic_elevation(rows, cols),
        width_mm=footprint.width_mm,
        height_mm=footprint.height_mm,
        scale=0.01,
        vertical_exaggeration=1.5,
        base_thickness_mm=3.0,
    )


def test_footprint_terrain_mesh_is_watertight(footprint_case):
    name, footprint = footprint_case
    topo = assert_watertight(triangles_of(_footprint_mesh(footprint)), name)
    # Volume must at least reach the flat base slab under the footprint.
    assert topo.volume_mm3 > footprint.area_mm2 * 3.0 * 0.9, topo.describe(name)


def test_footprint_terrain_mesh_survives_a_flat_plateau(footprint_case):
    """Constant elevation makes neighbouring samples coincide in Z.

    Delaunay still triangulates in XY, but a flat top is where a naive weld or
    a degenerate-triangle bug would show up first.
    """
    name, footprint = footprint_case
    mesh = _generate_footprint_terrain_mesh(
        footprint=footprint,
        processed_elevation=np.full((40, 46), 250.0),
        width_mm=footprint.width_mm,
        height_mm=footprint.height_mm,
        scale=0.01,
        vertical_exaggeration=1.5,
        base_thickness_mm=3.0,
    )
    assert_watertight(triangles_of(mesh), f"{name}-flat")


def test_footprint_side_walls_face_outward(footprint_case):
    """Regression: perimeter walls were wound inward while top and base faced out.

    The surface stayed closed, so a hole check alone missed it — every wall
    normal pointed into the solid and each perimeter edge was traversed the
    same way by the wall and by the cap sharing it.
    """
    name, footprint = footprint_case
    tris = triangles_of(_footprint_mesh(footprint))
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])

    # Walls are the vertical faces: zero Z component in the normal.
    is_wall = np.abs(normals[:, 2]) < 1e-9
    assert is_wall.sum() >= 2 * len(footprint.perimeter_points()) - 2, "no side walls found"

    centre = np.array([footprint.width_mm / 2.0, footprint.height_mm / 2.0])
    radial = tris[is_wall].mean(axis=1)[:, :2] - centre
    outward = np.einsum("ij,ij->i", normals[is_wall][:, :2], radial)
    assert np.all(outward > 0), f"{name}: {int(np.sum(outward <= 0))} wall normals point inward"


# --------------------------------------------------------------------------
# 3. End-to-end: generate an OBJ and re-parse it
# --------------------------------------------------------------------------


EXPORT_CASES = {
    "rectangle": Footprint.rectangle(80.0, 60.0),
    "hexagon": Footprint.hexagon(80.0, 80.0 / 1.1547, "flat"),
    "circle": Footprint.circle(70.0, 70.0),
}


@pytest.mark.parametrize("case", sorted(EXPORT_CASES))
def test_exported_terrain_body_is_watertight(case, synthetic_dem, synthetic_gpx, tmp_path):
    footprint = EXPORT_CASES[case]
    groups = _generate(synthetic_dem, synthetic_gpx, tmp_path / f"{case}.obj", footprint)

    assert "terrain" in groups, f"exported groups: {sorted(groups)}"
    topo = assert_watertight(groups["terrain"], f"{case}/terrain")
    assert topo.face_count > 1000, topo.describe(case)


def test_exported_body_is_watertight_over_dem_voids(synthetic_gappy_dem, synthetic_gpx, tmp_path):
    """Nodata patches must not tear the body open — or float the track.

    A void left as -9999 would drag its corner of the mesh 9 km below the base
    slab, so this also pins the fill: every vertex stays within the model's
    own height range, and the track still rides on the surface above the hole.
    """
    groups = _generate(
        synthetic_gappy_dem, synthetic_gpx, tmp_path / "voids.obj", Footprint.rectangle(80.0, 60.0)
    )

    assert_watertight(groups["terrain"], "voids/terrain")
    assert_watertight(groups["track"], "voids/track")
    for name, tris in groups.items():
        assert np.all(np.isfinite(tris)), f"voids/{name} has non-finite vertices"
        z = tris[:, :, 2]
        assert z.min() >= -1e-6, f"voids/{name} dips below the print bed: {z.min():.3f}mm"
        assert z.max() < 200.0, f"voids/{name} spikes to {z.max():.1f}mm — nodata leaked through"


def test_exported_body_is_watertight_across_a_tile_seam(
    synthetic_tile_pair, synthetic_gpx, tmp_path
):
    """Two tiles merged under the model box must mesh as one continuous body."""
    groups = _generate(
        dem=None,
        gpx=synthetic_gpx,
        out=tmp_path / "seam.obj",
        footprint=Footprint.rectangle(80.0, 60.0),
        map_tif=synthetic_tile_pair,
    )

    assert_watertight(groups["terrain"], "seam/terrain")
    shells = split_shells(groups["terrain"])
    assert len(shells) == 1, f"tile seam split the terrain into {len(shells)} shells"
    assert_watertight(groups["track"], "seam/track")


# --------------------------------------------------------------------------
# 3b. The track ribbon — same standard, over the sweep's corner cases
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", sorted(TRACK_SHAPES))
def test_exported_track_is_watertight(shape, synthetic_dem, tmp_path):
    """Every GPX shape must sweep into closed solids.

    Checked per shell: a corner too sharp for the sweep to turn is bridged by
    separate full-width beads that overlap the tube either side, so a track
    with switchbacks ships as several overlapping bodies — the slicer unions
    them, exactly as it does the water layer's lakes.

    Regressions this pins down, all present before the tube rewrite:
      * straight — a ring was emitted on both sides of every join, so with no
        turn the two coincided and the strip between them was zero-area;
      * zigzag — the sharp-corner fillet strip was wound the opposite way from
        the segment strips, leaving hundreds of contradicting edges;
      * every shape — the end caps were sampled separately from the tube and
        sat a fraction of a millimetre off it, tearing the ends open.
    """
    gpx = _write_gpx(tmp_path / f"{shape}.gpx", TRACK_SHAPES[shape]())
    groups = _generate(
        synthetic_dem, gpx, tmp_path / f"{shape}.obj", Footprint.rectangle(80.0, 60.0)
    )

    assert "track" in groups, f"exported groups: {sorted(groups)}"
    assert_watertight_shells(groups["track"], f"{shape}/track")


@pytest.mark.parametrize("shape", ["wiggle", "straight", "self-crossing"])
def test_a_track_without_sharp_corners_stays_one_body(shape, synthetic_dem, tmp_path):
    """Beads are for corners the sweep cannot turn — nothing else may split.

    A gently curving track, a straight one and one that merely crosses itself
    must all still come out as a single continuous tube.
    """
    gpx = _write_gpx(tmp_path / f"{shape}-one.gpx", TRACK_SHAPES[shape]())
    groups = _generate(
        synthetic_dem, gpx, tmp_path / f"{shape}-one.obj", Footprint.rectangle(80.0, 60.0)
    )

    shells = split_shells(groups["track"])
    assert len(shells) == 1, f"{shape}: track split into {len(shells)} bodies"


@pytest.mark.parametrize("shape", sorted(TRACK_SHAPES))
def test_exported_track_does_not_fold_over_itself(shape, synthetic_dem, tmp_path):
    """Closed is not enough: no wall may face the wall next to it.

    A sweep whose offset outgrows its turn radius produces a mesh that passes
    every check above — closed, manifold, consistently wound, positive volume —
    while its surface doubles back through itself. The slicer meets inward
    normals there and prints a pinch or a void.
    """
    gpx = _write_gpx(tmp_path / f"{shape}-fold.gpx", TRACK_SHAPES[shape]())
    groups = _generate(
        synthetic_dem,
        gpx,
        tmp_path / f"{shape}-fold.obj",
        Footprint.rectangle(TRACK_MODEL_W_MM, TRACK_MODEL_H_MM),
    )
    assert_no_folds(groups["track"], f"{shape}/track")


@pytest.mark.parametrize("radius_mm", [5.0, 1.0, 0.75, 0.4, 0.1, 0.02])
def test_a_switchback_tighter_than_the_ribbon_still_prints(radius_mm, synthetic_dem, tmp_path):
    """The ribbon narrows through a tight U-turn instead of turning inside out.

    0.75mm is half of the 1.5mm ribbon these tests print — the radius at which
    the inner edge used to stop advancing. Below it the sweep pinches the
    cross-section down (and its height with it) rather than folding, and the
    tube stays a closed solid the whole way.
    """
    gpx = _write_gpx(tmp_path / f"switchback{radius_mm}.gpx", _switchback_points(radius_mm))
    groups = _generate(
        synthetic_dem,
        gpx,
        tmp_path / f"switchback{radius_mm}.obj",
        Footprint.rectangle(TRACK_MODEL_W_MM, TRACK_MODEL_H_MM),
    )

    assert_watertight_shells(groups["track"], f"switchback {radius_mm}mm")
    assert_no_folds(groups["track"], f"switchback {radius_mm}mm")


@pytest.mark.xfail(
    strict=True,
    reason="open defect: the swept ribbon still folds where a real track doubles back",
)
def test_stacked_switchbacks_do_not_fold(synthetic_dem, tmp_path):
    """A staircase of switchbacks must not turn the ribbon inside out.

    This fails today. It is kept because it is the only synthetic reproduction
    of a fold that real routes produce in quantity, and `strict` means it turns
    into a failure the moment someone fixes the sweep, which is the signal to
    delete this marker.

    `test_a_switchback_tighter_than_the_ribbon_still_prints` covers one U-turn
    in isolation, which is why this survived: the fold needs two turns close
    enough that the arm between them is shorter than the sweep's own reach, so
    the cross-sections of one turn meet the next. A real mountain route is
    nothing but that, and one exported from an 80 x 60 mm model of a 90 km
    alpine track carries 37 folded pairs.
    """
    gpx = _write_gpx(tmp_path / "stacked.gpx", _stacked_switchbacks())
    groups = _generate(
        synthetic_dem,
        gpx,
        tmp_path / "stacked.obj",
        Footprint.rectangle(TRACK_MODEL_W_MM, TRACK_MODEL_H_MM),
    )

    assert_watertight_shells(groups["track"], "stacked switchbacks")
    assert_no_folds(groups["track"], "stacked switchbacks")


def _uncovered_track_xy(
    groups: dict[str, Triangles], show_mm: float = 0.5, reach_mm: float = 1.0
) -> FloatArray:
    """Track positions with nothing standing `show_mm` proud of the terrain nearby.

    Clearance is measured at the track's own vertices — the terrain group is a
    grid, so interpolating it there is exact. A vertex on the ribbon's lower
    edge is *meant* to sit below the surface, so what is asserted is that every
    part of the track has a proud vertex within `reach_mm`: the ribbon shows
    somewhere on every patch of ground it crosses.
    """
    terrain = groups["terrain"].reshape(-1, 3)
    # The terrain group is a closed solid, so its base and side walls carry the
    # same XY as the surface above them. Interpolating over all of it would
    # blend the two; the highest z at each XY is the surface the track rides.
    columns, inverse = np.unique(np.round(terrain[:, :2], 6), axis=0, return_inverse=True)
    tops = np.full(len(columns), -np.inf)
    np.maximum.at(tops, inverse, terrain[:, 2])
    surface = LinearNDInterpolator(columns, tops)
    track = groups["track"].reshape(-1, 3)
    clearance = np.nan_to_num(track[:, 2] - surface(track[:, 0], track[:, 1]), nan=-np.inf)

    proud = track[clearance > show_mm][:, :2]
    if not len(proud):
        return track[:, :2]
    nearest, _ = cKDTree(proud).query(track[:, :2])
    return np.asarray(track[nearest > reach_mm][:, :2], dtype=np.float64)


@pytest.mark.parametrize("shape", ["wiggle", "zigzag", "hairpin"])
def test_the_track_shows_above_the_terrain_everywhere(shape, synthetic_dem, tmp_path):
    """The ribbon has to be visible along its whole length, corners included.

    A cross-section pinched by the curvature clamp used to lose its height with
    its width, so the ridge at every switchback tip sat *below* the hillside —
    watertight, fold-free and invisible. Beads carry full height through those
    stretches instead.
    """
    gpx = _write_gpx(tmp_path / f"{shape}-show.gpx", TRACK_SHAPES[shape]())
    groups = _generate(
        synthetic_dem, gpx, tmp_path / f"{shape}-show.obj", Footprint.rectangle(80.0, 60.0)
    )

    # The ribbon stands 1.0mm proud in these runs; half of that is the margin
    # for the terrain triangle it is sampled against being a chord of the DEM.
    buried = _uncovered_track_xy(groups)
    assert not len(buried), (
        f"{shape}: track sinks into the terrain at {len(buried)} places, "
        f"first at ({buried[0][0]:.2f}, {buried[0][1]:.2f})mm"
    )


def test_a_tight_switchback_keeps_most_of_its_volume(synthetic_dem, tmp_path):
    """Pinching the tip may not cost the ribbon the rest of its body.

    The clamp only ever shrinks geometry, so an over-eager version of it would
    quietly file the whole track down. The arms either side of the tip are far
    longer than the tip, so the two runs must land close together.
    """
    volumes = []
    for radius_mm in (5.0, 0.1):
        gpx = _write_gpx(tmp_path / f"volume{radius_mm}.gpx", _switchback_points(radius_mm))
        groups = _generate(
            synthetic_dem,
            gpx,
            tmp_path / f"volume{radius_mm}.obj",
            Footprint.rectangle(TRACK_MODEL_W_MM, TRACK_MODEL_H_MM),
        )
        volumes.append(analyze(groups["track"]).volume_mm3)

    wide, tight = volumes
    # The tight run is shorter by the tip it no longer goes around, so it may
    # not match — but it must stay the same order of ribbon.
    assert tight > 0.5 * wide, f"tight switchback lost too much body: {tight:.1f} vs {wide:.1f}mm^3"


def test_revisit_nudge_stays_invisible(synthetic_dem, tmp_path):
    """The anti-weld nudge on retraced stretches must not widen the ribbon.

    An out-and-back along one straight line lays two tubes on the same centre
    line. Separating them is allowed to move geometry by a few hundredths of a
    millimetre — a 0.4mm nozzle cannot render more — but not by a visible margin.
    """
    gpx = _write_gpx(tmp_path / "retrace.gpx", _retrace_points())
    groups = _generate(
        synthetic_dem, gpx, tmp_path / "retrace.obj", Footprint.rectangle(80.0, 60.0)
    )
    track_y = groups["track"].reshape(-1, 3)[:, 1]
    # Track runs due east, so its Y extent is exactly the ribbon width plus
    # whatever the nudge added.
    spread_mm = float(track_y.max() - track_y.min())
    assert 1.5 <= spread_mm < 1.5 + 0.15, f"ribbon spread {spread_mm:.3f}mm"


def test_track_is_a_single_connected_solid(synthetic_dem, synthetic_gpx, tmp_path):
    """One tube, not a pile of loose prisms: genus 0 and one shell."""
    groups = _generate(
        synthetic_dem, synthetic_gpx, tmp_path / "one_shell.obj", Footprint.rectangle(80.0, 60.0)
    )
    topo = analyze(groups["track"])
    assert topo.euler_characteristic == 2, topo.describe("track")


def test_track_volume_matches_its_cross_section(synthetic_dem, tmp_path):
    """Sanity on the sweep itself: volume ~ path length x cross-section area.

    Catches a tube that closed cleanly but was built at the wrong width, height
    or scale — topology alone would happily accept that.
    """
    width_mm, height_mm = 1.5, 1.0
    points = _straight_points()
    gpx = _write_gpx(tmp_path / "volume.gpx", points)
    groups = _generate(synthetic_dem, gpx, tmp_path / "volume.obj", Footprint.rectangle(80.0, 60.0))
    topo = analyze(groups["track"])

    # Flat bottom + semicircular top, sunk TRACK_EMBEDDING_DEPTH_MM into terrain.
    rise_mm = height_mm - TRACK_EMBEDDING_DEPTH_MM
    section_mm2 = math.pi * (width_mm / 2) * rise_mm / 2 + width_mm * abs(TRACK_EMBEDDING_DEPTH_MM)

    xy = groups["track"].reshape(-1, 3)[:, :2]
    span_mm = float(np.hypot(*(xy.max(axis=0) - xy.min(axis=0))))
    expected_mm3 = section_mm2 * span_mm

    # Generous band: the path follows terrain slope (longer than its XY span)
    # and the polygonal arc under-fills the true semicircle.
    assert 0.5 * expected_mm3 < topo.volume_mm3 < 1.6 * expected_mm3, (
        f"{topo.describe('track')} vs expected ~{expected_mm3:.1f}mm^3"
    )


def test_track_rides_on_the_terrain_surface(synthetic_dem, synthetic_gpx, tmp_path):
    """The ribbon stays glued to the terrain: sunk by the embedding depth, no more.

    A track floating over a dip prints as a bridge in mid-air; one that sinks
    deeper than the embedding depth disappears into the hillside.
    """
    groups = _generate(
        synthetic_dem, synthetic_gpx, tmp_path / "ride.obj", Footprint.rectangle(80.0, 60.0)
    )
    track_z = groups["track"][:, :, 2]
    terrain_z = groups["terrain"][:, :, 2]

    assert track_z.min() > terrain_z.min(), "track dips below the model base"
    # Highest track point cannot exceed the highest terrain point by more than
    # the requested track height (1.0mm here) plus export rounding.
    assert track_z.max() <= terrain_z.max() + 1.0 + 0.01


@pytest.mark.parametrize("case", sorted(EXPORT_CASES))
def test_exported_water_layer_is_watertight(
    case, synthetic_coast_dem, synthetic_gpx, tmp_path, monkeypatch
):
    """Third printable body: the ocean slab, end to end through the exporter.

    Overpass is stubbed out to return nothing, which is what the generator
    already treats as "derive the sea from the DEM instead" — so the water layer
    is exercised offline, exactly as it behaves for a coast OSM has no data for.

    Every footprint shape runs, because only the non-rectangular ones clip the
    water polygon against the model outline, and that clip is where the layer
    stopped being closed: a rectangle exercises none of it.
    """
    from src.core import water_generator as water_module

    class _NoOsm:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def query_water_features(self, *_args: object, **_kwargs: object) -> None:
            return None

    monkeypatch.setattr(water_module, "OverpassClient", _NoOsm)

    groups = _generate(
        synthetic_coast_dem,
        synthetic_gpx,
        tmp_path / f"water_{case}.obj",
        EXPORT_CASES[case],
        water_detail=1,
    )
    assert "water" in groups, f"exported groups: {sorted(groups)}"
    # A coast breaks into several bays, and each is its own printed body — so
    # the standard is per shell, not one shell for the whole layer.
    shells = assert_watertight_shells(groups["water"], f"{case}/water")
    assert len(shells) >= 1
    assert_watertight(groups["terrain"], f"{case}/terrain")
    assert_watertight(groups["track"], f"{case}/track")


def _wide_river_payload(
    footprint: Footprint, centerline_mm, width_m: int = 900
) -> dict[str, object]:
    """An Overpass answer holding one river, placed by model-space coordinates."""
    nodes = []
    for index, (x_mm, y_mm) in enumerate(centerline_mm):
        nodes.append(
            {
                "type": "node",
                "id": 100 + index,
                "lat": LAT_TOP - (y_mm / footprint.height_mm) * (LAT_TOP - LAT_BOTTOM),
                "lon": LON_LEFT + (x_mm / footprint.width_mm) * (LON_RIGHT - LON_LEFT),
            }
        )
    return {
        "elements": [
            *nodes,
            {
                "type": "way",
                "id": 500,
                "nodes": [node["id"] for node in nodes],
                "tags": {"waterway": "river", "width": str(width_m)},
            },
        ]
    }


RIVER_CASES = {
    "hexagon": (
        Footprint.hexagon(80.0, 80.0 / 1.1547, "flat"),
        [(10.0, 34.641), (40.0, 34.641), (79.0, 34.641)],
    ),
    # A radial river never overhangs a circle — the outline curves away from it
    # faster than the ribbon widens. It takes a chord running near-tangent to
    # the edge, which is the same near-parallel contact the hexagon offers along
    # a whole edge.
    # Long enough that the old 1.5mm inset, not the river, decided where it
    # stopped — the ends then sat 19.8mm off centre, where the outline has
    # already turned in under the ribbon's outer edge.
    "circle": (Footprint.circle(70.0, 70.0), [(12.0, 62.0), (35.0, 62.0), (58.0, 62.0)]),
}


@pytest.mark.parametrize("case", sorted(RIVER_CASES))
def test_a_wide_river_reaching_the_footprint_edge_exports_closed(
    case, synthetic_coast_dem, synthetic_gpx, tmp_path, monkeypatch
):
    """The exporter used to trim water triangles that left the outline.

    Cutting triangles out of a closed shell cannot do anything but open it: this
    input lost three quarters of the river's faces and came back with 184
    boundary edges and negative volume. Containment now belongs to the clip that
    builds the ribbon, so the shell arrives closed and there is nothing to trim.
    """
    from src.core import water_generator as water_module

    footprint, centerline_mm = RIVER_CASES[case]
    payload = _wide_river_payload(footprint, centerline_mm)

    class _CannedOsm:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def query_water_features(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return payload

    monkeypatch.setattr(water_module, "OverpassClient", _CannedOsm)

    groups = _generate(
        synthetic_coast_dem,
        synthetic_gpx,
        tmp_path / f"river_{case}.obj",
        footprint,
        water_detail=3,
    )
    assert "water" in groups, f"the river vanished — exported groups: {sorted(groups)}"
    assert_watertight_shells(groups["water"], f"{case}/wide river")

    xy = groups["water"].reshape(-1, 3)[:, :2]
    outside = ~shapely.contains_xy(footprint.polygon.buffer(0.01), xy[:, 0], xy[:, 1])
    assert not outside.any(), f"{int(outside.sum())}/{len(xy)} water vertices left the footprint"


def test_exported_file_faces_outward_as_written(synthetic_dem, synthetic_gpx, tmp_path):
    """Judge the OBJ in its own coordinates, the way a slicer loads it.

    Regression: the writer swaps Y and Z (model space is Z-up, OBJ is Y-up)
    without reversing the corner order. Swapping two axes is a mirror, so every
    face kept its winding in a flipped frame and the whole model shipped inside
    out — summit faces pointing down, negative volume, "flipped normals" in
    slicers. The tests above could not see it: they undo the swap first.
    """
    out = tmp_path / "raw.obj"
    _generate(synthetic_dem, synthetic_gpx, out, Footprint.rectangle(80.0, 60.0))

    for name, tris in parse_obj_groups_raw(out).items():
        for index, shell in enumerate(split_shells(tris)):
            topo = analyze(shell)
            assert topo.volume_mm3 > 0, f"{name}[{index}] is inside out — {topo.describe(name)}"

        # OBJ is Y-up: the highest faces of the terrain must point up, not down.
        if name == "terrain":
            normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            height = tris[:, :, 1]
            summit = np.all(height > height.min() + 0.9 * (height.max() - height.min()), axis=1)
            assert summit.sum() > 0
            assert np.all(normals[summit][:, 1] > 0), "summit faces point into the model"


def test_exported_terrain_body_is_watertight_without_a_track(synthetic_dem, tmp_path):
    """Terrain-only export (no GPX) must close just as well."""
    groups = _generate(
        synthetic_dem, None, tmp_path / "no_track.obj", Footprint.rectangle(80.0, 60.0)
    )
    assert set(groups) == {"terrain"}
    assert_watertight(groups["terrain"], "terrain-only")


def test_exported_terrain_base_is_flat_and_closed(synthetic_dem, synthetic_gpx, tmp_path):
    """The print bed face: one flat plane at z=0, and terrain strictly above it."""
    groups = _generate(
        synthetic_dem, synthetic_gpx, tmp_path / "base.obj", Footprint.rectangle(80.0, 60.0)
    )
    z = groups["terrain"][:, :, 2]
    assert z.min() == pytest.approx(0.0, abs=1e-6)
    assert z.max() > 3.0, "terrain never rises above the 3mm base slab"
    on_bed = np.all(np.isclose(z, 0.0, atol=1e-6), axis=1)
    assert on_bed.sum() > 0, "no faces lie on the print bed"


def test_exported_model_has_no_nan_or_infinite_vertices(synthetic_dem, synthetic_gpx, tmp_path):
    """A single NaN vertex silently deletes geometry in most slicers."""
    groups = _generate(
        synthetic_dem, synthetic_gpx, tmp_path / "finite.obj", Footprint.rectangle(80.0, 60.0)
    )
    for name, tris in groups.items():
        assert np.all(np.isfinite(tris)), f"{name} has non-finite vertices"


def test_exported_geometry_stays_inside_the_footprint(synthetic_dem, synthetic_gpx, tmp_path):
    """Nothing — terrain or track — may poke outside the requested model outline."""
    footprint = Footprint.hexagon(80.0, 80.0 / 1.1547, "flat")
    groups = _generate(synthetic_dem, synthetic_gpx, tmp_path / "inside.obj", footprint)
    min_x, min_y, max_x, max_y = footprint.bbox
    tol = 0.01  # mm, float32 export rounding
    for name, tris in groups.items():
        xy = tris.reshape(-1, 3)[:, :2]
        assert xy[:, 0].min() >= min_x - tol, f"{name} exceeds -X"
        assert xy[:, 0].max() <= max_x + tol, f"{name} exceeds +X"
        assert xy[:, 1].min() >= min_y - tol, f"{name} exceeds -Y"
        assert xy[:, 1].max() <= max_y + tol, f"{name} exceeds +Y"


# --------------------------------------------------------------------------
# 4. Real FABDEM tiles + real GPX — opt-in, skipped wherever the data is absent
# --------------------------------------------------------------------------


def _tiles_covering(bounds_csv: Path, lat_bottom, lat_top, lon_left, lon_right) -> list[str]:
    """Tile paths from a FABDEM `bounds.csv` index whose extent overlaps the box."""
    hits = []
    with bounds_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                float(row["lat_min"]) < lat_top
                and float(row["lat_max"]) > lat_bottom
                and float(row["lon_min"]) < lon_right
                and float(row["lon_max"]) > lon_left
            ):
                tile = bounds_csv.parent / row["tif_file"]
                if tile.exists():
                    hits.append(str(tile))
    return hits


# No marks here: `_real_cases` is a helper, not a test, and pytest ignores
# marks on anything it does not collect. The test below carries them.
def _real_cases(limit: int) -> list[tuple[Path, list[str], tuple[float, float, float, float]]]:
    """Up to `limit` GPX tracks from the library that the local tiles cover."""
    from src.utils.gpx_utils import get_gpx_bounds

    bounds_csv = REAL_FABDEM_DIR / "bounds.csv"
    cases = []
    for gpx in sorted(REAL_GPX_DIR.glob("*.gpx")):
        try:
            track = get_gpx_bounds(str(gpx))
        except ValueError, OSError:
            continue
        pad = 0.01
        box = (
            track["lat_min"] - pad,
            track["lat_max"] + pad,
            track["lon_min"] - pad,
            track["lon_max"] + pad,
        )
        # Keep the run small: skip continent-spanning tracks.
        if box[1] - box[0] > 0.5 or box[3] - box[2] > 0.5:
            continue
        tiles = _tiles_covering(bounds_csv, box[0], box[1], box[2], box[3])
        if tiles:
            cases.append((gpx, tiles, box))
        if len(cases) == limit:
            break
    return cases


# --------------------------------------------------------------------------
# Real routes, measured in bulk
#
# These need the external library and never run in CI. They exist because the
# synthetic cases above cannot see what a real route does: a track with
# hundreds of switchbacks folds where a hand-built one does not. They name no
# route, so nothing about where anyone walked enters this repository.
# --------------------------------------------------------------------------


def _generate_real(
    gpx: Path,
    tiles: list[str],
    box: tuple[float, float, float, float],
    tmp_path: Path,
) -> dict[str, Triangles]:
    """One real route through the full pipeline, at the size these models print."""
    lat_bottom, lat_top, lon_left, lon_right = box
    return _generate(
        dem=None,
        gpx=gpx,
        out=tmp_path / f"{gpx.stem}.obj",
        footprint=Footprint.rectangle(90.0, 70.0),
        map_tif=tiles if len(tiles) > 1 else tiles[0],
        lat_bottom=lat_bottom,
        lat_top=lat_top,
        lon_left=lon_left,
        lon_right=lon_right,
        terrain_resolution=80,
    )


REAL_SAMPLE_SIZE = 12
"""How many routes the bulk checks take. Enough to average out one odd track."""

REAL_FOLD_RATE_CEILING = 7.5
"""Folded pairs per 1000 track faces, over the whole sample.

Measured at 7.31 with the sample and resolution this test uses. A ratchet on
an open defect, not a target: it may only ever come down. Raising it to make a
change fit is the change telling you it made real models worse, which is how a
tuning pass on `MIN_SWEPT_WIDTH_FRACTION` was caught buying six per cent on
one sample while making five routes of twelve worse.
"""


@pytest.mark.slow
@pytest.mark.skipif(
    not (REAL_FABDEM_DIR / "bounds.csv").exists() or not REAL_GPX_DIR.is_dir(),
    reason=f"no tile and GPX library at ${REAL_DATA_ENV} (it lives outside the repo)",
)
def test_real_routes_export_closed_bodies(tmp_path):
    """Every group of every real model is a closed solid.

    This is the guarantee the project states, measured where it matters. It
    holds today across the sample, which the synthetic cases alone could not
    establish.
    """
    cases = _real_cases(REAL_SAMPLE_SIZE)
    if not cases:
        pytest.skip("no GPX in the library is covered by the local tiles")

    for gpx, tiles, box in cases:
        groups = _generate_real(gpx, tiles, box, tmp_path)
        for name, triangles in groups.items():
            if name == "terrain":
                assert_watertight(triangles, f"{gpx.stem}/{name}")
            else:
                assert_watertight_shells(triangles, f"{gpx.stem}/{name}")


@pytest.mark.slow
@pytest.mark.skipif(
    not (REAL_FABDEM_DIR / "bounds.csv").exists() or not REAL_GPX_DIR.is_dir(),
    reason=f"no tile and GPX library at ${REAL_DATA_ENV} (it lives outside the repo)",
)
def test_real_terrain_never_overhangs(tmp_path):
    """No terrain face tips below horizontal.

    `find_folds` reports hundreds of folded pairs in the terrain of a real
    model, every one of them within half a millimetre of the outline. They are
    not folds: the detector walks edge-adjacent faces and cannot tell an
    inside-out surface from the genuinely sharp edge where a steep slope meets
    the vertical skirt. This is the property that would actually be violated by
    a broken heightfield, and it holds: measured over 4.6 million faces in
    twelve real models, nothing overhangs. Check this, not the fold count.
    """
    cases = _real_cases(REAL_SAMPLE_SIZE)
    if not cases:
        pytest.skip("no GPX in the library is covered by the local tiles")

    for gpx, tiles, box in cases:
        terrain = _generate_real(gpx, tiles, box, tmp_path)["terrain"]
        normals = np.cross(terrain[:, 1] - terrain[:, 0], terrain[:, 2] - terrain[:, 0])
        areas = np.linalg.norm(normals, axis=1) / 2.0
        unit_z = normals[:, 2] / np.where(areas == 0.0, 1.0, 2.0 * areas)
        # The flat base points straight down; anything between that and level
        # is an overhang no printer should be asked for.
        overhang = (unit_z < -0.02) & (unit_z > -0.98)
        assert not overhang.any(), (
            f"{gpx.stem}: {int(overhang.sum())} overhanging terrain faces, "
            f"steepest nz={float(unit_z[overhang].min()):.4f}"
        )


@pytest.mark.slow
@pytest.mark.skipif(
    not (REAL_FABDEM_DIR / "bounds.csv").exists() or not REAL_GPX_DIR.is_dir(),
    reason=f"no tile and GPX library at ${REAL_DATA_ENV} (it lives outside the repo)",
)
def test_real_routes_do_not_fold_more_than_they_did(tmp_path):
    """The rate of folded geometry in the track does not climb.

    `test_stacked_switchbacks_do_not_fold` reproduces one fold; this measures
    all of them. A sweep change that looks right on a hand-built path and
    raises this number is making real models worse, which is how a tuning pass
    on `MIN_SWEPT_WIDTH_FRACTION` was caught doing nothing.
    """
    cases = _real_cases(REAL_SAMPLE_SIZE)
    if not cases:
        pytest.skip("no GPX in the library is covered by the local tiles")

    folds = faces = 0
    for gpx, tiles, box in cases:
        track = _generate_real(gpx, tiles, box, tmp_path)["track"]
        folds += find_folds(track).count
        faces += len(track)

    assert faces, "the sample produced no track geometry"
    rate = 1000.0 * folds / faces
    assert rate <= REAL_FOLD_RATE_CEILING, (
        f"folded pairs per 1000 track faces rose to {rate:.2f}, "
        f"ceiling {REAL_FOLD_RATE_CEILING} ({folds} folds over {faces} faces, "
        f"{len(cases)} routes)"
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not (REAL_FABDEM_DIR / "bounds.csv").exists() or not REAL_GPX_DIR.is_dir(),
    reason=f"no tile and GPX library at ${REAL_DATA_ENV} (it lives outside the repo)",
)
def test_real_fabdem_models_are_watertight(tmp_path):
    """Same checks over real data — nodata patches, tile seams, retraced trails.

    Several tracks rather than one: real GPX exposes shapes no synthetic path
    covers, above all out-and-back trails that revisit their own coordinates.
    """
    cases = _real_cases(limit=5)
    if not cases:
        pytest.skip("no GPX track in the library is covered by the local FABDEM tiles")

    for gpx, tiles, (lat_bottom, lat_top, lon_left, lon_right) in cases:
        groups = _generate(
            dem=None,
            gpx=gpx,
            out=tmp_path / f"{gpx.stem}.obj",
            footprint=Footprint.rectangle(80.0, 60.0),
            map_tif=tiles if len(tiles) > 1 else tiles[0],
            lat_bottom=lat_bottom,
            lat_top=lat_top,
            lon_left=lon_left,
            lon_right=lon_right,
            terrain_resolution=80,
        )
        assert_watertight(groups["terrain"], f"{gpx.name}/terrain")
        assert_watertight_shells(groups["track"], f"{gpx.name}/track")
        for name, tris in groups.items():
            assert np.all(np.isfinite(tris)), f"{gpx.name}/{name} has non-finite vertices"
