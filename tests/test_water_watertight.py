# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Water layer topology: every water body must be a closed solid, offline.

The water layer is not decoration — it prints. A lake, a river ribbon or the
ocean slab is extruded to `WATER_THICKNESS_MM` and has to come out of the
generator as a closed manifold with outward normals, exactly like the terrain
body and the track (see [test_watertight.py]).

Everything here drives `WaterLayerGenerator` methods directly with hand-built
features and a synthetic elevation grid, so no Overpass query is ever made —
the autouse fixture below makes any attempt a hard failure.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import shapely
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, Polygon
from src.core import water_generator as water_module
from src.core.footprint import Footprint
from src.core.water_generator import (
    FOOTPRINT_CLIP_GRID_MM,
    RIVER_EDGE_MARGIN_MM,
    WATER_THICKNESS_MM,
    PointXY,
    WaterLayerGenerator,
    _clip_to_footprint,
)
from src.osm.water_features import WaterFeature

from tests.mesh_topology import (
    FloatArray,
    Triangles,
    analyze,
    assert_watertight,
    assert_watertight_shells,
    triangles_of,
)

LAT_MIN, LAT_MAX = 50.80, 50.90
LON_MIN, LON_MAX = 16.65, 16.77
GRID = 120
MODEL_WIDTH_MM, MODEL_HEIGHT_MM = 80.0, 60.0


@pytest.fixture(autouse=True)
def _no_overpass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any Overpass call from these code paths is a bug, not a slow test."""

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("water geometry must not query Overpass")

    monkeypatch.setattr(water_module, "OverpassClient", _explode)


def _hill(rows: int = GRID, cols: int = GRID) -> FloatArray:
    y, x = np.mgrid[0:rows, 0:cols]
    hill = 300.0 + 200.0 * np.exp(-(((x - cols / 2) / 30) ** 2 + ((y - rows / 2) / 30) ** 2))
    return np.asarray(hill, dtype=np.float64)


def _coast(rows: int = GRID, cols: int = GRID) -> FloatArray:
    """Sea in the west, land rising east — a wavy coastline in between."""
    y, x = np.mgrid[0:rows, 0:cols]
    coast = -50.0 + 800.0 * (x / cols) ** 1.5 + 15.0 * np.sin(y / 7.0)
    return np.asarray(coast, dtype=np.float64)


def _generator(elevation: FloatArray, footprint: Footprint | None = None) -> WaterLayerGenerator:
    rows, cols = elevation.shape
    transform = from_origin(
        LON_MIN, LAT_MAX, (LON_MAX - LON_MIN) / cols, (LAT_MAX - LAT_MIN) / rows
    )
    generator = WaterLayerGenerator(
        lat_min=LAT_MIN,
        lon_min=LON_MIN,
        lat_max=LAT_MAX,
        lon_max=LON_MAX,
        model_width_mm=MODEL_WIDTH_MM,
        model_height_mm=MODEL_HEIGHT_MM,
        base_thickness_mm=3.0,
        scale=0.01,
        vertical_exaggeration=1.5,
        lod_level=5,
        terrain_elevation=elevation,
        terrain_transform=transform,
        terrain_resolution=rows,
        footprint=footprint,
    )
    generator._load_elevation_data()
    return generator


def _triangles(vertices: object, faces: object) -> Triangles:
    """Index a (vertices, faces) pair into the (N, 3, 3) form the checks take."""
    indexed = np.asarray(vertices, dtype=np.float64)[np.asarray(faces)]
    return np.asarray(indexed, dtype=np.float64)


# --------------------------------------------------------------------------
# Water slabs from polygons
# --------------------------------------------------------------------------


SQUARE = Polygon([(10, 10), (40, 10), (40, 35), (10, 35)])
L_SHAPE = Polygon([(10, 10), (40, 10), (40, 20), (25, 20), (25, 35), (10, 35)])
WITH_ISLAND = Polygon(
    [(10, 10), (40, 10), (40, 35), (10, 35)],
    [[(20, 18), (30, 18), (30, 27), (20, 27)]],
)
# Same square, wound the other way: OSM and shapely both hand out either
# orientation, and the slab used to be built inside out from a CW ring.
CLOCKWISE_SQUARE = Polygon([(10, 10), (10, 35), (40, 35), (40, 10)])


def _comb(teeth: int = 7) -> Polygon:
    """A deeply concave outline — the shape a real coastline actually has.

    Plain Delaunay triangulates the convex hull of the vertices, so on an
    outline like this some triangles straddle the boundary: the caps stop
    matching the walls built from the ring and the slab tears open along the
    notches. A smooth square or L never shows it.
    """
    coords = [(0.0, 0.0), (60.0, 0.0)]
    step = 60.0 / (2 * teeth)
    for i in range(2 * teeth):
        x = 60.0 - i * step
        coords.append((x, 30.0 if i % 2 == 0 else 8.0))
    coords.append((0.0, 30.0))
    return Polygon(coords)


def _star(points: int = 9) -> Polygon:
    coords = []
    for i in range(2 * points):
        radius = 25.0 if i % 2 == 0 else 9.0
        angle = np.pi * i / points
        coords.append((30.0 + radius * np.cos(angle), 30.0 + radius * np.sin(angle)))
    return Polygon(coords)


def _ragged_ring(seed: int = 0, points: int = 60, jag: float = 0.55) -> list[tuple[float, float]]:
    """A jagged closed ring, deterministic from `seed`.

    The tidy comb and star above are too well behaved to break unconstrained
    Delaunay; this one has the fine-grained raggedness of a digitised coastline
    and tears open ~30 triangles' worth of boundary without a constrained
    triangulation.
    """
    radii = 20.0 * (1.0 - jag * np.random.default_rng(seed).random(points))
    return [
        (
            30.0 + radius * np.cos(2 * np.pi * i / points),
            30.0 + radius * np.sin(2 * np.pi * i / points),
        )
        for i, radius in enumerate(radii)
    ]


COMB = _comb()
STAR = _star()
RAGGED = Polygon(_ragged_ring())
# Island sits in the comb's uninterrupted lower band, clear of the teeth.
COMB_WITH_ISLAND = Polygon(COMB.exterior.coords, [[(20, 2), (30, 2), (30, 6), (20, 6)]])

SLABS = {
    "square": (SQUARE, 2),
    "clockwise-square": (CLOCKWISE_SQUARE, 2),
    "non-convex": (L_SHAPE, 2),
    # A lake with an island is a torus: one tunnel through the slab, Euler 0.
    "with-island": (WITH_ISLAND, 0),
    "comb-coastline": (COMB, 2),
    "ragged-coastline": (RAGGED, 2),
    "star": (STAR, 2),
    "comb-with-island": (COMB_WITH_ISLAND, 0),
}


@pytest.mark.parametrize("case", sorted(SLABS))
def test_water_polygon_slab_is_watertight(case):
    polygon, expected_euler = SLABS[case]
    result = _generator(_hill())._create_water_polygon_mesh(polygon, 5.0, 4.4)
    assert result is not None

    vertices, faces, _ = result
    topo = analyze(_triangles(vertices, faces))
    assert topo.is_watertight, topo.describe(case)
    assert topo.euler_characteristic == expected_euler, topo.describe(case)
    assert topo.volume_mm3 > 0, f"slab built inside out — {topo.describe(case)}"


def test_water_slab_volume_is_area_times_thickness():
    """Guards the extrusion itself, which topology checks cannot see."""
    result = _generator(_hill())._create_water_polygon_mesh(SQUARE, 5.0, 4.4)
    assert result is not None
    vertices, faces, _ = result
    topo = analyze(_triangles(vertices, faces))
    assert topo.volume_mm3 == pytest.approx(SQUARE.area * 0.6, rel=1e-6)


def test_water_slab_clipped_to_a_hexagon_stays_watertight():
    """Non-rectangular footprints clip the polygon before meshing."""
    footprint = Footprint.hexagon(MODEL_WIDTH_MM, MODEL_WIDTH_MM / 1.1547, "flat")
    # Deliberately overhangs the hexagon's western point.
    polygon = Polygon([(-10, 20), (45, 20), (45, 45), (-10, 45)])
    result = _generator(_hill(), footprint)._create_water_polygon_mesh(polygon, 5.0, 4.4)
    assert result is not None

    vertices, faces, _ = result
    tris = _triangles(vertices, faces)
    assert_watertight(tris, "hexagon-clipped slab")
    assert tris.reshape(-1, 3)[:, 0].min() >= -0.01, "water escaped the footprint"


def _outward_normal(footprint: Footprint, start: PointXY, end: PointXY) -> FloatArray:
    """Unit vector perpendicular to a footprint edge, pointing out of the model."""
    edge = np.array(end, dtype=np.float64) - np.array(start, dtype=np.float64)
    normal = np.array([-edge[1], edge[0]]) / float(np.hypot(*edge))
    midpoint = (np.array(start, dtype=np.float64) + np.array(end, dtype=np.float64)) / 2.0
    return normal if not footprint.polygon.contains(Point(midpoint + normal * 1e-6)) else -normal


def _longest_edge(footprint: Footprint) -> tuple[PointXY, PointXY]:
    """The footprint's longest straight edge — where a graze is most likely."""
    corners = list(footprint.polygon.exterior.coords)
    edges = list(itertools.pairwise(corners))
    return max(edges, key=lambda e: float(np.hypot(e[1][0] - e[0][0], e[1][1] - e[0][1])))


def _grazing_polygon(footprint: Footprint, eps_mm: float = 1e-9) -> Polygon:
    """A water outline with one vertex `eps_mm` outside a footprint edge.

    This is the accident the DEM coastline hits and the clip could not survive.
    The vertex is close enough to the edge to be a crossing, so the overlay adds
    its own node for it, but not equal to it, so the original vertex is kept too:
    the clipped ring comes back with a ~1e-9mm edge. Shapely holds the pair
    apart, the exporter's `%.6f` cannot, and the cap triangle and wall quad
    spanning the pair collapse onto one another.

    No footprint shape is immune - aimed at a circle's chord this breaks a
    circle. What a long straight edge changes is how often it happens by
    accident: a hexagon offers 40mm of one exactly-straight line for some
    coastline vertex to graze, while a 256-segment circle bends away from any
    given chord within a millimetre. That is why the DEM sea broke on the
    hexagon and not on the circle, and why both are parametrized here.
    """
    start, end = _longest_edge(footprint)
    normal = _outward_normal(footprint, start, end)
    midpoint = (np.array(start, dtype=np.float64) + np.array(end, dtype=np.float64)) / 2.0
    inward = midpoint - normal * 18.0
    along = (np.array(end, dtype=np.float64) - np.array(start, dtype=np.float64)) / 2.0
    return Polygon(
        [
            tuple(midpoint + normal * eps_mm),  # the grazing vertex
            tuple(inward + along * 0.35),
            tuple(inward),
            tuple(inward - along * 0.35),
        ]
    )


@pytest.mark.parametrize("shape", ["hexagon", "circle"])
def test_a_water_slab_grazing_the_footprint_edge_stays_watertight(shape):
    """The clip must not hand the mesher two vertices the exporter cannot separate.

    Without the snap in `_clip_to_footprint` this comes back with four zero-area
    faces, three non-manifold edges and Euler 4 — the hexagonal sea's failure,
    reduced to the one vertex that causes it.
    """
    footprint = (
        Footprint.hexagon(MODEL_WIDTH_MM, MODEL_WIDTH_MM / 1.1547, "flat")
        if shape == "hexagon"
        else Footprint.circle(MODEL_WIDTH_MM, MODEL_WIDTH_MM)
    )
    polygon = _grazing_polygon(footprint)
    result = _generator(_hill(), footprint)._create_water_polygon_mesh(polygon, 5.0, 4.4)
    assert result is not None

    topo = analyze(_triangles(result[0], result[1]))
    assert topo.is_watertight, topo.describe(f"{shape} grazing slab")
    assert topo.euler_characteristic == 2, topo.describe(f"{shape} grazing slab")
    assert topo.volume_mm3 > 0, topo.describe(f"{shape} grazing slab")


# --------------------------------------------------------------------------
# Mask -> polygon extraction
# --------------------------------------------------------------------------


def test_adjacent_water_cells_merge_into_one_polygon():
    """Regression: neighbouring cells used to miss each other by a float ulp.

    Each cell derived its own bounds as centre +/- half a step, so touching
    rectangles differed in the last bit, `unary_union` merged nothing, and a
    solid bay came back as hundreds of loose one-cell squares — each smoothed
    into a blob and walled on all four sides.
    """
    generator = _generator(_hill())
    lats = np.linspace(LAT_MIN, LAT_MAX, 40)
    lons = np.linspace(LON_MIN, LON_MAX, 40)
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:15, 5:20] = True

    polygons = generator._extract_polygons_from_mask(mask, lats, lons)
    assert len(polygons) == 1, f"{len(polygons)} pieces instead of one merged block"
    assert not polygons[0].interiors


def test_enclosed_island_becomes_a_hole():
    """A dry cell ringed by water must punch a hole, not split the polygon."""
    generator = _generator(_hill())
    lats = np.linspace(LAT_MIN, LAT_MAX, 40)
    lons = np.linspace(LON_MIN, LON_MAX, 40)
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:20, 5:20] = True
    mask[11:14, 11:14] = False  # island

    polygons = generator._extract_polygons_from_mask(mask, lats, lons)
    assert len(polygons) == 1
    assert len(polygons[0].interiors) == 1


# --------------------------------------------------------------------------
# Ocean, lake and river surfaces
# --------------------------------------------------------------------------


def test_ocean_surface_is_watertight():
    """Sea level detection runs off the DEM alone — no OSM, no network."""
    mesh = _generator(_coast())._generate_ocean_surface()
    assert mesh is not None
    assert_watertight(triangles_of(mesh), "ocean")


def test_ocean_surface_is_one_slab_not_a_field_of_cells():
    """The coast is a single connected water body, so it meshes as one shell.

    Euler 2 per shell means the count doubles with every stray piece; the old
    per-cell extraction produced hundreds and tens of thousands of faces.
    """
    mesh = _generator(_coast())._generate_ocean_surface()
    assert mesh is not None
    topo = analyze(triangles_of(mesh))
    assert topo.euler_characteristic == 2, topo.describe("ocean")
    assert topo.face_count < 4000, topo.describe("ocean")


def test_ocean_sits_at_sea_level_with_printable_thickness():
    generator = _generator(_coast())
    mesh = generator._generate_ocean_surface()
    assert mesh is not None
    z = triangles_of(mesh)[:, :, 2]
    assert z.max() == pytest.approx(generator.base_thickness_mm)
    assert z.max() - z.min() == pytest.approx(WATER_THICKNESS_MM)


def _lake_feature() -> WaterFeature:
    ring = [
        (16.70 + 0.012 * np.cos(2 * np.pi * i / 32), 50.84 + 0.009 * np.sin(2 * np.pi * i / 32))
        for i in range(33)
    ]
    return WaterFeature(1, {"natural": "water"}, ring)


def _river_feature() -> WaterFeature:
    coords = [(16.67 + 0.004 * i, 50.82 + 0.0035 * i + 0.002 * np.sin(i)) for i in range(18)]
    return WaterFeature(2, {"waterway": "river", "width": "30"}, coords)


def test_lake_surface_is_watertight():
    result = _generator(_hill())._create_terrain_following_surface(_lake_feature())
    assert result is not None
    vertices, faces, _ = result
    assert_watertight(_triangles(vertices, faces), "lake")


def _ragged_lake_feature() -> WaterFeature:
    """The ragged-coastline shape as an OSM lake, in geographic coordinates."""
    ring = [
        (16.68 + (x - 30.0) * 0.0012, 50.845 + (y - 30.0) * 0.0009) for x, y in _ragged_ring(seed=3)
    ]
    ring.append(ring[0])
    return WaterFeature(4, {"natural": "water"}, ring)


def test_ragged_lake_surface_is_watertight():
    result = _generator(_hill())._create_terrain_following_surface(_ragged_lake_feature())
    assert result is not None
    vertices, faces, _ = result
    assert_watertight(_triangles(vertices, faces), "ragged lake")


def _fjord_feature() -> WaterFeature:
    """A lake with deep concave inlets — same trap as the comb slab above."""
    ring = []
    for i in range(24):
        lon = 16.68 + 0.0035 * i
        ring.append((lon, 50.845 + (0.006 if i % 2 == 0 else 0.0015)))
    ring.append((16.68 + 0.0035 * 23, 50.83))
    ring.append((16.68, 50.83))
    ring.append(ring[0])
    return WaterFeature(3, {"natural": "water"}, ring)


def test_concave_lake_surface_is_watertight():
    result = _generator(_hill())._create_terrain_following_surface(_fjord_feature())
    assert result is not None
    vertices, faces, _ = result
    assert_watertight(_triangles(vertices, faces), "fjord lake")


@pytest.mark.parametrize("shape", ["hexagon", "circle"])
def test_a_lake_grazing_the_footprint_edge_stays_watertight(shape):
    """The OSM lake path clips against the footprint too, and broke the same way.

    Same graze as the slab above, routed through `_create_terrain_following_surface`
    instead: this path has its own `intersection` call, its own triangulation and
    its own wall walk, so the slab's coverage says nothing about it.
    """
    footprint = (
        Footprint.hexagon(MODEL_WIDTH_MM, MODEL_HEIGHT_MM, "flat")
        if shape == "hexagon"
        else Footprint.circle(MODEL_WIDTH_MM, MODEL_HEIGHT_MM)
    )
    generator = _generator(_hill(), footprint)
    ring_mm = list(_grazing_polygon(footprint).exterior.coords)
    feature = WaterFeature(
        5, {"natural": "water"}, [generator._model_to_geo(x, y) for x, y in ring_mm]
    )

    result = generator._create_terrain_following_surface(feature)
    assert result is not None

    topo = analyze(_triangles(result[0], result[1]))
    assert topo.is_watertight, topo.describe(f"{shape} grazing lake")
    assert topo.euler_characteristic == 2, topo.describe(f"{shape} grazing lake")
    assert topo.volume_mm3 > 0, topo.describe(f"{shape} grazing lake")


def test_lake_surface_is_flat():
    """Standing water gets one elevation; only flowing water follows terrain."""
    result = _generator(_hill())._create_terrain_following_surface(_lake_feature())
    assert result is not None
    vertices, faces, _ = result
    z = _triangles(vertices, faces)[:, :, 2]
    assert z.max() - z.min() == pytest.approx(WATER_THICKNESS_MM)


def test_river_ribbon_is_watertight():
    result = _generator(_hill())._create_river_surface(_river_feature())
    assert result is not None
    assert_watertight(_triangles(result[0], result[1]), "river")


def _oblique_edge(polygon: Polygon) -> tuple[PointXY, PointXY]:
    """The first edge of an outline that is neither horizontal nor vertical.

    The river's own clip inset shares its top and bottom edges with the earlier
    geographic-bbox clip, so a graze aimed there is cut by the bbox first and
    never reaches the footprint. An oblique edge belongs to the footprint alone.
    """
    corners = list(polygon.exterior.coords)
    for start, end in itertools.pairwise(corners):
        if abs(end[0] - start[0]) > 1e-9 and abs(end[1] - start[1]) > 1e-9:
            return start, end
    raise AssertionError("outline has no oblique edge")


def test_a_river_grazing_the_footprint_inset_is_dropped_not_meshed():
    """The third clip site: the river clips its centerline, not a ring.

    A centerline that dips outside the inset and touches back within a hair of
    it comes back doubled at the touch, and the ribbon then lays two
    cross-sections on one spot. A sharp crossing gives the two different miters
    and they survive; this shallow touch leaves them parallel, so they weld
    together and take the ribbon's walls with them.

    What the touch leaves inside the inset is a fragment ~2e-5mm long, so the
    snap collapses it and the river is dropped. That is the whole answer here:
    the assertion is that nothing is meshed, not that something closed is - a
    fragment that short has no printable form to come out closed in.
    """
    footprint = Footprint.hexagon(MODEL_WIDTH_MM, MODEL_HEIGHT_MM, "flat")
    generator = _generator(_hill(), footprint)

    inset = footprint.polygon.buffer(-RIVER_EDGE_MARGIN_MM)
    start, end = _oblique_edge(inset)
    along = np.array(end, dtype=np.float64) - np.array(start, dtype=np.float64)
    along /= float(np.hypot(*along))
    outward = np.array([-along[1], along[0]])
    midpoint = (np.array(start, dtype=np.float64) + np.array(end, dtype=np.float64)) / 2.0
    if inset.contains(Point(midpoint + outward * 1e-6)):
        outward = -outward

    # Out 3mm, back to within 1e-6mm of the boundary, out again.
    centerline_mm = [
        midpoint + outward * 3.0 - along * 0.5,
        midpoint - outward * 1e-6,
        midpoint + outward * 3.0 + along * 0.5,
    ]

    feature = WaterFeature(
        6,
        {"waterway": "river", "width": "5"},
        [generator._model_to_geo(x, y) for x, y in centerline_mm],
    )
    assert generator._create_river_surface(feature) is None, (
        "a sub-grid touch was meshed into a ribbon instead of being dropped"
    )


def test_the_footprint_clip_leaves_no_pair_the_exporter_cannot_separate():
    """The guarantee the three clip sites are built on, asserted on its own.

    Every consumer downstream — cap triangulation, wall walk, ribbon cross-
    sections — assumes consecutive boundary vertices are far enough apart to
    survive `%.6f`. This is the one place that promise is stated directly, so a
    future change to the snap has something to fail against.
    """
    footprint = Footprint.hexagon(MODEL_WIDTH_MM, MODEL_HEIGHT_MM, "flat")
    grazing = [
        _grazing_polygon(footprint).exterior,
        _grazing_polygon(footprint, eps_mm=1e-13).exterior,
    ]
    for outline in grazing:
        clipped = _clip_to_footprint(LineString(outline.coords), footprint.polygon)
        for part in getattr(clipped, "geoms", [clipped]):
            gaps = [math.dist(a, b) for a, b in itertools.pairwise(part.coords)]
            assert all(gap >= FOOTPRINT_CLIP_GRID_MM for gap in gaps), (
                f"clip left a {min(gaps):.3e}mm edge, below the "
                f"{FOOTPRINT_CLIP_GRID_MM:g}mm the exporter needs"
            )


def test_a_river_wider_than_its_edge_margin_stays_inside_the_footprint():
    """The ribbon has to fit, not just the centerline it is swept along.

    The centerline used to be clipped to a fixed 1.5mm inset and then widened by
    half the ribbon width on each side, so any river wider than 3mm hung over
    the outline. The exporter answered that by deleting the triangles that stuck
    out, which is the one thing that cannot be done to a closed shell.
    """
    footprint = Footprint.hexagon(MODEL_WIDTH_MM, MODEL_HEIGHT_MM, "flat")
    generator = _generator(_hill(), footprint)
    # 900m at scale 0.01 is a 9mm ribbon — six times the old 1.5mm inset. It
    # runs the full width, so both ends reach the hexagon's points, where a
    # 4.5mm half-width overhangs the two edges closing on them.
    centerline_mm = [(1.0, 30.0), (40.0, 30.0), (79.0, 30.0)]
    feature = WaterFeature(
        7,
        {"waterway": "river", "width": "900"},
        [generator._model_to_geo(x, y) for x, y in centerline_mm],
    )

    result = generator._create_river_surface(feature)
    assert result is not None, "a river across the middle of the model must survive"

    tris = _triangles(result[0], result[1])
    assert_watertight(tris, "wide river")
    xy = tris.reshape(-1, 3)[:, :2]
    outside = ~shapely.contains_xy(footprint.polygon.buffer(0.01), xy[:, 0], xy[:, 1])
    assert not outside.any(), f"{int(outside.sum())}/{len(xy)} river vertices left the footprint"


def _stub_overpass(monkeypatch: pytest.MonkeyPatch, payload: object) -> None:
    """Point the generator at a canned Overpass answer instead of the network."""

    class _Stub:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def query_water_features(self, *_args: object, **_kwargs: object) -> object:
            return payload

    monkeypatch.setattr(water_module, "OverpassClient", _Stub)


# A pond far too small to survive LOD 1 filtering (which keeps oceans only).
TINY_POND = {
    "elements": [
        {"type": "node", "id": 1, "lat": 50.849, "lon": 16.699},
        {"type": "node", "id": 2, "lat": 50.849, "lon": 16.700},
        {"type": "node", "id": 3, "lat": 50.850, "lon": 16.700},
        {"type": "node", "id": 4, "lat": 50.849, "lon": 16.699},
        {"type": "way", "id": 10, "nodes": [1, 2, 3, 4], "tags": {"natural": "water"}},
    ]
}


@pytest.mark.parametrize("payload", [None, {}, {"elements": []}, TINY_POND], ids=str)
def test_lod1_falls_back_to_the_dem_when_osm_offers_no_sea(payload, monkeypatch):
    """`--water-objects 1` is "sea from the DEM"; OSM is only a nicer shoreline.

    It used to bail out whenever OSM answered but nothing survived filtering, so
    a coast Overpass knows nothing about (or a run with no connectivity) came
    out with no sea at all.
    """
    generator = _generator(_coast())
    generator.lod_level = 1
    _stub_overpass(monkeypatch, payload)

    mesh = generator.generate_water_mesh()
    assert mesh is not None, "LOD 1 produced no sea from a DEM that has one"
    assert_watertight_shells(triangles_of(mesh), "ocean fallback")


@pytest.mark.parametrize("payload", [None, TINY_POND], ids=str)
def test_higher_lods_do_not_invent_a_coastline(payload, monkeypatch):
    """LOD >= 2 is inland water, which the terrain cannot infer — so: nothing."""
    generator = _generator(_coast())
    generator.lod_level = 3
    _stub_overpass(monkeypatch, payload)

    assert generator.generate_water_mesh() is None


def test_river_follows_the_terrain():
    """A river is not flat: its top rides the slope it flows down."""
    result = _generator(_hill())._create_river_surface(_river_feature())
    assert result is not None
    z = _triangles(result[0], result[1])[:, :, 2]
    assert z.max() - z.min() > WATER_THICKNESS_MM * 2
