# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Water layer over real OSM data, replayed from local fixtures.

[test_water_watertight.py] drives the mesh builders with hand-made polygons.
This file covers what only real data reaches: LOD filtering, the split into
coastlines / closed areas / linear waterways, coastline stitching into sea
polygons, and the ribbon merge — over answers Overpass actually returned for
tracks in the GPX library.

The fixtures in [data/](data/) are trimmed Overpass responses (only the ways
that survive filtering, coordinates rounded to ~10cm), so these tests need no
network, no API key and no elevation tiles: the DEM is synthetic and the
Overpass client is stubbed with the fixture. Rebuild one by re-running the
query for the bbox its `_comment` field records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from rasterio.transform import from_origin
from src.core import water_generator as water_module
from src.core.water_generator import WaterLayerGenerator

from tests.mesh_topology import (
    FloatArray,
    analyze,
    assert_watertight_shells,
    split_shells,
    triangles_of,
)

DATA_DIR = Path(__file__).parent / "data"
GRID = 80


def _fixture(name: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((DATA_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return payload


def _terrain(bbox: dict[str, float], sea_level_west: bool = False) -> FloatArray:
    """A hill; optionally with the western third below sea level."""
    y, x = np.mgrid[0:GRID, 0:GRID]
    surface = 120.0 + 600.0 * np.exp(-(((x - GRID / 2) / 22) ** 2 + ((y - GRID / 2) / 22) ** 2))
    if sea_level_west:
        surface = surface - 400.0 * np.clip(1.0 - x / (GRID / 3), 0.0, 1.0)
    del bbox
    return np.asarray(surface, dtype=np.float64)


def _generator(fixture: dict[str, Any], lod: int, elevation: FloatArray) -> WaterLayerGenerator:
    bbox = fixture["bbox"]
    transform = from_origin(
        bbox["lon_min"],
        bbox["lat_max"],
        (bbox["lon_max"] - bbox["lon_min"]) / GRID,
        (bbox["lat_max"] - bbox["lat_min"]) / GRID,
    )
    return WaterLayerGenerator(
        lat_min=bbox["lat_min"],
        lon_min=bbox["lon_min"],
        lat_max=bbox["lat_max"],
        lon_max=bbox["lon_max"],
        model_width_mm=80.0,
        model_height_mm=60.0,
        base_thickness_mm=3.0,
        scale=0.01,
        vertical_exaggeration=1.5,
        lod_level=lod,
        terrain_elevation=elevation,
        terrain_transform=transform,
        terrain_resolution=GRID,
    )


@pytest.fixture
def replay(monkeypatch: pytest.MonkeyPatch):
    """Serve one fixture to the generator in place of Overpass."""

    def _install(name: str, lod: int, *, sea_level_west: bool = False) -> WaterLayerGenerator:
        payload = _fixture(name)

        class _Stub:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            def query_water_features(self, *_args: object, **_kwargs: object) -> dict[str, Any]:
                return payload

        monkeypatch.setattr(water_module, "OverpassClient", _Stub)
        generator = _generator(payload, lod, _terrain(payload["bbox"], sea_level_west))
        generator._load_elevation_data()
        return generator

    return _install


# --------------------------------------------------------------------------
# Closed water areas — LOD 2 lakes
# --------------------------------------------------------------------------


def test_real_mountain_lakes_are_watertight(replay):
    """Two Pyrenean lakes (Estanys de Juclar / Fontargente) at LOD 2."""
    mesh = replay("pyrenees_lakes", 2).generate_water_mesh()
    assert mesh is not None, "no water mesh built from real lake polygons"

    shells = assert_watertight_shells(triangles_of(mesh), "pyrenees lakes")
    assert len(shells) >= 2, f"expected one body per lake, got {len(shells)}"


def test_real_lake_surfaces_are_flat(replay):
    """Standing water gets one elevation per lake — no terrain-following ripple."""
    mesh = replay("pyrenees_lakes", 2).generate_water_mesh()
    assert mesh is not None
    for shell in split_shells(triangles_of(mesh)):
        z = shell[:, :, 2]
        # Top and bottom faces of a flat lake: exactly WATER_THICKNESS_MM apart.
        assert z.max() - z.min() == pytest.approx(water_module.WATER_THICKNESS_MM, abs=1e-6)


# --------------------------------------------------------------------------
# Linear waterways — LOD 4 ribbons
# --------------------------------------------------------------------------


def test_real_rivers_mesh_into_watertight_ribbons(replay):
    """Nineteen Waal-floodplain waterways plus one water area, LOD 4."""
    mesh = replay("waal_waterways", 4).generate_water_mesh()
    assert mesh is not None

    shells = assert_watertight_shells(triangles_of(mesh), "waal waterways")
    assert len(shells) >= 5, f"only {len(shells)} bodies from 20 features"


def test_two_rivers_meeting_stay_watertight(replay):
    """A confluence must not tear the water shell open.

    Reduced from a real Overpass answer for the Waal at Nijmegen to the two
    ways that do it: the Spiegelwaal rejoins the Waal, the merge welds their
    ribbons, and three faces end up sharing an edge. A shell like that is what
    `--water-objects 3` and above produces wherever one river meets another.
    """
    mesh = replay("waal_confluence", 4).generate_water_mesh()
    assert mesh is not None, "the confluence produced no water at all"
    assert_watertight_shells(triangles_of(mesh), "waal confluence")


def test_river_ribbons_follow_the_terrain(replay):
    """Flowing water rides the slope; a flat result would mean lake handling."""
    mesh = replay("pyrenees_lakes", 4).generate_water_mesh()
    assert mesh is not None
    z = triangles_of(mesh)[:, :, 2]
    assert z.max() - z.min() > water_module.WATER_THICKNESS_MM * 2


# --------------------------------------------------------------------------
# Coastlines — LOD 1
# --------------------------------------------------------------------------


def test_real_coastline_becomes_a_watertight_sea(replay):
    """Seven Atlantic coastline ways near Lion's Head, stitched into a sea polygon.

    The terrain here is entirely above sea level, so the DEM fallback can
    produce nothing: any mesh at all proves the coastline path ran.
    """
    mesh = replay("capetown_coast", 1).generate_water_mesh()
    assert mesh is not None, "coastline data produced no sea"
    assert_watertight_shells(triangles_of(mesh), "cape town sea")


def test_sea_area_polygon_is_flat_and_watertight(replay):
    """`natural=sea` closed areas take the flat-ocean path, not the lake path."""
    mesh = replay("sea_polygon", 1).generate_water_mesh()
    assert mesh is not None

    assert_watertight_shells(triangles_of(mesh), "sea area")
    z = triangles_of(mesh)[:, :, 2]
    assert z.max() - z.min() == pytest.approx(water_module.WATER_THICKNESS_MM, abs=1e-6)


def test_lod1_ignores_inland_water(replay):
    """Level 1 is oceans only: lakes and rivers must not sneak in.

    With no coastline in the fixture and dry terrain, the answer is nothing —
    not a lake, and not an invented sea.
    """
    assert replay("pyrenees_lakes", 1).generate_water_mesh() is None


def test_fixtures_carry_their_provenance():
    """Each fixture says where it came from, so it can be regenerated."""
    for path in sorted(DATA_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["_comment"], path.name
        assert set(payload["bbox"]) == {"lat_min", "lat_max", "lon_min", "lon_max"}
        assert payload["elements"], path.name


def test_fixture_water_stays_inside_the_model(replay):
    """Clipping to the bbox holds for real geometry too, not just synthetic rings."""
    mesh = replay("waal_waterways", 4).generate_water_mesh()
    assert mesh is not None
    xy = triangles_of(mesh).reshape(-1, 3)[:, :2]
    assert xy[:, 0].min() >= -0.01
    assert xy[:, 0].max() <= 80.01
    assert xy[:, 1].min() >= -0.01
    assert xy[:, 1].max() <= 60.01


def test_analyze_reports_positive_volume_for_every_fixture(replay):
    """Blanket check: nothing in any fixture exports an inside-out body."""
    for name, lod in (("pyrenees_lakes", 4), ("waal_waterways", 4), ("capetown_coast", 1)):
        mesh = replay(name, lod).generate_water_mesh()
        assert mesh is not None, name
        for shell in split_shells(triangles_of(mesh)):
            assert analyze(shell).volume_mm3 > 0, name
