# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Auto-calculated vertical exaggeration in `generate_terrain_stl`.

The auto path divides the available Z height by the tile's elevation range. A
perfectly flat tile makes that range zero, and the earlier "is this tile flat
enough to be a bug" guard does not catch every route to it: it only fires when
the tile also sits below 1 m of elevation, so a flat plateau at 300 m walks
straight into the division. Without a guard the result is `inf`, the 1.0-4.0
clamp lands on 4.0, and the only trace is a numpy RuntimeWarning nobody reads.

The DEMs here are constant or a plain ramp, so this file asserts nothing about
the mesh - see tests/test_watertight.py for the topology checks, which need a
DEM with real relief.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from src.core.footprint import Footprint
from src.core.mesh_generator import generate_terrain_stl

from tests.mesh_topology import FloatArray

# Same synthetic site as tests/test_watertight.py, at the same ~55 m pixel, with
# a margin so the model box never reaches a tile edge.
LAT_BOTTOM, LAT_TOP = 50.80, 50.90
LON_LEFT, LON_RIGHT = 16.65, 16.77
PIXEL_DEG = 0.0005
MARGIN_PX = 10
GRID_ROWS = int((LAT_TOP - LAT_BOTTOM) / PIXEL_DEG) + 2 * MARGIN_PX
GRID_COLS = int((LON_RIGHT - LON_LEFT) / PIXEL_DEG) + 2 * MARGIN_PX

# 30 mm of Z on a 3 mm base leaves 27 mm for relief.
MODEL_Z_MM = 30.0
BASE_MM = 3.0


def _write_dem(path: Path, elevation: FloatArray) -> Path:
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
        transform=from_origin(
            LON_LEFT - MARGIN_PX * PIXEL_DEG,
            LAT_TOP + MARGIN_PX * PIXEL_DEG,
            PIXEL_DEG,
            PIXEL_DEG,
        ),
        nodata=-9999.0,
    ) as dst:
        dst.write(elevation.astype("float32"), 1)
    return path


def _flat(elevation_m: float) -> FloatArray:
    """Zero elevation range - the case that used to divide by zero."""
    return np.full((GRID_ROWS, GRID_COLS), elevation_m, dtype=np.float64)


def _ramp(base_m: float, relief_m: float) -> FloatArray:
    """A north-south slope, so the range is exactly `relief_m`."""
    rows = np.linspace(0.0, relief_m, GRID_ROWS)
    return np.repeat(rows[:, None], GRID_COLS, axis=1) + base_m


def _auto_exaggeration(dem: Path, out: Path, capsys: pytest.CaptureFixture[str]) -> float:
    """Run the pipeline with exaggeration left to auto and return what it chose."""
    footprint = Footprint.rectangle(100.0, 80.0)
    generate_terrain_stl(
        lat_top=LAT_TOP,
        lon_left=LON_LEFT,
        lat_bottom=LAT_BOTTOM,
        lon_right=LON_RIGHT,
        map_tif=str(dem),
        output=str(out),
        track_gpx=None,
        model_width_mm=footprint.width_mm,
        model_height_mm=footprint.height_mm,
        vertical_exaggeration=None,  # the auto path under test
        model_z_height_mm=MODEL_Z_MM,
        base_thickness_mm=BASE_MM,
        track_width_mm=1.5,
        track_height_mm=1.0,
        include_water=False,
        terrain_resolution=60,
        terrain_upsample=1,
        smoothing_iterations=1,
        osm_water_detail=0,
        footprint=footprint,
    )
    printed = capsys.readouterr().out
    marker = "Auto-calculated vertical exaggeration: "
    assert marker in printed, "the auto path did not run"
    return float(printed.split(marker)[1].split("x")[0])


@pytest.mark.parametrize("elevation_m", [300.0, 1500.0])
def test_flat_tile_takes_the_exaggeration_floor(tmp_path, capsys, elevation_m):
    """A zero elevation range gives 1.0, not the top of the clamp.

    Both altitudes clear the low-and-flat guard that raises earlier, so they
    reach the division. Altitude itself is irrelevant; only the range matters.
    """
    dem = _write_dem(tmp_path / f"flat_{int(elevation_m)}.tif", _flat(elevation_m))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exaggeration = _auto_exaggeration(dem, tmp_path / "flat.obj", capsys)

    assert exaggeration == pytest.approx(1.0)
    assert not [w for w in caught if "divide by zero" in str(w.message)]


def test_tile_with_relief_still_uses_the_full_clamp(tmp_path, capsys):
    """The guard must not flatten the ordinary case to 1.0."""
    dem = _write_dem(tmp_path / "relief.tif", _ramp(300.0, 400.0))

    exaggeration = _auto_exaggeration(dem, tmp_path / "relief.obj", capsys)

    assert 1.0 < exaggeration <= 4.0, "400 m of relief on a 27 mm Z budget should exaggerate"


def test_shallow_relief_saturates_at_the_documented_cap(tmp_path, capsys):
    """AGENTS.md: auto exaggeration caps at 4.0; above that is manual only."""
    # 5 m of relief over ~11 km asks for far more than 4x to fill 27 mm.
    dem = _write_dem(tmp_path / "shallow.tif", _ramp(300.0, 5.0))

    exaggeration = _auto_exaggeration(dem, tmp_path / "shallow.obj", capsys)

    assert exaggeration == pytest.approx(4.0)
