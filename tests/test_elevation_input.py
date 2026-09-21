# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Terrain the tool refuses, flattens, or cannot find.

Every case here is a real place: a depression below sea level, a coast flat
enough to be called broken, and a tile index written on another operating
system.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from src.core.footprint import Footprint
from src.utils.srtm_scanner import find_tifs_for_bounds

from tests.test_watertight import (
    GRID_COLS,
    GRID_ROWS,
    _generate,
    _write_dem,
)


def _flat_field(value: float, relief: float = 0.0) -> np.ndarray:
    field = np.full((GRID_ROWS, GRID_COLS), value, dtype=np.float64)
    if relief:
        ramp = np.linspace(0.0, relief, GRID_COLS, dtype=np.float64)
        field += ramp[None, :]
    return field


def _z_extent(triangles) -> float:
    return float(triangles[:, :, 2].max() - triangles[:, :, 2].min())


class TestBelowSeaLevel:
    @pytest.mark.xfail(
        strict=True,
        reason="open defect: sea level reaches the water layer only as a flat 0m plateau",
    )
    def test_a_depression_keeps_its_depth(self, tmp_path):
        """The Dead Sea is 430 m down and prints as a hole, not a plate.

        Clamping every negative elevation to zero turns a depression into flat
        ground, silently: the run says so in one line among a hundred, and
        there is no flag to keep the depth.
        """
        dem = _write_dem(tmp_path / "depression.tif", _flat_field(-60.0, relief=120.0))
        groups = _generate(dem, None, tmp_path / "d.obj", Footprint.rectangle(80.0, 60.0))
        sunk = _z_extent(groups["terrain"])

        dem_up = _write_dem(tmp_path / "raised.tif", _flat_field(40.0, relief=120.0))
        groups_up = _generate(dem_up, None, tmp_path / "u.obj", Footprint.rectangle(80.0, 60.0))
        raised = _z_extent(groups_up["terrain"])

        assert sunk == pytest.approx(raised, rel=0.02), (
            "the same 120 m of relief must print the same height whether it "
            f"straddles sea level or not: {sunk:.3f} mm against {raised:.3f} mm"
        )


class TestFlatTerrain:
    def test_a_low_flat_coast_is_not_called_broken(self, tmp_path):
        """A polder, a barrier island or a salt flat is real terrain.

        The guard tests absolute metres after clipping below-sea-level ground
        to zero, so a genuinely flat coast reads as missing FABDEM data and the
        user is sent to re-download tiles that are fine.
        """
        dem = _write_dem(tmp_path / "polder.tif", _flat_field(0.2, relief=0.4))
        groups = _generate(dem, None, tmp_path / "p.obj", Footprint.rectangle(80.0, 60.0))
        assert "terrain" in groups
        assert len(groups["terrain"]) > 0


class TestTileIndexPaths:
    def test_an_index_written_on_windows_still_resolves(self, tmp_path):
        """`bounds.csv` records a relative path, and its separator is local.

        A CSV written on Windows holds `tiles\\N45E006.tif`. Joined to a data
        directory on any other system that names no file, so every tile reads
        as missing while sitting right there.
        """
        csv_path = tmp_path / "bounds.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["tif_file", "lat_min", "lat_max", "lon_min", "lon_max"])
            writer.writerow(["tiles\\N45E006_FABDEM_V1-2.tif", "45.0", "46.0", "6.0", "7.0"])

        found = find_tifs_for_bounds(45.2, 45.8, 6.2, 6.8, str(csv_path))

        assert found, "the tile covering these bounds was not found at all"
        assert all("\\" not in name for name in found), (
            f"a backslash survived into the resolved tile path: {found}"
        )


class TestTileOverlap:
    @staticmethod
    def _index(tmp_path: Path, rows: list[tuple[str, float, float, float, float]]) -> str:
        path = tmp_path / "bounds.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["tif_file", "lat_min", "lat_max", "lon_min", "lon_max"])
            for name, lat_min, lat_max, lon_min, lon_max in rows:
                writer.writerow([name, lat_min, lat_max, lon_min, lon_max])
        return str(path)

    def test_a_tile_touching_the_edge_is_not_an_overlap(self, tmp_path):
        """FABDEM tiles abut exactly, so a box ending on a tile line touches two.

        The second shares a line with the box and no area. Reading it yields a
        window zero pixels high, the merge rejects the tile as empty, and the
        whole run fails over a tile it never needed. A bounding box that stops
        at a whole degree is the ordinary case, not a corner one.
        """
        index = self._index(
            tmp_path,
            [
                ("tiles/N45E007.tif", 45.0, 46.0, 7.0, 8.0),
                ("tiles/N46E007.tif", 46.0, 47.0, 7.0, 8.0),
            ],
        )

        found = find_tifs_for_bounds(45.9, 46.0, 7.6, 7.7, index)

        assert found == ["tiles/N45E007.tif"], (
            f"a tile sharing only the line lat=46.0 was included: {found}"
        )

    def test_a_box_spanning_two_tiles_takes_both(self, tmp_path):
        """The seam case still has to work: real overlap on both sides."""
        index = self._index(
            tmp_path,
            [
                ("tiles/N45E007.tif", 45.0, 46.0, 7.0, 8.0),
                ("tiles/N46E007.tif", 46.0, 47.0, 7.0, 8.0),
            ],
        )

        found = find_tifs_for_bounds(45.9, 46.1, 7.6, 7.7, index)

        assert sorted(found) == ["tiles/N45E007.tif", "tiles/N46E007.tif"], found
