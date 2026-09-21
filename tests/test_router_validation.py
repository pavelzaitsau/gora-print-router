# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Argument validation in `src.core.router.generate_model`, plus the degenerate
track spans that reach `calculate_terrain_bounds`.

Every test here stops inside the validation block, in the tile lookup that
follows it, or in the pure geometry math, so none of them need FABDEM tiles,
network access or GPX fixtures on disk. `data_dir` points at an empty
`tmp_path` wherever a test runs far enough to look for elevation tiles, which
makes the "no elevation data" failure the deterministic end of the road rather
than a property of whatever `data/bounds.csv` happens to hold locally.

Only `TestBboxAtOrigin` passes a bbox corner of 0.0. Everywhere else the bbox
sits far from zero, because 0.0 is the exact input a past bug read as "not
specified": a test using it would fail on that bug before reaching its own
subject, and would keep passing for the wrong reason once the subject broke.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from src.core.router import generate_model
from src.utils.gpx_utils import calculate_terrain_bounds

# A bbox in the Alps: no coordinate near zero, no tile index to find it in.
AWAY_LAT_MIN, AWAY_LAT_MAX = 45.0, 46.0
AWAY_LON_MIN, AWAY_LON_MAX = 6.0, 7.0
# The centre generate_model names once the tile lookup comes up empty.
AWAY_CENTRE = r"\(45\.5000, 6\.5000\)"


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


def _generate_away_from_origin(
    tmp_path: Path,
    *,
    bbox_lat_min: float = AWAY_LAT_MIN,
    bbox_lat_max: float = AWAY_LAT_MAX,
    bbox_lon_min: float = AWAY_LON_MIN,
    bbox_lon_max: float = AWAY_LON_MAX,
    map_tif: str | list[str] | None = None,
    water_objects: int = 0,
    auto_rotate: bool = False,
    data_dir: Path | None = None,
) -> None:
    """Generate a terrain-only model from a bbox that has no zero coordinate.

    Each keyword defaults to a valid value, so a test overrides exactly the one
    it is about and the failure it asserts on comes from that override rather
    than from another argument. `data_dir` defaults to `tmp_path`, which holds
    no tile index, so a call that clears validation still ends at the tile
    lookup - that terminal FileNotFoundError is what the AWAY_CENTRE tests
    assert on when nothing invalid was passed at all.
    """
    generate_model(
        track_gpx=None,
        include_track=False,
        map_tif=map_tif,
        bbox_lat_min=bbox_lat_min,
        bbox_lat_max=bbox_lat_max,
        bbox_lon_min=bbox_lon_min,
        bbox_lon_max=bbox_lon_max,
        water_objects=water_objects,
        auto_rotate=auto_rotate,
        min_model_size_x_mm=100,
        min_model_size_y_mm=100,
        data_dir=str(data_dir if data_dir is not None else tmp_path),
    )


class TestBboxAtOrigin:
    """A bbox coordinate of 0.0 is a coordinate, not a missing argument."""

    def test_bbox_on_the_equator_is_accepted(self, tmp_path):
        # Getting as far as the tile lookup proves the bbox was seen. The
        # centre in the message is the bbox centre, so it also proves the
        # 0.0 corner reached the arithmetic rather than a default.
        with pytest.raises(FileNotFoundError, match=r"\(0\.5000, 0\.5000\)"):
            generate_model(
                track_gpx=None,
                include_track=False,
                bbox_lat_min=0.0,
                bbox_lat_max=1.0,
                bbox_lon_min=0.0,
                bbox_lon_max=1.0,
                min_model_size_x_mm=100,
                min_model_size_y_mm=100,
                data_dir=str(tmp_path),
            )

    def test_bbox_away_from_the_origin_fails_the_same_way(self, tmp_path):
        """The nudged bbox is the control: same path, same error."""
        with pytest.raises(FileNotFoundError, match=r"\(0\.5500, 0\.5500\)"):
            generate_model(
                track_gpx=None,
                include_track=False,
                bbox_lat_min=0.1,
                bbox_lat_max=1.0,
                bbox_lon_min=0.1,
                bbox_lon_max=1.0,
                min_model_size_x_mm=100,
                min_model_size_y_mm=100,
                data_dir=str(tmp_path),
            )


class TestBboxValidation:
    def test_no_track_and_no_bbox_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Either provide a GPX track file"):
            generate_model(
                track_gpx=None,
                include_track=False,
                min_model_size_x_mm=100,
                min_model_size_y_mm=100,
                data_dir=str(tmp_path),
            )

    def test_partial_bbox_is_rejected(self, tmp_path):
        """Three of four coordinates is all-or-none's failure case."""
        gpx = _write_gpx(tmp_path / "track.gpx", [(0.2, 0.2), (0.3, 0.3)])
        with pytest.raises(ValueError, match="must be specified together"):
            generate_model(
                track_gpx=str(gpx),
                bbox_lat_min=0.0,
                bbox_lat_max=1.0,
                bbox_lon_min=0.0,
                min_model_size_x_mm=100,
                min_model_size_y_mm=100,
                data_dir=str(tmp_path),
            )

    @pytest.mark.parametrize(
        ("lat_min", "lat_max"),
        [(46.0, 46.0), (47.0, 46.0)],
        ids=["equal", "inverted"],
    )
    def test_latitude_range_must_increase(self, tmp_path, lat_min, lat_max):
        """The guard is `>=`, so equal bounds are as invalid as swapped ones."""
        with pytest.raises(ValueError, match=r"bbox-lat-min .* must be less than bbox-lat-max"):
            _generate_away_from_origin(tmp_path, bbox_lat_min=lat_min, bbox_lat_max=lat_max)

    @pytest.mark.parametrize(
        ("lon_min", "lon_max"),
        [(7.0, 7.0), (8.0, 7.0)],
        ids=["equal", "inverted"],
    )
    def test_longitude_range_must_increase(self, tmp_path, lon_min, lon_max):
        with pytest.raises(ValueError, match=r"bbox-lon-min .* must be less than bbox-lon-max"):
            _generate_away_from_origin(tmp_path, bbox_lon_min=lon_min, bbox_lon_max=lon_max)

    def test_auto_rotate_with_manual_bbox_is_rejected(self, tmp_path):
        """A manual bbox fixes the orientation, so auto-rotate has no say."""
        with pytest.raises(ValueError, match="--auto-rotate cannot be used together"):
            _generate_away_from_origin(tmp_path, auto_rotate=True)


class TestWaterObjectsValidation:
    """`--water-objects` takes 0-5; both ends of that range are inclusive."""

    @pytest.mark.parametrize("level", [-1, 6])
    def test_level_outside_zero_to_five_is_rejected(self, tmp_path, level):
        with pytest.raises(ValueError, match="--water-objects must be between 0 and 5"):
            _generate_away_from_origin(tmp_path, water_objects=level)

    @pytest.mark.parametrize("level", [0, 5])
    def test_level_at_either_end_is_accepted(self, tmp_path, level):
        """Reaching the tile lookup is the proof: validation let the level by.

        Nothing here talks to Overpass. Level 5 would want it, but the run dies
        on missing elevation data long before any water layer is built.
        """
        with pytest.raises(FileNotFoundError, match=AWAY_CENTRE):
            _generate_away_from_origin(tmp_path, water_objects=level)


class TestMapTifShape:
    """`map_tif` takes one path or a list of them; the list is the multi-tile form.

    The boundary itself is type-checked. `ignore_errors` on `src.core.router`
    suppresses errors raised *inside* that module, not the signature it
    exports, so every caller outside the ratchet is held to
    `str | list[str] | None` - including these tests, which is why the
    `type: ignore` below is load-bearing and `warn_unused_ignores` proves it.

    What no type checker sees is the interior: the `isinstance` dispatch over
    the two shapes, the reassignments from the tile index (`map_tif =
    existing_tiles` / `existing_tiles[0]`), and the handoff at
    src/core/router.py:679, where narrowing to `str | list[str]` rests on a
    runtime `if map_tif is None: raise`. These three tests cover that interior
    and the messages it produces.
    """

    def test_missing_single_tile_is_named(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"nowhere\.tif"):
            _generate_away_from_origin(tmp_path, map_tif=str(tmp_path / "nowhere.tif"))

    def test_missing_tile_inside_a_list_is_named(self, tmp_path):
        """The list form is checked element by element, not rejected wholesale."""
        present = tmp_path / "present.tif"
        present.write_bytes(b"")  # this branch looks at existence and nothing else
        with pytest.raises(FileNotFoundError, match=r"absent\.tif"):
            _generate_away_from_origin(
                tmp_path, map_tif=[str(present), str(tmp_path / "absent.tif")]
            )

    def test_anything_but_a_path_or_a_list_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="map_tif must be a string or list"):
            _generate_away_from_origin(tmp_path, map_tif=42)  # type: ignore[arg-type]


def _write_tile_index(data_dir: Path, tif_file: str, size_bytes: int) -> Path:
    """A one-row `bounds.csv` covering the away bbox, plus the tile it names.

    Columns match `load_results_from_csv` in src/utils/srtm_scanner.py.
    """
    (data_dir / "bounds.csv").write_text(
        "tif_file,lat_min,lat_max,lon_min,lon_max\n"
        f"{tif_file},{AWAY_LAT_MIN - 1},{AWAY_LAT_MAX + 1},"
        f"{AWAY_LON_MIN - 1},{AWAY_LON_MAX + 1}\n",
        encoding="utf-8",
    )
    tile = data_dir / tif_file
    tile.write_bytes(b"\0" * size_bytes)
    return tile


class TestTileIndexSizeFilter:
    """Auto-detected tiles must exist *and* clear 10000 bytes.

    A truncated or placeholder download passes `os.path.exists` and then fails
    deep inside rasterio, so the tile lookup screens on size first. Only the
    auto-detect branch does this; an explicit `map_tif` is checked for
    existence alone, which is what TestMapTifShape covers.
    """

    def test_tile_below_the_size_floor_counts_as_missing(self, tmp_path):
        _write_tile_index(tmp_path, "stub.tif", size_bytes=9_000)
        with pytest.raises(FileNotFoundError, match=r"Missing FABDEM tiles.*stub\.tif"):
            _generate_away_from_origin(tmp_path, data_dir=tmp_path)

    def test_tile_above_the_size_floor_is_taken(self, tmp_path):
        """The control: same tile, one byte over the floor, and it gets used."""
        _write_tile_index(tmp_path, "stub.tif", size_bytes=10_001)
        # Reaching rasterio is the proof that the screen passed the tile on:
        # 10001 bytes of zeros are not a GeoTIFF, and only a tile that was
        # taken gets far enough to be opened and rejected as one.
        with pytest.raises(ValueError, match="Error reading elevation data"):
            _generate_away_from_origin(tmp_path, data_dir=tmp_path)


def _bounds_without_zero_division(
    gpx: Path, *, border_mm: float, model_size_mm: tuple[float, float]
) -> dict[str, float]:
    """Terrain bounds, asserting nothing divided by zero on the way.

    The two axes used to fail differently, which is why the warning matters as
    much as the return value. Longitude is scaled by `111320 * np.cos(...)`, a
    numpy float, so a zero-width track divided to `inf` with a RuntimeWarning
    and `min()` happened to pick the surviving axis. Latitude is scaled by a
    plain `int`, so a zero-height track raised ZeroDivisionError outright. Only
    one of the two was visible, and neither was intended.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bounds = calculate_terrain_bounds(
            str(gpx), border_mm=border_mm, model_size_mm=model_size_mm
        )
    divides = [str(w.message) for w in caught if "divide by zero" in str(w.message)]
    assert not divides, f"degenerate axis still divides by zero: {divides}"
    return bounds


class TestDegenerateTrackSpan:
    """A track along one meridian or parallel has zero extent in one axis."""

    def test_due_north_south_track_scales_from_latitude(self, tmp_path):
        gpx = _write_gpx(tmp_path / "meridian.gpx", [(46.0 + 0.01 * i, 7.0) for i in range(10)])
        bounds = _bounds_without_zero_division(gpx, border_mm=10.0, model_size_mm=(100.0, 100.0))

        assert bounds["track_width_m"] == 0.0
        assert bounds["track_height_m"] > 0.0
        # 80mm of usable height over the track's north-south run.
        assert bounds["scale"] == pytest.approx(80.0 / bounds["track_height_m"])
        assert bounds["terrain_width_m"] > 0.0
        assert bounds["lon_right"] > bounds["lon_left"]

    def test_due_east_west_track_scales_from_longitude(self, tmp_path):
        gpx = _write_gpx(tmp_path / "parallel.gpx", [(46.0, 7.0 + 0.01 * i) for i in range(10)])
        bounds = _bounds_without_zero_division(gpx, border_mm=10.0, model_size_mm=(100.0, 100.0))

        assert bounds["track_height_m"] == 0.0
        assert bounds["scale"] == pytest.approx(80.0 / bounds["track_width_m"])
        assert bounds["terrain_height_m"] > 0.0
        assert bounds["lat_top"] > bounds["lat_bottom"]

    @pytest.mark.parametrize(
        ("name", "points"),
        [
            ("meridian", [(46.0 + 0.01 * i, 7.0) for i in range(10)]),
            ("parallel", [(46.0, 7.0 + 0.01 * i) for i in range(10)]),
        ],
    )
    def test_degenerate_axis_survives_a_zero_border(self, tmp_path, name, points):
        """With no border there is nothing to pad the zero axis but the aspect."""
        gpx = _write_gpx(tmp_path / f"{name}.gpx", points)
        bounds = _bounds_without_zero_division(gpx, border_mm=0.0, model_size_mm=(100.0, 50.0))

        assert bounds["terrain_width_m"] > 0.0
        assert bounds["terrain_height_m"] > 0.0
        assert bounds["terrain_width_m"] == pytest.approx(bounds["terrain_height_m"] * 2.0)

    def test_single_point_track_raises_value_error(self, tmp_path):
        """Neither axis has extent, so no scale exists to fall back to."""
        gpx = _write_gpx(tmp_path / "point.gpx", [(46.0, 7.0), (46.0, 7.0)])
        with pytest.raises(ValueError, match="Track spans no distance"):
            calculate_terrain_bounds(str(gpx), border_mm=10.0, model_size_mm=(100.0, 100.0))


class TestHexagonAutoRotate:
    """A hexagon stays regular when --auto-rotate swaps the axes."""

    @staticmethod
    def _bbox_aspect(footprint) -> float:
        x_min, y_min, x_max, y_max = footprint.polygon.bounds
        return float(x_max - x_min) / float(y_max - y_min)

    def test_swapping_the_axes_keeps_the_hexagon_regular(self, tmp_path, monkeypatch):
        """The swap changes the axes; the orientation is derived before it.

        `--model-size-x` is the long diagonal, so the bounding box is derived
        from the resolved orientation. `--auto-rotate` then swaps width and
        height without re-deriving, and the polygon that reaches the mesh is a
        stretched hexagon while the log still reports the regular size.
        """
        import csv as _csv

        import numpy as _np
        import src.core.router as router

        from tests.test_watertight import _write_dem, _write_gpx

        dem = _write_dem(tmp_path / "tile.tif", _np.full((220, 260), 500.0, dtype=_np.float64))
        index = tmp_path / "bounds.csv"
        with index.open("w", newline="", encoding="utf-8") as handle:
            writer = _csv.writer(handle)
            writer.writerow(["tif_file", "lat_min", "lat_max", "lon_min", "lon_max"])
            writer.writerow([dem.name, "50.0", "51.0", "16.0", "17.0"])

        # A track far wider than it is tall, so auto-rotate wants to swap.
        gpx = _write_gpx(
            tmp_path / "wide.gpx",
            [(50.50 + 0.0004 * (i % 3), 16.30 + 0.004 * i) for i in range(40)],
        )

        captured: dict[str, object] = {}

        def _capture(*args, **kwargs):
            captured.update(kwargs)
            captured["_positional"] = args
            raise RuntimeError("stop after the footprint is decided")

        monkeypatch.setattr(router, "generate_terrain_stl", _capture)

        with pytest.raises(RuntimeError, match="stop after the footprint"):
            router.generate_model(
                track_gpx=str(gpx),
                data_dir=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                model_shape="hexagon",
                hex_orientation="pointy",
                min_model_size_x_mm=200.0,
                auto_rotate=True,
            )

        footprint = captured.get("footprint")
        assert footprint is not None, f"generate_terrain_stl saw: {sorted(captured)}"

        aspect = self._bbox_aspect(footprint)
        regular = {"pointy": 3.0**0.5 / 2.0, "flat": 2.0 / 3.0**0.5}
        assert any(abs(aspect - want) < 0.01 for want in regular.values()), (
            f"hexagon bounding box aspect {aspect:.3f} is neither "
            f"{regular['pointy']:.3f} (pointy) nor {regular['flat']:.3f} (flat): "
            "the swap left it stretched"
        )
