# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What a failed run leaves behind, and what it tells the person who ran it.

None of this changes a model. All of it decides whether a stranger can work
out what went wrong and whether their disk is the same afterwards.
"""

from __future__ import annotations

import csv
import os
import warnings
from pathlib import Path

import pytest
from src.osm.overpass_client import OverpassClient
from src.utils.gpx_utils import parse_gpx_track
from src.utils.srtm_scanner import load_results_from_csv


class TestNothingIsLeftBehind:
    def test_a_run_that_cannot_find_tiles_creates_no_output_directory(self, tmp_path):
        """The output directory is made before the tiles are resolved.

        Every failed attempt leaves an empty directory next to the user's GPX,
        named after the track or after a hash of the bounding box.
        """
        from src.core.router import generate_model

        out = tmp_path / "out"
        with pytest.raises((FileNotFoundError, ValueError)):
            generate_model(
                track_gpx=None,
                include_track=False,
                output_dir=str(out),
                data_dir=str(tmp_path / "no-tiles-here"),
                bbox_lat_min=45.0,
                bbox_lat_max=45.2,
                bbox_lon_min=6.0,
                bbox_lon_max=6.2,
                min_model_size_x_mm=80.0,
                min_model_size_y_mm=60.0,
            )

        leftovers = list(out.rglob("*")) if out.exists() else []
        assert not leftovers, f"the failed run left {[p.name for p in leftovers]} behind"

    def test_an_interrupted_export_leaves_no_temporary_file(self, tmp_path, monkeypatch):
        """The OBJ is written to a temp file and renamed, which is right.

        Nothing removes that temp file when the write fails, so a full disk or
        a Ctrl-C leaves `tmp*.obj.tmp` beside the model for good.
        """
        import numpy as np
        from src.utils.obj_exporter import Mesh, OBJExporter

        data = np.zeros(1, dtype=[("vectors", np.float32, (3, 3))])
        data["vectors"][0] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        exporter = OBJExporter()
        exporter.add_mesh(Mesh(data), "terrain")

        real_replace = os.replace

        def _fail(*args, **kwargs):
            raise OSError("no space left on device")

        monkeypatch.setattr(os, "replace", _fail)
        with pytest.raises(OSError, match="no space left"):
            exporter.save(str(tmp_path / "model.obj"))
        monkeypatch.setattr(os, "replace", real_replace)

        strays = list(tmp_path.glob("*.tmp"))
        assert not strays, f"temporary files survived the failure: {[p.name for p in strays]}"


class TestCacheDirectory:
    def test_no_cache_directory_when_caching_is_off(self, tmp_path):
        """`--no-osm-cache` should not create the directory it will not use.

        The constructor makes it unconditionally, so a read-only working
        directory raises `PermissionError` from a constructor, before any
        query is attempted.
        """
        target = tmp_path / "osm_cache"
        OverpassClient(cache_dir=str(target), use_cache=False)
        assert not target.exists(), "the cache directory was created despite --no-osm-cache"


class TestFailuresNameTheirCause:
    def test_an_unreadable_gpx_is_not_reported_as_invalid_gpx(self, tmp_path):
        """A directory, a permission error and bad XML are different problems.

        All three currently arrive as "Invalid GPX file", which sends the
        reader to check their file format when the file may be fine.
        """
        a_directory = tmp_path / "not-a-file.gpx"
        a_directory.mkdir()

        with pytest.raises(OSError, match=r".") as failure:
            parse_gpx_track(str(a_directory))

        assert "Invalid GPX" not in str(failure.value), (
            f"a directory was reported as bad GPX: {failure.value}"
        )

    def test_a_damaged_index_is_reported_not_swallowed(self, tmp_path, capsys):
        """A corrupt `bounds.csv` reads as "no elevation data for these bounds".

        That sends the user to re-run the scanner, which is right by accident.
        A malformed index and an index that simply misses the area are not the
        same thing and should not read the same.
        """
        index = tmp_path / "bounds.csv"
        with index.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["tif_file", "lat_min", "lat_max", "lon_min", "lon_max"])
            writer.writerow(["tiles/x.tif", "not-a-number", "46.0", "6.0", "7.0"])

        with pytest.raises(ValueError, match=r"bounds\.csv"):
            load_results_from_csv(str(index))


class TestNumpyDeprecations:
    def test_the_water_layer_raises_no_numpy_deprecation(self, monkeypatch):
        """`np.cross` on 2-vectors is deprecated and will be removed.

        The suite emits tens of thousands of these. When NumPy drops the call,
        the water layer stops computing triangle areas.
        """
        import io
        import json
        from contextlib import redirect_stdout

        import numpy as np
        import src.core.water_generator as water_module
        from rasterio.transform import from_origin
        from src.core.water_generator import WaterLayerGenerator

        payload = json.loads(Path("tests/data/waal_confluence.json").read_text(encoding="utf-8"))
        bbox = payload["bbox"]
        grid = 60

        class _Stub:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def query_water_features(self, *args: object, **kwargs: object) -> dict[str, object]:
                return dict(payload)

        rows, cols = np.mgrid[0:grid, 0:grid]
        surface = 5.0 + 8.0 * np.sin(cols / grid * 3.0) + 4.0 * np.cos(rows / grid * 2.0)
        transform = from_origin(
            bbox["lon_min"],
            bbox["lat_max"],
            (bbox["lon_max"] - bbox["lon_min"]) / grid,
            (bbox["lat_max"] - bbox["lat_min"]) / grid,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            monkeypatch.setattr(water_module, "OverpassClient", _Stub)
            if True:
                generator = WaterLayerGenerator(
                    lat_min=bbox["lat_min"],
                    lon_min=bbox["lon_min"],
                    lat_max=bbox["lat_max"],
                    lon_max=bbox["lon_max"],
                    model_width_mm=80.0,
                    model_height_mm=60.0,
                    base_thickness_mm=3.0,
                    scale=0.01,
                    vertical_exaggeration=1.5,
                    lod_level=4,
                    terrain_elevation=np.asarray(surface, dtype=np.float64),
                    terrain_transform=transform,
                    terrain_resolution=grid,
                )
                with redirect_stdout(io.StringIO()):
                    generator._load_elevation_data()
                    generator.generate_water_mesh()

        deprecated = [w for w in caught if "2-dimensional vectors" in str(w.message)]
        assert not deprecated, (
            f"{len(deprecated)} NumPy deprecation warnings from the water layer; "
            "np.cross on 2-vectors is scheduled for removal"
        )


class TestOutputPath:
    def test_a_root_output_directory_is_not_silently_swapped(self, tmp_path):
        """`--output /` must not quietly become the current directory.

        Stripping the trailing separator empties a bare root, and the fallback
        that stops `makedirs("")` then points the model at wherever the shell
        happens to be.
        """
        from src.core.router import _resolve_output_dir

        assert _resolve_output_dir("/", None) == "/"
        assert _resolve_output_dir(None, "track.gpx") == "."
        assert _resolve_output_dir(None, str(Path("sub") / "track.gpx")) == "sub"


class TestRasterHandles:
    def test_a_tile_that_fails_to_open_does_not_leak_the_earlier_ones(self, tmp_path, monkeypatch):
        """The tiles were opened in a comprehension, outside the try.

        Fail on the third and the first two stay open for the life of the
        process. A model over a seam opens several tiles, and a partly
        downloaded library hits exactly this.
        """
        from typing import Any

        import numpy as np
        import rasterio
        from src.core.footprint import Footprint

        from tests.test_watertight import GRID_COLS, GRID_ROWS, _generate, _write_dem

        tiles = [
            str(_write_dem(tmp_path / f"t{i}.tif", np.full((GRID_ROWS, GRID_COLS), 100.0)))
            for i in range(3)
        ]

        opened: list[Any] = []
        real_open = rasterio.open
        seen = {"n": 0}

        def _flaky(path: Any, *args: Any, **kwargs: Any) -> Any:
            seen["n"] += 1
            if seen["n"] == 3:
                raise OSError("tile is truncated")
            handle = real_open(path, *args, **kwargs)
            opened.append(handle)
            return handle

        monkeypatch.setattr("rasterio.open", _flaky)

        with pytest.raises((OSError, ValueError)):
            _generate(
                dem=None,
                gpx=None,
                out=tmp_path / "m.obj",
                footprint=Footprint.rectangle(80.0, 60.0),
                map_tif=tiles,
            )

        still_open = [h for h in opened if not h.closed]
        assert not still_open, f"{len(still_open)} raster handles were left open"


class TestErrorMessages:
    def test_a_tile_failure_is_reported_once(self, tmp_path, monkeypatch):
        """The merge wrapper sat inside the read wrapper.

        Anything that went wrong while merging arrived as "Error reading
        elevation data: Error merging tiles: <cause>", burying the cause behind
        two layers of the same sentence.
        """
        from typing import Any

        import numpy as np
        import rasterio
        from src.core.footprint import Footprint

        from tests.test_watertight import GRID_COLS, GRID_ROWS, _generate, _write_dem

        tiles = [
            str(_write_dem(tmp_path / f"e{i}.tif", np.full((GRID_ROWS, GRID_COLS), 100.0)))
            for i in range(2)
        ]

        real_open = rasterio.open

        class _Exploding:
            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def __getattr__(self, name: str) -> Any:
                if name == "read":
                    raise RuntimeError("the tile header is nonsense")
                return getattr(self._inner, name)

        monkeypatch.setattr("rasterio.open", lambda p, *a, **k: _Exploding(real_open(p, *a, **k)))

        with pytest.raises(ValueError, match="nonsense") as failure:
            _generate(
                dem=None,
                gpx=None,
                out=tmp_path / "m.obj",
                footprint=Footprint.rectangle(80.0, 60.0),
                map_tif=tiles,
            )

        message = str(failure.value)
        assert message.count("Error ") <= 1, f"the cause is wrapped twice: {message}"


class TestAdvertisedThresholds:
    def test_the_config_exposes_no_threshold_it_does_not_apply(self):
        """A run prints `W_min=...m (minimum printable waterway width)`.

        Nothing filters on it. `filter_by_lod` tests `min_river_width_m` and
        nothing else, so a reader watching the log concludes that narrower
        waterways were dropped when they were not. The printable floor that is
        real is `MIN_WATERWAY_WIDTH_MM`, and it widens a ribbon rather than
        discarding it, which is the better behaviour: a river network with gaps
        is worse than one drawn slightly too wide.
        """
        from src.osm.water_features import WaterFilterConfig

        config = WaterFilterConfig(
            bbox_area_km2=100.0, model_width_mm=80.0, bbox_width_km=10.0, lod_level=4
        )

        assert not hasattr(config, "w_min"), (
            "w_min is computed and printed as a threshold but filters nothing"
        )


class TestNetworkPatience:
    def test_a_refused_connection_is_not_waited_out(self, monkeypatch):
        """Every mirror gets the full query timeout plus thirty seconds.

        A server that refuses the connection outright is known bad in
        milliseconds, but the client waits the same 210 seconds for it as for
        one that is thinking. With two mirrors that is seven minutes before the
        run gives up, after the terrain is built and thrown away.
        """
        from typing import Any

        import src.osm.overpass_client as overpass

        seen: list[Any] = []

        class _Response:
            status_code = 200

            @staticmethod
            def json() -> dict[str, list[object]]:
                return {"elements": []}

        def _capture(url: str, **kwargs: Any) -> _Response:
            seen.append(kwargs.get("timeout"))
            return _Response()

        monkeypatch.setattr(overpass.requests, "post", _capture)

        client = overpass.OverpassClient(cache_dir="unused", use_cache=False)
        client.query_water_features(45.0, 6.0, 45.1, 6.1)

        assert seen, "no request was made"
        timeout = seen[0]
        assert isinstance(timeout, tuple), (
            f"timeout is {timeout!r}: one number covers both connecting and reading, "
            "so a refused connection waits as long as a slow query"
        )
        connect, _read = timeout
        assert connect <= 15, f"connect timeout {connect}s is longer than any handshake needs"
