# gora-print-router

Generate 3D-printable OBJ terrain from GPX tracks and FABDEM elevation rasters. Python 3.14,
managed by `uv`. About 8.1k lines in [src/](src/).

## Commands

```bash
uv sync                      # install deps, dev group included
uv run pytest                # tests, testpaths = tests/
uv run ruff check            # lint
uv run ruff format           # format
uv run mypy                  # type check, strict
uv run reuse lint            # every file carries copyright and licensing
uv run python main.py TRACK.gpx --output ./models
uv run python src/utils/srtm_scanner.py --mode fabdem   # rebuild the tile index into data/bounds.csv
```

Run all four gates before you call the work done: `ruff check`, `mypy`, `reuse lint`, `pytest`.

**Baseline:** every gate is green. `ruff check` 0, `ruff format --check` clean, `mypy --strict` 0,
`reuse lint` compliant with REUSE 3.3, `pytest` 267 passed and 1 skipped without an external data
library. Keep them there. A new error is yours.

## Tests

The one skipped test reads real FABDEM tiles and GPX tracks. It is marked `slow` and runs only
where `GORA_ROUTER_TEST_DATA` points at a directory holding `FABDEM_V1.2/` with its `bounds.csv`,
and `gpx/`. CI never has that data, so anything only real tiles can show needs a synthetic
counterpart: `synthetic_gappy_dem` covers nodata voids and `synthetic_tile_pair` covers a merge
seam under the model box. New real-data coverage needs one too.

[tests/test_watertight.py](tests/test_watertight.py) and
[tests/test_water_watertight.py](tests/test_water_watertight.py) assert that every exported group,
`terrain`, `track` and `water`, is a closed manifold: no boundary edges, no non-manifold edges, no
flipped winding, Euler 2 per shell, positive volume. `water` is checked per shell because each bay
and each lake is its own body. The helpers live in [tests/mesh_topology.py](tests/mesh_topology.py).

`find_folds` is for the track and nothing else. Pointed at a real terrain group it reports hundreds
of folded pairs, all within half a millimetre of the outline, and every one is a false positive: it
walks edge-adjacent faces and cannot tell an inside-out surface from the sharp edge where a steep
slope meets the vertical skirt. Measured over 4.6 million faces in twelve real models, no terrain
face overhangs at all. `test_real_terrain_never_overhangs` checks the property that would actually
break. Do not chase the terrain fold count.

Watertightness holds on real data too: a run over real tiles and a real Overpass answer passes
`assert_watertight_shells` for every group. Folds do not. A real track still carries face pairs that
point at each other where it doubles back, which `assert_no_folds` rejects, and the README lists it
under known issues. One mechanism is covered by `test_stacked_switchbacks_do_not_fold`; the rest have
no synthetic reproduction yet, and writing one is the way to fix them. Measuring a real export is the
only way to know: run the tool, then feed the OBJ back through
[tests/mesh_topology.py](tests/mesh_topology.py).

Closed is necessary and not sufficient. A swept ribbon whose offset outgrows its turn radius folds
inside out and passes every one of those checks, and one that answers a fold by losing height sinks
out of sight and passes them too. So `find_folds` and `assert_no_folds` walk edge-adjacent faces
and fail where two of them point at each other, folds on faces under `MIN_FOLD_AREA_MM2` are
reported rather than failed because they cannot reach the print, and
`test_the_track_shows_above_the_terrain_everywhere` interpolates the terrain under every track
vertex and demands something standing proud nearby. Sweep geometry is unit-tested without a DEM in
[tests/test_track_sweep.py](tests/test_track_sweep.py).

The water tests build features by hand and stub `OverpassClient`, so they never touch the network.
Keep it that way. [tests/test_water_osm_fixtures.py](tests/test_water_osm_fixtures.py) replays
trimmed real Overpass answers from [tests/data/](tests/data/) for the paths only real data reaches:
LOD filtering, coastline stitching, ribbon merge. `tests/data/` is exempt from the root `data`
ignore, so keep the `/data` anchor in [.gitignore](.gitignore).

## The mypy ratchet

`mypy` reaches zero only because two legacy modules sit in a ratchet list, `ignore_errors = true`
in the `[[tool.mypy.overrides]]` block at the end of [pyproject.toml](pyproject.toml):
`src.core.router` holds 34 unfixed strict errors and `src.osm.water_features` holds 23.

Two rules. Never add a module to that list. Clearing one module, which means deleting its line and
fixing what surfaces, is a standalone pull request and never travels with feature work.
`srtm_scanner`, `obj_exporter`, `mesh_generator` and `water_generator` were cleared this way.

Reuse the types those passes introduced instead of re-typing bare dicts and lists at the call
sites: `TileBounds`, `TileRecord` and `FabdemTile` in `srtm_scanner` for the tile index, `Face` and
`_Group` in `obj_exporter` for OBJ output, `LatLon` and `PointXY` in `mesh_generator` and
`Vertex` in `water_generator` for coordinates. Degrees against millimetres is carried by
the alias, not by the variable name alone.

## Layout

| Path | Role |
| --- | --- |
| [main.py](main.py) | Entry shim. Calls `src.core.router.main`. |
| [src/core/router.py](src/core/router.py) | CLI argparse and the `generate_model()` orchestrator. Border solving, auto-rotate, argument validation. |
| [src/core/mesh_generator.py](src/core/mesh_generator.py) | Biggest module, 2.4k lines. Track mesh, terrain mesh, clipping, smoothing, upsampling. |
| [src/core/water_generator.py](src/core/water_generator.py) | `WaterLayerGenerator`. Water surfaces and waterway ribbons. |
| [src/core/footprint.py](src/core/footprint.py) | `Footprint`. Rectangle, hexagon and circle model outlines. |
| [src/osm/](src/osm/) | Overpass client, cache, water feature extraction, LOD filtering. |
| [src/utils/srtm_scanner.py](src/utils/srtm_scanner.py) | FABDEM and SRTM tile discovery, the `data/bounds.csv` index. |
| [src/utils/obj_exporter.py](src/utils/obj_exporter.py) | `Mesh`, `OBJExporter`, the combined-mesh writer. |
| [src/utils/gpx_utils.py](src/utils/gpx_utils.py) | GPX parsing, bounds, terrain-bounds maths. |
| [tests/](tests/) | pytest. Imports as `from src.core.router import ...`. |

## Pipeline

GPX parse, terrain bounds, tile lookup through `bounds.csv`, DEM read with rasterio,
resample and upsample, Laplacian smoothing, footprint clip, terrain mesh, track ribbon mesh,
optional OSM water layer, OBJ export.

## Domain rules

Breaking one of these produces a model that is silently wrong rather than an error.

- **Units.** Geography in degrees, elevation in metres, the model in millimetres. Name variables
  with the unit suffix the existing code uses: `_mm`, `_m`, `_deg`.
- **A manual bbox forbids `--auto-rotate`**, and `--terrain-border` is ignored once the bbox is
  explicit.
- **Hexagon aspect is not 1:1.** A regular pointy-top bbox aspect is √3/2, about 0.866; flat-top is
  2/√3, about 1.155. A manual bbox has to match the resolved orientation or the terrain comes out
  stretched. The `--model-size-y` defaults are derived per shape, so do not "fix" them to `y = x`.
- **Automatic vertical exaggeration stops at 4.0.** A higher value has to be passed explicitly.
- **`--water-objects` runs 0 to 5.** 0 is off. 1 is ocean and sea: it tries OSM coastlines for a
  truer shoreline and falls back to sea-level detection on the DEM when the answer is empty or
  filtered to nothing. The fallback does not cover an unreachable Overpass: `query_water_features`
  raises `RuntimeError` once every mirror has failed, nothing between there and `main` catches it,
  and the run aborts. 2 and above need Overpass and return nothing without it, because terrain
  cannot infer inland water. Levels 2 to 5 are physical
  size thresholds, not arbitrary tiers.
- **A duplicate CLI flag is a hard error** by design, because argparse would otherwise keep the last
  one without saying so. See `_reject_duplicate_args` in
  [router.py](src/core/router.py).
- **The track sweep answers to the ribbon's own width.** A ribbon of half-width `w` can only follow
  a turn whose radius stays above `w`. Below that its inner edge runs backwards, the wall quads fold
  inside out, and the mesh stays perfectly watertight while printing as a pinch or a void. At print
  scale that bites constantly: on an 80 mm model of a 10 km box, half of a 1.5 mm ribbon is about
  90 m of ground, so every real switchback is tighter than the ribbon is wide. The order in
  `generate_track_mesh` is therefore fixed. Thin the path to what the ribbon can express
  (`_resample_path`, Douglas-Peucker at one half-width, because a GPX simplified to 10 m lands its
  points eight hundredths of a millimetre apart and the jitter alone would force the clamp on), round
  corners into arcs (`_round_sharp_joins`), subdivide for the terrain (`_subdivide_path`), clamp
  every offset so no quad can invert (`_sweep_offsets`), then sweep. Subdividing before rounding
  leaves each corner a tenth of the radius its real arms could give.
- **The `track` group can be several overlapping bodies.** Stretches the clamp pinched below half
  width are replaced by beads: straight, full-width, full-height tubes bridging the swept ends. They
  are separate closed shells that the slicer unions, exactly like the water layer's lakes, so the
  track is checked with `assert_watertight_shells` rather than `assert_watertight`. A track without
  corners sharp enough to bead is still a single shell, and a test pins that down.
- **The track must stay visible.** Never buy fold-free geometry by shrinking the ribbon's height. A
  cross-section that loses its height along with its width sinks under the hillside, which is a
  worse bug than the fold it fixes. Beads exist for this reason: they carry full height through the
  tips.
- **Every water clip against a non-rectangular footprint goes through `_clip_to_footprint`**, never
  a bare `polygon.intersection(footprint.polygon)`. An intersection returns two vertices a few float
  ulps apart wherever a water vertex grazes the outline: the vertex counts as inside so the overlay
  keeps it, and GEOS adds its own node for the crossing at the same spot. Shapely holds the pair
  apart and builds a cap triangle and a wall quad across it. The OBJ writer's `%.6f` cannot, so both
  collapse to zero area and their edges fold into the neighbours, which is non-manifold and leaves
  the shell open. `_clip_to_footprint` snaps its result to `FOOTPRINT_CLIP_GRID_MM`, 1e-4 mm, which
  is 100 times the OBJ's own resolution and about 1000 times below one printed layer. That removes
  the pair before the caps and the walls disagree. A long straight edge is what makes the graze
  likely, so a hexagon offers 40 mm of one exact line for some coastline vertex to land on while a
  circle of a few hundred segments bends away from any chord within a millimetre. It therefore reached production
  breaking hexagons only.
- **A river's clip inset owes the ribbon's own half-width**, not just `RIVER_EDGE_MARGIN_MM`. What
  has to fit inside the model is the swept ribbon, not the centerline it is swept along, so anything
  wider than twice the margin used to hang over the outline. Nothing downstream catches that: the
  exporter used to drop water triangles that left the footprint, which cut a closed shell open, and
  on a river along a hexagon edge it deleted three quarters of the faces and left 184 boundary
  edges. Containment is settled at the clip or not at all.
- **Winding order matters.** OBJ faces are CCW, and `_densify_polygon_ccw` and `_signed_area` exist
  for that. The exporter writes `v x z y`, because model space is Z-up and OBJ is Y-up, and it
  **reverses each face's corner order** to compensate: swapping two axes mirrors the frame, so
  keeping the original order ships the model inside out. Judge an exported file with
  `parse_obj_groups_raw` in [tests/mesh_topology.py](tests/mesh_topology.py); `parse_obj_groups`
  undoes both and hands back the generator's own mesh.

## Working in this repo

- **Never read [mesh_generator.py](src/core/mesh_generator.py) or
  [water_generator.py](src/core/water_generator.py) whole.** They cost about 30k and 22k tokens.
  `grep -n` for the symbol, then read with an offset and a limit.
- `data/`, `osm_cache/` and `.venv/` hold gigabytes of tiles and cached JSON. They are excluded from
  search in VS Code and belong out of any grep.
- Elevation tiles and GPX files live outside the repository, behind `GORA_ROUTER_TEST_DATA` for the
  tests and behind your own `.vscode/launch.json` for a debug run. Copy
  [.vscode/launch.json.example](.vscode/launch.json.example) to make one. Do not assume that data
  exists in CI, because it never does.
- Adding a CLI flag means touching three places: `add_argument` in `router.py`, the
  `generate_model()` signature and its docstring, and [README.md](README.md).
- **Always import as `src.core.x` or `src.utils.x`**, never `core.x`. Modules inside `src/` import
  each other with the `src.` prefix, so a `sys.path`-based `core.router` import loads the same file
  a second time under a different module name, which gives two copies of module-level state and
  `isinstance` failures across the pair.
- Every source file carries a three-line SPDX header naming both copyright holders and
  `AGPL-3.0-or-later`; [REUSE.toml](REUSE.toml) covers the files that take no comments. A new file
  needs one or the other, and `reuse lint` fails until it has it. The Overpass fixtures under
  `tests/data/` stay `ODbL-1.0` and © OpenStreetMap contributors: relicensing a derived extract
  under AGPL is not ours to do.
- `ruff` is the only linter and formatter. Dependencies live in
  [pyproject.toml](pyproject.toml), pinned by `uv.lock`.
- Line endings are LF everywhere, enforced by [.gitattributes](.gitattributes) and
  [.editorconfig](.editorconfig). Do not set `end_of_line = crlf`. It puts CRLF in the index and
  turns every edit into a whole-file diff.
- Commit messages follow Conventional Commits, in English, with no attribution trailer.

## Style

Type hints on every new function, because `mypy` runs strict. Google-style docstrings with an
`Args:` block, matching `generate_model`. Line length 100, owned by `ruff`. Comments explain why a
geometric constant or a guard exists. That convention is load-bearing here, so keep it.
