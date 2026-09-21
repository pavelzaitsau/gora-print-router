# Góra Print Router

[![CI](https://github.com/pavelzaitsau/gora-print-router/actions/workflows/ci.yml/badge.svg)](https://github.com/pavelzaitsau/gora-print-router/actions/workflows/ci.yml)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-green)](LICENSE)
[![REUSE 3.3](https://img.shields.io/badge/REUSE-3.3-brightgreen)](https://reuse.software/)

Turn a GPX track into a 3D-printable OBJ terrain model.

The tool cuts the track's area out of a
[FABDEM v1.2](https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn) elevation raster,
builds a terrain mesh under a rectangular, hexagonal or circular footprint, sweeps the track over
it as a raised ribbon, and writes one OBJ file a slicer can print. Lakes, rivers and coastlines are
optional, and come from [OpenStreetMap](https://www.openstreetmap.org/) through the
[Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API).

This is the model generator behind Góra Print.

## Setup

```bash
uv sync                                                  # dependencies, from uv.lock
# download FABDEM tiles, extract them under data/tiles/
uv run python src/utils/srtm_scanner.py --mode fabdem    # writes data/bounds.csv
```

Every run resolves its tiles through `data/bounds.csv`. Skip the last step and the run stops
before any mesh work, naming the scanner command it needs.

## Usage

```bash
uv run python main.py track.gpx
```

The settings behind a printed 90 mm by 70 mm model, with tiles and output outside the repository:

```bash
uv run python main.py track.gpx --output out --data-dir ~/fabdem \
  --model-size-x 90 --model-size-y 70 --auto-rotate --terrain-border 10 \
  --track-width 1.5 --track-height 1 \
  --vertical-exaggeration 4.5 --base-thickness 3 --max-points 2000 \
  --terrain-resolution 200 --terrain-upsample 3 \
  --smoothing-iterations 4 --smoothing-strength 0.7
```

Terrain alone, over a manual bounding box, and a hexagonal coaster:

```bash
uv run python main.py --no-track --model-size-x 150 --model-size-y 150 \
  --bbox-lat-min 45.0 --bbox-lat-max 46.0 --bbox-lon-min 6.0 --bbox-lon-max 7.0
uv run python main.py track.gpx --model-shape hexagon --model-size-x 100
```

Sizes are in millimetres. The preset names are the Góra Print sizes.

| Preset | `--model-size-x` | `--model-size-y` | `--track-width` | `--track-height` | `--terrain-border` |
| --- | :---: | :---: | :---: | :---: | :---: |
| Standard | 80 | 60 | 1 | 1 | 5 |
| Large | 90 | 70 | 1.5 | 1 | 10 |
| Custom | 130 | 130 | 1.5 | 1.5 | 10 |

## Options

`uv run python main.py --help` prints the full help text. A duplicate flag is an error, not a
silent win for the last value.

| Flag | Values | Default | Effect |
| --- | --- | --- | --- |
| `track_gpx` | path | none | Positional GPX file; it also sets the terrain bounds |
| `--output` | path | the GPX directory | Output directory |
| `--data-dir` | path | `data` | Elevation tiles and `bounds.csv` |
| `--no-track` | flag | off | Skip the track. A run without a GPX path needs all four `--bbox-*` flags |
| `--bbox-lat-min`, `--bbox-lat-max` | degrees | from the track | Southern and northern edge |
| `--bbox-lon-min`, `--bbox-lon-max` | degrees | from the track | Western and eastern edge |
| `--model-shape` | `rectangle`, `hexagon`, `circle` | `rectangle` | Footprint; a circle becomes an oval when X and Y differ |
| `--hex-orientation` | `flat`, `pointy`, `auto` | `auto` | `auto` reads the track aspect |
| `--model-size-x` | mm | auto | Bounding box width, except for a hexagon where it is the long diagonal. Required for a hexagon or a circle |
| `--model-size-y` | mm | derived | Depth; a circle takes X, a hexagon derives from its orientation |
| `--min-model-height-z` | mm | `0` | Minimum total height including the base. At `0` the model targets 10 mm |
| `--vertical-exaggeration` | 1.5 to 3.0 typical | auto | Elevation multiplier; the automatic value stops at 4.0 |
| `--terrain-border` | mm | `10.0` | Margin around the track; ignored with a manual bbox |
| `--base-thickness` | mm, 2.0 and up | `2.0` | Solid base under the terrain |
| `--auto-rotate` | flag | off | Swap X and Y to match the terrain aspect; a manual bbox forbids it |
| `--auto-rotate-tolerance` | fraction | `0.05` | Improvement needed before the swap |
| `--track-width` | mm, 1.5 to 3.5 typical | `2.5` | Ribbon width |
| `--track-height` | mm, 1.5 and up | `3.5` | Ribbon rise above the terrain |
| `--simplification-tolerance` | m, 5 to 30 typical | `15.0` | Douglas-Peucker threshold |
| `--max-points` | 200 to 500 typical | unlimited | Point cap; it overrides the tolerance |
| `--terrain-resolution` | px, 100 to 250 | `200` | Grid on the long side, before upsampling; 200 alone gives ~160k faces |
| `--terrain-upsample` | 1 to 4 | `2` | 2 gives 4x the faces, 4 gives 16x |
| `--terrain-smoothing`, `--no-terrain-smoothing` | flag | on | Laplacian smoothing. Off also disables upsampling |
| `--smoothing-iterations` | 0 to 5 | `2` | Passes; 1 to 3 recommended |
| `--smoothing-strength` | 0.0 to 1.0 | `0.3` | Strength of one pass; 0.2 to 0.4 recommended |
| `--water-objects` | 0 to 5 | `0` | Water detail; see below |
| `--no-osm-cache` | flag | cache on | Query Overpass fresh |
| `--osm-cache-dir` | path | `osm_cache` | Cached Overpass answers |

> **A hexagon's bounding box is not square.** Pointy-top has aspect 0.866 and flat-top 1.155, so
> `--model-size-y` is never simply `--model-size-x`. A manual bounding box has to match the
> resolved orientation, or the terrain comes out stretched, which shows in the print and not in the
> log.

## Water levels

Each level includes the ones below it. The thresholds are physical sizes, not arbitrary tiers.

| Level | Adds | Network |
| --- | --- | --- |
| `0` | nothing, water disabled | no |
| `1` | oceans and seas | optional |
| `2` | lakes above the smaller of 5 km² and 0.5% of the bounding box | Overpass |
| `3` | rivers at least 40 m wide and 3 km long | Overpass |
| `4` | rivers and streams at least 20 m wide, any length | Overpass |
| `5` | every waterway at least 3 m wide and 500 m long, minus the intermittent | Overpass |

Level 1 tries OpenStreetMap coastlines first and falls back to sea-level detection on the raster
when the answer is empty or filtered to nothing. That fallback does not cover an unreachable
Overpass: when every mirror fails the run stops with an error, so water at any level needs the
network unless a cached answer is already on disk. Level 2 and above return nothing without
Overpass, because terrain alone cannot tell an inland lake from a flat field.

A lake or river sits 0.4 mm above the terrain and is 0.5 mm thick; an ocean detected from the
raster sits at sea level instead. A river is inset from the model edge by 1.5 mm or by its own
half-width, whichever is larger, and no waterway is drawn narrower than 1.0 mm.

Labels, track markers, stands and split models are applied by hand after generation and have no
flag. Colour is a filament choice, not a property of the OBJ.

## Development

```bash
uv run ruff check && uv run ruff format --check   # lint and format
uv run mypy                                       # types, strict
uv run reuse lint                                 # copyright and licensing
uv run pytest                                     # tests
```

CI runs these on every push and every pull request, including one from a fork.
[AGENTS.md](AGENTS.md) carries the domain rules, including the ones whose violation produces a
model that is wrong rather than an error.

One test needs real tiles. Point `GORA_ROUTER_TEST_DATA` at a directory holding `FABDEM_V1.2/`
with its `bounds.csv`, and `gpx/`; leave it unset and the test skips, which is what CI does. Copy
`.vscode/launch.json.example` to `.vscode/launch.json` to debug from VS Code.

## Known issues

These are open. Each says what you will see and what to do instead.

| Issue | What happens | Workaround |
| --- | --- | --- |
| GPX routes are not read | A file exported as a *route* rather than a *track*, which is what Komoot, Garmin Connect courses and most planning tools produce, fails with `No track points found in GPX file` | Re-export as a track, or convert the route to a track in any GPX editor |
| Errors arrive as tracebacks | Every failure, including the ones with a clear message, prints a Python stack trace | Read the last line; it carries the message |
| Tile lookup fails on Windows | `bounds.csv` written on Windows records `tiles\name.tif`, and the lookup compares it against `tiles/name.tif`, so every tile reads as missing | Generate `bounds.csv` on Linux or macOS and copy it over |
| Console output fails on Windows when redirected | The progress lines use `✓`, `⚠` and `→`, which the ANSI code page cannot encode | Set `PYTHONIOENCODING=utf-8`, or do not redirect |
| Land below sea level is flattened | Every negative elevation is clamped to 0 m, so a Dead Sea or Death Valley model prints as a flat plate where the depression should be | None |
| Flat coastal terrain is rejected | An area whose relief is under 0.5 m and whose peak is under 1 m, such as a polder or a barrier island, fails with `missing or invalid FABDEM data` although the tiles are fine | Extend the bounding box to include higher ground |
| Hexagon with `--auto-rotate` comes out stretched | The axes are swapped after the bounding box was derived from the hexagon orientation, and the log still reports the undistorted size | Drop `--auto-rotate` for a hexagon, or pass `--model-size-y` explicitly |

### One the test suite still cannot see

The exported bodies are closed on real data: a run over real tiles and a real Overpass answer now
passes `assert_watertight_shells` for `terrain`, `track` and `water` alike. Folds are a separate
property, and the track does not yet meet it.

**Open: the track ribbon still folds on real tracks.** The `track` group is closed,
outward-facing and prints, yet carries face pairs that point at each other where the path doubles
back: 30 of them on an 80 x 60 mm model of a 90 km alpine route. `assert_no_folds` rejects that.
One mechanism is now covered and fixed, a staircase of switchbacks closer together than the ribbon
is wide, and `test_stacked_switchbacks_do_not_fold` pins it. Real routes reach the rest.

The way in is the same: reproduce a real fold synthetically before touching the sweep. A fold that
only real data shows is a fold nobody can keep fixed.

## License

Copyright (C) 2025-2026 Pavel Zaitsau and Góra Print. Released under the GNU Affero General Public
License, version 3 or later. See [LICENSE](LICENSE).

Build and editor configuration is CC0-1.0 rather than AGPL, so it can be copied into another
project without dragging copyleft along. [REUSE.toml](REUSE.toml) lists exactly which files.

A model this tool produces is derived from the elevation data you feed it, and from OpenStreetMap
data where the water layer is enabled. Both carry their own terms, independent of this license, and
this license grants you nothing in respect of either.

**FABDEM v1.2 is licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/):
non-commercial use only, attribution required, derivatives under the same terms.** A model you
generate from it inherits those terms. The University of Bristol names a contact for commercial
licensing on the [dataset record](https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn).
The tool does not check what you feed it, so the obligation is yours.

OpenStreetMap ships under the [Open Database License](https://www.openstreetmap.org/copyright), and
the Overpass answers replayed as fixtures in [tests/data/](tests/data/) are © OpenStreetMap
contributors under it.
