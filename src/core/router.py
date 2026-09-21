# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
FABDEM/SRTM to OBJ Terrain Generator

Generates 3D-printable OBJ terrain models from FABDEM or SRTM elevation data with optional GPX track overlay.
"""

import argparse
import contextlib
import hashlib
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np

from src.core.footprint import VALID_SHAPES, Footprint
from src.core.mesh_generator import generate_terrain_stl
from src.utils.gpx_utils import (
    calculate_terrain_bounds,
    get_gpx_bounds,
    parse_gpx_track,
)
from src.utils.srtm_scanner import find_tif_for_coordinates, find_tifs_for_bounds


def _solve_effective_border_mm(
    track_gpx: str,
    model_w_mm: float,
    model_h_mm: float,
    requested_border_mm: float,
    model_shape: str,
    hex_orientation: str,
    terrain_resolution: int,
) -> float:
    """
    Pick a `border_mm` value to feed calculate_terrain_bounds so that the
    GPX track, after projection to mm-space, lies entirely inside the model
    footprint polygon with at least `requested_border_mm` of clearance.

    Binary-searches the relationship `effective_border ↦ track_clearance`.
    Increasing the border makes calculate_terrain_bounds reserve more
    rectangular padding → scale shrinks → track points move closer to the
    polygon centre → clearance to the polygon edge grows.

    For rectangle footprints requested_border_mm is enough by construction;
    this function is only called for hex / circle.

    Args:
        track_gpx: Path to GPX file.
        model_w_mm, model_h_mm: Model bbox dimensions (post hex-y-derivation).
        requested_border_mm: Clearance the user wants between track and the
            actual polygon edge.
        model_shape, hex_orientation: Footprint identity.
        terrain_resolution: Circle segment count seed.

    Returns:
        The border_mm to pass to calculate_terrain_bounds (>= requested).
    """
    import shapely

    # Build the target footprint once.
    footprint = Footprint.build(
        model_shape,
        model_w_mm,
        model_h_mm,
        hex_orientation=(hex_orientation if model_shape == "hexagon" else "flat"),
        circle_segments=max(64, terrain_resolution * 2),
    )
    inner = footprint.polygon.buffer(-requested_border_mm)
    if inner.is_empty:
        # Requested border doesn't even fit inside the polygon — fall back to
        # rectangle behaviour and let the caller fail with a clearer error.
        return requested_border_mm

    # Pre-project track points to (dx_m, dy_m) offsets from the GPX bbox
    # centre. Mesh Y is inverted vs latitude (north = up in lat, north = -Y
    # in mesh space relative to the row order used downstream).
    gpx_bounds = get_gpx_bounds(track_gpx)
    track_pts = parse_gpx_track(track_gpx)
    lat_c = (gpx_bounds["lat_min"] + gpx_bounds["lat_max"]) / 2
    lon_c = (gpx_bounds["lon_min"] + gpx_bounds["lon_max"]) / 2
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * float(np.cos(np.radians(lat_c)))
    arr = np.asarray(track_pts, dtype=float)
    dx_m = (arr[:, 1] - lon_c) * m_per_deg_lon
    dy_m = (arr[:, 0] - lat_c) * m_per_deg_lat

    cx_mm = model_w_mm / 2.0
    cy_mm = model_h_mm / 2.0

    def track_fits(border_mm: float) -> bool:
        """Project track with calculate_terrain_bounds-style scale; test fit."""
        usable_w = model_w_mm - 2.0 * border_mm
        usable_h = model_h_mm - 2.0 * border_mm
        if usable_w <= 0 or usable_h <= 0:
            return False
        # calculate_terrain_bounds picks scale = min(usable_w/track_w_m,
        # usable_h/track_h_m). Reproduce that here.
        track_w_m = (gpx_bounds["lon_max"] - gpx_bounds["lon_min"]) * m_per_deg_lon
        track_h_m = (gpx_bounds["lat_max"] - gpx_bounds["lat_min"]) * m_per_deg_lat
        if track_w_m <= 0 or track_h_m <= 0:
            return True
        scale = min(usable_w / track_w_m, usable_h / track_h_m)
        xs = cx_mm + dx_m * scale
        ys = cy_mm - dy_m * scale
        return bool(shapely.contains_xy(inner, xs, ys).all())

    # If the rectangle-equivalent border already fits, no boost needed.
    if track_fits(requested_border_mm):
        return requested_border_mm

    # Upper bound: pad with the worst bbox-corner inset; this is the most
    # conservative value (track confined to the largest inscribed
    # axis-aligned rectangle minus border) and is guaranteed to fit.
    corners = [(0.0, 0.0), (model_w_mm, 0.0), (0.0, model_h_mm), (model_w_mm, model_h_mm)]
    worst_inset = max(0.0, max(-footprint.signed_distance(*c) for c in corners))
    lo = requested_border_mm
    hi = requested_border_mm + worst_inset
    # Sanity: make sure the upper bound actually fits; if not, return it
    # anyway (geometric edge case — degenerate aspect).
    if not track_fits(hi):
        return hi

    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if track_fits(mid):
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-3:
            break
    return hi


def calculate_aspect_ratio_distance(model_ar: float, terrain_ar: float) -> float:
    """Calculate symmetric distance between two aspect ratios using log difference."""
    return abs(math.log(model_ar) - math.log(terrain_ar))


def should_swap_model_sizes(
    terrain_bounds: dict[str, float],
    model_size_x: float,
    model_size_y: float,
    tolerance: float = 0.05,
) -> bool:
    """
    Determine if model sizes should be swapped based on aspect ratio closeness.

    Args:
        terrain_bounds: Terrain bounds dictionary from calculate_terrain_bounds
        model_size_x: Model width in mm (use 160.0 if originally 0)
        model_size_y: Model height in mm (use 160.0 if originally 0)
        tolerance: Minimum improvement ratio required to swap (default 5%)

    Returns:
        True if swapping improves aspect ratio match by more than tolerance
    """
    # Use track dimensions, not terrain — terrain is always forced to match model AR
    track_width_m = terrain_bounds.get("track_width_m") or terrain_bounds["terrain_width_m"]
    track_height_m = terrain_bounds.get("track_height_m") or terrain_bounds["terrain_height_m"]

    if track_width_m <= 0 or track_height_m <= 0:
        return False

    terrain_ar = track_width_m / track_height_m

    # Calculate model aspect ratios (before and after swap)
    model_ar_original = model_size_x / model_size_y
    model_ar_swapped = model_size_y / model_size_x

    # Calculate distances
    dist_original = calculate_aspect_ratio_distance(model_ar_original, terrain_ar)
    dist_swapped = calculate_aspect_ratio_distance(model_ar_swapped, terrain_ar)

    # Swap only if improvement exceeds tolerance
    improvement = dist_original - dist_swapped
    return improvement > tolerance and dist_swapped < dist_original


def generate_model(
    track_gpx: str | None = None,
    map_tif: str | list[str] | None = None,
    output_dir: str | None = None,
    min_model_size_x_mm: float = 0.0,
    min_model_size_y_mm: float = 0.0,
    terrain_border_mm: float = 10.0,
    vertical_exaggeration: float | None = None,
    min_model_height_z_mm: float = 0.0,
    base_thickness_mm: float = 2.0,
    track_width_mm: float = 2.5,
    track_height_mm: float = 3.5,
    include_track: bool = True,
    bbox_lat_min: float | None = None,
    bbox_lat_max: float | None = None,
    bbox_lon_min: float | None = None,
    bbox_lon_max: float | None = None,
    simplification_tolerance: float = 15.0,
    max_points: int | None = None,
    terrain_resolution: int = 200,
    terrain_smoothing: bool = True,
    smoothing_iterations: int = 2,
    smoothing_strength: float = 0.3,
    terrain_upsample: int = 2,
    water_objects: int = 0,
    use_osm_cache: bool = True,
    auto_rotate: bool = False,
    auto_rotate_tolerance: float = 0.05,
    data_dir: str = "data",
    osm_cache_dir: str = "osm_cache",
    model_shape: str = "rectangle",
    hex_orientation: str = "auto",
) -> None:
    """
    Generate a 3D terrain model with optional GPX track.

    Args:
        track_gpx: Path to GPX track file (optional, required unless --no-track is specified)
        map_tif: Path to a FABDEM/SRTM GeoTIFF, or a list of paths when the bounds
            span several tiles. Auto-detected from the tile index if None, which is
            also how the list form usually arises.
        output_dir: Output directory for OBJ files (optional, defaults to GPX location)
        min_model_size_x_mm: Model width (X dimension) in millimeters (0 = auto-calculated from terrain)
        min_model_size_y_mm: Model depth (Y dimension) in millimeters (0 = auto-calculated from terrain)
        terrain_border_mm: Border of terrain without track in mm
        vertical_exaggeration: Terrain height multiplier (auto-calculated if None, range: 1.0-4.0)
        min_model_height_z_mm: Minimum model Z-axis height in mm (0 = auto-calculated)
        base_thickness_mm: Base thickness in mm
        track_width_mm: Track width in mm
        track_height_mm: Track height above terrain in mm (total protrusion height)
        include_track: Whether to include track (default: True)
        bbox_lat_min: Minimum latitude for bounding box (optional)
        bbox_lat_max: Maximum latitude for bounding box (optional)
        bbox_lon_min: Minimum longitude for bounding box (optional)
        bbox_lon_max: Maximum longitude for bounding box (optional)
        simplification_tolerance: Track simplification threshold in meters (Douglas-Peucker algorithm)
        max_points: Maximum number of track points after simplification (optional)
        terrain_resolution: Terrain mesh grid resolution (maximum dimension in pixels)
        terrain_smoothing: Enable Laplacian smoothing to reduce blocky grid artifacts
        smoothing_iterations: Number of Laplacian smoothing passes (0-5)
        smoothing_strength: Smoothing strength factor (0.0-1.0)
        terrain_upsample: Terrain mesh upsampling multiplier (1-4)
        water_objects: Water feature detail level (0=disabled, 1-5=physical size-based thresholds)
                       1=oceans/seas only, 2=+large lakes (≥5km²), 3=+major rivers (≥40m/3km),
                       4=+rivers/streams (≥20m), 5=+all waterways (≥3m/500m)
        use_osm_cache: Whether to use OSM query caching (default: True)
        auto_rotate: If True, automatically swap model_size_x and model_size_y if it improves aspect ratio match
        auto_rotate_tolerance: Minimum improvement required for auto-rotate to swap (default: 0.05 = 5%)
        data_dir: Path to data directory containing elevation tiles and bounds.csv (default: "data")
        osm_cache_dir: Path to OSM cache directory (default: "osm_cache")
        model_shape: Footprint shape — "rectangle", "hexagon", or "circle" (default: "rectangle").
        hex_orientation: Hexagon orientation — "flat", "pointy", or "auto" (chosen from track aspect). Ignored unless model_shape="hexagon".

    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If parameters are invalid
    """

    # Which bbox coordinates the caller supplied. Test against None, not
    # truthiness: 0.0 is a valid latitude and longitude, so a bbox on the
    # equator or the Greenwich meridian would otherwise read as unspecified.
    bbox_params = [bbox_lat_min, bbox_lat_max, bbox_lon_min, bbox_lon_max]
    bbox_specified = [p is not None for p in bbox_params]

    # Validate required parameters
    if not track_gpx and include_track:
        raise ValueError(
            "track_gpx is required when generating track overlay (remove --no-track or provide GPX file)"
        )

    # If no track is provided, bounding box must be specified
    if not track_gpx and not all(bbox_specified):
        raise ValueError(
            "Either provide a GPX track file or specify bounding box coordinates (--bbox-lat-min, --bbox-lat-max, --bbox-lon-min, --bbox-lon-max)"
        )

    # Validate object parameters
    if water_objects < 0 or water_objects > 5:
        raise ValueError("--water-objects must be between 0 and 5")

    # Numeric bounds. Without these the values reach scipy and numpy, where a
    # bad one is a stack trace from a library the user never called, a run that
    # never ends, or a model that is quietly wrong. Each bound below marks the
    # point where one of those three starts.
    if terrain_resolution < 10:
        raise ValueError(
            f"--terrain-resolution must be at least 10, got {terrain_resolution}. "
            "A smaller grid has too few points to interpolate and fails inside scipy."
        )
    if terrain_resolution > 1000:
        raise ValueError(
            f"--terrain-resolution must be at most 1000, got {terrain_resolution}. "
            "Above that the smoothing pass takes hours and the mesh exceeds what a slicer loads."
        )
    if not 1 <= terrain_upsample <= 4:
        raise ValueError(
            f"--terrain-upsample must be between 1 and 4, got {terrain_upsample}. "
            "Each step squares the face count."
        )
    if not 0 <= smoothing_iterations <= 10:
        raise ValueError(
            f"--smoothing-iterations must be between 0 and 10, got {smoothing_iterations}"
        )
    if not 0.0 <= smoothing_strength <= 1.0:
        raise ValueError(
            f"--smoothing-strength must be between 0.0 and 1.0, got {smoothing_strength}. "
            "Above 1.0 the blend overshoots and the terrain oscillates instead of smoothing."
        )
    if base_thickness_mm < 0.5:
        raise ValueError(
            f"--base-thickness must be at least 0.5mm, got {base_thickness_mm}. "
            "Below that the terrain meets the base at its lowest point and the shell stops "
            "being manifold; 2.0 and up is the printable range."
        )
    if simplification_tolerance <= 0:
        raise ValueError(
            f"--simplification-tolerance must be positive, got {simplification_tolerance}"
        )
    if max_points is not None and max_points < 2:
        raise ValueError(
            f"--max-points must be at least 2, got {max_points}. "
            "Fewer leaves a straight line between the track's endpoints."
        )
    if auto_rotate_tolerance < 0:
        raise ValueError(
            f"--auto-rotate-tolerance must not be negative, got {auto_rotate_tolerance}"
        )

    # Geographic bounds, where they were given explicitly.
    for name, value in (
        ("--bbox-lat-min", bbox_lat_min),
        ("--bbox-lat-max", bbox_lat_max),
    ):
        if value is not None and not -90.0 <= value <= 90.0:
            raise ValueError(f"{name} must be between -90 and 90, got {value}")
    for name, value in (
        ("--bbox-lon-min", bbox_lon_min),
        ("--bbox-lon-max", bbox_lon_max),
    ):
        if value is not None and not -180.0 <= value <= 180.0:
            raise ValueError(f"{name} must be between -180 and 180, got {value}")

    # A border eats from both sides, so twice it has to leave something behind.
    # The default is 10mm, which alone rules out any model under 20mm, and the
    # failure would otherwise surface from deep in the bounds solver.
    explicit_sizes = [d for d in (min_model_size_x_mm, min_model_size_y_mm) if d > 0]
    if explicit_sizes and 2 * terrain_border_mm >= min(explicit_sizes):
        raise ValueError(
            f"--terrain-border {terrain_border_mm}mm leaves nothing of a "
            f"{min(explicit_sizes)}mm model: it is taken from both sides. "
            f"Use --terrain-border below {min(explicit_sizes) / 2}."
        )

    # Validate shape parameters
    if model_shape not in VALID_SHAPES:
        raise ValueError(f"--model-shape must be one of {VALID_SHAPES}, got {model_shape!r}")
    if hex_orientation not in ("flat", "pointy", "auto"):
        raise ValueError(
            f"--hex-orientation must be one of 'flat', 'pointy', 'auto', got {hex_orientation!r}"
        )

    # For non-rectangle shapes --model-size-x must be explicit (the shape
    # has no natural auto-size). --model-size-y handling depends on shape:
    #
    #   circle   : default y = x → produces a circle. y > 0 stretches into
    #              an ellipse aligned with model X.
    #   hexagon  : default y is derived from x AND the hex orientation so the
    #              result is a *regular* hexagon, not a stretched one. The
    #              orientation may be "auto", in which case the y default is
    #              applied later (after orientation resolution from the track
    #              bbox). Setting --model-size-y explicitly still wins.
    #
    # The previous behaviour ("y defaults to x" for every shape) made
    # hexagons look distorted because a regular hex's bbox aspect is
    # sqrt(3)/2 ≈ 0.866 (pointy-top) or 2/sqrt(3) ≈ 1.155 (flat-top), never
    # 1:1.
    user_provided_y = min_model_size_y_mm > 0
    if model_shape != "rectangle":
        if min_model_size_x_mm <= 0:
            raise ValueError(f"--model-shape {model_shape} requires --model-size-x to be > 0")
        if model_shape == "circle" and not user_provided_y:
            min_model_size_y_mm = min_model_size_x_mm

    # Resolve hex orientation early so we can derive --model-size-y when only
    # --model-size-x is provided. The orientation depends on the track bbox
    # aspect (auto mode) — read GPX bounds here. Without this step the
    # auto-rotate XOR check below would reject `--model-size-x N --auto-rotate`
    # for hexagon (only one axis provided), even though that's the canonical
    # way to request "a regular hex N mm wide".
    resolved_hex_orientation = hex_orientation
    if model_shape == "hexagon":
        if hex_orientation == "auto":
            if track_gpx and os.path.exists(track_gpx):
                gpx_bounds_tmp = get_gpx_bounds(track_gpx)
                lat_avg_tmp = (gpx_bounds_tmp["lat_min"] + gpx_bounds_tmp["lat_max"]) / 2
                m_per_deg_lat = 111320.0
                m_per_deg_lon = 111320.0 * np.cos(np.radians(lat_avg_tmp))
                track_w_m = (
                    abs(gpx_bounds_tmp["lon_max"] - gpx_bounds_tmp["lon_min"]) * m_per_deg_lon
                )
                track_h_m = (
                    abs(gpx_bounds_tmp["lat_max"] - gpx_bounds_tmp["lat_min"]) * m_per_deg_lat
                )
            elif all(bbox_specified):
                lat_avg_tmp = (bbox_lat_min + bbox_lat_max) / 2
                m_per_deg_lat = 111320.0
                m_per_deg_lon = 111320.0 * np.cos(np.radians(lat_avg_tmp))
                track_w_m = abs(bbox_lon_max - bbox_lon_min) * m_per_deg_lon
                track_h_m = abs(bbox_lat_max - bbox_lat_min) * m_per_deg_lat
            else:
                track_w_m = 1.0
                track_h_m = 1.0
            resolved_hex_orientation = Footprint.pick_hex_orientation(track_w_m, track_h_m)

        # Per the public spec, for hexagon --model-size-x is the LONG DIAGONAL
        # (vertex-to-opposite-vertex through the centre = 2·edge_length). The
        # bbox derived from that depends on orientation:
        #   flat-top   : bbox_w = diag,         bbox_h = diag * sqrt(3)/2
        #   pointy-top : bbox_w = diag * sqrt(3)/2, bbox_h = diag
        # We overwrite min_model_size_x/y in-place so downstream code (terrain
        # bounds, auto-rotate, mesh generation) sees true bbox dimensions
        # rather than the user-facing "diagonal".
        # If the user explicitly passed --model-size-y, treat both axes as
        # literal bbox dimensions (a deliberately stretched hex).
        if not user_provided_y:
            diag = min_model_size_x_mm
            short = diag * (math.sqrt(3.0) / 2.0)
            if resolved_hex_orientation == "flat":
                min_model_size_x_mm = diag  # bbox width = long axis
                min_model_size_y_mm = short  # bbox height = flat-to-flat
            else:  # pointy-top
                min_model_size_x_mm = short  # bbox width = flat-to-flat
                min_model_size_y_mm = diag  # bbox height = long axis

    # Validate bbox parameters - all or none must be specified
    if any(bbox_specified) and not all(bbox_specified):
        raise ValueError(
            "All bounding box coordinates must be specified together: --bbox-lat-min, --bbox-lat-max, --bbox-lon-min, --bbox-lon-max"
        )

    if all(bbox_specified):
        if bbox_lat_min >= bbox_lat_max:
            raise ValueError(
                f"bbox-lat-min ({bbox_lat_min}) must be less than bbox-lat-max ({bbox_lat_max})"
            )
        if bbox_lon_min >= bbox_lon_max:
            raise ValueError(
                f"bbox-lon-min ({bbox_lon_min}) must be less than bbox-lon-max ({bbox_lon_max})"
            )

    # Validate auto-rotate mutual exclusivity
    if auto_rotate and all(bbox_specified):
        raise ValueError(
            "--auto-rotate cannot be used together with manual bbox arguments "
            "(--bbox-lat-min, --bbox-lat-max, --bbox-lon-min, --bbox-lon-max)"
        )

    # Check if exactly one model size is provided (non-zero) when auto-rotate is enabled
    if auto_rotate:
        x_provided = min_model_size_x_mm > 0
        y_provided = min_model_size_y_mm > 0

        if x_provided != y_provided:  # XOR: exactly one is provided
            raise ValueError(
                "When using --auto-rotate, both --model-size-x and --model-size-y "
                "must be provided together (both >0) or both left as default (0)"
            )

    # Handle auto-rotate with both sizes at 0
    if auto_rotate and min_model_size_x_mm == 0 and min_model_size_y_mm == 0:
        warnings.warn(
            "--auto-rotate is enabled but both model sizes are 0 (auto). "
            "Auto-rotate will be ignored.",
            UserWarning,
            stacklevel=2,
        )
        auto_rotate = False

    # Validate file existence
    # Validate GPX file exists if provided
    if track_gpx and not os.path.exists(track_gpx):
        raise FileNotFoundError(f"GPX file not found: {track_gpx}")

    # Auto-detect map_tif if not provided
    if map_tif is None:
        # Get center coordinates from GPX or bbox
        if track_gpx:
            gpx_bounds = get_gpx_bounds(track_gpx)
            center_lat = (gpx_bounds["lat_min"] + gpx_bounds["lat_max"]) / 2
            center_lon = (gpx_bounds["lon_min"] + gpx_bounds["lon_max"]) / 2
        elif all(bbox_specified):
            center_lat = (bbox_lat_min + bbox_lat_max) / 2
            center_lon = (bbox_lon_min + bbox_lon_max) / 2
        else:
            raise ValueError("Either track_gpx or bounding box must be provided")

        # Try to find FABDEM tile
        fabdem_csv = os.path.join(data_dir, "bounds.csv")

        found_fabdem = False

        if os.path.exists(fabdem_csv):
            matching_tifs = find_tif_for_coordinates(center_lat, center_lon, fabdem_csv)
            if matching_tifs:
                found_fabdem = True

        if not found_fabdem:
            raise FileNotFoundError(
                f"No elevation data found for coordinates ({center_lat:.4f}, {center_lon:.4f}). "
                f"Please run 'uv run python src/utils/srtm_scanner.py --mode fabdem' "
                f"to scan FABDEM tiles."
            )
        # map_tif will be set after we calculate terrain bounds
        # to ensure we get all tiles needed for the full area
    else:
        # User provided explicit map_tif path - validate it exists
        if isinstance(map_tif, str):
            if not os.path.exists(map_tif):
                raise FileNotFoundError(f"Elevation data file not found: {map_tif}")
        elif isinstance(map_tif, list):
            for tif in map_tif:
                if not os.path.exists(tif):
                    raise FileNotFoundError(f"Elevation data file not found: {tif}")
        else:
            raise ValueError(f"map_tif must be a string or list, got {type(map_tif)}")

    # Ensure output directory exists
    output_dir = _resolve_output_dir(output_dir, track_gpx)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate output directory and filename
    # Create a folder for the model and put all files inside
    if track_gpx:
        model_name = Path(track_gpx).stem
    else:
        # Use hash of bbox coordinates to create consistent filename
        bbox_str = f"{bbox_lat_min:.6f}_{bbox_lat_max:.6f}_{bbox_lon_min:.6f}_{bbox_lon_max:.6f}"
        bbox_hash = hashlib.md5(bbox_str.encode()).hexdigest()[:8]
        model_name = f"terrain_{bbox_hash}"

    # Create model directory
    model_dir = os.path.join(output_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Output path points to {model_name}.obj in model directory
    output = os.path.join(model_dir, f"draft-{model_name}.obj")

    # Prepare model sizes for auto-rotate decision (use 160.0 default for comparison)
    effective_model_x = min_model_size_x_mm if min_model_size_x_mm > 0 else 160.0
    effective_model_y = min_model_size_y_mm if min_model_size_y_mm > 0 else 160.0

    # Effective border for non-rectangle footprints.
    #
    # calculate_terrain_bounds reserves `border_mm` on each side of a
    # rectangular bbox. For hex/circle the inscribed polygon doesn't reach
    # the bbox corners, so a track that fits inside the rectangle with
    # `terrain_border_mm` clearance can still cross the actual polygon edge.
    #
    # Strategy: binary-search for the largest *effective* border, fed into
    # the rectangular calculate_terrain_bounds, such that EVERY GPX track
    # point ends up inside footprint.buffer(-terrain_border_mm) when
    # projected into the model's mm-space. Inflating the rectangular border
    # is equivalent to shrinking the scale, which moves track points closer
    # to the hex centre. This is much less conservative than always
    # adding the worst bbox-corner inset (which would shrink the track to a
    # tiny inscribed rectangle even for tracks that don't reach the corners).
    effective_border_mm = terrain_border_mm
    if model_shape != "rectangle" and track_gpx and os.path.exists(track_gpx):
        effective_border_mm = _solve_effective_border_mm(
            track_gpx=track_gpx,
            model_w_mm=effective_model_x,
            model_h_mm=effective_model_y,
            requested_border_mm=terrain_border_mm,
            model_shape=model_shape,
            hex_orientation=resolved_hex_orientation,
            terrain_resolution=terrain_resolution,
        )
        print(
            f"  Footprint border boost: "
            f"requested {terrain_border_mm:.2f}mm → effective "
            f"{effective_border_mm:.2f}mm "
            f"(ensures track stays inside {model_shape} with clearance)"
        )

    # Calculate terrain bounds from GPX with border or use manual bounding box
    if all(bbox_specified):
        # Use manual bounding box
        lat_avg = (bbox_lat_max + bbox_lat_min) / 2
        meters_per_deg_lat = 111320
        meters_per_deg_lon = 111320 * np.cos(np.radians(lat_avg))

        requested_width_m = abs(bbox_lon_max - bbox_lon_min) * meters_per_deg_lon
        requested_height_m = abs(bbox_lat_max - bbox_lat_min) * meters_per_deg_lat

        # Create bounds dict
        bounds = {
            "lat_top": bbox_lat_max,
            "lat_bottom": bbox_lat_min,
            "lon_left": bbox_lon_min,
            "lon_right": bbox_lon_max,
            "terrain_width_m": requested_width_m,
            "terrain_height_m": requested_height_m,
        }
    else:
        # Calculate from GPX
        if not track_gpx:
            raise ValueError("Internal error: track_gpx required for terrain bounds calculation")

        bounds = calculate_terrain_bounds(
            track_gpx,
            border_mm=effective_border_mm,
            model_size_mm=(effective_model_x, effective_model_y),
        )

    # Auto-rotate decision and application
    actual_model_width_mm = min_model_size_x_mm
    actual_model_height_mm = min_model_size_y_mm
    axes_swapped = False

    if auto_rotate:
        swap_needed = should_swap_model_sizes(
            bounds, effective_model_x, effective_model_y, auto_rotate_tolerance
        )

        if swap_needed:
            print("Auto-rotate: Swapping model dimensions for better aspect ratio match")
            print(f"  Original: {min_model_size_x_mm}mm x {min_model_size_y_mm}mm")

            # Swap the model sizes
            actual_model_width_mm, actual_model_height_mm = (
                actual_model_height_mm,
                actual_model_width_mm,
            )
            effective_model_x, effective_model_y = effective_model_y, effective_model_x
            axes_swapped = True

            print(f"  Swapped:  {actual_model_width_mm}mm x {actual_model_height_mm}mm")

            # Recalculate terrain bounds with swapped sizes
            if not all(bbox_specified):
                bounds = calculate_terrain_bounds(
                    track_gpx,
                    border_mm=effective_border_mm,
                    model_size_mm=(effective_model_x, effective_model_y),
                )

    # Build footprint (after auto-rotate so width/height are final).
    footprint_w = actual_model_width_mm if actual_model_width_mm > 0 else effective_model_x
    footprint_h = actual_model_height_mm if actual_model_height_mm > 0 else effective_model_y

    resolved_hex_orientation = hex_orientation
    if model_shape == "hexagon" and hex_orientation == "auto":
        track_w_m = bounds.get("track_width_m") or bounds["terrain_width_m"]
        track_h_m = bounds.get("track_height_m") or bounds["terrain_height_m"]
        resolved_hex_orientation = Footprint.pick_hex_orientation(track_w_m, track_h_m)

    # A derived hexagon box is not square: pointy-top is 0.866 wide for its
    # height, flat-top 1.155. Swapping the axes turns one into the other, so
    # the orientation has to follow or the polygon comes out stretched while
    # the log still reports the regular size. A box the caller pinned with
    # --model-size-y is theirs, stretched or not, and is left alone.
    if model_shape == "hexagon" and axes_swapped and not user_provided_y:
        resolved_hex_orientation = "flat" if resolved_hex_orientation == "pointy" else "pointy"
        print(f"  Auto-rotate: hexagon is now {resolved_hex_orientation}-top")

    footprint = Footprint.build(
        model_shape,
        footprint_w,
        footprint_h,
        hex_orientation=resolved_hex_orientation if model_shape == "hexagon" else "flat",
        circle_segments=max(64, terrain_resolution * 2),
    )

    shape_label = model_shape
    if model_shape == "hexagon":
        shape_label = f"hexagon ({resolved_hex_orientation}-top)"
    print(f"Footprint: {shape_label} {footprint.width_mm:.1f}mm x {footprint.height_mm:.1f}mm")

    # Find all TIF tiles needed for the terrain bounds
    if map_tif is None:
        fabdem_csv = os.path.join(data_dir, "bounds.csv")
        if os.path.exists(fabdem_csv):
            all_tiles = find_tifs_for_bounds(
                bounds["lat_bottom"],
                bounds["lat_top"],
                bounds["lon_left"],
                bounds["lon_right"],
                fabdem_csv,
            )

            if all_tiles:
                all_tile_paths = [os.path.join(data_dir, tile) for tile in all_tiles]
                existing_tiles = []
                existing_rel = []
                for tile_path in all_tile_paths:
                    if os.path.exists(tile_path):
                        file_size = os.path.getsize(tile_path)
                        if file_size > 10000:
                            existing_tiles.append(tile_path)
                            # Normalize path separators for cross-platform comparison
                            rel_path = os.path.relpath(tile_path, data_dir).replace(os.sep, "/")
                            existing_rel.append(rel_path)

                missing_tiles = [tile for tile in all_tiles if tile not in existing_rel]

                if missing_tiles:
                    raise FileNotFoundError(
                        "Missing FABDEM tiles for requested bounds: " + ", ".join(missing_tiles)
                    )

                if len(existing_tiles) > 1:
                    total_expected = len(all_tiles)
                    if len(existing_tiles) < total_expected:
                        missing_count = total_expected - len(existing_tiles)
                        raise FileNotFoundError(
                            f"Missing {missing_count} FABDEM tiles for requested bounds: "
                            + ", ".join(missing_tiles)
                        )
                    map_tif = existing_tiles
                elif len(existing_tiles) == 1:
                    map_tif = existing_tiles[0]
                else:
                    raise FileNotFoundError("FABDEM tiles referenced but none are available.")

    # Ensure we have valid map_tif
    if map_tif is None:
        raise FileNotFoundError(
            f"No elevation data found for bounds "
            f"({bounds['lat_bottom']:.6f}, {bounds['lon_left']:.6f}) to "
            f"({bounds['lat_top']:.6f}, {bounds['lon_right']:.6f})"
        )

    # Generate terrain with track
    generate_terrain_stl(
        bounds["lat_top"],
        bounds["lon_left"],
        bounds["lat_bottom"],
        bounds["lon_right"],
        map_tif,
        output,
        track_gpx=track_gpx if include_track else None,
        model_width_mm=actual_model_width_mm,
        model_height_mm=actual_model_height_mm,
        vertical_exaggeration=vertical_exaggeration,
        model_z_height_mm=min_model_height_z_mm,
        base_thickness_mm=base_thickness_mm,
        track_width_mm=track_width_mm,
        track_height_mm=track_height_mm,
        include_terrain=True,
        include_water=True,
        simplification_tolerance=simplification_tolerance,
        max_points=max_points,
        terrain_resolution=terrain_resolution,
        terrain_smoothing=terrain_smoothing,
        smoothing_iterations=smoothing_iterations,
        smoothing_strength=smoothing_strength,
        terrain_upsample=terrain_upsample,
        osm_water_detail=water_objects,
        osm_use_cache=use_osm_cache,
        osm_cache_dir=osm_cache_dir,
        footprint=footprint,
    )


def _reject_duplicate_args(argv: list, parser: argparse.ArgumentParser) -> None:
    """
    Fail fast if the same option flag appears more than once in argv.

    argparse normally accepts a repeated flag and silently keeps the last
    value. That hides bugs like two un-commented preset rows in a launch.json
    both passing `--model-size-x`. We scan argv for any token that matches a
    known option string (long or short) and raise SystemExit on the first
    duplicate, listing all offenders.

    Args:
        argv: Argument tokens (excluding the program name).
        parser: The configured ArgumentParser used to enumerate option strings.

    Raises:
        SystemExit: When any option string appears more than once.
    """
    known_opts: set[str] = set()
    for action in parser._actions:
        for opt in action.option_strings:
            known_opts.add(opt)

    seen: dict[str, int] = {}
    for tok in argv:
        # `--flag=value` is a single token; strip the value for matching.
        key = tok.split("=", 1)[0]
        if key in known_opts:
            seen[key] = seen.get(key, 0) + 1

    duplicates = sorted(k for k, count in seen.items() if count > 1)
    if duplicates:
        listing = ", ".join(f"{k} (×{seen[k]})" for k in duplicates)
        parser.error(
            "duplicate option(s) on command line: "
            + listing
            + ". Remove the extra occurrence(s) — argparse would silently "
            "keep only the last value, which usually hides a real bug "
            "(e.g. two un-commented preset rows in launch.json)."
        )


def _resolve_output_dir(output_dir: str | None, track_gpx: str | None) -> str:
    """Where the model goes when `--output` is not given.

    `dirname` is empty for a bare filename such as `track.gpx`, and stripping a
    trailing separator empties a bare root. `makedirs("")` raises, so both need
    an answer, but they need different ones: an empty dirname means "here",
    while `/` is a directory the caller named and must not be quietly swapped
    for wherever the shell happens to be.
    """
    if output_dir is None:
        output_dir = os.path.dirname(track_gpx) if track_gpx else "."
    trimmed = output_dir.rstrip("/\\")
    if trimmed:
        return trimmed
    return output_dir or "."


def _survive_a_narrow_console() -> None:
    """Stop the progress output from killing a run it only meant to narrate.

    The run prints check marks, warning signs and arrows. On Windows a
    redirected stdout uses the ANSI code page rather than UTF-8, and encoding
    one of those raises `UnicodeEncodeError` partway through a multi-minute
    run, after the terrain is built and before anything is written. Replacing
    the character it cannot encode costs one glyph; raising costs the model.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(Exception):
            reconfigure(errors="replace")


def main():
    """Main entry point for the router application."""
    _survive_a_narrow_console()
    parser = argparse.ArgumentParser(
        description="Generate 3D-printable OBJ terrain models from FABDEM/SRTM elevation data with GPX track overlay. "
        "Creates a single integrated model with track conforming to terrain surface.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  Basic usage (auto-sized model with track):
    python main.py track.gpx

  Custom size with simplified track:
    python main.py track.gpx --model-size-x 120 --model-size-y 120 --max-points 300

  Rectangular model (200mm x 100mm):
    python main.py track.gpx --model-size-x 200 --model-size-y 100

  Auto-rotate for best aspect ratio match:
    python main.py track.gpx --model-size-x 200 --model-size-y 120 --auto-rotate

  Manual bounding box with high detail:
    python main.py track.gpx --bbox-lat-min 45.0 --bbox-lat-max 46.0 \\
                     --bbox-lon-min 6.0 --bbox-lon-max 7.0 --terrain-resolution 250

  Terrain only (no track, requires bounding box):
    python main.py --no-track --bbox-lat-min 39.0 --bbox-lat-max 40.0 \\
                   --bbox-lon-min -120.0 --bbox-lon-max -119.0 \\
                   --model-size-x 150 --model-size-y 150

  With ocean/water features at sea level:
    python main.py track.gpx --water-objects 1

  High detail model with smoothing:
    python main.py track.gpx --terrain-resolution 250 --terrain-upsample 3 \\
                   --smoothing-iterations 3 --smoothing-strength 0.4
""",
    )

    parser.add_argument(
        "track_gpx",
        nargs="?",
        help="Path to GPX track file. Required unless --no-track is specified. "
        "Track will be overlaid on terrain and used to auto-calculate bounds if no manual bbox specified.",
    )

    parser.add_argument(
        "--model-shape",
        type=str,
        default="rectangle",
        choices=["rectangle", "hexagon", "circle"],
        help="Footprint shape of the model. 'circle' produces an oval when "
        "--model-size-x != --model-size-y. 'hexagon' uses --hex-orientation. Default: rectangle.",
    )

    parser.add_argument(
        "--hex-orientation",
        type=str,
        default="auto",
        choices=["flat", "pointy", "auto"],
        help="Hexagon orientation: 'flat' (flat edges top/bottom), 'pointy' (vertices top/bottom), "
        "or 'auto' (choose by track bbox aspect). Ignored unless --model-shape=hexagon. Default: auto.",
    )

    parser.add_argument(
        "--model-size-x",
        type=float,
        default=0.0,
        help="Model width (X-axis, East-West) in millimeters, except for "
        "--model-shape=hexagon, where it is the long diagonal and the bounding box comes out "
        "narrower. If 0 or not specified, auto-calculated to fit terrain with borders while "
        "maintaining aspect ratio. Required for non-rectangle shapes. Default: 0 (auto)",
    )

    parser.add_argument(
        "--model-size-y",
        type=float,
        default=0.0,
        help="Model depth (Y-axis, North-South) in millimeters. If 0 or not specified: "
        "for rectangle, auto-calculated; for a circle, equal to --model-size-x; for a "
        "hexagon, derived from the resolved orientation, which is 0.866 or 1.155 times "
        "--model-size-x, never equal to it. Default: 0 (auto)",
    )

    parser.add_argument(
        "--min-model-height-z",
        type=float,
        default=0.0,
        help="Model height budget (Z-axis, vertical) in millimeters, including base and terrain. "
        "The automatic vertical exaggeration is the largest that still fits inside it, so the "
        "model comes out at most this tall and shorter wherever the 4.0 cap binds. Ignored when "
        "--vertical-exaggeration is given. Default: 0, meaning a 10mm budget",
    )

    parser.add_argument(
        "--terrain-border",
        type=float,
        default=10.0,
        help="Extra terrain border around GPX track in millimeters (on model, not geographic). "
        "Provides context and improves print stability. Ignored when using manual bbox. Default: 10.0",
    )

    parser.add_argument(
        "--vertical-exaggeration",
        type=float,
        default=None,
        help="Elevation multiplier to emphasize terrain features. Higher values create more dramatic relief. "
        "If not specified, auto-calculated to fit within --min-model-height-z (typical range: 1.5-3.0). Default: None (auto)",
    )

    parser.add_argument(
        "--base-thickness",
        type=float,
        default=2.0,
        help="Solid base platform thickness in millimeters. Provides structural strength and print stability. "
        "Minimum recommended: 2.0. Default: 2.0",
    )

    parser.add_argument(
        "--track-width",
        type=float,
        default=2.5,
        help="GPX track width in millimeters. Controls route visibility. "
        "Typical range: 1.5-3.5. Too wide may overwhelm terrain details. Default: 2.5",
    )

    parser.add_argument(
        "--track-height",
        type=float,
        default=3.5,
        help="Track protrusion height above terrain surface in millimeters. "
        "Minimum 1.5 recommended for visibility. Default: 3.5",
    )

    parser.add_argument("--output", help="Output directory path.")

    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Generate terrain-only model (no track overlay). Requires manual bounding box (--bbox-* parameters). "
        "Outputs a single OBJ file with terrain mesh only.",
    )

    parser.add_argument(
        "--bbox-lat-min",
        type=float,
        help="Minimum latitude in decimal degrees (e.g., 45.5 or -12.3). All four bbox parameters must be specified together. "
        "Overrides automatic bounds calculation from GPX track.",
    )

    parser.add_argument(
        "--bbox-lat-max",
        type=float,
        help="Maximum latitude in decimal degrees (e.g., 46.5). Must be greater than --bbox-lat-min. "
        "Defines northern edge of terrain area.",
    )

    parser.add_argument(
        "--bbox-lon-min",
        type=float,
        help="Minimum longitude in decimal degrees (e.g., 6.0 or -120.5). Must be less than --bbox-lon-max. "
        "Defines western edge of terrain area.",
    )

    parser.add_argument(
        "--bbox-lon-max",
        type=float,
        help="Maximum longitude in decimal degrees (e.g., 7.5). Must be greater than --bbox-lon-min. "
        "Defines eastern edge of terrain area.",
    )

    parser.add_argument(
        "--simplification-tolerance",
        type=float,
        default=15.0,
        help="Track simplification threshold in meters (Douglas-Peucker algorithm). Higher values = simpler geometry with fewer points. "
        "Lower values preserve more detail. Typical range: 5-30. Overridden by --max-points if specified. Default: 15.0",
    )

    parser.add_argument(
        "--max-points",
        type=int,
        help="Maximum number of track points after simplification. Automatically adjusts --simplification-tolerance to achieve target. "
        "Recommended: 200-300 (moderate detail), 300-500 (high detail). Default: None (unlimited)",
    )

    parser.add_argument(
        "--terrain-resolution",
        type=int,
        default=200,
        help="Terrain mesh grid resolution (maximum dimension in pixels). Controls polygon density and file size. "
        "Higher = more detail but larger files and slower processing. Typical: 100 (~34k faces), 150 (~76k), 200 (~160k), 250 (~250k). Default: 200",
    )

    parser.add_argument(
        "--terrain-smoothing",
        action="store_true",
        default=True,
        help="Enable Laplacian smoothing to reduce blocky grid artifacts (enabled by default). "
        "Use --no-terrain-smoothing for sharper, more angular terrain. Note that turning "
        "smoothing off also disables --terrain-upsample.",
    )

    parser.add_argument(
        "--no-terrain-smoothing",
        action="store_false",
        dest="terrain_smoothing",
        help="Disable terrain smoothing. Results in more angular terrain with visible grid "
        "patterns. This also disables --terrain-upsample, so the mesh keeps the raster resolution.",
    )

    parser.add_argument(
        "--smoothing-iterations",
        type=int,
        default=2,
        help="Number of Laplacian smoothing passes (0-5). Higher values create smoother terrain but may reduce fine detail. "
        "Recommended: 1-3. Default: 2",
    )

    parser.add_argument(
        "--smoothing-strength",
        type=float,
        default=0.3,
        help="Smoothing strength factor (0.0-1.0). Higher values apply more aggressive smoothing per iteration. "
        "Recommended: 0.2-0.4. Default: 0.3",
    )

    parser.add_argument(
        "--terrain-upsample",
        type=int,
        default=2,
        help="Terrain mesh upsampling multiplier (1-4). Higher values create finer mesh detail but significantly increase face count. "
        "1=no upsampling, 2=4x faces, 3=9x faces, 4=16x faces. Default: 2",
    )

    parser.add_argument(
        "--water-objects",
        type=int,
        default=0,
        choices=range(0, 6),
        metavar="0-5",
        help="Water feature rendering with physical size-based filtering: "
        "0=disabled (default), "
        "1=oceans/seas only (from coastline data), "
        "2=level 1 + large lakes (≥5km² OR 0.5%% of bbox, whichever is smaller), "
        "3=level 2 + major rivers (≥40m width AND ≥3km length), "
        "4=level 3 + rivers/streams (≥20m width), "
        "5=level 4 + all waterways (≥3m width AND ≥500m length). "
        "Features: flat lake surfaces, terrain-following rivers at +0.4mm, 1.5mm edge margin "
        "or the river's own half-width, minimum 1.0mm printable width. The lake threshold "
        "adapts to bbox size. Default: 0",
    )

    parser.add_argument(
        "--no-osm-cache",
        action="store_false",
        dest="use_osm_cache",
        help="Disable OpenStreetMap query caching (forces fresh API calls). "
        "Use when testing or when you need the absolute latest OSM data. Caching is enabled by default.",
    )

    parser.add_argument(
        "--auto-rotate",
        action="store_true",
        help="Automatically swap model-size-x and model-size-y if it improves the aspect ratio "
        "match with the track, falling back to the terrain where the track has no extent. "
        "Cannot be used with manual bbox arguments. Both model sizes must be provided together "
        "(both >0) or both left as default.",
    )

    parser.add_argument(
        "--auto-rotate-tolerance",
        type=float,
        default=0.05,
        help="Minimum improvement required for auto-rotate to swap dimensions (default: 0.05 = 5%%). "
        "Higher values require more significant improvement before swapping.",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory containing elevation tiles and bounds.csv (default: 'data'). "
        "Use this to specify a custom location for FABDEM/SRTM data.",
    )

    parser.add_argument(
        "--osm-cache-dir",
        type=str,
        default="osm_cache",
        help="Path to OpenStreetMap cache directory (default: 'osm_cache'). "
        "Use this to specify a custom location for cached OSM queries.",
    )

    # Reject duplicate CLI flags before parsing so accidentally-uncommented
    # blocks in launch configs (e.g. two `--model-size-x` from two preset
    # rows in .vscode/launch.json) fail loudly instead of silently letting
    # argparse keep the last value.
    _reject_duplicate_args(sys.argv[1:], parser)

    args = parser.parse_args()

    options = {
        "track_gpx": args.track_gpx,
        "map_tif": None,
        "output_dir": args.output,
        "min_model_size_x_mm": args.model_size_x,
        "min_model_size_y_mm": args.model_size_y,
        "terrain_border_mm": args.terrain_border,
        "vertical_exaggeration": args.vertical_exaggeration,
        "min_model_height_z_mm": args.min_model_height_z,
        "base_thickness_mm": args.base_thickness,
        "track_width_mm": args.track_width,
        "track_height_mm": args.track_height,
        "include_track": not args.no_track,
        "bbox_lat_min": args.bbox_lat_min,
        "bbox_lat_max": args.bbox_lat_max,
        "bbox_lon_min": args.bbox_lon_min,
        "bbox_lon_max": args.bbox_lon_max,
        "simplification_tolerance": args.simplification_tolerance,
        "max_points": args.max_points,
        "terrain_resolution": args.terrain_resolution,
        "terrain_smoothing": args.terrain_smoothing,
        "smoothing_iterations": args.smoothing_iterations,
        "smoothing_strength": args.smoothing_strength,
        "terrain_upsample": args.terrain_upsample,
        "water_objects": args.water_objects,
        "use_osm_cache": args.use_osm_cache,
        "auto_rotate": args.auto_rotate,
        "auto_rotate_tolerance": args.auto_rotate_tolerance,
        "data_dir": args.data_dir,
        "osm_cache_dir": args.osm_cache_dir,
        "model_shape": args.model_shape,
        "hex_orientation": args.hex_orientation,
    }

    # The messages these raise are written for the person who typed the
    # command. A traceback above them says the tool broke, which is the wrong
    # story for a path that does not exist or a flag out of range.
    try:
        generate_model(**options)
    except (ValueError, FileNotFoundError, RuntimeError) as failure:
        print(f"error: {failure}", file=sys.stderr)
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
