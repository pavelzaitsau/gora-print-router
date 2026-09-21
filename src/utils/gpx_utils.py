# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Utilities for parsing and processing GPX track files."""

import gpxpy
import numpy as np


def parse_gpx_track(gpx_file: str) -> list[tuple[float, float]]:
    """
    Parse GPX file and extract track points as (lat, lon) pairs.

    Args:
        gpx_file: Path to the GPX file

    Returns:
        List of (latitude, longitude) tuples

    Raises:
        FileNotFoundError: If GPX file doesn't exist
        ValueError: If GPX file is invalid or empty
    """
    try:
        with open(gpx_file, encoding="utf-8") as f:
            gpx = gpxpy.parse(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"GPX file not found: {gpx_file}") from exc
    except OSError:
        # A directory, a permission denial, a broken symlink. The file may be
        # perfectly good GPX; the problem is that it cannot be opened, and
        # saying "invalid GPX" sends the reader to check the wrong thing.
        raise
    except Exception as e:
        raise ValueError(f"Invalid GPX file: {gpx_file}: {e}") from e

    # A recorded activity is a <trk>; a planned one is a <rte>. Komoot, Garmin
    # Connect courses and most route planners export the second, and to whoever
    # exported it the file plainly holds a path. Prefer the track where a file
    # carries both: it is the measured line, not the proposed one.
    points = [
        (point.latitude, point.longitude)
        for track in gpx.tracks
        for segment in track.segments
        for point in segment.points
    ]
    if not points:
        points = [
            (point.latitude, point.longitude) for route in gpx.routes for point in route.points
        ]

    if not points:
        raise ValueError(f"No track or route points found in GPX file: {gpx_file}")

    return points


def get_gpx_bounds(gpx_file: str) -> dict[str, float]:
    """
    Get the bounding box of a GPX track.

    Args:
        gpx_file: Path to the GPX file

    Returns:
        Dictionary with keys: lat_min, lat_max, lon_min, lon_max, points_count

    Raises:
        FileNotFoundError: If GPX file doesn't exist
        ValueError: If GPX file is invalid or has no points
    """
    points = parse_gpx_track(gpx_file)

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]

    return {
        "lat_min": min(lats),
        "lat_max": max(lats),
        "lon_min": min(lons),
        "lon_max": max(lons),
        "points_count": len(points),
    }


def calculate_terrain_bounds(
    gpx_file: str, border_mm: float, model_size_mm: tuple[float, float]
) -> dict[str, float]:
    """
    Calculate terrain bounds with padding around GPX track.

    A track that runs along a single meridian or parallel has zero extent in
    one axis. The scale then comes from the other axis alone, and the
    degenerate axis is padded by the border and the model aspect ratio.

    Args:
        gpx_file: Path to GPX file
        border_mm: Border size in mm (terrain without track)
        model_size_mm: Target model size in mm

    Returns:
        Dictionary with terrain bounds and metadata:
        - lat_top, lat_bottom, lon_left, lon_right: Terrain boundaries
        - track_bounds: Original GPX bounds
        - track_width_m, track_height_m: Track dimensions in meters
        - terrain_width_m, terrain_height_m: Terrain dimensions in meters
        - scale: Calculated scale factor
        - border_mm: Border size used

    Raises:
        ValueError: If parameters are invalid, the GPX is empty, or every
            track point shares the same coordinate (no axis has extent, so
            no scale can be derived)
    """
    if border_mm < 0:
        raise ValueError(f"Border must be non-negative, got {border_mm}")
    model_width_mm, model_height_mm = model_size_mm
    if model_width_mm <= 0 or model_height_mm <= 0:
        raise ValueError(
            f"Model size must be positive, got ({model_width_mm}mm, {model_height_mm}mm)"
        )
    if 2 * border_mm >= min(model_width_mm, model_height_mm):
        raise ValueError(
            f"Border ({border_mm}mm) too large for model size ({model_width_mm}mm x {model_height_mm}mm)"
        )

    # Get GPX bounds
    gpx_bounds = get_gpx_bounds(gpx_file)

    # Calculate real-world dimensions of the track
    lat_avg = (gpx_bounds["lat_max"] + gpx_bounds["lat_min"]) / 2
    meters_per_deg_lat = 111320
    meters_per_deg_lon = 111320 * np.cos(np.radians(lat_avg))

    track_width_m = (gpx_bounds["lon_max"] - gpx_bounds["lon_min"]) * meters_per_deg_lon
    track_height_m = (gpx_bounds["lat_max"] - gpx_bounds["lat_min"]) * meters_per_deg_lat

    # Calculate the scale needed for the track to fit within the model rectangle
    # while respecting the border on all sides.
    usable_width_mm = model_width_mm - 2 * border_mm
    usable_height_mm = model_height_mm - 2 * border_mm

    # A track running due north-south has zero width, and one running due
    # east-west zero height. Scale from the axis that still has extent instead
    # of dividing by zero; a track with no extent at all gives nothing to
    # scale from.
    if track_width_m <= 0 and track_height_m <= 0:
        raise ValueError(
            "Track spans no distance: all GPX points share the same coordinate, "
            "so no scale can be derived"
        )
    if track_width_m <= 0:
        scale = usable_height_mm / track_height_m
    elif track_height_m <= 0:
        scale = usable_width_mm / track_width_m
    else:
        # Use the smaller scale so the track fits both dimensions
        scale = min(usable_width_mm / track_width_m, usable_height_mm / track_height_m)

    # Calculate how much border we need in geographic coordinates (degrees)
    border_lat_deg = (border_mm / scale) / meters_per_deg_lat
    border_lon_deg = (border_mm / scale) / meters_per_deg_lon

    # Apply padding to create initial terrain bounds
    terrain_bounds = {
        "lat_top": gpx_bounds["lat_max"] + border_lat_deg,
        "lat_bottom": gpx_bounds["lat_min"] - border_lat_deg,
        "lon_left": gpx_bounds["lon_min"] - border_lon_deg,
        "lon_right": gpx_bounds["lon_max"] + border_lon_deg,
        "track_bounds": gpx_bounds,
        "track_width_m": track_width_m,
        "track_height_m": track_height_m,
        "scale": scale,
        "border_mm": border_mm,
    }

    # Calculate final terrain dimensions
    terrain_width_m = (
        terrain_bounds["lon_right"] - terrain_bounds["lon_left"]
    ) * meters_per_deg_lon
    terrain_height_m = (
        terrain_bounds["lat_top"] - terrain_bounds["lat_bottom"]
    ) * meters_per_deg_lat

    # Adjust bounds to match the model aspect ratio by expanding the shorter dimension
    desired_ratio = model_width_mm / model_height_mm

    # A degenerate axis reaches here unpadded when the border is zero. Name
    # the ratio it implies instead of dividing by that zero: the expansion
    # below then widens the flat axis to the model aspect, which is what the
    # division used to arrive at through numpy's inf and a RuntimeWarning.
    if terrain_height_m <= 0:
        current_ratio = float("inf")
    elif terrain_width_m <= 0:
        current_ratio = 0.0
    else:
        current_ratio = terrain_width_m / terrain_height_m

    if abs(current_ratio - desired_ratio) > 1e-6:
        if current_ratio < desired_ratio:
            # Terrain is too tall/narrow; expand longitude span
            target_width_m = terrain_height_m * desired_ratio
            extra_width_m = target_width_m - terrain_width_m
            extra_lon_deg = (extra_width_m / meters_per_deg_lon) / 2
            terrain_bounds["lon_left"] -= extra_lon_deg
            terrain_bounds["lon_right"] += extra_lon_deg
            terrain_width_m = target_width_m
        else:
            # Terrain is too wide/short; expand latitude span
            target_height_m = terrain_width_m / desired_ratio
            extra_height_m = target_height_m - terrain_height_m
            extra_lat_deg = (extra_height_m / meters_per_deg_lat) / 2
            terrain_bounds["lat_bottom"] -= extra_lat_deg
            terrain_bounds["lat_top"] += extra_lat_deg
            terrain_height_m = target_height_m

    terrain_bounds["terrain_width_m"] = terrain_width_m
    terrain_bounds["terrain_height_m"] = terrain_height_m

    return terrain_bounds
