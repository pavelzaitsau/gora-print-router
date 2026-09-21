# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mesh generation utilities for terrain and track OBJ models."""

import contextlib
import itertools

import numpy as np
import rasterio
import shapely
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.windows import from_bounds
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import zoom
from scipy.spatial import Delaunay, cKDTree

from src.core.footprint import Footprint
from src.core.water_generator import WaterLayerGenerator
from src.utils.gpx_utils import parse_gpx_track
from src.utils.obj_exporter import Mesh, save_combined_mesh_obj, save_mesh_obj

# A geographic point, degrees, in the order this module passes it around.
LatLon = tuple[float, float]
# A model-space point, millimetres, in the XY plane of the print bed.
PointXY = tuple[float, float]

# Track embedding configuration
# Negative value = track base embedded below terrain surface
# Track rises from this embedded base to track_height_mm above terrain
TRACK_EMBEDDING_DEPTH_MM = -0.6


def _clip_segment_to_bbox(
    p1: tuple[float, float],
    p2: tuple[float, float],
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> list[tuple[float, float]]:
    """
    Clip a line segment to a bounding box using Cohen-Sutherland algorithm.

    Args:
        p1: First point (lat, lon)
        p2: Second point (lat, lon)
        lat_min, lat_max, lon_min, lon_max: Bounding box boundaries

    Returns:
        List of clipped points (empty if segment is completely outside,
        [p1_clipped, p2_clipped] if segment intersects or is inside bbox)
    """

    # Cohen-Sutherland outcodes
    INSIDE = 0  # 0000
    LEFT = 1  # 0001
    RIGHT = 2  # 0010
    BOTTOM = 4  # 0100
    TOP = 8  # 1000

    def compute_outcode(lat: float, lon: float) -> int:
        code = INSIDE
        if lon < lon_min:
            code |= LEFT
        elif lon > lon_max:
            code |= RIGHT
        if lat < lat_min:
            code |= BOTTOM
        elif lat > lat_max:
            code |= TOP
        return code

    lat1, lon1 = p1
    lat2, lon2 = p2
    outcode1 = compute_outcode(lat1, lon1)
    outcode2 = compute_outcode(lat2, lon2)

    while True:
        # Both points inside
        if outcode1 == 0 and outcode2 == 0:
            return [(lat1, lon1), (lat2, lon2)]

        # Both points outside on same side
        if (outcode1 & outcode2) != 0:
            return []

        # At least one point outside - clip
        outcode_out = outcode1 if outcode1 != 0 else outcode2

        # Find intersection point
        # Initialize to avoid linter warnings
        lat = 0.0
        lon = 0.0

        if outcode_out & TOP:
            lat = lat_max
            lon = lon1 + (lon2 - lon1) * (lat_max - lat1) / (lat2 - lat1)
        elif outcode_out & BOTTOM:
            lat = lat_min
            lon = lon1 + (lon2 - lon1) * (lat_min - lat1) / (lat2 - lat1)
        elif outcode_out & RIGHT:
            lon = lon_max
            lat = lat1 + (lat2 - lat1) * (lon_max - lon1) / (lon2 - lon1)
        elif outcode_out & LEFT:
            lon = lon_min
            lat = lat1 + (lat2 - lat1) * (lon_min - lon1) / (lon2 - lon1)

        # Replace point outside with intersection point
        if outcode_out == outcode1:
            lat1, lon1 = lat, lon
            outcode1 = compute_outcode(lat1, lon1)
        else:
            lat2, lon2 = lat, lon
            outcode2 = compute_outcode(lat2, lon2)


def clip_track_to_bbox(
    track_points: list[tuple[float, float]],
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> list[tuple[float, float]]:
    """
    Clip a track to bounding box boundaries.

    Segments that cross bbox boundaries are clipped at the intersection point.
    Segments completely outside bbox are discarded.
    Result is a continuous path of all track portions inside bbox.

    Args:
        track_points: List of (lat, lon) track points
        lat_min, lat_max, lon_min, lon_max: Bounding box boundaries

    Returns:
        List of clipped track points (may be empty if track doesn't intersect bbox)
    """
    if len(track_points) < 2:
        return track_points

    clipped_points: list[LatLon] = []

    for i in range(len(track_points) - 1):
        p1 = track_points[i]
        p2 = track_points[i + 1]

        # Clip segment to bbox
        segment = _clip_segment_to_bbox(p1, p2, lat_min, lat_max, lon_min, lon_max)

        if segment:
            # Add first point if this is the start or if it's different from last added
            if not clipped_points or segment[0] != clipped_points[-1]:
                clipped_points.append(segment[0])
            # Always add second point
            clipped_points.append(segment[1])

    return clipped_points


def simplify_track_points(points: list[LatLon], tolerance: float = 10.0) -> list[LatLon]:
    """
    Simplify track points using Ramer-Douglas-Peucker.

    Iterative implementation with a vectorised perpendicular-distance pass over
    each sub-array. The previous version allocated six numpy arrays *per point
    per recursion level*, which made a 1300-point track take many seconds on
    each call; multiplied by the `--max-points` retry loop that re-runs RDP
    with growing tolerance, the simplification could hang for minutes. The new
    implementation pre-converts all coordinates to a local metric frame once,
    then for each (lo, hi) sub-range computes every interior point's distance
    to the chord [lo, hi] in a single numpy expression.

    Args:
        points: List of (lat, lon) tuples.
        tolerance: Max perpendicular distance from chord, in metres.

    Returns:
        Simplified list of (lat, lon) tuples in original order.
    """
    n = len(points)
    if n < 3:
        return list(points)

    # Project once to local metres around the route's mean latitude.
    arr = np.asarray(points, dtype=float)
    lat_mean = float(arr[:, 0].mean())
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * float(np.cos(np.radians(lat_mean)))
    xy = np.empty_like(arr)
    xy[:, 0] = arr[:, 0] * m_per_deg_lat
    xy[:, 1] = arr[:, 1] * m_per_deg_lon

    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True

    # Iterative RDP with an explicit stack of (lo, hi) index ranges.
    stack: list[tuple[int, int]] = [(0, n - 1)]
    eps2 = tolerance * tolerance  # compare squared distances to skip a sqrt

    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue

        p_lo = xy[lo]
        p_hi = xy[hi]
        seg = p_hi - p_lo
        seg_len2 = float(seg[0] * seg[0] + seg[1] * seg[1])

        # Sub-range of interior points (exclusive of endpoints).
        sub = xy[lo + 1 : hi]

        if seg_len2 == 0.0:
            # Degenerate chord — distance is just Euclidean to p_lo.
            d2 = (sub[:, 0] - p_lo[0]) ** 2 + (sub[:, 1] - p_lo[1]) ** 2
        else:
            # Squared perpendicular distance from each sub point to the
            # infinite line through p_lo–p_hi:
            #   d = |(p - p_lo) × seg| / |seg|
            dx = sub[:, 0] - p_lo[0]
            dy = sub[:, 1] - p_lo[1]
            cross = dx * seg[1] - dy * seg[0]
            d2 = (cross * cross) / seg_len2

        max_local = int(np.argmax(d2))
        max_d2 = float(d2[max_local])
        max_idx = lo + 1 + max_local

        if max_d2 > eps2:
            keep[max_idx] = True
            stack.append((lo, max_idx))
            stack.append((max_idx, hi))
        # else: drop all interior points in this range

    return [tuple(p) for p in arr[keep]]


def sample_z_from_terrain_mesh(
    x_mm: float,
    y_mm: float,
    terrain_vertices: np.ndarray,
    terrain_rows: int,
    terrain_cols: int,
    width_mm: float,
    height_mm: float,
) -> float:
    """
    Sample Z coordinate from terrain mesh using barycentric interpolation.

    This gives the exact elevation of the final terrain mesh surface at
    any XY position by finding the containing triangle and interpolating.

    Args:
        x_mm, y_mm: Query position in model space
        terrain_vertices: Terrain mesh vertex array (only top surface)
        terrain_rows, terrain_cols: Grid dimensions
        width_mm, height_mm: Model dimensions

    Returns:
        Z coordinate at the given XY position from terrain mesh
    """
    # Find grid cell containing this point
    col_f = (x_mm / width_mm) * (terrain_cols - 1)
    row_f = (y_mm / height_mm) * (terrain_rows - 1)

    # Clamp to valid range
    col_f = np.clip(col_f, 0, terrain_cols - 1.001)
    row_f = np.clip(row_f, 0, terrain_rows - 1.001)

    col = int(col_f)
    row = int(row_f)

    # Get fractional part for interpolation
    u = col_f - col
    v = row_f - row

    # Ensure we don't go out of bounds
    if col >= terrain_cols - 1:
        col = terrain_cols - 2
        u = 1.0
    if row >= terrain_rows - 1:
        row = terrain_rows - 2
        v = 1.0

    # Get the four corner vertices of the grid cell
    # Grid layout: v0--v1
    #              |  /|
    #              | / |
    #              |/  |
    #              v2--v3
    idx_v0 = row * terrain_cols + col
    idx_v1 = row * terrain_cols + (col + 1)
    idx_v2 = (row + 1) * terrain_cols + col
    idx_v3 = (row + 1) * terrain_cols + (col + 1)

    v0 = terrain_vertices[idx_v0]
    v1 = terrain_vertices[idx_v1]
    v2 = terrain_vertices[idx_v2]
    v3 = terrain_vertices[idx_v3]

    # Determine which triangle the point is in
    # Triangle 1: v0, v1, v2
    # Triangle 2: v1, v3, v2
    if u + v <= 1.0:
        # Point is in triangle v0-v1-v2
        # Barycentric coordinates: w0 = 1-u-v, w1 = u, w2 = v
        z = v0[2] * (1 - u - v) + v1[2] * u + v2[2] * v
    else:
        # Point is in triangle v1-v3-v2, whose corners sit at (u,v) = (1,0),
        # (1,1) and (0,1). Solving P = w1*v1 + w3*v3 + w2*v2 with the weights
        # summing to 1 gives w1 = 1-v, w3 = u+v-1, w2 = 1-u. Each weight
        # belongs to the corner *opposite* the edge it measures — pairing them
        # the other way round leaves the two triangles disagreeing along the
        # diagonal, which crinkled the track riding on this surface.
        z = v1[2] * (1 - v) + v3[2] * (u + v - 1) + v2[2] * (1 - u)

    return float(z)


def upsample_elevation_grid(
    elevation_data: np.ndarray,
    upsample_factor: int = 2,
) -> np.ndarray:
    """
    Upsample elevation grid using bilinear interpolation for smoother terrain.

    Args:
        elevation_data: Original elevation data array (rows x cols)
        upsample_factor: Multiplier for grid resolution (2 = 2x finer grid)

    Returns:
        Upsampled elevation data with smoother transitions between cells
    """
    if upsample_factor <= 1:
        return elevation_data

    rows, cols = elevation_data.shape

    # Create coordinate arrays for original grid
    y_orig = np.arange(rows)
    x_orig = np.arange(cols)

    # Create interpolator using bilinear interpolation
    interpolator = RectBivariateSpline(y_orig, x_orig, elevation_data, kx=1, ky=1)

    # Create finer grid coordinates
    new_rows = (rows - 1) * upsample_factor + 1
    new_cols = (cols - 1) * upsample_factor + 1
    y_new = np.linspace(0, rows - 1, new_rows)
    x_new = np.linspace(0, cols - 1, new_cols)

    # Interpolate to new grid. RectBivariateSpline is untyped, so pin the result.
    upsampled_data: np.ndarray = interpolator(y_new, x_new)

    return upsampled_data


def apply_laplacian_smoothing(
    elevation_data: np.ndarray,
    iterations: int = 2,
    smoothing_factor: float = 0.3,
) -> np.ndarray:
    """
    Apply Laplacian smoothing to reduce high-frequency terrain artifacts.

    This preserves overall terrain shape while smoothing grid-like patterns.
    Edge vertices are preserved to maintain watertight mesh boundaries.

    Args:
        elevation_data: Elevation data array (rows x cols)
        iterations: Number of smoothing iterations (default: 2)
        smoothing_factor: Smoothing strength 0-1 (default: 0.3, higher = more smoothing)

    Returns:
        Smoothed elevation data
    """
    if iterations <= 0 or smoothing_factor <= 0:
        return elevation_data

    smoothed = elevation_data.copy()
    rows, cols = smoothed.shape

    for _ in range(iterations):
        new_smoothed = smoothed.copy()

        # Apply Laplacian smoothing to interior points only
        # Edge points are preserved to maintain watertight boundaries
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # Calculate Laplacian (average of 4-neighbors minus center)
                neighbors_avg = (
                    smoothed[i - 1, j]  # North
                    + smoothed[i + 1, j]  # South
                    + smoothed[i, j - 1]  # West
                    + smoothed[i, j + 1]  # East
                ) / 4.0

                # Blend between original and neighbor average
                new_smoothed[i, j] = (
                    smoothed[i, j] * (1.0 - smoothing_factor) + neighbors_avg * smoothing_factor
                )

        smoothed = new_smoothed

    return smoothed


def _clip_track_to_footprint(
    track_points: list[LatLon],
    footprint: Footprint,
    lat_bottom: float,
    lat_top: float,
    lon_left: float,
    lon_right: float,
    width_mm: float,
    height_mm: float,
    margin_mm: float = 0.0,
) -> list[LatLon]:
    """
    Clip a (lat, lon) track to the footprint polygon (mesh-space).

    Segments fully outside the polygon are dropped. Segments crossing the
    boundary are clipped at the intersection. Result preserves order.

    `margin_mm` pulls the clip boundary inwards. What has to fit inside the
    model is the swept ribbon, not the centreline it is swept along, so the
    caller passes half the ribbon width: without it the outer edge hangs over
    the outline with no base under it. A model barely wider than the ribbon
    insets to nothing, and a track grazing the edge beats no track at all, so
    an empty or broken inset falls back to the outline itself.
    """
    if not track_points:
        return []

    def to_mesh(lat: float, lon: float) -> PointXY:
        x = (lon - lon_left) / (lon_right - lon_left) * width_mm
        y = (lat - lat_bottom) / (lat_top - lat_bottom) * height_mm
        return x, y

    # Inverse map: x_mm/width_mm = (lon-lon_left)/(lon_right-lon_left)
    def from_mesh(xy: PointXY) -> LatLon:
        x, y = xy
        lon = lon_left + (x / width_mm) * (lon_right - lon_left)
        lat = lat_bottom + (y / height_mm) * (lat_top - lat_bottom)
        return (lat, lon)

    poly = footprint.polygon
    if margin_mm > 0.0:
        inset = poly.buffer(-margin_mm)
        if not inset.is_empty and inset.is_valid:
            poly = inset
    out: list[LatLon] = []

    # Vectorised inside-polygon test for all (lat, lon) points.
    xs = np.array([to_mesh(p[0], p[1])[0] for p in track_points])
    ys = np.array([to_mesh(p[0], p[1])[1] for p in track_points])
    inside_mask = shapely.contains_xy(poly, xs, ys)

    from shapely.geometry import LineString

    for i in range(len(track_points) - 1):
        p1 = track_points[i]
        p2 = track_points[i + 1]
        in1 = bool(inside_mask[i])
        in2 = bool(inside_mask[i + 1])

        if in1 and in2:
            if not out or out[-1] != p1:
                out.append(p1)
            out.append(p2)
        elif not in1 and not in2:
            # Both outside; segment may still cross the polygon — handle below.
            x1, y1 = to_mesh(*p1)
            x2, y2 = to_mesh(*p2)
            seg = LineString([(x1, y1), (x2, y2)])
            inter = poly.intersection(seg)
            if inter.is_empty:
                continue
            # Convert intersection back to lat/lon (linear).
            geoms = (
                [inter]
                if inter.geom_type == "LineString"
                else list(getattr(inter, "geoms", [inter]))
            )
            for g in geoms:
                if g.geom_type != "LineString" or g.is_empty:
                    continue
                coords = list(g.coords)
                if len(coords) < 2:
                    continue
                first_xy, last_xy = coords[0], coords[-1]

                a = from_mesh(first_xy)
                b = from_mesh(last_xy)
                if not out or out[-1] != a:
                    out.append(a)
                out.append(b)
        else:
            # One in, one out — find intersection with polygon boundary.
            x1, y1 = to_mesh(*p1)
            x2, y2 = to_mesh(*p2)
            seg = LineString([(x1, y1), (x2, y2)])
            inter = poly.intersection(seg)
            if inter.is_empty or inter.geom_type != "LineString":
                # Fall back to whichever endpoint is inside.
                if in1 and (not out or out[-1] != p1):
                    out.append(p1)
                continue
            coords = list(inter.coords)
            if len(coords) < 2:
                if in1 and (not out or out[-1] != p1):
                    out.append(p1)
                continue

            a = from_mesh(coords[0])
            b = from_mesh(coords[-1])
            if not out or out[-1] != a:
                out.append(a)
            out.append(b)

    return out


# --- Ribbon sweep geometry -------------------------------------------------
# A ribbon of half-width w can only follow a turn whose radius of curvature
# stays above w. Below that its inner edge runs backwards: the wall quads
# between two cross-sections fold inside out and the tube passes through
# itself. The mesh stays closed — every watertight check still passes — but the
# slicer meets inward-facing normals and prints a pinch or a void, which is what
# a switchback tighter than the ribbon used to produce. At print scale that is
# not exotic: on an 80mm model of a 10km box, half a 1.5mm ribbon is ~90m on the
# ground, so every real hairpin is tighter than the ribbon is wide.
PATH_STEP_FRACTION = 0.5
"""Closest two path points may be, as a fraction of the half-width.

A GPS trace simplified in metres arrives with its points far closer together
than the ribbon is wide, and the jitter between them turns into curvature the
sweep then has to clamp the ribbon away to survive."""
PATH_TOLERANCE_FRACTION = 1.0
"""How far thinning the path may move it, as a fraction of the half-width.

One half-width is the honest ceiling: the ribbon drawn on the thinned path
still covers the trace it came from. Anything finer than that is detail a
ribbon 200m wide on the ground cannot show anyway."""
JOIN_ROUNDING_TURN_DEG = 30.0
"""Corners turning more than this are replaced by an arc before sweeping.

One cross-section per path point means a corner asks the cross-section to
rotate by the whole turn in a single step: the strip between its neighbours
twists, and at a reversal it twists through itself. An arc spreads that
rotation over several steps.
"""
JOIN_RADIUS_FACTOR = 1.15
"""Arc radius as a multiple of the half-width — just above the offset limit,
so rounding a corner does not itself trip the curvature clamp below."""
MIN_ARC_STEP_MM = 0.02
"""Shortest arc step. Above the exporter's write precision and above the
revisit search radius, so a tight tip cannot be mistaken for a retrace."""
MIN_JOIN_RADIUS_MM = 0.03
"""Smallest arc the sweep will draw. A tangent arc thinner than this is shorter
than one write step, so the tip keeps this radius and rejoins the outgoing arm
with a kink instead — invisible next to a 0.4mm nozzle, and the only way an
exact out-and-back reversal gets an arc at all."""
MAX_MITRE_STRETCH = 2.0
"""Mitre limit for the gentle joins that keep a single cross-section."""
OFFSET_SAFETY = 0.95
"""Offsets stay this fraction below the length that would invert an edge."""
MIN_HALF_WIDTH_MM = 0.02
"""...but never collapse onto the centre line: coincident vertices weld into a
non-manifold pinch."""
JOIN_MAX_SHIFT_FRACTION = 0.25
"""How far a rounded corner may pull the path off the original tip, as a
fraction of the half-width. Kept small because a pinched tip is no longer a
problem: the beads bridge it at full width, so fidelity wins over sweeping."""
JOIN_TRIM_FRACTION = 0.45
"""A rounded join never eats more than this much of either adjacent segment, so
two neighbouring corners cannot trade places along the segment between them."""
OFFSET_RELAXATION_PASSES = 8
"""Each clamp lowers two offsets, which can put their other joins over the
limit; the sweep re-checks this many times before giving up."""
MIN_SWEPT_WIDTH_FRACTION = 0.7
"""Narrowest the swept tube may get, as a fraction of the requested half-width.

Below this the cross-section is a knife edge: nothing a nozzle can lay down,
and — because the ring's height goes with its width — a ridge sunk into the
hillside. Those points are covered by beads instead.

Do not tune this to chase folds. Measured over twelve real routes, the total
folded pairs in the track group go 1110 at 0.7, 1044 at 0.8, 1047 at 0.85 and
1060 at 0.9: a shallow basin worth about six per cent, with individual routes
getting worse either way. The folds come from somewhere else."""
BEAD_OVERLAP_FRACTION = 0.35
"""How far past its anchors a bead reaches, as a fraction of the half-width.
Enough that neighbouring beads share solid instead of touching at a point,
small enough that a bevelled corner does not grow a lip."""
WIDTH_SLOPE_LIMIT = 0.5
"""How much the half-width may change per millimetre travelled. Keeps a
narrowed ring from cutting a notch its neighbours' walls fold across."""


def _unit(vec_x: float, vec_y: float) -> PointXY:
    """Normalise a 2D vector; a zero-length one keeps its direction of +X."""
    length = float(np.hypot(vec_x, vec_y))
    if length < 1e-12:
        return (1.0, 0.0)
    return (vec_x / length, vec_y / length)


def _turn_rad(before: PointXY, after: PointXY) -> float:
    """Angle between two unit directions, 0 (straight on) to pi (reversal)."""
    return float(np.arccos(np.clip(before[0] * after[0] + before[1] * after[1], -1.0, 1.0)))


def _round_sharp_joins(
    path: list[PointXY], join_radius_mm: float, max_shift_mm: float
) -> list[PointXY]:
    """Replace sharp corners with tangent arcs so the cross-section never spins in place.

    The arc is tangent to both arms and has radius `join_radius_mm`, shrunk
    whenever the neighbouring segments are too short to give it room — a hairpin
    between two long arms still ends in a near-zero radius, and the offset clamp
    in `_sweep_offsets` narrows the ribbon there rather than inverting it.

    Below `MIN_JOIN_RADIUS_MM` the arc stops being tangent: it keeps that radius
    and rejoins the outgoing arm with a kink of a few hundredths of a
    millimetre. A tangent arc there would be shorter than one write step, and a
    turn left unrounded flips the cross-section end for end — an exact
    out-and-back reversal is the case that has no arc at all.

    The radius is also held down to whatever keeps the arc within
    `max_shift_mm` of the corner it replaces. A tangent arc pulls the path
    `r * (1/sin(interior/2) - 1)` off the tip, which near a reversal is many
    times the radius: rounding a 160 degree corner at the full radius would
    quietly move the route by several millimetres. The ribbon narrows through
    the tighter arc instead, and that narrowing hides inside the overlap the
    two arms already have.

    Args:
        path: Polyline positions in model space, millimetres, no repeats.
        join_radius_mm: Target radius for a rounded corner.
        max_shift_mm: How far the arc may pull the path off the original corner.

    Returns:
        The polyline with an arc substituted for every corner sharper than
        `JOIN_ROUNDING_TURN_DEG`. Endpoints are never moved.
    """
    if len(path) < 3:
        return list(path)

    max_turn = np.radians(JOIN_ROUNDING_TURN_DEG)
    out: list[PointXY] = [path[0]]
    for corner_idx in range(1, len(path) - 1):
        (ax, ay), (cx, cy), (bx, by) = path[corner_idx - 1], path[corner_idx], path[corner_idx + 1]
        arm_in = float(np.hypot(cx - ax, cy - ay))
        arm_out = float(np.hypot(bx - cx, by - cy))
        dir_in = _unit(cx - ax, cy - ay)
        dir_out = _unit(bx - cx, by - cy)
        turn = _turn_rad(dir_in, dir_out)
        if turn <= max_turn:
            out.append((cx, cy))
            continue

        # Tangent length for a circle of radius r meeting arms at interior
        # angle (pi - turn): t = r / tan((pi - turn) / 2). Trimmed to what the
        # arms can spare, which is what shrinks the radius on a hairpin.
        half_interior = (np.pi - turn) / 2
        tan_half = float(np.tan(half_interior))
        sin_half = float(np.sin(half_interior))
        trim_ideal = join_radius_mm / tan_half if tan_half > 1e-9 else float("inf")
        # r * (1/sin - 1) <= max_shift, solved for r and turned back into a
        # tangent length.
        radius_for_shift = (
            max_shift_mm * sin_half / (1.0 - sin_half) if sin_half < 1.0 - 1e-9 else float("inf")
        )
        trim_for_shift = radius_for_shift / tan_half if tan_half > 1e-9 else float("inf")
        trim = min(
            trim_ideal,
            trim_for_shift,
            JOIN_TRIM_FRACTION * arm_in,
            JOIN_TRIM_FRACTION * arm_out,
        )
        radius = max(trim * tan_half, MIN_JOIN_RADIUS_MM)
        if trim < MIN_JOIN_RADIUS_MM:
            # Both arms are shorter than the smallest arc worth drawing: this is
            # a sub-resolution wobble, and rounding it would move the path
            # further than the feature itself measures.
            out.append((cx, cy))
            continue
        steps = min(int(np.ceil(turn / max_turn)), int(radius * turn / MIN_ARC_STEP_MM))
        if steps < 2:
            out.append((cx, cy))
            continue

        start = (cx - dir_in[0] * trim, cy - dir_in[1] * trim)
        # Cross product picks the inside of the turn; an exact reversal has no
        # inside, so either hand does — the tip loops around one way or the other.
        cross = dir_in[0] * dir_out[1] - dir_in[1] * dir_out[0]
        turn_sign = 1.0 if cross >= 0 else -1.0
        centre = (
            start[0] + turn_sign * (-dir_in[1]) * radius,
            start[1] + turn_sign * dir_in[0] * radius,
        )
        start_angle = float(np.arctan2(start[1] - centre[1], start[0] - centre[0]))
        out.append(start)
        for step in range(1, steps + 1):
            angle = start_angle + turn_sign * turn * step / steps
            out.append(
                (
                    centre[0] + radius * float(np.cos(angle)),
                    centre[1] + radius * float(np.sin(angle)),
                )
            )

    out.append(path[-1])
    return out


def _resample_path(path: list[PointXY], min_step_mm: float, tolerance_mm: float) -> list[PointXY]:
    """Drop path detail finer than the ribbon can express.

    A GPX simplified to ten metres lands its points a few hundredths of a
    millimetre apart at model scale, an order of magnitude below the ribbon's
    own width, and every one of them carries GPS jitter. Sweeping that asks the
    cross-section to swing about between points closer together than it is
    wide, which the curvature clamp can only answer by shrinking the ribbon to
    nothing. Thinning the path first is what keeps it at full width.

    Douglas-Peucker first, so corners survive and only noise inside
    `tolerance_mm` goes, then a minimum spacing pass for the points a corner
    leaves bunched. Both ends are always kept.

    Args:
        path: Polyline positions in model space, millimetres.
        min_step_mm: Closest two kept points may be.
        tolerance_mm: Largest deviation the thinning may introduce.

    Returns:
        The thinned polyline.
    """
    if len(path) < 3:
        return list(path)

    points = np.asarray(path, dtype=np.float64)
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        chord = points[last] - points[first]
        chord_len = float(np.hypot(*chord))
        offsets = points[first + 1 : last] - points[first]
        if chord_len < 1e-12:
            distances = np.hypot(offsets[:, 0], offsets[:, 1])
        else:
            distances = np.abs(chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0]) / chord_len
        worst = int(np.argmax(distances))
        if float(distances[worst]) <= tolerance_mm:
            continue
        split = first + 1 + worst
        keep[split] = True
        stack.append((first, split))
        stack.append((split, last))

    thinned: list[PointXY] = [path[0]]
    for index in np.flatnonzero(keep)[1:-1]:
        if float(np.hypot(*(points[index] - np.asarray(thinned[-1])))) >= min_step_mm:
            thinned.append(path[int(index)])
    thinned.append(path[-1])
    return thinned


def _subdivide_path(path: list[PointXY], max_step_mm: float) -> list[PointXY]:
    """Split every segment longer than `max_step_mm` into equal pieces.

    Args:
        path: Polyline positions in model space, millimetres.
        max_step_mm: Longest segment to leave alone.

    Returns:
        The polyline with sub-points inserted; the original points are kept.
    """
    if len(path) < 2 or max_step_mm <= 0:
        return list(path)

    out: list[PointXY] = [path[0]]
    for (x_a, y_a), (x_b, y_b) in itertools.pairwise(path):
        pieces = int(np.ceil(float(np.hypot(x_b - x_a, y_b - y_a)) / max_step_mm))
        for piece in range(1, pieces):
            fraction = piece / pieces
            out.append((x_a + fraction * (x_b - x_a), y_a + fraction * (y_b - y_a)))
        out.append((x_b, y_b))
    return out


def _sweep_offsets(path: list[PointXY], half_width_mm: float) -> list[PointXY]:
    """Half-width vectors for every cross-section of a swept ribbon.

    The ring at `path[i]` spans `path[i] +- result[i]`. Interior points get the
    mitred normal (the two segment normals averaged) stretched by the usual
    1/cos(turn/2) so the ribbon keeps its width through a bend, capped at
    `MAX_MITRE_STRETCH`.

    Every offset is then relaxed until no wall quad can invert. Between two
    cross-sections the edge on either side advances by
    `L - (w_i*|n_i.d| + w_j*|n_j.d|)`; once that goes negative the quad folds
    inside out, so the two offsets are scaled until it does not.

    Args:
        path: Polyline positions in model space, millimetres, no repeats.
        half_width_mm: Requested half-width of the ribbon.

    Returns:
        One offset vector per path point, never longer than
        `half_width_mm * MAX_MITRE_STRETCH` and never shorter than
        `MIN_HALF_WIDTH_MM`.
    """
    if len(path) < 2:
        return []

    directions: list[PointXY] = []
    lengths: list[float] = []
    for (x_a, y_a), (x_b, y_b) in itertools.pairwise(path):
        lengths.append(float(np.hypot(x_b - x_a, y_b - y_a)))
        directions.append(_unit(x_b - x_a, y_b - y_a))

    normals: list[PointXY] = []
    widths: list[float] = []
    for point_idx in range(len(path)):
        before = directions[point_idx - 1] if point_idx > 0 else None
        after = directions[point_idx] if point_idx < len(directions) else None
        if before is None or after is None:
            segment = after if before is None else before
            assert segment is not None
            normals.append((-segment[1], segment[0]))
            widths.append(half_width_mm)
            continue
        mitre_x, mitre_y = before[0] + after[0], before[1] + after[1]
        mitre_len = float(np.hypot(mitre_x, mitre_y))
        if mitre_len < 1e-6:  # reversal: the mitre direction is undefined
            normals.append((-before[1], before[0]))
            widths.append(half_width_mm)
            continue
        # |before + after| == 2*cos(turn/2) for unit directions, so the
        # cross-section has to grow by 1/cos(turn/2) to keep the ribbon the
        # requested width through the corner.
        direction = (mitre_x / mitre_len, mitre_y / mitre_len)
        normals.append((-direction[1], direction[0]))
        widths.append(half_width_mm * min(MAX_MITRE_STRETCH, 2.0 / mitre_len))

    for _ in range(OFFSET_RELAXATION_PASSES):
        clamped = False
        for join_idx, direction in enumerate(directions):
            lead = abs(normals[join_idx][0] * direction[0] + normals[join_idx][1] * direction[1])
            trail = abs(
                normals[join_idx + 1][0] * direction[0] + normals[join_idx + 1][1] * direction[1]
            )
            needed = widths[join_idx] * lead + widths[join_idx + 1] * trail
            allowed = lengths[join_idx] * OFFSET_SAFETY
            if needed <= allowed or needed < 1e-12:
                continue
            scale = allowed / needed
            widths[join_idx] = max(MIN_HALF_WIDTH_MM, widths[join_idx] * scale)
            widths[join_idx + 1] = max(MIN_HALF_WIDTH_MM, widths[join_idx + 1] * scale)
            clamped = True
        if not clamped:
            break

    # A half-width that jumps between neighbouring cross-sections zigzags the
    # edge even when every quad still advances: the walls on either side of a
    # narrowed ring face each other, which is the accordion a pinched hairpin
    # tip used to print as. Limit how fast the half-width may change per
    # millimetre travelled; like the clamp above this only ever shrinks it, so
    # the two cannot fight.
    for _ in range(OFFSET_RELAXATION_PASSES):
        limited = False
        for join_idx, length in enumerate(lengths):
            room = WIDTH_SLOPE_LIMIT * length
            for near, far in ((join_idx, join_idx + 1), (join_idx + 1, join_idx)):
                if widths[far] > widths[near] + room:
                    widths[far] = widths[near] + room
                    limited = True
        if not limited:
            break

    return [
        (normal[0] * width, normal[1] * width)
        for normal, width in zip(normals, widths, strict=True)
    ]


def generate_track_mesh(
    track_gpx: str,
    lat_top: float,
    lon_left: float,
    lat_bottom: float,
    lon_right: float,
    terrain_vertices: np.ndarray,
    rows: int,
    cols: int,
    grid_transform: Affine,
    width_mm: float,
    height_mm: float,
    base_thickness_mm: float,
    track_width_mm: float,
    track_height_mm: float,
    simplification_tolerance: float = 15.0,
    max_points: int | None = None,
    footprint: Footprint | None = None,
) -> Mesh:
    """
    Create a D-shaped track mesh that conforms to the terrain surface.

    Args:
        track_gpx: Path to GPX file
        lat_top: Top latitude boundary
        lon_left: Left longitude boundary
        lat_bottom: Bottom latitude boundary
        lon_right: Right longitude boundary
        terrain_vertices: Final terrain mesh vertices (top surface only)
        rows: Number of rows in terrain grid
        cols: Number of columns in terrain grid
        grid_transform: Affine transform for the elevation grid
        width_mm: Model width in mm
        height_mm: Model height in mm
        base_thickness_mm: Base thickness in mm
        track_width_mm: Track width in mm
        track_height_mm: Track height above terrain in mm (total protrusion height)
        simplification_tolerance: Minimum distance between points in meters
        max_points: Maximum number of track points

    Returns:
        Mesh object with the D-shaped track. There is no "no track" return —
        an empty or non-intersecting GPX raises instead, so the caller can
        treat the result as a mesh unconditionally.
    """
    # Parse GPX file
    try:
        track_points = parse_gpx_track(track_gpx)

        # Validate points
        if len(track_points) == 0:
            raise RuntimeError(
                f"CRITICAL ERROR: GPX file contains ZERO points!\n"
                f"  File: {track_gpx}\n"
                "  This file appears to be empty or invalid."
            )

        # Log track bounds
        if track_points:
            track_lats = [p[0] for p in track_points]
            track_lons = [p[1] for p in track_points]
            print("  Track geographic bounds:")
            print(f"    Lat: {min(track_lats):.6f} to {max(track_lats):.6f}")
            print(f"    Lon: {min(track_lons):.6f} to {max(track_lons):.6f}")
    except (FileNotFoundError, ValueError) as e:
        raise RuntimeError(
            f"CRITICAL ERROR: Failed to load GPX file!\n  File: {track_gpx}\n  Error: {e}"
        ) from e

    if len(track_points) < 2:
        raise RuntimeError(
            f"CRITICAL ERROR: GPX file has too few points!\n"
            f"  File: {track_gpx}\n"
            f"  Points: {len(track_points)}\n"
            "  Minimum required: 2 points for a track"
        )

    # Clip track to bbox boundaries instead of simple filtering
    # This preserves track segments that cross bbox edges
    print("\n  Terrain bbox for clipping:")
    print(f"    Lat: {lat_bottom:.6f} to {lat_top:.6f}")
    print(f"    Lon: {lon_left:.6f} to {lon_right:.6f}")

    clipped_points = clip_track_to_bbox(track_points, lat_bottom, lat_top, lon_left, lon_right)

    # Polygon clip, inset by half the ribbon width. Every shape needs it: a
    # rectangle's own bbox clip cuts the centreline, which still leaves half a
    # ribbon hanging over the edge.
    if footprint is not None:
        before_n = len(clipped_points)
        inset_points = _clip_track_to_footprint(
            clipped_points,
            footprint,
            lat_bottom,
            lat_top,
            lon_left,
            lon_right,
            width_mm,
            height_mm,
            margin_mm=track_width_mm / 2.0,
        )
        if len(inset_points) >= 2:
            clipped_points = inset_points
        else:
            # A track that runs along the outline has nothing left once the
            # ribbon's half-width is taken off both sides. An overhanging lip
            # is a poor print; no track at all is not a model. Keep the track,
            # clip it to the outline, and say what the print will show.
            print(
                "  \u26a0 Track runs along the model edge: the ribbon will overhang it. "
                "Raise --terrain-border, or widen the model."
            )
            clipped_points = _clip_track_to_footprint(
                clipped_points,
                footprint,
                lat_bottom,
                lat_top,
                lon_left,
                lon_right,
                width_mm,
                height_mm,
            )
        dropped = before_n - len(clipped_points)
        if before_n > 0:
            pct = 100.0 * dropped / before_n
            if pct > 5.0:
                print(
                    f"  WARNING: Track clipped to {footprint.shape} footprint dropped "
                    f"{dropped}/{before_n} points ({pct:.1f}%). Consider larger "
                    f"--model-size-x / --model-size-y or a different shape."
                )
            else:
                print(
                    f"  Track clipped to {footprint.shape} footprint: "
                    f"{dropped}/{before_n} points removed ({pct:.1f}%)"
                )

    if len(clipped_points) < 2:
        raise RuntimeError(
            f"CRITICAL ERROR: Track does not intersect terrain bounds!\n"
            f"  Track bbox: ({min(p[0] for p in track_points):.6f}, {min(p[1] for p in track_points):.6f}) to\n"
            f"              ({max(p[0] for p in track_points):.6f}, {max(p[1] for p in track_points):.6f})\n"
            f"  Terrain bbox: ({lat_bottom:.6f}, {lon_left:.6f}) to ({lat_top:.6f}, {lon_right:.6f})\n"
            "\n"
            "The track and terrain do not overlap. Please check your --bbox coordinates.\n"
            "The bbox should contain at least part of the GPX track."
        )

    filtered_points = clipped_points

    # Simplify track points
    if max_points is not None:
        base_tolerance = simplification_tolerance
        simplified_points = simplify_track_points(filtered_points, tolerance=base_tolerance)

        if len(simplified_points) > max_points:
            current_tolerance = max(base_tolerance, 0.1)
            while len(simplified_points) > max_points and current_tolerance < 1000.0:
                current_tolerance *= 1.5
                simplified_points = simplify_track_points(
                    filtered_points, tolerance=current_tolerance
                )
    else:
        simplified_points = simplify_track_points(
            filtered_points, tolerance=simplification_tolerance
        )

    # HARD VALIDATION: Ensure simplification didn't eliminate entire track
    if len(simplified_points) < 2:
        raise RuntimeError(
            f"CRITICAL ERROR: Track simplification reduced points below minimum!\n"
            f"  Original points: {len(track_points)}\n"
            f"  After clipping: {len(filtered_points)}\n"
            f"  After simplification: {len(simplified_points)}\n"
            f"  Simplification tolerance: {simplification_tolerance}m\n"
            f"  Max points limit: {max_points}\n"
            "\n"
            "The simplification was too aggressive. Try:\n"
            "  - Reducing --simplification-tolerance\n"
            "  - Increasing --max-points\n"
            "  - Using a longer GPX track"
        )

    # Use simplified points directly without additional smoothing
    smoothed_points = simplified_points

    print(f"  ✓ Final track: {len(smoothed_points)} points ready for 3D projection")

    # Create interpolator for elevation data using affine transform for pixel centers
    lon_coords = np.array([grid_transform * (j, 0) for j in range(cols)])[:, 0]
    lat_coords = np.array([grid_transform * (0, i) for i in range(rows)])[:, 1]

    # Geographic bounds from actual pixel centers
    lon_left_actual = lon_coords[0]
    lon_right_actual = lon_coords[-1]
    lat_top_actual = lat_coords[0]
    lat_bottom_actual = lat_coords[-1]

    print("\n" + "=" * 60)
    print("TRACK COORDINATE SYSTEM")
    print("=" * 60)
    print("Geographic bounds (requested):")
    print(f"  Lat: {lat_bottom:.6f} to {lat_top:.6f}")
    print(f"  Lon: {lon_left:.6f} to {lon_right:.6f}")
    print("Pixel-center bounds (elevation grid):")
    print(f"  Lat: {lat_bottom_actual:.6f} to {lat_top_actual:.6f}")
    print(f"  Lon: {lon_left_actual:.6f} to {lon_right_actual:.6f}")
    print(f"Model space: {width_mm:.2f}mm x {height_mm:.2f}mm")
    print("Using requested bounds for track->terrain alignment")
    print("=" * 60 + "\n")

    # Helper function to get elevation at any lat/lon from FINAL terrain mesh
    def get_terrain_z(lat: float, lon: float, apply_clearance: bool = True) -> float:
        """Get terrain Z coordinate from final terrain mesh at given lat/lon.

        Args:
            lat, lon: Geographic coordinates
            apply_clearance: If True, applies track embedding offset (-0.5mm below surface)
        """
        # Convert lat/lon to model XY coordinates
        x_mm = ((lon - lon_left) / (lon_right - lon_left)) * width_mm
        y_mm = ((lat_top - lat) / (lat_top - lat_bottom)) * height_mm

        # Sample Z from actual terrain mesh using barycentric interpolation
        z_terrain = sample_z_from_terrain_mesh(
            x_mm, y_mm, terrain_vertices, rows, cols, width_mm, height_mm
        )

        # Apply track embedding offset (negative = below surface)
        if apply_clearance:
            z_terrain += TRACK_EMBEDDING_DEPTH_MM

        return z_terrain

    # Direct XY-based terrain sampling (preferred - avoids round-trip conversion errors)
    def get_terrain_z_at_xy(x_mm: float, y_mm: float, apply_clearance: bool = True) -> float:
        """Get terrain Z directly from XY coordinates without lat/lon conversion.

        This avoids floating-point precision errors from round-trip XY→lat/lon→XY conversions.

        Args:
            x_mm, y_mm: Position in model space
            apply_clearance: If True, applies track embedding offset (-0.5mm below surface)
        """
        z_terrain = sample_z_from_terrain_mesh(
            x_mm, y_mm, terrain_vertices, rows, cols, width_mm, height_mm
        )

        if apply_clearance:
            z_terrain += TRACK_EMBEDDING_DEPTH_MM

        return z_terrain

    # Helper to sample terrain Z (no longer need slope-aware offset since we sample from mesh)
    def get_terrain_z_with_slope_offset(lat: float, lon: float) -> float:
        """
        Get terrain Z from final mesh with track embedding offset applied.

        Now that we sample from the actual terrain mesh triangles,
        we get the exact surface elevation without interpolation errors.
        The embedding offset is applied to position track base 0.5mm below surface.
        """
        return get_terrain_z(lat, lon, apply_clearance=True)

    track_points_mm: list[PointXY] = [
        (
            ((lon - lon_left) / (lon_right - lon_left)) * width_mm,
            ((lat_top - lat) / (lat_top - lat_bottom)) * height_mm,
        )
        for lat, lon in smoothed_points
    ]

    # Long segments are subdivided so the track bottom follows the terrain and
    # does not float over dips. Max subsegment length: track_width_mm * 2 keeps
    # terrain variation within each subsegment small next to the embedding
    # depth. This runs *after* the corners are rounded: rounding a corner needs
    # room on both arms, and cutting the arms into 3mm pieces first would leave
    # every corner with a radius a tenth of what its real arms could give.
    MAX_SEGMENT_LENGTH_MM = track_width_mm * 2

    # Helper function to convert model XY back to lat/lon for elevation sampling
    def xy_to_latlon(x_mm: float, y_mm: float) -> LatLon:
        """Convert model XY coordinates back to lat/lon.

        Uses REQUESTED bounds (same as terrain) for accurate coordinate mapping.
        """
        # Map model coordinates back to lat/lon using REQUESTED bounds
        lon = lon_left + (x_mm / width_mm) * (lon_right - lon_left)
        lat = lat_top - (y_mm / height_mm) * (lat_top - lat_bottom)
        return lat, lon

    # Generate the track as one closed tube swept along the polyline.
    #
    # Exactly one cross-section ("ring") per polyline point, stitched to its
    # neighbours with a single winding convention and closed with a fan at each
    # end. The result is a closed two-manifold: every edge is shared by exactly
    # two triangles, every normal points outward.
    #
    # Two earlier constructions are deliberately gone:
    #   * a fresh ring on both sides of every join — at a straight join the two
    #     rings coincided and the connecting strip collapsed into zero-area
    #     slivers, and the fillet strip was wound the other way round from the
    #     segment strips (holes, flipped faces, negative volume);
    #   * separately sampled end caps (`create_rounded_cap`) whose vertices sat
    #     a fraction of a millimetre off the tube's own ring, leaving a seam.
    # At a corner the cross-section is mitred (the two segment normals are
    # averaged) instead of emitting two rings at the same point: two rings at
    # one position share their apex vertex exactly, which pinches the surface
    # into a non-manifold point. Sharp corners are rounded into short arcs
    # first (`_round_sharp_joins`) and every offset is capped so no wall quad
    # can fold inside out (`_sweep_offsets`) — see those two for why.
    num_arc_segments = 8  # Number of segments for the rounded top
    half_width_mm = track_width_mm / 2
    # Two path points closer than this stitch into zero-area quads. 0.01mm is
    # the same cull the segment builder above uses.
    MIN_PATH_STEP_MM = 0.01

    # --- Path: polyline positions, no repeats ---
    path: list[tuple[float, float]] = []
    for x_mm, y_mm in track_points_mm:
        if path and np.hypot(x_mm - path[-1][0], y_mm - path[-1][1]) < MIN_PATH_STEP_MM:
            continue
        path.append((x_mm, y_mm))

    # --- Cross-section orientation per path point ---
    # A single point cannot form a tube; an empty mesh makes the guard further
    # down report it as a track-generation failure.
    if len(path) < 2:
        path = []

    # Thin the path to what the ribbon can express, round the corners the sweep
    # cannot turn on the spot, subdivide what is left for the terrain, then drop
    # any offset that would make its wall quad run backwards. Rounding and
    # clamping only ever shrink geometry, so the ribbon narrows at a switchback
    # instead of inverting.
    path = _resample_path(
        path,
        half_width_mm * PATH_STEP_FRACTION,
        half_width_mm * PATH_TOLERANCE_FRACTION,
    )
    path = _round_sharp_joins(
        path,
        half_width_mm * JOIN_RADIUS_FACTOR,
        half_width_mm * JOIN_MAX_SHIFT_FRACTION,
    )
    path = _subdivide_path(path, MAX_SEGMENT_LENGTH_MM)
    perps: list[tuple[float, float]] = _sweep_offsets(path, half_width_mm)

    # --- Cross-section profile ---
    # Closed loop: bottom-left, bottom-right, then the arc back over the top
    # towards the left. The arc sample at angle pi is skipped — it lands exactly
    # on the bottom-left corner and would duplicate that vertex, collapsing one
    # quad per ring into a zero-area sliver.
    profile: list[tuple[float, float]] = [(-1.0, 0.0), (1.0, 0.0)]
    profile += [
        (float(np.cos(np.pi * i / num_arc_segments)), float(np.sin(np.pi * i / num_arc_segments)))
        for i in range(1, num_arc_segments)
    ]
    ring_size = len(profile)

    # Arc rises from the embedded base to track_height_mm above the terrain.
    arc_rise_mm = track_height_mm - TRACK_EMBEDDING_DEPTH_MM

    # --- Retraced stretches ---
    # Out-and-back trails come back over the exact same coordinates. Two rings
    # built at one position share every vertex, so the two stretches of tube
    # fuse into a non-manifold pinch instead of passing over each other. Nudging
    # the later visit sideways keeps the surfaces apart; 0.02mm is two orders of
    # magnitude below a 0.4mm nozzle, so nothing moves visibly.
    # Collisions are hunted between ring vertices rather than between path
    # points: two passes running a track-width apart also collide, and a nudged
    # ring can land on a third one, so the search repeats until it comes up dry.
    REVISIT_RADIUS_MM = 0.01  # far above the exporter's 1e-6mm write precision
    WELD_RADIUS_MM = 5e-4  # what a viewer/slicer treats as the same vertex
    REVISIT_NUDGE_MM = 0.02
    MAX_REVISIT_PASSES = 6

    def ring_xy(point_idx: int, nudge_mm: float) -> list[tuple[float, float]]:
        """Cross-section footprint of one ring, without the terrain sampling."""
        x_mm, y_mm = path[point_idx]
        perp_x, perp_y = perps[point_idx]
        nudge_x = nudge_y = 0.0
        if nudge_mm:
            perp_len = float(np.hypot(perp_x, perp_y))
            nudge_x = perp_x / perp_len * nudge_mm
            nudge_y = perp_y / perp_len * nudge_mm
        return [
            (x_mm + nudge_x + perp_scale * perp_x, y_mm + nudge_y + perp_scale * perp_y)
            for perp_scale, _rise in profile
        ]

    # The nudge also has to stay small next to the step between two rings. On a
    # pinched switchback the rings sit hundredths of a millimetre apart, and a
    # sideways shove longer than the step ahead tips the quad between them
    # inside out — the very fold `_sweep_offsets` exists to prevent. A quarter
    # of the local step is always well above the weld radius, so the two
    # surfaces still separate.
    steps_mm = [
        min(
            float(np.hypot(*(np.subtract(path[i], path[max(i - 1, 0)])))) or float("inf"),
            float(np.hypot(*(np.subtract(path[min(i + 1, len(path) - 1)], path[i]))))
            or float("inf"),
        )
        for i in range(len(path))
    ]
    nudge_caps = [max(2 * WELD_RADIUS_MM, 0.25 * step) for step in steps_mm]

    nudges = [0.0] * len(path)
    for _ in range(MAX_REVISIT_PASSES):
        if not path:
            break
        candidate_xy = np.array(
            [xy for point_idx in range(len(path)) for xy in ring_xy(point_idx, nudges[point_idx])]
        )
        collided: set[int] = set()
        for a, b in cKDTree(candidate_xy).query_pairs(REVISIT_RADIUS_MM):
            ring_a, slot_a = divmod(a, ring_size)
            ring_b, slot_b = divmod(b, ring_size)
            if abs(profile[slot_a][1] - profile[slot_b][1]) > 1e-9:
                continue  # different heights above the terrain: no shared vertex
            # Neighbouring rings are stitched, so they are *meant* to sit close;
            # only a genuine coincidence (which collapses the quad between them
            # into a sliver) counts, hence the much tighter threshold there.
            if (
                abs(ring_a - ring_b) <= 1
                and float(np.hypot(*(candidate_xy[a] - candidate_xy[b]))) > WELD_RADIUS_MM
            ):
                continue
            collided.add(max(ring_a, ring_b))
        if not collided:
            break
        for ring_idx in collided:
            nudges[ring_idx] = min(nudges[ring_idx] + REVISIT_NUDGE_MM, nudge_caps[ring_idx])

    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    # Two bodies that share a single vertex are one connected component with a
    # pinch point rather than two solids — an out-and-back that turns round
    # twice over the same spot does exactly that. Vertices already taken are
    # therefore lifted a quarter of a micron: above what any viewer welds,
    # three orders of magnitude below what a nozzle lays down.
    VERTEX_SEPARATION_MM = 2.5e-4
    taken: set[tuple[int, int, int]] = set()

    def build_ring(centre: PointXY, offset: PointXY) -> list[int]:
        """One cross-section: `centre` +- `offset`, riding on the terrain."""
        ring: list[int] = []
        for perp_scale, rise_fraction in profile:
            vertex_x = centre[0] + perp_scale * offset[0]
            vertex_y = centre[1] + perp_scale * offset[1]
            vertex_z = (
                get_terrain_z_at_xy(vertex_x, vertex_y, apply_clearance=True)
                + arc_rise_mm * rise_fraction
            )
            key = (
                round(vertex_x / VERTEX_SEPARATION_MM),
                round(vertex_y / VERTEX_SEPARATION_MM),
                round(vertex_z / VERTEX_SEPARATION_MM),
            )
            while key in taken:
                vertex_z += VERTEX_SEPARATION_MM
                key = (key[0], key[1], key[2] + 1)
            taken.add(key)
            ring.append(len(vertices))
            vertices.append([vertex_x, vertex_y, vertex_z])
        return ring

    # --- Runs and beads ---
    # A cross-section the curvature clamp has pinched below half the requested
    # width is not a track any more: it is a knife edge, invisible on the print
    # and impossible to extrude. Those stretches leave the swept tube and are
    # bridged by *beads* instead: straight, full-width, full-height tubes strung
    # between the swept ends, overlapping them and each other. A slicer unions
    # the lot, so a switchback tip prints as a full-width bevelled corner rather
    # than a needle sunk into the hillside. This is why the track group can ship
    # as several overlapping shells, the same way the water layer ships one body
    # per lake.
    centres: list[PointXY] = []
    for (x_mm, y_mm), (perp_x, perp_y), nudge_mm in zip(path, perps, nudges, strict=True):
        perp_len = float(np.hypot(perp_x, perp_y))
        nudge_x = perp_x / perp_len * nudge_mm if nudge_mm else 0.0
        nudge_y = perp_y / perp_len * nudge_mm if nudge_mm else 0.0
        centres.append((x_mm + nudge_x, y_mm + nudge_y))

    min_swept_width_mm = half_width_mm * MIN_SWEPT_WIDTH_FRACTION
    swept = [float(np.hypot(*offset)) >= min_swept_width_mm for offset in perps]
    # A run of one ring is not a tube — its two end caps would land on the same
    # cross-section and enclose nothing — so a lone survivor becomes a bead too.
    for point_idx in range(len(swept)):
        before = swept[point_idx - 1] if point_idx > 0 else False
        after = swept[point_idx + 1] if point_idx + 1 < len(swept) else False
        if swept[point_idx] and not before and not after:
            swept[point_idx] = False

    def bead_anchors(first_idx: int, last_idx: int) -> list[PointXY]:
        """Points a pinched stretch is bridged between, ends included.

        The stretch is anchored on the swept ring either side of it so the
        beads overlap the tube they replace, and cut up every `half_width_mm`
        of travel so a long tight curve is followed rather than chorded across.
        """
        anchors = [centres[first_idx - 1] if first_idx > 0 else centres[first_idx]]
        travelled_mm = 0.0
        for point_idx in range(first_idx, last_idx + 1):
            travelled_mm += float(
                np.hypot(*np.subtract(centres[point_idx], centres[max(point_idx - 1, 0)]))
            )
            if travelled_mm >= half_width_mm:
                anchors.append(centres[point_idx])
                travelled_mm = 0.0
        tail = centres[last_idx + 1] if last_idx + 1 < len(centres) else centres[last_idx]
        anchors.append(tail)
        return anchors

    def build_bead(start: PointXY, end: PointXY, fallback: PointXY) -> list[list[int]] | None:
        """A straight full-width tube from `start` to `end`, overlapping both.

        `fallback` is the ribbon direction to use when the two ends coincide —
        an out-and-back reversal bridges a stretch that goes nowhere.
        """
        span_x, span_y = end[0] - start[0], end[1] - start[1]
        length_mm = float(np.hypot(span_x, span_y))
        along = _unit(span_x, span_y) if length_mm > 1e-9 else fallback
        # Reach past both anchors so consecutive beads share solid, not a point:
        # two tubes meeting exactly end to end leave a notch on the outside of
        # the turn between them.
        margin_mm = half_width_mm * BEAD_OVERLAP_FRACTION
        offset = (-along[1] * half_width_mm, along[0] * half_width_mm)
        head = (start[0] - along[0] * margin_mm, start[1] - along[1] * margin_mm)
        tail = (end[0] + along[0] * margin_mm, end[1] + along[1] * margin_mm)
        if float(np.hypot(tail[0] - head[0], tail[1] - head[1])) < MIN_PATH_STEP_MM:
            return None
        return [build_ring(head, offset), build_ring(tail, offset)]

    runs: list[list[list[int]]] = []
    swept_run: list[list[int]] = []
    point_idx = 0
    while point_idx < len(centres):
        if swept[point_idx]:
            swept_run.append(build_ring(centres[point_idx], perps[point_idx]))
            point_idx += 1
            continue

        if swept_run:
            runs.append(swept_run)
            swept_run = []
        stretch_end = point_idx
        while stretch_end + 1 < len(centres) and not swept[stretch_end + 1]:
            stretch_end += 1
        anchors = bead_anchors(point_idx, stretch_end)
        fallback = _unit(perps[point_idx][1], -perps[point_idx][0])
        for start, end in itertools.pairwise(anchors):
            bead = build_bead(start, end, fallback)
            if bead is not None:
                runs.append(bead)
        point_idx = stretch_end + 1
    if swept_run:
        runs.append(swept_run)

    # --- Tube walls ---
    # The ring runs counter-clockwise in the (normal, up) plane and the rings
    # advance along the track, so this order puts every normal on the outside.
    for run in runs:
        for ring_a, ring_b in itertools.pairwise(run):
            for i in range(ring_size):
                j = (i + 1) % ring_size
                faces.append([ring_a[i], ring_b[j], ring_b[i]])
                faces.append([ring_a[i], ring_a[j], ring_b[j]])

        # --- End caps ---
        # A fan over the terminal ring itself, so the cap shares the tube's
        # vertices instead of sitting beside them. The two fans wind opposite
        # ways: the start cap faces backwards along the track, the end cap
        # forwards.
        for i in range(1, ring_size - 1):
            faces.append([run[0][0], run[0][i + 1], run[0][i]])
            faces.append([run[-1][0], run[-1][i], run[-1][i + 1]])
    if not vertices:
        error_msg = (
            "CRITICAL ERROR: Track geometry generation failed!\n"
            f"  GPX file: {track_gpx}\n"
            f"  Simplified points: {len(smoothed_points)}\n"
            "  Result: NO VERTICES GENERATED\n"
            "\n"
            "This is a critical failure. Track mesh MUST be created when GPX is provided.\n"
            "Possible causes:\n"
            "  - D-track geometry construction failed\n"
            "  - All vertices filtered out\n"
            "  - Track width/height too small\n"
        )
        raise RuntimeError(error_msg)

    vertices_array = np.array(vertices)

    # Create mesh
    track_mesh = Mesh(np.zeros(len(faces), dtype=Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            track_mesh.vectors[i][j] = vertices_array[face[j]]

    # Validate mesh
    if len(track_mesh.data) == 0:
        raise RuntimeError(
            f"CRITICAL ERROR: Track mesh has ZERO faces after creation!\n"
            f"  Vertices created: {len(vertices)}\n"
            f"  Faces list length: {len(faces)}\n"
            f"  Mesh data length: {len(track_mesh.data)}"
        )

    # Validate track fits within model bounds
    track_x_min = vertices_array[:, 0].min()
    track_x_max = vertices_array[:, 0].max()
    track_y_min = vertices_array[:, 1].min()
    track_y_max = vertices_array[:, 1].max()

    if track_x_min < 0 or track_x_max > width_mm or track_y_min < 0 or track_y_max > height_mm:
        # The clip insets by half the ribbon width, so this should not happen.
        # Report rather than raise: an overhang prints as an unsupported lip,
        # which is worse than ugly but not worth discarding a finished mesh.
        print(
            f"  \u26a0 Track reaches outside the model: "
            f"x {track_x_min:.2f}..{track_x_max:.2f} of 0..{width_mm:.2f}, "
            f"y {track_y_min:.2f}..{track_y_max:.2f} of 0..{height_mm:.2f}"
        )

    # Validate track surface alignment - check for any penetration below terrain
    print("\n" + "=" * 60)
    print("TRACK-TERRAIN ALIGNMENT VALIDATION")
    print("  (Track samples from FINAL terrain mesh triangles)")
    print("=" * 60)

    penetration_count = 0
    max_penetration = 0.0
    min_clearance = float("inf")
    max_clearance = float("-inf")

    for vertex in vertices_array:
        x_mm, y_mm, z_track = vertex

        # Sample terrain Z directly from mesh at this XY position
        z_terrain = sample_z_from_terrain_mesh(
            x_mm, y_mm, terrain_vertices, rows, cols, width_mm, height_mm
        )

        # Calculate clearance (positive = above terrain, negative = below)
        clearance = z_track - z_terrain

        if clearance < min_clearance:
            min_clearance = clearance
        if clearance > max_clearance:
            max_clearance = clearance

        # Track bottom should be at embedding depth (-0.5mm)
        # Allow small tolerance for numerical precision beyond intended embedding
        EMBEDDING_TOLERANCE_MM = 0.1  # Allow 0.1mm beyond intended embedding
        if clearance < (TRACK_EMBEDDING_DEPTH_MM - EMBEDDING_TOLERANCE_MM):
            penetration_count += 1
            if abs(clearance) > max_penetration:
                max_penetration = abs(clearance)

    print(f"Track vertices: {len(vertices_array)}")
    print(f"Clearance range: {min_clearance:.3f} to {max_clearance:.3f} mm")
    print(
        f"Expected: {TRACK_EMBEDDING_DEPTH_MM:.3f}mm (base embedded) to {track_height_mm:.3f}mm (top)"
    )

    if penetration_count > 0:
        penetration_pct = (penetration_count / len(vertices_array)) * 100
        print(
            f"⚠ WARNING: {penetration_count} vertices ({penetration_pct:.1f}%) exceed intended embedding"
        )
        print(f"  Max penetration beyond embedding: {max_penetration:.3f} mm")
        print(f"  Track embedding depth: {TRACK_EMBEDDING_DEPTH_MM:.3f} mm")
        print("  Note: Excessive penetration may indicate terrain sampling issues")
    else:
        print("✓ Track geometry within expected range")
        print(
            f"  Track embedded: {TRACK_EMBEDDING_DEPTH_MM:.3f} mm, rises: {track_height_mm:.3f} mm above terrain"
        )

    print("=" * 60 + "\n")

    return track_mesh


def _sample_elevation_bilinear(
    elev: np.ndarray,
    x_mm: float,
    y_mm: float,
    width_mm: float,
    height_mm: float,
) -> float:
    """Bilinear sample of `elev` at mesh-space (x_mm, y_mm)."""
    rows, cols = elev.shape
    col_f = (x_mm / width_mm) * (cols - 1)
    row_f = (y_mm / height_mm) * (rows - 1)
    col_f = float(np.clip(col_f, 0.0, cols - 1))
    row_f = float(np.clip(row_f, 0.0, rows - 1))
    c0 = int(np.floor(col_f))
    r0 = int(np.floor(row_f))
    c1 = min(c0 + 1, cols - 1)
    r1 = min(r0 + 1, rows - 1)
    fc = col_f - c0
    fr = row_f - r0
    z00 = float(elev[r0, c0])
    z01 = float(elev[r0, c1])
    z10 = float(elev[r1, c0])
    z11 = float(elev[r1, c1])
    return (1.0 - fr) * ((1.0 - fc) * z00 + fc * z01) + fr * ((1.0 - fc) * z10 + fc * z11)


def _densify_polygon_ccw(perimeter: list[PointXY], target_edge_mm: float) -> list[PointXY]:
    """Insert sub-points on each polygon edge so segments are <= target_edge_mm."""
    if target_edge_mm <= 0:
        return list(perimeter)
    out = []
    n = len(perimeter)
    for k in range(n):
        a = perimeter[k]
        b = perimeter[(k + 1) % n]
        seg_len = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        n_sub = max(1, int(np.ceil(seg_len / target_edge_mm)))
        for i in range(n_sub):
            t = i / n_sub
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def _generate_footprint_terrain_mesh(
    footprint: Footprint,
    processed_elevation: np.ndarray,
    width_mm: float,
    height_mm: float,
    scale: float,
    vertical_exaggeration: float,
    base_thickness_mm: float,
) -> Mesh:
    """
    Build a watertight terrain mesh constrained to a non-rectangular footprint.

    Sample points = interior regular-grid cells inside the polygon + densified
    perimeter. Convex shapes (hexagon, circle/oval) triangulate cleanly with
    unconstrained Delaunay because all simplices fall inside the convex hull.

    Returns:
        Mesh containing top surface, mirrored base, and perimeter side walls.
    """
    rows, cols = processed_elevation.shape

    # --- Interior grid sample points (vectorised inside-test) ---
    xs = np.linspace(0.0, width_mm, cols)
    ys = np.linspace(0.0, height_mm, rows)
    grid_x, grid_y = np.meshgrid(xs, ys)

    # Tiny inset so grid points exactly on the boundary aren't counted twice
    # (and so Delaunay doesn't get coincident interior + boundary samples).
    inset_eps = max(width_mm, height_mm) * 1e-4
    interior_poly = footprint.polygon.buffer(-inset_eps)
    if interior_poly.is_empty:
        raise ValueError("Footprint too small for terrain inset; increase --model-size-x/y.")

    mask = shapely.contains_xy(interior_poly, grid_x, grid_y)
    interior_xy = np.column_stack([grid_x[mask], grid_y[mask]])
    interior_z = processed_elevation[mask].astype(float)

    # --- Boundary points (CCW perimeter vertices, no densification) ---
    # Densifying midpoints would put them exactly on the chord between two
    # polygon corners; qhull treats those as interior collinear points and
    # drops them from the convex hull, which breaks watertightness because
    # the wall walks the densified path while the top mesh's boundary skips
    # the midpoints. Using only the corner vertices guarantees the Delaunay
    # convex hull matches the wall perimeter exactly.
    perim = footprint.perimeter_points()
    boundary_xy = np.array(perim, dtype=float)
    boundary_z = np.array(
        [
            _sample_elevation_bilinear(processed_elevation, x, y, width_mm, height_mm)
            for x, y in perim
        ],
        dtype=float,
    )

    # --- Combine ---
    all_xy = np.vstack([interior_xy, boundary_xy])
    all_z = np.concatenate([interior_z, boundary_z])
    n_total = len(all_xy)
    n_interior = len(interior_xy)
    n_boundary = len(boundary_xy)

    print(f"  Footprint mesh samples: {n_interior} interior + {n_boundary} boundary = {n_total}")

    # --- Delaunay ---
    # Rectangle/hexagon/circle (oval) are all convex, so the Delaunay
    # triangulation of (interior grid + dense perimeter) covers exactly the
    # polygon — no filtering needed. Filtering by centroid-inside is fragile
    # near the boundary because the polygon is a piecewise-linear approximation
    # (especially for circle) and would punch holes in the mesh.
    tri = Delaunay(all_xy)
    kept = tri.simplices  # (M, 3) int
    print(f"  Footprint mesh triangles: {len(kept)}")

    # --- Vertices: top + base ---
    z_top = all_z * scale * vertical_exaggeration + base_thickness_mm
    top_verts = np.column_stack([all_xy[:, 0], all_xy[:, 1], z_top])
    base_verts = np.column_stack([all_xy[:, 0], all_xy[:, 1], np.zeros(n_total)])
    vertices = np.vstack([top_verts, base_verts])

    # --- Faces ---
    faces: list[list[int]] = []
    # Ensure top triangles CCW from above (outward normal +Z).
    p0 = all_xy[kept[:, 0]]
    p1 = all_xy[kept[:, 1]]
    p2 = all_xy[kept[:, 2]]
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (
        p2[:, 0] - p0[:, 0]
    )
    for idx, s in enumerate(kept):
        if cross[idx] >= 0:
            top_face = [int(s[0]), int(s[1]), int(s[2])]
        else:
            top_face = [int(s[0]), int(s[2]), int(s[1])]
        faces.append(top_face)
        # Base = same triangle, reversed winding, offset by n_total.
        faces.append(
            [
                n_total + top_face[0],
                n_total + top_face[2],
                n_total + top_face[1],
            ]
        )

    # --- Perimeter side walls (vertical extrusion) ---
    # Boundary vertices live at indices [n_interior .. n_total-1] in CCW order.
    for k in range(n_boundary):
        top0 = n_interior + k
        top1 = n_interior + (k + 1) % n_boundary
        bot0 = n_total + top0
        bot1 = n_total + top1
        # CCW seen from outside the solid. For a CCW perimeter walk the interior
        # lies left of top0->top1, so the outward normal is (edge x down): the
        # triangles must run top0 -> bottom -> top1, not top0 -> top1 -> bottom.
        # The reversed order still closes the surface, but leaves every wall
        # normal pointing inward and every perimeter edge wound against the top
        # and base caps that share it.
        faces.append([top0, bot1, top1])
        faces.append([top0, bot0, bot1])

    # --- Build Mesh ---
    mesh = Mesh(np.zeros(len(faces), dtype=Mesh.dtype))
    for fi, face in enumerate(faces):
        for v in range(3):
            mesh.vectors[fi][v] = vertices[face[v]]

    return mesh


def generate_terrain_stl(
    lat_top: float,
    lon_left: float,
    lat_bottom: float,
    lon_right: float,
    map_tif: str | list[str],
    output: str,
    track_gpx: str | None,
    model_width_mm: float,
    model_height_mm: float,
    vertical_exaggeration: float | None,
    model_z_height_mm: float,
    base_thickness_mm: float,
    track_width_mm: float,
    track_height_mm: float,
    include_terrain: bool = True,
    include_water: bool = True,
    simplification_tolerance: float = 15.0,
    max_points: int | None = None,
    terrain_resolution: int = 200,
    terrain_smoothing: bool = True,
    smoothing_iterations: int = 2,
    smoothing_strength: float = 0.3,
    terrain_upsample: int = 2,
    osm_water_detail: int = 0,
    osm_use_cache: bool = True,
    osm_cache_dir: str = "osm_cache",
    footprint: Footprint | None = None,
) -> None:
    """
    Generate terrain OBJ model with optional GPX track overlay.

    Args:
        lat_top: Top latitude boundary
        lon_left: Left longitude boundary
        lat_bottom: Bottom latitude boundary
        lon_right: Right longitude boundary
        map_tif: Path to a GeoTIFF elevation file, or a list of paths that are
            merged into one raster when the bounds span several tiles
        output: Output OBJ file path
        track_gpx: Optional path to GPX track file
        model_width_mm: Target model width (X-axis) in mm
        model_height_mm: Target model height (Y-axis) in mm
        vertical_exaggeration: Terrain height multiplier. None auto-calculates the
            largest value that still fits model_z_height_mm, clamped to 1.0-4.0.
            A flat tile takes the 1.0 floor, having no relief to exaggerate;
            anything above 4.0 has to be passed explicitly.
        model_z_height_mm: Maximum model Z-axis height in mm
        base_thickness_mm: Base thickness in mm
        track_width_mm: Track width in mm
        track_height_mm: Track height above terrain in mm
        include_terrain: Whether to include terrain surface (default: True)
        include_water: Whether to generate water surfaces (default: True)
        terrain_smoothing: Whether to apply terrain smoothing (default: True)
        smoothing_iterations: Number of Laplacian smoothing passes (default: 2)
        smoothing_strength: Smoothing strength 0-1 (default: 0.3)
        terrain_upsample: Upsampling factor for terrain mesh (default: 2)
        osm_water_detail: OpenStreetMap water detail level (0=off, 1-5 physical size-based)
        osm_use_cache: Whether to use OSM query caching (default: True)
        osm_cache_dir: Path to OSM cache directory (default: "osm_cache")

    Raises:
        FileNotFoundError: If map_tif doesn't exist
        ValueError: If parameters are invalid
    """
    # Validate parameters
    if model_width_mm <= 0:
        raise ValueError(f"Model width must be positive, got {model_width_mm}")

    if model_height_mm <= 0:
        raise ValueError(f"Model height must be positive, got {model_height_mm}")

    if vertical_exaggeration is not None and vertical_exaggeration <= 0:
        raise ValueError(f"Vertical exaggeration must be positive, got {vertical_exaggeration}")

    # Auto-calculate model_z_height_mm if not specified
    if model_z_height_mm <= 0:
        model_z_height_mm = 10.0

    if base_thickness_mm < 0:
        raise ValueError(f"Base thickness must be non-negative, got {base_thickness_mm}")

    if track_width_mm <= 0:
        raise ValueError(f"Track width must be positive, got {track_width_mm}")

    if track_height_mm <= 0:
        raise ValueError(f"Track height must be positive, got {track_height_mm}")

    if lat_bottom >= lat_top:
        raise ValueError(f"Invalid latitude bounds: {lat_bottom} >= {lat_top}")

    if lon_left >= lon_right:
        raise ValueError(f"Invalid longitude bounds: {lon_left} >= {lon_right}")

    # Handle both single TIF file (string) and multiple TIF files (list)
    map_tif_list = [map_tif] if isinstance(map_tif, str) else map_tif

    # Read the elevation data
    try:
        if len(map_tif_list) == 1:
            # Single tile
            with rasterio.open(map_tif_list[0]) as src:
                window = from_bounds(lon_left, lat_bottom, lon_right, lat_top, src.transform)
                elevation_data = src.read(1, window=window)
                grid_transform = src.window_transform(window)

                nodata_value = src.nodata

                # Validate tile stats
                if nodata_value is not None:
                    masked = np.ma.masked_equal(elevation_data, nodata_value)
                else:
                    masked = np.ma.masked_invalid(elevation_data)

                valid_values = masked.compressed()
                if valid_values.size == 0:
                    raise ValueError(
                        "No valid elevation samples found in FABDEM tile for requested bounds."
                    )

                print(
                    f"Tile stats [{map_tif_list[0]}]: min={valid_values.min():.2f}m, max={valid_values.max():.2f}m"
                )
        else:
            # Multiple tiles - merge them
            # Opened inside the try. A comprehension outside it leaves every
            # tile already open when a later one fails, for the life of the
            # process, and a partly downloaded library hits exactly that.
            src_files = []

            try:
                for tif in map_tif_list:
                    src_files.append(rasterio.open(tif))

                # Per-tile validation before merge
                invalid_tiles = []
                for tile_path, src in zip(map_tif_list, src_files, strict=False):
                    window = from_bounds(lon_left, lat_bottom, lon_right, lat_top, src.transform)
                    tile_data = src.read(1, window=window)
                    nodata_value = src.nodata
                    if nodata_value is not None:
                        masked = np.ma.masked_equal(tile_data, nodata_value)
                    else:
                        masked = np.ma.masked_invalid(tile_data)

                    valid_values = masked.compressed()
                    if valid_values.size == 0:
                        invalid_tiles.append(tile_path)
                        pass  # No valid elevation samples
                    else:
                        pass  # Valid tile

                if invalid_tiles:
                    raise ValueError(
                        "One or more FABDEM tiles contain no valid elevation data for the requested bounds: "
                        + ", ".join(invalid_tiles)
                    )

                # Merge tiles into a single dataset
                base_res = src_files[0].res
                mosaic, mosaic_transform = merge(
                    src_files,
                    bounds=(lon_left, lat_bottom, lon_right, lat_top),
                    res=base_res,
                    resampling=Resampling.bilinear,
                    method="first",
                    target_aligned_pixels=True,
                )

                # Close all source files
                for src in src_files:
                    src.close()

                # Extract the elevation band
                elevation_data = mosaic[0]
                grid_transform = mosaic_transform

                # Get nodata value from first tile
                with rasterio.open(map_tif_list[0]) as src:
                    nodata_value = src.nodata

                # Merged elevation data ready

                # Check for nodata coverage to warn about missing tiles
                if nodata_value is not None:
                    nodata_count = np.sum(elevation_data == nodata_value)
                    total_pixels = elevation_data.size
                    nodata_percentage = (nodata_count / total_pixels) * 100
                    if nodata_percentage > 0.1:
                        print(
                            f"Warning: merged mosaic has {nodata_percentage:.1f}% nodata "
                            f"(likely ocean or missing tiles). Filling with 0m (sea level)."
                        )
                        elevation_data = np.where(
                            elevation_data == nodata_value, 0.0, elevation_data
                        )
                        nodata_value = None  # already replaced

            except Exception as e:
                # Make sure to close files on error
                for src in src_files:
                    with contextlib.suppress(Exception):
                        src.close()
                raise ValueError(f"Error merging tiles: {e}") from e

        # Common code for both single and multi-tile paths
        # Calculate real-world dimensions in meters using REQUESTED bounds
        lat_avg = (lat_top + lat_bottom) / 2
        meters_per_deg_lat = 111320  # approximately constant
        meters_per_deg_lon = 111320 * np.cos(np.radians(lat_avg))

        width_m = abs(lon_right - lon_left) * meters_per_deg_lon
        height_m = abs(lat_top - lat_bottom) * meters_per_deg_lat
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Elevation data file not found: {map_tif_list}") from exc
    except ValueError:
        # Already named by whoever raised it, the merge handler above included.
        # Wrapping it again buries the cause behind two layers of the same
        # sentence: "Error reading elevation data: Error merging tiles: ...".
        raise
    except Exception as e:
        raise ValueError(f"Error reading elevation data: {e}") from e

    # Handle nodata values - temporarily mark as NaN to find real elevation range
    nodata_mask = None
    if nodata_value is not None:
        nodata_mask = elevation_data == nodata_value
        if nodata_mask.any():
            nodata_count = nodata_mask.sum()
            elevation_data = np.where(nodata_mask, np.nan, elevation_data)
            print(f"Marked {nodata_count} nodata values ({nodata_value}) for later replacement")

    # Handle extreme negative values that might be nodata markers
    # Common nodata values: -32768, -9999, etc.
    extreme_negative_threshold = -1000
    extreme_mask = elevation_data < extreme_negative_threshold
    if extreme_mask.any() and not np.isnan(elevation_data[extreme_mask]).all():
        extreme_count = np.sum(extreme_mask & ~np.isnan(elevation_data))
        elevation_data = np.where(extreme_mask, np.nan, elevation_data)
        print(
            f"Marked {extreme_count} extreme negative values (< {extreme_negative_threshold}m) for later replacement"
        )

    # Handle legitimately negative values (below sea level)
    # These will be clipped to 0 after we find the real minimum
    valid_mask = ~np.isnan(elevation_data)
    # Below-sea-level ground is clipped to 0, which flattens a real depression:
    # the Dead Sea and Death Valley print as a plate. Do not simply remove
    # this. Sea level reaches the water layer only as "the terrain reads 0",
    # and that holds because this clip makes the sea floor one flat plateau.
    # Keep the true depths and the plateau becomes a slope, smoothing pulls the
    # shoreline cells above zero, and ocean detection loses two thirds of its
    # area: measured at 1169 water cells with the clip and 396 without, on the
    # same tile. Fixing the depression means carrying absolute sea level
    # through to `WaterLayerGenerator` instead of inferring it, and that is a
    # change to the interface, not to this line.
    if valid_mask.any():
        valid_data = elevation_data[valid_mask]
        negative_mask_valid = valid_data < 0
        if negative_mask_valid.any():
            negative_count = negative_mask_valid.sum()
            min_negative = valid_data[negative_mask_valid].min()
            elevation_data = np.where((elevation_data < 0) & valid_mask, 0, elevation_data)
            print(
                f"Clipped {negative_count} below-sea-level values (min: {min_negative:.1f}m) to 0m"
            )

    # Get statistics from VALID (non-NaN) data only
    valid_mask = ~np.isnan(elevation_data)
    if not valid_mask.any():
        raise ValueError("No valid elevation data found after cleaning")

    valid_elevations = elevation_data[valid_mask]
    elevation_min = float(valid_elevations.min())
    elevation_max = float(valid_elevations.max())
    elevation_range = elevation_max - elevation_min
    elevation_std = float(np.std(valid_elevations))

    print(f"Cleaned elevation range (valid data): {elevation_min:.1f}m to {elevation_max:.1f}m")
    print(
        f"Elevation stats: min={elevation_min:.2f}m, max={elevation_max:.2f}m, range={elevation_range:.2f}m, std={elevation_std:.2f}m"
    )
    # A polder, a salt flat or a barrier island is flat and real. The test that
    # used to sit here read absolute metres, so anything low and level was
    # called missing data and the user was sent to re-download tiles that were
    # fine. What no tile ever produces is a single repeated value, which is
    # what an empty or unreadable raster looks like.
    if elevation_range == 0.0 and elevation_max == 0.0:
        raise ValueError(
            "Every pixel in the area reads 0m. That is not terrain: the tiles "
            "are missing, empty or unreadable."
        )
    if elevation_range < 0.5:
        print(
            f"  \u26a0 Relief across the area is only {elevation_range:.2f}m. "
            "The model will be nearly flat; raise --vertical-exaggeration or widen the area."
        )

    # Normalize elevation BEFORE downsampling to prevent interpolation artifacts
    # When nodata (0m) pixels are near real elevation, downsampling can create false low values
    if elevation_min > 0.01:  # Use small threshold to handle floating point errors
        elevation_data = np.where(valid_mask, elevation_data - elevation_min, elevation_data)
        print(f"✓ Normalized elevation by subtracting {elevation_min:.2f}m (shifted minimum to 0m)")
        # Update stats after normalization
        elevation_min = 0.0
        elevation_max = elevation_data[valid_mask].max()
        elevation_range = elevation_max
    elif elevation_min < -0.01:
        print(f"Warning: Unexpected negative elevations (min: {elevation_min:.2f}m)")
        elevation_data = np.where(valid_mask, elevation_data - elevation_min, elevation_data)
        elevation_min = 0.0
        elevation_max = elevation_data[valid_mask].max()
        elevation_range = elevation_max

    # NOW replace NaN values (nodata) with 0 (= normalized minimum elevation)
    if np.isnan(elevation_data).any():
        nan_count = np.isnan(elevation_data).sum()
        elevation_data = np.nan_to_num(elevation_data, nan=0.0)
        print(f"Replaced {nan_count} nodata/NaN pixels with 0m (normalized minimum)")

    # Downsample elevation data to reduce polygon count
    original_shape = elevation_data.shape
    if terrain_resolution > 0:
        # Calculate target shape maintaining aspect ratio
        aspect_ratio = original_shape[1] / original_shape[0]
        if aspect_ratio >= 1.0:
            # Wider than tall
            target_cols = terrain_resolution
            target_rows = max(10, int(terrain_resolution / aspect_ratio))
        else:
            # Taller than wide
            target_rows = terrain_resolution
            target_cols = max(10, int(terrain_resolution * aspect_ratio))

        # Only downsample if target is smaller than original
        if target_rows < original_shape[0] or target_cols < original_shape[1]:
            scale_row = target_rows / original_shape[0]
            scale_col = target_cols / original_shape[1]
            elevation_data = zoom(elevation_data, (scale_row, scale_col), order=1)
            # Update transform to reflect new pixel size after resampling
            grid_transform = grid_transform * Affine.scale(
                original_shape[1] / target_cols,
                original_shape[0] / target_rows,
            )
            print(
                f"Downsampled elevation: {original_shape} -> {elevation_data.shape} (resolution: {terrain_resolution})"
            )

    # Final elevation range (after normalization and downsampling)
    elevation_range = elevation_data.max()
    if elevation_range < 0.1:
        print(
            f"Warning: Very flat terrain detected (range: {elevation_range:.3f}m). Model may appear flat."
        )

    # Calculate model dimensions based on actual DEM bounds to ensure perfect alignment
    # between terrain and track coordinate systems
    # The actual bounds may differ slightly from requested bounds due to pixel alignment

    print("\n" + "=" * 60)
    print("TERRAIN COORDINATE SYSTEM")
    print("=" * 60)
    print("Geographic bounds (requested):")
    print(f"  Lat: {lat_bottom:.6f} to {lat_top:.6f}")
    print(f"  Lon: {lon_left:.6f} to {lon_right:.6f}")
    print(f"Real-world dimensions: {width_m:.0f}m x {height_m:.0f}m")
    print(f"Model dimensions (requested): {model_width_mm:.2f}mm x {model_height_mm:.2f}mm")

    # Determine scale based on target model size and actual geographic dimensions
    # If both dimensions are specified (> 0), use them exactly even if it distorts aspect ratio
    # If only one dimension is specified, calculate the other to preserve aspect ratio
    if model_width_mm > 0 and model_height_mm > 0:
        # Both dimensions specified - use them exactly (may distort terrain)
        scale_x = model_width_mm / width_m
        scale_y = model_height_mm / height_m
        width_mm = model_width_mm
        height_mm = model_height_mm
        # Use average scale for reporting, but actual scaling is non-uniform
        scale = (scale_x + scale_y) / 2
        print(f"Using exact specified dimensions: {width_mm:.2f}mm x {height_mm:.2f}mm")
        print(f"Scale: X={scale_x:.6f} mm/m, Y={scale_y:.6f} mm/m (non-uniform)")
    else:
        # At least one dimension is auto (0) - preserve aspect ratio
        scale_x = model_width_mm / width_m if model_width_mm > 0 else 0
        scale_y = model_height_mm / height_m if model_height_mm > 0 else 0

        if scale_x > 0:
            scale = scale_x
            width_mm = model_width_mm
            height_mm = height_m * scale
        elif scale_y > 0:
            scale = scale_y
            width_mm = width_m * scale
            height_mm = model_height_mm
        else:
            # Both are 0, shouldn't happen but fallback
            scale = 0.018685  # some default
            width_mm = width_m * scale
            height_mm = height_m * scale

        print(f"Scale: {scale:.6f} mm/m (uniform, preserving aspect ratio)")
        print(f"Final model size: {width_mm:.2f}mm x {height_mm:.2f}mm")

    print(f"Terrain mesh spans: (0,0) to ({width_mm:.2f}, {height_mm:.2f})mm")
    print("=" * 60 + "\n")

    # Diagnostic: project bbox corners into mesh space
    def _geo_to_mesh(lon: float, lat: float) -> tuple[float, float]:
        x = ((lon - lon_left) / (lon_right - lon_left)) * width_mm
        y = ((lat_top - lat) / (lat_top - lat_bottom)) * height_mm
        return x, y

    nw = _geo_to_mesh(lon_left, lat_top)
    ne = _geo_to_mesh(lon_right, lat_top)
    sw = _geo_to_mesh(lon_left, lat_bottom)
    se = _geo_to_mesh(lon_right, lat_bottom)
    print("Projected bbox corners (mesh space):")
    print(f"  NW -> ({nw[0]:.2f}, {nw[1]:.2f})")
    print(f"  NE -> ({ne[0]:.2f}, {ne[1]:.2f})")
    print(f"  SW -> ({sw[0]:.2f}, {sw[1]:.2f})")
    print(f"  SE -> ({se[0]:.2f}, {se[1]:.2f})")

    # Auto-calculate vertical exaggeration if not specified
    if vertical_exaggeration is None:
        # Calculate max possible exaggeration to fit within height constraint
        # max_height = elevation_range * scale * v_exag + base_thickness_mm
        # v_exag = (max_height - base_thickness_mm) / (elevation_range * scale)
        # `elevation_range` is `elevation_data.max()`, which equals max - min
        # only because normalisation above shifted the minimum to 0 - and that
        # shift is skipped when the minimum already sits within +/-0.01.
        #
        # A flat tile therefore has no range left to exaggerate, and dividing
        # by it yields inf, which the clamp below would silently turn into 4.0.
        # Take the 1.0 floor instead: nothing to exaggerate, so exaggerate by 1.
        denominator = float(elevation_range * scale)
        if denominator <= 0.0:
            max_exag = 1.0
        else:
            max_exag = (model_z_height_mm - base_thickness_mm) / denominator
        # Clamp to range [1.0, 4.0] and choose the highest possible value
        vertical_exaggeration = min(4.0, max(1.0, max_exag))
        print(f"\nAuto-calculated vertical exaggeration: {vertical_exaggeration:.2f}x")
        print(f"  (Range: 1.0-4.0, constrained by model Z height: {model_z_height_mm / 10:.1f}cm)")

    elevation_mm = elevation_range * scale * vertical_exaggeration
    print("\nModel dimensions:")
    print(
        f"  Size: {width_mm:.1f}mm x {height_mm:.1f}mm ({width_mm / 10:.1f}cm x {height_mm / 10:.1f}cm)"
    )
    print("\nElevation calculation breakdown:")
    print(f"  Elevation range (normalized): {elevation_range:.2f}m")
    print(f"  Horizontal scale: {scale:.6f} mm/m")
    print(f"  Vertical exaggeration: {vertical_exaggeration:.2f}x")
    print(
        f"  Formula: {elevation_range:.2f}m × {scale:.6f} mm/m × {vertical_exaggeration:.2f} = {elevation_mm:.2f}mm"
    )
    print(f"  Max terrain height (above base): {elevation_mm:.2f}mm ({elevation_mm / 10:.2f}cm)")
    print(f"  Total model Z height (base + terrain): {base_thickness_mm + elevation_mm:.2f}mm")

    # Create mesh grid
    rows, cols = elevation_data.shape

    # Apply terrain smoothing if enabled
    processed_elevation = elevation_data.copy()
    terrain_rows = rows
    terrain_cols = cols
    processed_transform = grid_transform

    if terrain_smoothing and terrain_upsample > 1:
        processed_elevation = upsample_elevation_grid(
            processed_elevation, upsample_factor=terrain_upsample
        )
        terrain_rows = processed_elevation.shape[0]
        terrain_cols = processed_elevation.shape[1]
        processed_transform = processed_transform * Affine.scale(
            (cols - 1) / (terrain_cols - 1),
            (rows - 1) / (terrain_rows - 1),
        )

    if terrain_smoothing and smoothing_iterations > 0:
        processed_elevation = apply_laplacian_smoothing(
            processed_elevation,
            iterations=smoothing_iterations,
            smoothing_factor=smoothing_strength,
        )

    # ========== CRITICAL: Generate terrain mesh FIRST ==========
    # The track must sample Z from the FINAL terrain mesh, not from elevation grid.
    # This ensures perfect alignment even after smoothing/simplification.
    print("\n  Building final terrain mesh...")

    # Generate terrain mesh vertices (top surface only for now)
    terrain_vertices_top = []
    for i in range(terrain_rows):
        for j in range(terrain_cols):
            x = j * width_mm / (terrain_cols - 1)
            y = i * height_mm / (terrain_rows - 1)
            z = processed_elevation[i, j] * scale * vertical_exaggeration + base_thickness_mm
            terrain_vertices_top.append([x, y, z])

    terrain_vertices_array = np.array(terrain_vertices_top)
    print(f"  ✓ Terrain mesh vertices: {len(terrain_vertices_array)}")

    # Generate track wall mesh if provided - NOW using final terrain mesh
    track_mesh = None
    if track_gpx:
        print("  Generating track mesh from final terrain surface...")
        track_mesh = generate_track_mesh(
            track_gpx,
            lat_top,
            lon_left,
            lat_bottom,
            lon_right,
            terrain_vertices_array,  # Pass FINAL terrain mesh vertices
            terrain_rows,
            terrain_cols,
            processed_transform,
            width_mm,
            height_mm,
            base_thickness_mm,
            track_width_mm,
            track_height_mm,
            simplification_tolerance,
            max_points,
            footprint=footprint,
        )

        # Validate track mesh
        if track_mesh is None:
            raise RuntimeError(
                "CRITICAL ERROR: Track mesh is None!\n"
                f"  GPX file provided: {track_gpx}\n"
                "  Expected: Valid track mesh\n"
                "  Actual: None\n"
                "\n"
                "Track generation failed. This is NOT acceptable when GPX is provided.\n"
                "Check earlier error messages for details."
            )

        if len(track_mesh.data) == 0:
            raise RuntimeError(
                "CRITICAL ERROR: Track mesh has ZERO faces!\n"
                f"  GPX file: {track_gpx}\n"
                "  Track mesh object exists but contains no geometry.\n"
                "  This indicates a critical failure in mesh construction."
            )

    # Generate terrain mesh from already-created vertices
    if include_terrain and footprint is not None and footprint.shape != "rectangle":
        print(f"  Assembling footprint-constrained terrain mesh ({footprint.shape})...")
        terrain_mesh = _generate_footprint_terrain_mesh(
            footprint=footprint,
            processed_elevation=processed_elevation,
            width_mm=width_mm,
            height_mm=height_mm,
            scale=scale,
            vertical_exaggeration=vertical_exaggeration,
            base_thickness_mm=base_thickness_mm,
        )
    elif include_terrain:
        print("  Assembling final terrain mesh...")

        # Reuse the already-generated top surface vertices
        vertices = terrain_vertices_top.copy()
        faces = []

        # Generate vertices for the bottom surface (base)
        for i in range(terrain_rows):
            for j in range(terrain_cols):
                x = j * width_mm / (terrain_cols - 1)
                y = i * height_mm / (terrain_rows - 1)
                z = 0
                vertices.append([x, y, z])

        vertices_array = np.array(vertices)

        # Create faces for the top surface
        for i in range(terrain_rows - 1):
            for j in range(terrain_cols - 1):
                # Two triangles per grid square
                v1 = i * terrain_cols + j
                v2 = i * terrain_cols + j + 1
                v3 = (i + 1) * terrain_cols + j
                v4 = (i + 1) * terrain_cols + j + 1

                faces.append([v1, v2, v3])
                faces.append([v2, v4, v3])

        # Create faces for the bottom surface (reversed winding)
        base_offset = terrain_rows * terrain_cols
        for i in range(terrain_rows - 1):
            for j in range(terrain_cols - 1):
                v1 = base_offset + i * terrain_cols + j
                v2 = base_offset + i * terrain_cols + j + 1
                v3 = base_offset + (i + 1) * terrain_cols + j
                v4 = base_offset + (i + 1) * terrain_cols + j + 1

                faces.append([v1, v3, v2])
                faces.append([v2, v3, v4])

        # Create side walls
        # Left edge (j=0)
        for i in range(terrain_rows - 1):
            v1 = i * terrain_cols
            v2 = (i + 1) * terrain_cols
            v3 = base_offset + i * terrain_cols
            v4 = base_offset + (i + 1) * terrain_cols
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])

        # Right edge (j=terrain_cols-1)
        for i in range(terrain_rows - 1):
            v1 = i * terrain_cols + (terrain_cols - 1)
            v2 = (i + 1) * terrain_cols + (terrain_cols - 1)
            v3 = base_offset + i * terrain_cols + (terrain_cols - 1)
            v4 = base_offset + (i + 1) * terrain_cols + (terrain_cols - 1)
            faces.append([v1, v3, v2])
            faces.append([v2, v3, v4])

        # Front edge (i=0)
        for j in range(terrain_cols - 1):
            v1 = j
            v2 = j + 1
            v3 = base_offset + j
            v4 = base_offset + j + 1
            faces.append([v1, v3, v2])
            faces.append([v2, v3, v4])

        # Back edge (i=terrain_rows-1)
        for j in range(terrain_cols - 1):
            v1 = (terrain_rows - 1) * terrain_cols + j
            v2 = (terrain_rows - 1) * terrain_cols + j + 1
            v3 = base_offset + (terrain_rows - 1) * terrain_cols + j
            v4 = base_offset + (terrain_rows - 1) * terrain_cols + j + 1
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])

        # Create the mesh
        terrain_mesh = Mesh(np.zeros(len(faces), dtype=Mesh.dtype))
        for i, face in enumerate(faces):
            for j in range(3):
                terrain_mesh.vectors[i][j] = vertices_array[face[j]]

    else:
        terrain_mesh = None

    # ==================== OSM Feature Rendering ====================
    # Generate OpenStreetMap features (water, roads, trails) if requested
    # These are rendered as separate mesh components and combined with terrain

    water_mesh = None
    if osm_water_detail > 0:
        water_generator = WaterLayerGenerator(
            lat_min=lat_bottom,
            lon_min=lon_left,
            lat_max=lat_top,
            lon_max=lon_right,
            model_width_mm=width_mm,
            model_height_mm=height_mm,
            base_thickness_mm=base_thickness_mm,
            scale=scale,
            vertical_exaggeration=vertical_exaggeration,
            lod_level=osm_water_detail,
            terrain_elevation=processed_elevation,
            terrain_transform=processed_transform,
            terrain_resolution=terrain_resolution,
            scale_x=scale_x,
            scale_y=scale_y,
            use_cache=osm_use_cache,
            osm_cache_dir=osm_cache_dir,
            y_origin_south=False,
            footprint=footprint,
        )

        # Arrives already inside the footprint: every water body is clipped where
        # it is built, and the river's inset accounts for its own ribbon width.
        # Do not add a trim here — dropping triangles that poke out cuts a closed
        # shell open, which is the bug this replaced.
        water_mesh = water_generator.generate_water_mesh()

    # Calculate model dimensions from combined mesh
    all_vertices = []

    if track_mesh is not None:
        for face in track_mesh.data["vectors"]:
            for vertex in face:
                all_vertices.append(vertex)

    if terrain_mesh is not None:
        for face in terrain_mesh.data["vectors"]:
            for vertex in face:
                all_vertices.append(vertex)

    if water_mesh is not None:
        for face in water_mesh.data["vectors"]:
            for vertex in face:
                all_vertices.append(vertex)

    # Check polygon count and warn if too high
    total_triangles = 0
    if terrain_mesh is not None:
        total_triangles += len(terrain_mesh.data["vectors"])
    if track_mesh is not None:
        total_triangles += len(track_mesh.data["vectors"])
    if water_mesh is not None:
        total_triangles += len(water_mesh.data["vectors"])

    if total_triangles > 500000:
        print(f"⚠️  CRITICAL WARNING: Model has {total_triangles:,} triangles!")
        print("   This may cause severe performance issues in CAD software.")
        print("   Recommended: Reduce --terrain-resolution or --terrain-upsample")
    elif total_triangles > 100000:
        print(f"⚠️  WARNING: Model has {total_triangles:,} triangles.")
        print("   This may cause performance issues on some systems.")
        print("   Consider reducing --terrain-resolution if needed.")

    # Combine and save meshes
    if track_mesh is not None and terrain_mesh is not None:
        mesh_parts = [
            (terrain_mesh, "terrain", "terrain"),
            (track_mesh, "track", "track"),
        ]
        if water_mesh is not None:
            mesh_parts.append((water_mesh, "water", "water"))

        save_combined_mesh_obj(mesh_parts, output)
    elif track_mesh is not None:
        if water_mesh is not None:
            save_combined_mesh_obj(
                [(track_mesh, "track", "track"), (water_mesh, "water", "water")], output
            )
        else:
            save_mesh_obj(track_mesh, output, "track", "track")
    elif terrain_mesh is not None:
        if water_mesh is not None:
            save_combined_mesh_obj(
                [(terrain_mesh, "terrain", "terrain"), (water_mesh, "water", "water")],
                output,
            )
        else:
            save_mesh_obj(terrain_mesh, output, "terrain", "terrain")
    else:
        raise RuntimeError("No mesh generated (no terrain and no track)")

    # Print model dimensions
    if all_vertices:
        all_vertices_array = np.array(all_vertices)
        # OBJ format swaps Y and Z, so we need to account for that
        # Our system: X=left-right, Y=front-back, Z=up-down
        # In file vertices: (x, z, y) - so actual dimensions are: X, Z (as Y in file), Y (as Z in file)
        x_size = all_vertices_array[:, 0].max() - all_vertices_array[:, 0].min()
        y_size = all_vertices_array[:, 1].max() - all_vertices_array[:, 1].min()
        z_size = all_vertices_array[:, 2].max() - all_vertices_array[:, 2].min()
        print(f"Model size (mm): X={x_size:.2f} Y={y_size:.2f} Z={z_size:.2f}")

    print(output)
