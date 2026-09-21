# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Common utilities for OSM feature rendering.

Ribbon geometry for linear water features (rivers, canals, streams). Produces
indexed meshes with smooth left/right boundary edges (miter joints at corners)
and planar cross-sections (single Z per centerline sample). The previous
implementation sampled terrain independently at each edge of every cross-
section, producing twisted non-planar quads whose triangulation showed up as
visible diagonal creases — the "fish-scale" pattern seen in renders.
"""

from collections.abc import Callable

import numpy as np

# Miter limit — when adjacent segments meet at a very sharp angle the miter
# offset blows up. Above this distance multiplier we clamp the offset.
_MITER_LIMIT = 4.0


def densify_polyline(points: np.ndarray, max_segment: float) -> np.ndarray:
    """
    Insert evenly-spaced points so no segment is longer than ``max_segment``.

    Original vertices are preserved; extra vertices are added only across gaps
    wider than ``max_segment``. Used so a river ribbon samples terrain finely
    enough that its top surface keeps hugging the ground — without this, far-
    apart OSM nodes let the ribbon bridge straight over higher terrain between
    samples, which buries the strip and makes the river look broken.

    Args:
        points: (N, 2) polyline.
        max_segment: Max allowed distance between consecutive points (same
            units as ``points``). Non-positive disables densification.

    Returns:
        (M, 2) densified polyline, M >= N.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2 or max_segment <= 0:
        return pts

    out = [pts[0]]
    for i in range(1, len(pts)):
        p0 = pts[i - 1]
        p1 = pts[i]
        dist = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        if dist > max_segment:
            steps = int(np.ceil(dist / max_segment))
            for s in range(1, steps):
                out.append(p0 + (p1 - p0) * (s / steps))
        out.append(p1)

    return np.array(out)


def calculate_perpendicular_offset(
    points: np.ndarray, offset_distance: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute left/right miter-offset polylines from a centerline.

    Each vertex of the centerline gets an offset that bisects the angle between
    its incoming and outgoing edges, scaled so the resulting parallel polyline
    sits exactly `offset_distance` away from every centerline segment. End
    vertices use the single adjacent segment's perpendicular.

    Args:
        points: (N, 2) centerline coordinates.
        offset_distance: Half-width of the ribbon (perpendicular distance).

    Returns:
        (left_points, right_points), each (N, 2).
    """
    n = len(points)
    if n < 2:
        return points.copy(), points.copy()

    # Per-segment unit direction and unit perpendicular (left of travel).
    seg_dx = points[1:, 0] - points[:-1, 0]
    seg_dy = points[1:, 1] - points[:-1, 1]
    seg_len = np.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)
    seg_len = np.where(seg_len > 0, seg_len, 1.0)
    dir_x = seg_dx / seg_len
    dir_y = seg_dy / seg_len
    # Perpendicular (rotate +90°): (-dy, dx).
    perp_x = -dir_y
    perp_y = dir_x

    left = np.zeros_like(points)
    right = np.zeros_like(points)

    for i in range(n):
        if i == 0:
            px, py = perp_x[0], perp_y[0]
            scale = 1.0
        elif i == n - 1:
            px, py = perp_x[-1], perp_y[-1]
            scale = 1.0
        else:
            p_prev_x, p_prev_y = perp_x[i - 1], perp_y[i - 1]
            p_next_x, p_next_y = perp_x[i], perp_y[i]
            sum_x = p_prev_x + p_next_x
            sum_y = p_prev_y + p_next_y
            denom = 1.0 + p_prev_x * p_next_x + p_prev_y * p_next_y
            if denom < 1.0 / (_MITER_LIMIT * _MITER_LIMIT):
                # Reflex / very sharp angle — fall back to plain perpendicular
                # from the next segment (avoids absurdly long miter spikes).
                px, py = p_next_x, p_next_y
                scale = 1.0
            else:
                px = sum_x / denom
                py = sum_y / denom
                # Clamp to miter limit relative to half-width.
                mag = float(np.hypot(px, py))
                if mag > _MITER_LIMIT:
                    px *= _MITER_LIMIT / mag
                    py *= _MITER_LIMIT / mag
                scale = 1.0  # offset_distance multiplied in below

        left[i, 0] = points[i, 0] + px * offset_distance * scale
        left[i, 1] = points[i, 1] + py * offset_distance * scale
        right[i, 0] = points[i, 0] - px * offset_distance * scale
        right[i, 1] = points[i, 1] - py * offset_distance * scale

    return left, right


def create_ribbon_geometry(
    centerline_points: np.ndarray,
    width_mm: float,
    height_mm: float,
    elevation_sampler: Callable[[float, float], float],
    thickness_mm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a watertight 3D ribbon mesh from a 2D centerline.

    Cross-section is planar: top-left and top-right share a single Z per
    centerline sample (computed from the centerline position, not from each
    edge independently). This makes every top quad planar in 3D and removes
    the diagonal crease that produces "fish-scale" shading.

    Args:
        centerline_points: (N, 2) centerline in mesh-space (mm).
        width_mm: Total ribbon width.
        height_mm: Top surface offset above terrain in mm.
        elevation_sampler: f(x_mm, y_mm) -> terrain z in mm.
        thickness_mm: Optional thickness. The bottom is placed at
            (top_z - thickness_mm) but no lower than terrain (so the ribbon
            stays sitting on the surface, never floating or buried).

    Returns:
        (vertices, faces) ndarrays — indexed mesh.
    """
    half_width = width_mm / 2.0
    left, right = calculate_perpendicular_offset(centerline_points, half_width)
    n = len(centerline_points)
    if n < 2:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)

    # Per-cross-section Z values. Sample terrain at the centerline AND both
    # edges, then build a planar band that straddles the full cross-section:
    #   top = highest terrain across the width + height_mm
    #   bottom = lowest terrain across the width
    # A single Z per side keeps each quad planar (no fish-scale), while taking
    # the max/min over the width guarantees the ribbon clears terrain on the
    # uphill edge and reaches down to it on the downhill edge. Without this,
    # using only the centerline elevation lets terrain sloping perpendicular to
    # the flow poke up through the uphill edge and bury the river — which prints
    # as a broken / intermittent river on steep (vertically-exaggerated) ground.
    z_center = np.array(
        [
            elevation_sampler(float(centerline_points[i, 0]), float(centerline_points[i, 1]))
            for i in range(n)
        ],
        dtype=float,
    )
    z_left = np.array(
        [elevation_sampler(float(left[i, 0]), float(left[i, 1])) for i in range(n)],
        dtype=float,
    )
    z_right = np.array(
        [elevation_sampler(float(right[i, 0]), float(right[i, 1])) for i in range(n)],
        dtype=float,
    )
    z_top = np.maximum.reduce([z_center, z_left, z_right]) + height_mm
    z_bot = np.minimum.reduce([z_center, z_left, z_right])

    # Optional explicit thickness floor: ensure the band is at least this tall
    # so a river on flat ground still has printable vertical body.
    if thickness_mm is not None and thickness_mm > 0.0:
        z_bot = np.minimum(z_bot, z_top - thickness_mm)

    # Vertex layout per cross-section i (4 vertices, contiguous):
    #   4i+0 = top-left, 4i+1 = top-right,
    #   4i+2 = bottom-left, 4i+3 = bottom-right.
    vertices = np.empty((n * 4, 3), dtype=float)
    vertices[0::4, 0] = left[:, 0]
    vertices[0::4, 1] = left[:, 1]
    vertices[0::4, 2] = z_top
    vertices[1::4, 0] = right[:, 0]
    vertices[1::4, 1] = right[:, 1]
    vertices[1::4, 2] = z_top
    vertices[2::4, 0] = left[:, 0]
    vertices[2::4, 1] = left[:, 1]
    vertices[2::4, 2] = z_bot
    vertices[3::4, 0] = right[:, 0]
    vertices[3::4, 1] = right[:, 1]
    vertices[3::4, 2] = z_bot

    faces: list[list[int]] = []

    # Top surface: two triangles per quad, consistent diagonal LT_i → RT_{i+1}.
    # Since top quad is planar (single Z per cross-section), this choice no
    # longer produces a visible crease.
    for i in range(n - 1):
        b = i * 4
        lt_i, rt_i = b, b + 1
        lt_n, rt_n = b + 4, b + 5
        faces.append([lt_i, rt_i, rt_n])
        faces.append([lt_i, rt_n, lt_n])

    # Bottom surface (reverse winding so normals face down).
    for i in range(n - 1):
        b = i * 4
        lb_i, rb_i = b + 2, b + 3
        lb_n, rb_n = b + 6, b + 7
        faces.append([lb_i, rb_n, rb_i])
        faces.append([lb_i, lb_n, rb_n])

    # Left wall (outside normal = +left perpendicular).
    for i in range(n - 1):
        b = i * 4
        lt_i, lb_i = b, b + 2
        lt_n, lb_n = b + 4, b + 6
        faces.append([lt_i, lt_n, lb_i])
        faces.append([lt_n, lb_n, lb_i])

    # Right wall (outside normal = -left perpendicular).
    for i in range(n - 1):
        b = i * 4
        rt_i, rb_i = b + 1, b + 3
        rt_n, rb_n = b + 5, b + 7
        faces.append([rt_i, rb_i, rt_n])
        faces.append([rt_n, rb_i, rb_n])

    # Front cap (start, normal points backwards along centerline).
    lt_0, rt_0, lb_0, rb_0 = 0, 1, 2, 3
    faces.append([lt_0, lb_0, rt_0])
    faces.append([rt_0, lb_0, rb_0])

    # Back cap (end, normal points forwards).
    last = (n - 1) * 4
    lt_e, rt_e, lb_e, rb_e = last, last + 1, last + 2, last + 3
    faces.append([lt_e, rt_e, lb_e])
    faces.append([rt_e, rb_e, lb_e])

    return vertices, np.array(faces, dtype=int)
