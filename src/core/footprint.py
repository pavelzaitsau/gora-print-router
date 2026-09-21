# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Footprint: 2D model outline abstraction.

The model "footprint" is the 2D shape of the base/terrain when viewed from
above. Historically the router supported only rectangles defined by
--model-size-x / --model-size-y. This module generalises that to multiple
shapes (rectangle, hexagon, circle/oval) while keeping the rectangle case
identical to the previous behaviour.

Coordinates are in millimetres. Origin (0, 0) is the bbox bottom-left corner
so the footprint always lives in the first quadrant — matching the existing
mesh generator conventions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from shapely.affinity import translate
from shapely.geometry import Point, Polygon

ShapeName = str  # "rectangle" | "hexagon" | "circle"
HexOrientation = str  # "flat" | "pointy"

VALID_SHAPES: tuple[ShapeName, ...] = ("rectangle", "hexagon", "circle")
VALID_HEX_ORIENTATIONS: tuple[HexOrientation, ...] = ("flat", "pointy")


@dataclass(frozen=True)
class Footprint:
    """Immutable 2D footprint in mm coordinates."""

    polygon: Polygon
    width_mm: float
    height_mm: float
    shape: ShapeName
    # Extra metadata kept for hex (orientation) / circle (segment count).
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def rectangle(cls, width_mm: float, height_mm: float) -> Footprint:
        _validate_positive(width_mm, "width_mm")
        _validate_positive(height_mm, "height_mm")
        poly = Polygon(
            [
                (0.0, 0.0),
                (width_mm, 0.0),
                (width_mm, height_mm),
                (0.0, height_mm),
            ]
        )
        return cls(
            polygon=poly,
            width_mm=float(width_mm),
            height_mm=float(height_mm),
            shape="rectangle",
        )

    @classmethod
    def hexagon(
        cls,
        width_mm: float,
        height_mm: float,
        orientation: HexOrientation = "flat",
    ) -> Footprint:
        _validate_positive(width_mm, "width_mm")
        _validate_positive(height_mm, "height_mm")
        if orientation not in VALID_HEX_ORIENTATIONS:
            raise ValueError(
                f"orientation must be one of {VALID_HEX_ORIENTATIONS}, got {orientation!r}"
            )

        # Vertices defined inside the unit bbox [0,1] x [0,1] then scaled.
        if orientation == "flat":
            # Long axis horizontal — flat edges on left/right of a regular hex.
            # Six vertices around centre (0.5, 0.5).
            unit = [
                (1.0, 0.5),
                (0.75, 1.0),
                (0.25, 1.0),
                (0.0, 0.5),
                (0.25, 0.0),
                (0.75, 0.0),
            ]
        else:  # pointy
            # Long axis vertical — pointy vertices on top/bottom.
            unit = [
                (0.5, 1.0),
                (1.0, 0.75),
                (1.0, 0.25),
                (0.5, 0.0),
                (0.0, 0.25),
                (0.0, 0.75),
            ]

        scaled = [(x * width_mm, y * height_mm) for x, y in unit]
        poly = Polygon(scaled)
        return cls(
            polygon=poly,
            width_mm=float(width_mm),
            height_mm=float(height_mm),
            shape="hexagon",
            meta={"orientation": orientation},
        )

    @classmethod
    def circle(
        cls,
        width_mm: float,
        height_mm: float,
        segments: int = 256,
    ) -> Footprint:
        """Circle (width == height) or ellipse (width != height)."""
        _validate_positive(width_mm, "width_mm")
        _validate_positive(height_mm, "height_mm")
        if segments < 12:
            raise ValueError(f"segments must be >= 12, got {segments}")

        a = width_mm / 2.0  # semi-axis x
        b = height_mm / 2.0  # semi-axis y
        cx, cy = a, b  # centre at bbox centre, origin at bbox bottom-left

        verts: list[tuple[float, float]] = []
        for i in range(segments):
            theta = 2.0 * math.pi * i / segments
            verts.append((cx + a * math.cos(theta), cy + b * math.sin(theta)))
        poly = Polygon(verts)
        return cls(
            polygon=poly,
            width_mm=float(width_mm),
            height_mm=float(height_mm),
            shape="circle",
            meta={"segments": int(segments)},
        )

    # ------------------------------------------------------------------
    # Factory dispatch
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        shape: ShapeName,
        width_mm: float,
        height_mm: float,
        *,
        hex_orientation: HexOrientation = "flat",
        circle_segments: int = 256,
    ) -> Footprint:
        """Single entry point used by the CLI/router layer."""
        if shape == "rectangle":
            return cls.rectangle(width_mm, height_mm)
        if shape == "hexagon":
            return cls.hexagon(width_mm, height_mm, hex_orientation)
        if shape == "circle":
            return cls.circle(width_mm, height_mm, circle_segments)
        raise ValueError(f"shape must be one of {VALID_SHAPES}, got {shape!r}")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def contains(self, x: float, y: float) -> bool:
        # shapely ships no stubs, so `covers` is Any — narrow it at the boundary.
        return bool(self.polygon.covers(Point(x, y)))

    def perimeter_points(self) -> list[tuple[float, float]]:
        """Ordered CCW boundary vertices, no closing duplicate."""
        coords = list(self.polygon.exterior.coords)
        # Shapely closes the ring; drop the duplicate last point.
        if coords and coords[0] == coords[-1]:
            coords = coords[:-1]
        # Ensure CCW orientation (positive signed area).
        if _signed_area(coords) < 0:
            coords.reverse()
        return coords

    def signed_distance(self, x: float, y: float) -> float:
        """Positive inside, negative outside, 0 on boundary."""
        pt = Point(x, y)
        d = float(self.polygon.exterior.distance(pt))
        return d if self.polygon.covers(pt) else -d

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y) in mm."""
        min_x, min_y, max_x, max_y = self.polygon.bounds
        return (float(min_x), float(min_y), float(max_x), float(max_y))

    @property
    def area_mm2(self) -> float:
        return float(self.polygon.area)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def outset(self, mm: float) -> Footprint:
        """
        Minkowski outset by `mm`. Returns a new Footprint translated so its
        bbox bottom-left stays at (0, 0). Negative `mm` shrinks the shape.
        """
        if mm == 0.0:
            return self
        join_style = "round" if self.shape == "circle" else "mitre"
        quad_segs = max(32, self.meta.get("segments", 64) // 4)
        new_poly = self.polygon.buffer(
            mm,
            join_style=join_style,
            mitre_limit=20.0,
            quad_segs=quad_segs,
        )
        if new_poly.is_empty:
            raise ValueError(f"outset({mm}) produced empty polygon")

        min_x, min_y, max_x, max_y = new_poly.bounds
        # Translate so bbox bottom-left = origin (preserve invariant).
        new_poly = translate(new_poly, xoff=-min_x, yoff=-min_y)
        return Footprint(
            polygon=new_poly,
            width_mm=float(max_x - min_x),
            height_mm=float(max_y - min_y),
            shape=self.shape,
            meta=dict(self.meta),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def pick_hex_orientation(track_width: float, track_height: float) -> HexOrientation:
        """Pick hex orientation so the long axis aligns with the track bbox."""
        if track_width <= 0 or track_height <= 0:
            return "flat"
        return "flat" if track_width >= track_height else "pointy"


# ----------------------------------------------------------------------
# Module-private helpers
# ----------------------------------------------------------------------


def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _signed_area(coords: list[tuple[float, float]]) -> float:
    """Signed area of a ring (positive = CCW)."""
    n = len(coords)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        s += (x2 - x1) * (y2 + y1)
    return -s / 2.0
