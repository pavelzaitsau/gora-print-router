# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Geometry conversion utilities for converting between geographic and mesh coordinates.

Handles coordinate transformations needed for OSM data integration.
"""

import numpy as np


class GeometryConverter:
    """
    Convert between geographic (lat/lon) and mesh (mm) coordinate systems.

    Critical for aligning OSM features with terrain mesh.
    Supports non-uniform scaling (different X and Y scales).
    """

    def __init__(
        self,
        bbox_lat_min: float,
        bbox_lat_max: float,
        bbox_lon_min: float,
        bbox_lon_max: float,
        model_width_mm: float,
        model_height_mm: float,
        scale_x: float,
        scale_y: float,
    ):
        """
        Initialize coordinate converter.

        Args:
            bbox_lat_min: Minimum latitude of terrain
            bbox_lat_max: Maximum latitude of terrain
            bbox_lon_min: Minimum longitude of terrain
            bbox_lon_max: Maximum longitude of terrain
            model_width_mm: Model width in millimeters
            model_height_mm: Model height in millimeters
            scale_x: X-axis scale (millimeters per meter)
            scale_y: Y-axis scale (millimeters per meter)
        """
        self.bbox_lat_min = bbox_lat_min
        self.bbox_lat_max = bbox_lat_max
        self.bbox_lon_min = bbox_lon_min
        self.bbox_lon_max = bbox_lon_max
        self.model_width_mm = model_width_mm
        self.model_height_mm = model_height_mm
        self.scale_x = scale_x
        self.scale_y = scale_y

        # Calculate meters per degree at this latitude
        lat_avg = (bbox_lat_min + bbox_lat_max) / 2
        self.meters_per_deg_lat = 111320  # Roughly constant
        self.meters_per_deg_lon = 111320 * np.cos(np.radians(lat_avg))

    def geo_to_mesh(self, lon: float, lat: float) -> tuple[float, float]:
        """
        Convert geographic coordinates to mesh coordinates.

        Args:
            lon: Longitude in degrees
            lat: Latitude in degrees

        Returns:
            Tuple of (x_mm, y_mm) in mesh coordinate system
        """
        # Convert to meters from top-left corner (lat_max, lon_min)
        # Terrain mesh uses y=0 at the northern edge (lat_max) and increases southward.
        x_m = (lon - self.bbox_lon_min) * self.meters_per_deg_lon
        y_m = (self.bbox_lat_max - lat) * self.meters_per_deg_lat

        # Convert to millimeters with non-uniform scaling
        x_mm = x_m * self.scale_x
        y_mm = y_m * self.scale_y

        return x_mm, y_mm

    def mesh_to_geo(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        """
        Convert mesh coordinates to geographic coordinates.

        Args:
            x_mm: X coordinate in millimeters
            y_mm: Y coordinate in millimeters

        Returns:
            Tuple of (lon, lat) in degrees
        """
        # Convert to meters with non-uniform scaling
        x_m = x_mm / self.scale_x
        y_m = y_mm / self.scale_y

        # Convert to degrees
        lon = self.bbox_lon_min + (x_m / self.meters_per_deg_lon)
        lat = self.bbox_lat_max - (y_m / self.meters_per_deg_lat)

        return lon, lat

    def clip_polygon_to_bbox(
        self, coords: list[tuple[float, float]]
    ) -> list[tuple[float, float]] | None:
        """
        Clip a polygon to the bounding box.

        Args:
            coords: List of (lon, lat) tuples

        Returns:
            Clipped polygon coordinates or None if no intersection
        """
        from shapely import make_valid
        from shapely.geometry import Polygon, box

        try:
            # Create polygon from coordinates
            polygon = Polygon(coords)

            # Fix invalid polygons before clipping
            if not polygon.is_valid:
                polygon = make_valid(polygon)
                if polygon.is_empty:
                    return None

            # Create bounding box
            bbox = box(
                self.bbox_lon_min,
                self.bbox_lat_min,
                self.bbox_lon_max,
                self.bbox_lat_max,
            )

            # Clip polygon to bbox
            clipped = polygon.intersection(bbox)

            # Handle different geometry types
            if clipped.is_empty:
                return None

            # Fix clipped geometry if invalid
            if not clipped.is_valid:
                clipped = make_valid(clipped)
                if clipped.is_empty:
                    return None

            # Extract coordinates from clipped geometry
            if hasattr(clipped, "exterior"):
                # Single polygon
                coords_list = list(clipped.exterior.coords)
                # Ensure polygon is closed
                if coords_list[0] != coords_list[-1]:
                    coords_list.append(coords_list[0])
                return coords_list
            elif hasattr(clipped, "geoms"):
                # MultiPolygon - take the largest piece
                largest = max(clipped.geoms, key=lambda g: g.area)
                coords_list = list(largest.exterior.coords)
                # Ensure polygon is closed
                if coords_list[0] != coords_list[-1]:
                    coords_list.append(coords_list[0])
                return coords_list
            else:
                return None

        except Exception:
            # Return original coordinates if clipping fails
            return coords

    def convert_polygon(
        self, coords: list[tuple[float, float]], clip_to_bbox: bool = True
    ) -> list[tuple[float, float]]:
        """
        Convert a polygon from geographic to mesh coordinates.

        Args:
            coords: List of (lon, lat) tuples
            clip_to_bbox: Whether to clip polygon to bounding box

        Returns:
            List of (x_mm, y_mm) tuples
        """
        # Clip to bounding box if requested
        if clip_to_bbox:
            clipped = self.clip_polygon_to_bbox(coords)
            if clipped is None:
                return []
            coords = clipped

        return [self.geo_to_mesh(lon, lat) for lon, lat in coords]
