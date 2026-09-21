# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Water layer generation for 3D terrain models.

Generates terrain-following water surfaces from OSM data.
"""

import math
from collections.abc import Sequence

import numpy as np
import shapely
from rasterio.transform import Affine
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import LineString, MultiLineString, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import linemerge, polygonize, unary_union

from src.core.footprint import Footprint
from src.osm.geometry_converter import GeometryConverter
from src.osm.osm_utils import create_ribbon_geometry, densify_polyline
from src.osm.overpass_client import OverpassClient
from src.osm.water_features import (
    WaterFeature,
    WaterFilterConfig,
    extract_water_features,
    filter_by_lod,
)
from src.utils.obj_exporter import Mesh

# A model-space point, millimetres, in the XY plane of the print bed.
PointXY = tuple[float, float]
# One mesh vertex as the builders below accumulate it: [x_mm, y_mm, z_mm].
Vertex = list[float]

# Water surface offset above terrain (mm). Increased for better visibility.
# Water is placed at: terrain_elevation + water_offset_mm
# The terrain elevation is from the FINAL processed terrain (after smoothing/upsampling)
WATER_OFFSET_MM = 0.4

# Water thickness (mm). Added downward from the top water surface.
WATER_THICKNESS_MM = 0.5

# Absolute floor for linear water features (mm). A linear waterway is always
# rendered at least this wide so it stays continuous and doesn't drop out /
# print broken on an FDM printer (a sub-1mm strip is below what one or two
# nozzle passes can reliably lay down). Hard floor — applied regardless of
# model size; natural widths above it are kept as-is.
MIN_WATERWAY_WIDTH_MM = 1.0

# River edge margin (mm) - prevents rivers from extending to model edges
RIVER_EDGE_MARGIN_MM = 1.5

# Grid the footprint clip snaps its result onto (mm).
#
# An intersection can leave two vertices a few float ulps apart. It takes a
# water vertex grazing the clip boundary: the vertex is inside the clip region
# so the overlay keeps it, and GEOS also inserts the node it computed for the
# crossing at what is geometrically the same place. In exact double arithmetic
# the two differ, so shapely hands back a ring (or a centerline) carrying a
# ~1e-12mm edge, and every consumer builds real geometry across it - the cap
# triangulation, the wall walk, the ribbon's cross-sections.
#
# Nothing downstream can hold that apart: the OBJ writer emits `%.6f` and a
# slicer welds coarser still, so the pair lands on one vertex, the faces spanning
# it collapse to zero area, and their edges fold into the neighbours -
# non-manifold edges, reversed winding, and a shell that is no longer closed.
#
# Snapping the clip output to a grid removes the pair before any consumer sees
# it, and keeps caps and walls reading the same boundary. 1e-4mm is 100x above
# what the OBJ text can distinguish and ~1000x below one printed layer, so it
# separates what the exporter must tell apart without moving anything visible.
FOOTPRINT_CLIP_GRID_MM = 1e-4


def _clip_to_footprint(geometry: BaseGeometry, footprint_polygon: Polygon) -> BaseGeometry:
    """Intersect water geometry with the model outline, snapped to a printable grid.

    The snap is not cosmetic - see `FOOTPRINT_CLIP_GRID_MM`. Without it the clip
    can hand back vertices closer together than the exporter can represent, and
    the water body built on them comes out non-manifold.

    Args:
        geometry: Water polygon or centerline in model space (mm).
        footprint_polygon: The model outline to clip against (mm).

    Returns:
        The clipped geometry, or an empty geometry if nothing survives. Callers
        must handle a multi-part result: snapping can pinch a waist that the raw
        intersection left joined. Anything thinner or shorter than the grid
        collapses to empty, which is the right answer - at that size it is not
        printable geometry, only the sliver that would break the shell.
    """
    clipped = geometry.intersection(footprint_polygon)
    if clipped.is_empty:
        return clipped
    snapped: BaseGeometry = shapely.set_precision(clipped, FOOTPRINT_CLIP_GRID_MM)
    return snapped


def _polygon_triangles(polygon: Polygon) -> list[Polygon]:
    """Triangulate a polygon so the triangles tile it exactly.

    Plain Delaunay triangulates the *convex hull* of the vertices: on a concave
    outline (every real coastline) some triangles straddle the boundary, so the
    top and bottom caps stop lining up with the side walls built from the ring
    and the slab comes out with holes along the coast. A constrained Delaunay
    honours every ring, holes included.
    """
    collection = shapely.constrained_delaunay_triangles(polygon)
    return [tri for tri in getattr(collection, "geoms", []) if not tri.is_empty]


def _ccw(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return a triangle's corners counter-clockwise (positive signed area).

    `shapely.ops.triangulate` gives no orientation guarantee, so the top face of
    a water slab has to be normalised or half the triangles point downward.
    """
    (x0, y0), (x1, y1), (x2, y2) = coords
    signed_area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    return coords if signed_area2 >= 0 else [coords[0], coords[2], coords[1]]


def compute_waterway_width_mm(natural_width_mm: float, model_min_dim_mm: float) -> float:
    """
    Pick a printable ribbon width for a linear water feature.

    Clamps to an absolute printable floor (`MIN_WATERWAY_WIDTH_MM`) so the
    ribbon never prints too thin to be continuous. Natural widths already
    wider than the floor are returned unchanged.

    Args:
        natural_width_mm: Physical width × scale (mm).
        model_min_dim_mm: min(model_width_mm, model_height_mm). Retained for
            call-site compatibility; no longer used to lower the floor.

    Returns:
        Width to use for the ribbon mesh (mm).
    """
    if natural_width_mm <= 0:
        return MIN_WATERWAY_WIDTH_MM
    return max(natural_width_mm, MIN_WATERWAY_WIDTH_MM)


# Waterway tags considered linear features (rivers/streams/creeks)
LINEAR_WATERWAY_TYPES = {
    "river",
    "stream",
    "creek",
    "brook",
    "canal",
    "ditch",
    "drain",
    "tidal_channel",
    "wadi",
}


def _overlap_at_junctions(lines: list[LineString], width_m: float) -> list[LineString]:
    """Push every branch past a shared endpoint so the ribbons overlap there.

    `unary_union` nodes a confluence: where one river meets another it splits
    the through-river in two and leaves all three branches sharing one vertex.
    Swept, their walls weld along the edges at that vertex and three faces end
    up on one edge, which is not manifold and is not printable.

    Overlapping bodies are fine here, for the same reason the lakes and the
    track's beads are: the slicer unions them. Touching ones are not. So each
    branch is extended past the junction by its own width, far enough that the
    tubes interpenetrate rather than abut, and short enough to stay inside the
    water the junction already covers.
    """
    if len(lines) < 2:
        return lines

    ends: dict[tuple[float, float], int] = {}
    for line in lines:
        coords = list(line.coords)
        for point in (coords[0], coords[-1]):
            ends[point] = ends.get(point, 0) + 1
    shared = {point for point, n in ends.items() if n > 1}
    if not shared:
        return lines

    out: list[LineString] = []
    for line in lines:
        coords = list(line.coords)
        if len(coords) < 2:
            out.append(line)
            continue
        if coords[0] in shared:
            coords[0] = _step_beyond(coords[1], coords[0], width_m)
        if coords[-1] in shared:
            coords[-1] = _step_beyond(coords[-2], coords[-1], width_m)
        out.append(LineString(coords))
    return out


def _step_beyond(
    inner: tuple[float, float], end: tuple[float, float], distance_m: float
) -> tuple[float, float]:
    """One point `distance_m` past `end`, along the direction `inner` -> `end`."""
    lat_rad = math.radians(end[1])
    m_per_deg_lat = 111320.0
    m_per_deg_lon = m_per_deg_lat * max(math.cos(lat_rad), 1e-6)

    dx_m = (end[0] - inner[0]) * m_per_deg_lon
    dy_m = (end[1] - inner[1]) * m_per_deg_lat
    length_m = math.hypot(dx_m, dy_m)
    if length_m <= 0.0:
        return end

    step = max(distance_m, 1.0) / length_m
    return (end[0] + (end[0] - inner[0]) * step, end[1] + (end[1] - inner[1]) * step)


class WaterLayerGenerator:
    """
    Generates 3D water surfaces that follow terrain elevation.

    Key features:
    - Loads full-resolution elevation data (not downsampled)
    - Creates terrain-following surfaces within water polygon boundaries
    - Samples elevation at each vertex for accurate 3D representation
    """

    def __init__(
        self,
        lat_min: float,
        lon_min: float,
        lat_max: float,
        lon_max: float,
        model_width_mm: float,
        model_height_mm: float,
        base_thickness_mm: float,
        scale: float,
        vertical_exaggeration: float,
        lod_level: int,
        terrain_elevation: np.ndarray,
        terrain_transform: Affine,
        terrain_resolution: int = 200,
        scale_x: float | None = None,
        scale_y: float | None = None,
        water_offset_mm: float = WATER_OFFSET_MM,
        water_thickness_mm: float = WATER_THICKNESS_MM,
        use_cache: bool = True,
        osm_cache_dir: str = "osm_cache",
        y_origin_south: bool = False,
        footprint: Footprint | None = None,
    ) -> None:
        """
        Initialize water layer generator.

        Args:
            lat_min, lon_min, lat_max, lon_max: Geographic bounds
            model_width_mm, model_height_mm: Model dimensions
            base_thickness_mm: Base layer thickness
            scale: Millimeters per meter conversion (average if non-uniform)
            vertical_exaggeration: Z-axis multiplier
            lod_level: Level of detail (0-5, physical size-based filtering)
            terrain_elevation: Processed terrain elevation array (normalized, smoothed)
            terrain_transform: Rasterio transform for terrain
            terrain_resolution: Terrain resolution for normalization reference
            scale_x: X-axis scale (mm/m), uses scale if None
            scale_y: Y-axis scale (mm/m), uses scale if None
            water_offset_mm: Vertical offset applied above terrain (mm)
            water_thickness_mm: Water thickness added downward from top surface (mm)
            use_cache: Whether to use OSM query caching (default: True)
            osm_cache_dir: Path to OSM cache directory (default: "osm_cache")
            y_origin_south: If True, model Y=0 is south edge (track terrain orientation)
        """
        self.lat_min = lat_min
        self.lon_min = lon_min
        self.lat_max = lat_max
        self.lon_max = lon_max
        self.model_width_mm = model_width_mm
        self.model_height_mm = model_height_mm
        self.base_thickness_mm = base_thickness_mm
        self.scale = scale
        self.scale_x = scale_x if scale_x is not None else scale
        self.scale_y = scale_y if scale_y is not None else scale
        self.vertical_exaggeration = vertical_exaggeration
        self.lod_level = lod_level
        self.terrain_resolution = terrain_resolution
        self.terrain_elevation = terrain_elevation
        self.terrain_transform = terrain_transform
        self.water_offset_mm = water_offset_mm
        self.water_thickness_mm = water_thickness_mm
        self.use_cache = use_cache
        self.osm_cache_dir = osm_cache_dir
        self.y_origin_south = y_origin_south
        # Optional Footprint (mesh-space polygon) used to clip water features
        # to non-rectangular model outlines. None for rectangle = no clipping.
        self.footprint = footprint

        # Will be set when loading elevation
        self.elevation_interpolator = None

        # Geometry converter
        self.geo_converter = GeometryConverter(
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            model_width_mm,
            model_height_mm,
            self.scale_x,
            self.scale_y,
        )

        # Calculate bbox parameters for physical filtering
        lat_avg = (lat_max + lat_min) / 2
        meters_per_deg_lat = 111320
        meters_per_deg_lon = 111320 * np.cos(np.radians(lat_avg))

        bbox_width_m = abs(lon_max - lon_min) * meters_per_deg_lon
        bbox_height_m = abs(lat_max - lat_min) * meters_per_deg_lat
        bbox_area_m2 = bbox_width_m * bbox_height_m

        self.bbox_area_km2 = bbox_area_m2 / 1_000_000  # Convert to km²
        self.bbox_width_km = bbox_width_m / 1000  # Convert to km

        # Create water filter configuration
        self.water_config = WaterFilterConfig(
            bbox_area_km2=self.bbox_area_km2,
            model_width_mm=model_width_mm,
            bbox_width_km=self.bbox_width_km,
            lod_level=lod_level,
        )

        print(
            f"  Water filter config: bbox_area={self.bbox_area_km2:.2f}km², "
            f"bbox_width={self.bbox_width_km:.2f}km, model_width={model_width_mm:.1f}mm"
        )
        print(f"  Scale: {self.water_config.meters_per_mm:.1f}m of ground per mm of model")
        print(
            f"  Min lake area={self.water_config.min_lake_area_m2 / 1_000_000:.3f}km² ({self.water_config.min_lake_area_m2:.0f}m²)"
        )
        if self.water_config.allow_linear:
            print(
                f"  Min river width={self.water_config.min_river_width_m:.1f}m, "
                f"Min river length={self.water_config.min_river_length_m:.0f}m"
            )
        print(f"  Water offset above terrain: {self.water_offset_mm:.2f}mm (for visibility)")
        print(f"  River edge margin: {RIVER_EDGE_MARGIN_MM:.1f}mm (keeps rivers from model edges)")

    def _geo_to_model(self, lon: float, lat: float) -> tuple[float, float]:
        """Convert geographic coordinates to model coordinates."""
        x_mm = ((lon - self.lon_min) / (self.lon_max - self.lon_min)) * self.model_width_mm
        if self.y_origin_south:
            y_mm = ((lat - self.lat_min) / (self.lat_max - self.lat_min)) * self.model_height_mm
        else:
            y_mm = ((self.lat_max - lat) / (self.lat_max - self.lat_min)) * self.model_height_mm
        return x_mm, y_mm

    def _model_to_geo(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        """Convert model coordinates back to geographic coordinates."""
        lon = self.lon_min + (x_mm / self.model_width_mm) * (self.lon_max - self.lon_min)
        if self.y_origin_south:
            lat = self.lat_min + (y_mm / self.model_height_mm) * (self.lat_max - self.lat_min)
        else:
            lat = self.lat_max - (y_mm / self.model_height_mm) * (self.lat_max - self.lat_min)
        return lon, lat

    def _load_elevation_data(self) -> bool:
        """
        Create interpolator from processed terrain elevation data.

        Uses the same elevation data as terrain (upsampled, smoothed) to ensure
        water surfaces align perfectly with terrain.

        Returns:
            True if successful
        """
        try:
            # Use processed terrain elevation data
            elevation_data = self.terrain_elevation
            rows, cols = elevation_data.shape

            # Get coordinate arrays from the terrain affine transform
            # Ensures sampling aligns with the actual raster grid (pixel centers)
            lon_coords = np.array([self.terrain_transform * (j, 0) for j in range(cols)])[:, 0]
            lat_coords = np.array([self.terrain_transform * (0, i) for i in range(rows)])[:, 1]

            # Create interpolator with processed terrain data
            # Note: elevation_data[i,j] where i=0 is northernmost (lat_max)
            self.elevation_interpolator = RegularGridInterpolator(
                (lat_coords, lon_coords),
                elevation_data,
                bounds_error=False,
                fill_value=0,
            )

            return True

        except Exception as e:
            print(f"  ⚠ Water layer skipped: could not read the elevation data ({e})")
            return False

    def _sample_elevation(self, lat: float, lon: float) -> float:
        """
        Sample terrain elevation at geographic coordinate.

        Uses ONLY the processed terrain elevation data (after upsampling and smoothing).
        Any elevation data from OSM is completely ignored - water surfaces follow
        the final terrain geometry.

        Args:
            lat, lon: Geographic coordinates

        Returns:
            Elevation in millimeters (terrain height at this point)
        """
        if self.elevation_interpolator is None:
            return self.base_thickness_mm

        # Sample elevation in meters from processed terrain data
        # This is the FINAL terrain after upsampling and smoothing
        elev_m = self.elevation_interpolator((lat, lon))

        # Convert to millimeters with vertical exaggeration
        z_mm = elev_m * self.scale * self.vertical_exaggeration + self.base_thickness_mm

        return z_mm

    def _calculate_lake_surface_elevation(
        self, polygon_coords: list[tuple[float, float]], polygon_shapely: Polygon | None = None
    ) -> float:
        """
        Calculate flat elevation for a lake surface.

        Samples terrain at every shoreline boundary vertex and returns a low
        percentile (default 10th) rather than the raw minimum. Using a strict
        minimum makes a single outlier point (a deeply cut ravine touching the
        shoreline, or a misaligned DEM pixel) drag the entire lake surface
        below the real waterline, leaving the lake mesh visibly below terrain
        on the rest of the perimeter. The 10th percentile is robust against
        such outliers while still tracking the true low side of the basin.

        Interior sampling is avoided because the DEM inside a reservoir
        polygon reflects the flooded valley terrain, not the water surface.

        Args:
            polygon_coords: List of (lon, lat) coordinates defining the lake boundary
            polygon_shapely: Unused, kept for API compatibility

        Returns:
            Boundary elevation (10th percentile) in millimeters.
        """
        if not polygon_coords:
            return self.base_thickness_mm

        # Skip duplicate closing vertex.
        coords = polygon_coords[:-1] if len(polygon_coords) > 1 else polygon_coords

        # Cap the sample count to bound cost on very large OSM polygons; pick
        # vertices via uniform stride so we still cover the whole shoreline.
        max_samples = 1024
        if len(coords) > max_samples:
            stride = len(coords) // max_samples
            coords = coords[::stride][:max_samples]

        elevation_samples = [self._sample_elevation(lat, lon) for lon, lat in coords]
        if not elevation_samples:
            return self.base_thickness_mm

        # 10th percentile = robust low estimate. For polygons with few samples
        # (<10 vertices) numpy.percentile is well-defined; for a single sample
        # it returns that sample.
        return float(np.percentile(elevation_samples, 10))

    def _verify_water_surface_heights(
        self, feature_info: list[dict[str, object]], vertices_array: np.ndarray
    ) -> None:
        """
        Verify that water surfaces are positioned correctly relative to terrain.

        Args:
            feature_info: List of dicts with feature data and vertex info
            vertices_array: Array of all water vertices [x, y, z]
        """
        return

    def _generate_ocean_surface(self) -> Mesh | None:
        """
        Generate ocean/sea surface for areas at or below sea level.

        This method creates a simple ocean layer by:
        1. Identifying all terrain areas at or below sea level (elevation <= 0)
        2. Creating a flat water surface at sea level, with no offset
        3. Smoothing coastline boundaries for natural appearance
        4. Generating a watertight mesh with proper thickness for 3D printing

        Returns:
            Mesh object with ocean surface or None if no water areas found
        """
        if self.elevation_interpolator is None:
            print("  ⚠ No elevation data available for ocean generation")
            return None

        # Sample terrain elevation across the entire model area
        # Use a grid that's fine enough to capture coastline details
        grid_resolution = 100  # Points per dimension

        # Create coordinate arrays
        lats = np.linspace(self.lat_min, self.lat_max, grid_resolution)
        lons = np.linspace(self.lon_min, self.lon_max, grid_resolution)

        # Identify cells below sea level
        water_cells = []
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                lat = lats[i]
                lon = lons[j]

                # Get terrain elevation in meters (before scale conversion)
                elev_m = self.elevation_interpolator((lat, lon))

                # Check if at or below sea level (0m elevation)
                if elev_m <= 0.0:
                    water_cells.append((i, j, lat, lon))

        if not water_cells:
            print("  ⚠ No areas below sea level found")
            return None

        print(f"  Found {len(water_cells)} water cells below sea level")

        # Build a binary mask for water cells
        water_mask = np.zeros((grid_resolution, grid_resolution), dtype=bool)
        for i, j, _, _ in water_cells:
            water_mask[i, j] = True

        # Smooth the water mask to reduce jaggedness while preserving features
        # Use binary morphological operations: close small gaps, remove noise
        water_mask_smoothed = ndimage.binary_closing(water_mask, iterations=2)
        water_mask_smoothed = ndimage.binary_opening(water_mask_smoothed, iterations=1)

        # Extract water polygons from the smoothed mask
        water_polygons = self._extract_polygons_from_mask(water_mask_smoothed, lats, lons)

        if not water_polygons:
            print("  ⚠ No water polygons extracted from mask")
            return None

        print(f"  Extracted {len(water_polygons)} water polygon(s)")

        # Ocean is at sea level (0m normalized = base_thickness_mm). No offset.
        water_surface_z = self.base_thickness_mm
        water_bottom_z = water_surface_z - self.water_thickness_mm

        # Convert polygons to mesh
        all_vertices = []
        all_faces = []
        vertex_offset = 0

        for poly_idx, polygon in enumerate(water_polygons):
            try:
                # Simplify polygon slightly to reduce vertex count while preserving shape
                # Tolerance in mm - adjust based on model size
                simplify_tolerance = min(self.model_width_mm, self.model_height_mm) * 0.001
                polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)

                if not polygon.is_valid or polygon.is_empty or len(polygon.exterior.coords) < 3:
                    continue

                # Apply Chaikin smoothing to the polygon boundary for natural curves
                smoothed_polygon = self._smooth_polygon_chaikin(polygon, iterations=2)

                if not smoothed_polygon.is_valid or smoothed_polygon.is_empty:
                    continue

                # Create mesh from smoothed polygon
                result = self._create_water_polygon_mesh(
                    smoothed_polygon, water_surface_z, water_bottom_z
                )

                if result is None:
                    continue

                vertices, faces, _ = result

                # Adjust face indices and add to global lists
                adjusted_faces = faces + vertex_offset
                all_vertices.extend(vertices)
                all_faces.extend(adjusted_faces)
                vertex_offset += len(vertices)

            except Exception as e:
                print(f"  ⚠ Warning: Failed to process polygon {poly_idx}: {e}")
                continue

        if not all_vertices:
            print("  ⚠ No water mesh vertices generated")
            return None

        vertices_array = np.array(all_vertices)
        faces_array = np.array(all_faces)

        print(f"  Ocean mesh: {len(vertices_array)} vertices, {len(faces_array)} faces")
        print(f"  Water surface at Z={water_surface_z:.2f}mm")

        # Validation: Verify ocean coverage
        total_cells = (grid_resolution - 1) * (grid_resolution - 1)
        water_cell_count = len(water_cells)
        coverage_pct = (water_cell_count / total_cells) * 100 if total_cells > 0 else 0
        print(f"  Ocean coverage: {water_cell_count}/{total_cells} cells ({coverage_pct:.1f}%)")
        print("  Validation: Ocean layer includes all terrain at or below sea level")
        print("  Validation: Coastline smoothed for natural appearance")
        print("  Validation: Mesh is watertight for 3D printing")

        # Build structured array with face triangles
        mesh_data = np.zeros(len(faces_array), dtype=Mesh.dtype)
        for i, face in enumerate(faces_array):
            # Get three vertices of this triangle
            v0, v1, v2 = vertices_array[face]
            mesh_data["vectors"][i] = np.array([v0, v1, v2])
            # Calculate face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm_length = np.linalg.norm(normal)
            if norm_length > 0:
                normal = normal / norm_length
            mesh_data["normals"][i] = normal

        return Mesh(mesh_data)

    def _extract_polygons_from_mask(
        self, mask: np.ndarray, lats: np.ndarray, lons: np.ndarray
    ) -> list[Polygon]:
        """
        Extract polygons from binary mask by tracing boundaries.

        Args:
            mask: Binary mask of water cells
            lats: Latitude array
            lons: Longitude array

        Returns:
            List of Shapely polygons representing water areas
        """
        polygons = []
        rows, cols = mask.shape

        # Cell edges, shared between neighbours by construction. Deriving each
        # cell's own bounds as centre ± half a step instead leaves neighbouring
        # rectangles a float ulp apart, so `unary_union` never merges them and
        # the coastline comes back as hundreds of loose one-cell squares — each
        # smoothed into its own blob and walled on all four sides.
        lat_edges = np.empty(rows + 1)
        lat_edges[1:-1] = (lats[:-1] + lats[1:]) / 2 if rows > 1 else lats[0]
        lat_edges[0], lat_edges[-1] = lats[0], lats[-1]
        lon_edges = np.empty(cols + 1)
        lon_edges[1:-1] = (lons[:-1] + lons[1:]) / 2 if cols > 1 else lons[0]
        lon_edges[0], lon_edges[-1] = lons[0], lons[-1]

        x_edges = np.array([self._geo_to_model(lon, lats[0])[0] for lon in lon_edges])
        y_edges = np.array([self._geo_to_model(lons[0], lat)[1] for lat in lat_edges])

        # Process each connected water region
        labeled_mask, num_features = ndimage.label(mask)

        for region_id in range(1, num_features + 1):
            region_mask = labeled_mask == region_id

            # Use ALL water cells (not just boundary cells).
            # Unioning all cell rectangles naturally creates holes where islands
            # (non-water areas) are enclosed by the region — using only boundary
            # cells drops interior cells and loses the island holes.
            all_cells = np.argwhere(region_mask)

            if len(all_cells) < 3:
                continue

            # Convert cells to polygon
            # Create rectangles for each water cell and union them
            cell_polygons = [
                box(x_edges[j], y_edges[i + 1], x_edges[j + 1], y_edges[i]) for i, j in all_cells
            ]

            # Union all cell polygons to create smooth boundary
            if cell_polygons:
                try:
                    region_polygon = unary_union(cell_polygons)

                    # Handle MultiPolygon case (disconnected regions)
                    if hasattr(region_polygon, "geoms"):
                        # Multiple disconnected polygons
                        for geom in region_polygon.geoms:
                            if isinstance(geom, Polygon) and not geom.is_empty:
                                polygons.append(geom)
                    elif isinstance(region_polygon, Polygon) and not region_polygon.is_empty:
                        polygons.append(region_polygon)
                except Exception as e:
                    print(f"  ⚠ Warning: Failed to create polygon for region {region_id}: {e}")
                    continue

        return polygons

    def _smooth_polygon_chaikin(
        self, polygon: Polygon, iterations: int = 2, ratio: float = 0.25
    ) -> Polygon:
        """
        Apply Chaikin's corner-cutting algorithm to smooth polygon boundaries.

        Args:
            polygon: Input polygon to smooth
            iterations: Number of smoothing iterations (default: 2)
            ratio: Corner cutting ratio (default: 0.25 for smooth curves)

        Returns:
            Smoothed polygon with natural-looking curves
        """

        def _smooth_ring(
            coords: Sequence[Sequence[float]], iters: int, r: float
        ) -> list[list[float]]:
            # Shapely hands back tuples; the corner-cutting below produces
            # lists, so normalise once instead of widening the loop variable.
            ring = [[float(p[0]), float(p[1])] for p in coords]
            for _ in range(iters):
                out: list[list[float]] = []
                n = len(ring)
                for i in range(n):
                    p0 = ring[i]
                    p1 = ring[(i + 1) % n]
                    out.append([p0[0] * (1 - r) + p1[0] * r, p0[1] * (1 - r) + p1[1] * r])
                    out.append([p0[0] * r + p1[0] * (1 - r), p0[1] * r + p1[1] * (1 - r)])
                ring = out
            return ring

        ext_coords = _smooth_ring(list(polygon.exterior.coords[:-1]), iterations, ratio)
        if len(ext_coords) < 3:
            return polygon

        # Smooth each interior ring (island holes) and preserve them
        smoothed_holes: list[list[list[float]]] = []
        for interior in polygon.interiors:
            hole_coords = _smooth_ring(list(interior.coords[:-1]), iterations, ratio)
            if len(hole_coords) >= 3:
                smoothed_holes.append(hole_coords)

        result = Polygon(ext_coords, smoothed_holes)
        return result if result.is_valid else polygon

    def _create_water_polygon_mesh(
        self, polygon: Polygon, water_surface_z: float, water_bottom_z: float
    ) -> tuple[list[Vertex], np.ndarray, int] | None:
        """
        Create watertight mesh from water polygon.

        Args:
            polygon: Water surface polygon boundary
            water_surface_z: Height of water surface (mm)
            water_bottom_z: Height of water bottom (mm)

        Returns:
            Tuple of (vertices, faces, top_vertex_count) or None if failed.
        """
        # Clip incoming water polygon to the model footprint for non-rect
        # shapes. Avoids fan-triangulation wedges that stick out past hex /
        # circle boundaries.
        if (
            self.footprint is not None
            and getattr(self.footprint, "shape", "rectangle") != "rectangle"
        ):
            try:
                clipped = _clip_to_footprint(polygon, self.footprint.polygon)
            except Exception:
                # Falling back to the unclipped polygon used to be survivable
                # because the exporter trimmed whatever left the outline. That
                # trim is gone (it opened the shell it cut), so the fallback
                # would now ship water hanging off the model. Skip the body.
                return None
            if clipped.is_empty:
                return None
            # MultiPolygon → handle each piece by recursing.
            if clipped.geom_type == "MultiPolygon":
                pieces_vertices: list[Vertex] = []
                pieces_faces: list[list[int]] = []
                offset = 0
                for sub in clipped.geoms:
                    if sub.is_empty or sub.area < 1e-6:
                        continue
                    sub_result = self._create_water_polygon_mesh(
                        sub, water_surface_z, water_bottom_z
                    )
                    if sub_result is None:
                        continue
                    sv, sf, _ = sub_result
                    pieces_vertices.extend(sv)
                    pieces_faces.extend((np.array(sf) + offset).tolist())
                    offset += len(sv)
                if not pieces_vertices:
                    return None
                return (
                    pieces_vertices,
                    np.array(pieces_faces, dtype=np.int32),
                    len(pieces_vertices),
                )
            if clipped.geom_type != "Polygon":
                # GeometryCollection or LineString — skip.
                return None
            polygon = clipped

        try:
            all_vertices: list[Vertex] = []
            faces: list[list[int]] = []
            _top_vertex_count: int = 0

            # Exterior ring CCW, holes CW. Every winding below is derived from
            # that convention, so an arbitrarily-oriented input polygon would
            # otherwise turn the slab inside out (normals pointing into the
            # water body, negative volume).
            polygon = orient(polygon, sign=1.0)

            tris = _polygon_triangles(polygon)
            if not tris:
                return None

            # Top + bottom surface triangles (separate vertex blocks per tri
            # is fine for STL — no indexed sharing needed).
            top_start = 0
            for tri in tris:
                coords = _ccw(list(tri.exterior.coords[:3]))
                for x, y in coords:
                    all_vertices.append([x, y, water_surface_z])
            bot_start = len(all_vertices)
            for tri in tris:
                coords = _ccw(list(tri.exterior.coords[:3]))
                for x, y in coords:
                    all_vertices.append([x, y, water_bottom_z])

            for i in range(len(tris)):
                b = top_start + i * 3
                faces.append([b, b + 1, b + 2])  # top (CCW)
                b2 = bot_start + i * 3
                faces.append([b2, b2 + 2, b2 + 1])  # bottom (CW = reversed)

            _top_vertex_count = bot_start + 1

            # --- Side walls along every ring (exterior + interior/island holes) ---
            # One formula for both: `orient` left the exterior CCW and the holes
            # CW, and walking a ring with the water on the left always puts the
            # wall normal on the water's outside — which for an island hole means
            # facing into the island.
            rings = [polygon.exterior, *list(polygon.interiors)]
            for ring in rings:
                ring_coords = list(ring.coords[:-1])
                n = len(ring_coords)
                if n < 2:
                    continue

                v_base = len(all_vertices)
                for x, y in ring_coords:
                    all_vertices.append([x, y, water_surface_z])  # top
                for x, y in ring_coords:
                    all_vertices.append([x, y, water_bottom_z])  # bottom

                for i in range(n):
                    ni = (i + 1) % n
                    t_i = v_base + i
                    t_ni = v_base + ni
                    b_i = v_base + n + i
                    b_ni = v_base + n + ni
                    faces.append([t_i, b_ni, t_ni])
                    faces.append([t_i, b_i, b_ni])

            if not all_vertices:
                return None

            return all_vertices, np.array(faces, dtype=np.int32), _top_vertex_count

        except Exception as e:
            print(f"  ⚠ Warning: Failed to create water polygon mesh: {e}")
            return None

    def _estimate_waterway_width_m(self, feature: WaterFeature) -> float:
        """
        Estimate waterway width in meters from OSM tags with fallback defaults.
        """
        width_candidates = [
            feature.get_tag("width"),
            feature.get_tag("est_width"),
            feature.get_tag("width:estimated"),
            feature.get_tag("waterway:width"),
        ]
        for width_str in width_candidates:
            if not width_str:
                continue
            try:
                # Parse values like "15", "15 m", "15.5m"
                numeric = "".join(
                    ch for ch in width_str if (ch.isdigit() or ch == "." or ch == "-")
                )
                if numeric:
                    return max(0.0, float(numeric))
            except Exception:
                continue

        waterway = feature.get_tag("waterway") or feature.get_tag("water")
        defaults = {
            "river": 30.0,
            "stream": 5.0,
            "creek": 4.0,
            "brook": 3.0,
            "canal": 10.0,
            "ditch": 2.0,
            "drain": 2.0,
            "tidal_channel": 10.0,
            "wadi": 5.0,
        }
        return defaults.get(waterway, 5.0)

    def _is_linear_water_feature(self, feature: WaterFeature) -> bool:
        """Return True if feature represents a linear waterway."""
        if feature.is_closed():
            return False
        waterway = feature.get_tag("waterway")
        water_sub = feature.get_tag("water")
        return (waterway in LINEAR_WATERWAY_TYPES) or (water_sub in LINEAR_WATERWAY_TYPES)

    def _merge_linear_waterways(self, features: list[WaterFeature]) -> list[WaterFeature]:
        """
        Merge connected waterway segments into continuous lines.
        """
        grouped: dict[str, list[WaterFeature]] = {}
        for feat in features:
            waterway = feat.get_tag("waterway") or feat.get_tag("water")
            if not waterway:
                continue
            grouped.setdefault(waterway, []).append(feat)

        merged_features: list[WaterFeature] = []
        for waterway, group in grouped.items():
            lines: list[LineString] = []
            widths: list[float] = []
            for feat in group:
                if len(feat.coords) < 2:
                    continue
                rounded = [(round(lon, 7), round(lat, 7)) for lon, lat in feat.coords]
                lines.append(LineString(rounded))
                width_m = self._estimate_waterway_width_m(feat)
                if width_m > 0:
                    widths.append(width_m)

            if not lines:
                continue

            merged_lines = []
            unioned = unary_union(lines)
            if unioned.geom_type == "LineString":
                merged_lines = [unioned]
            elif unioned.geom_type == "MultiLineString":
                merged_lines = list(unioned.geoms)
            elif unioned.geom_type == "GeometryCollection":
                merged_lines = [g for g in unioned.geoms if g.geom_type == "LineString"]

            if merged_lines:
                try:
                    merged = linemerge(merged_lines)
                    if merged.geom_type == "LineString":
                        merged_lines = [merged]
                    elif merged.geom_type == "MultiLineString":
                        merged_lines = list(merged.geoms)
                except ValueError:
                    pass

            width_m = max(widths) if widths else self._estimate_waterway_width_m(group[0])
            merged_lines = _overlap_at_junctions(merged_lines, width_m)
            for line in merged_lines:
                coords = list(line.coords)
                if len(coords) < 2:
                    continue
                tags = {"waterway": waterway, "width": str(width_m)}
                merged_features.append(type(group[0])(group[0].osm_id, tags, coords))

        return merged_features

    def _build_sea_polygons_from_coastline(
        self, coastline_features: list[WaterFeature]
    ) -> list[Polygon]:
        """
        Build filled ocean polygons from OSM coastline segments.

        Algorithm:
        1. Clip coastline segments to bounding box
        2. Combine coastlines with bbox boundary to form closed rings
        3. Use polygonize to create filled polygons from the line network
        4. Select ocean polygons (touching bbox, lowest elevation)
        5. Preserve island holes (interior rings = land areas inside ocean)

        Returns:
            List of Shapely Polygon objects in geographic coordinates.
            Interior rings represent islands that must be cut out of the ocean mesh.
        """
        if not coastline_features:
            return []

        print(f"    Building ocean polygons from {len(coastline_features)} coastline segments...")

        # Create bounding box polygon
        bbox_poly = box(self.lon_min, self.lat_min, self.lon_max, self.lat_max)

        # Clip all coastline segments to the bounding box
        lines = []
        clipped_segments = 0
        for feat in coastline_features:
            if len(feat.coords) < 2:
                continue
            line = LineString(feat.coords)
            clipped = line.intersection(bbox_poly)
            if clipped.is_empty:
                continue
            geoms = (
                [clipped]
                if clipped.geom_type == "LineString"
                else list(clipped.geoms)
                if clipped.geom_type == "MultiLineString"
                else []
            )
            lines.extend(geoms)
            clipped_segments += 1

        print(f"    Clipped {clipped_segments} coastline segments to bbox")

        if not lines:
            print("    \u26a0 No coastline segments within bounding box")
            return []

        # Merge all coastline segments into a single geometry (chain or multi-line)
        print("    Merging coastlines and splitting bbox into land/sea parts...")
        from shapely.ops import split as shapely_split

        coastline_merged = linemerge(lines)

        # Stitch all parts into a single continuous chain, bridging gaps (e.g. harbors)
        # by connecting the nearest free endpoints with straight lines.
        def _stitch_parts(merged: LineString | MultiLineString) -> LineString:
            if merged.geom_type == "LineString":
                return merged
            parts_list = list(merged.geoms)
            print(f"    MultiLineString: {len(parts_list)} parts — stitching gaps")
            ordered = list(parts_list[0].coords)
            used = {0}
            for _ in range(len(parts_list) - 1):
                cur_end = ordered[-1]
                best_idx: int | None = None
                best_dist, best_rev = float("inf"), False
                for i, part in enumerate(parts_list):
                    if i in used:
                        continue
                    cs = list(part.coords)
                    d_s = (cs[0][0] - cur_end[0]) ** 2 + (cs[0][1] - cur_end[1]) ** 2
                    d_e = (cs[-1][0] - cur_end[0]) ** 2 + (cs[-1][1] - cur_end[1]) ** 2
                    if d_s <= d_e:
                        d, rev = d_s, False
                    else:
                        d, rev = d_e, True
                    if d < best_dist:
                        best_dist, best_idx, best_rev = d, i, rev
                if best_idx is None:
                    break
                cs = list(parts_list[best_idx].coords)
                if best_rev:
                    cs = list(reversed(cs))
                ordered.extend(cs)  # straight-line bridge + next segment
                used.add(best_idx)
            return LineString(ordered)

        coastline_main = _stitch_parts(coastline_merged)
        print(f"    Stitched coastline: {len(list(coastline_main.coords))} coords")

        # shapely_split requires the splitter to CROSS the polygon boundary —
        # endpoints touching the boundary are treated as interior-only.
        # Extend both endpoints well beyond the bbox so the line enters/exits
        # cleanly. 1e-4° (~11m) was too short for coastlines that approach the
        # bbox at a shallow angle or touch a corner; 0.01° (~1km, scales with
        # cos(lat) for lon) reliably crosses for any realistic bbox.
        def _extend_line(line: LineString, extension: float = 0.01) -> LineString:
            coords = list(line.coords)
            if len(coords) < 2:
                return line
            dx0, dy0 = coords[0][0] - coords[1][0], coords[0][1] - coords[1][1]
            d0 = (dx0**2 + dy0**2) ** 0.5
            if d0 > 0:
                coords[0] = (
                    coords[0][0] + dx0 / d0 * extension,
                    coords[0][1] + dy0 / d0 * extension,
                )
            dx1, dy1 = coords[-1][0] - coords[-2][0], coords[-1][1] - coords[-2][1]
            d1 = (dx1**2 + dy1**2) ** 0.5
            if d1 > 0:
                coords[-1] = (
                    coords[-1][0] + dx1 / d1 * extension,
                    coords[-1][1] + dy1 / d1 * extension,
                )
            return LineString(coords)

        coastline_extended = _extend_line(coastline_main, extension=0.01)

        try:
            parts = shapely_split(bbox_poly, coastline_extended)
            polygons = list(parts.geoms) if not parts.is_empty else []
        except Exception as e:
            print(f"    \u26a0 Split failed ({e}), falling back to polygonize")
            merged_net = unary_union([*lines, bbox_poly.boundary])
            polygons = list(polygonize(merged_net))

        print(f"    Split produced {len(polygons)} polygon(s)")

        if not polygons:
            print("    \u26a0 No polygons produced from coastline split")
            return []

        # Select the sea polygon: lowest elevation representative point
        scored = []
        for poly in polygons:
            try:
                point = poly.representative_point()
                z_mm = float(self._sample_elevation(point.y, point.x))
                scored.append((z_mm, poly))
            except Exception as e:
                print(f"    \u26a0 Warning: Failed to score polygon: {e}")
                continue

        if not scored:
            print("    \u26a0 Failed to score any polygons")
            return []

        scored.sort(key=lambda item: item[0])  # Lowest elevation first
        min_z = scored[0][0]
        max_z = scored[-1][0]

        print(f"    Elevation range: {min_z:.2f}mm (lowest) to {max_z:.2f}mm (highest)")

        # Ocean = polygons at or near minimum (sea level) elevation.
        # Use a tight threshold: only select polygons clearly at sea level.
        if (max_z - min_z) < 0.5:
            # No clear contrast — all parts at similar elevation, pick the lowest
            selected = [scored[0][1]]
            print("    Low elevation contrast — selecting lowest polygon as ocean")
        else:
            threshold = min_z + (max_z - min_z) * 0.15
            selected = [poly for z_mm, poly in scored if z_mm <= threshold]
            print(
                f"    Selected {len(selected)} ocean polygon(s) below threshold {threshold:.2f}mm"
            )

        # Adjacent split parts are all sea: merge them before they are meshed.
        # Two slabs built from polygons that share a boundary weld into one
        # non-manifold surface (every shared edge carrying four triangles),
        # which is exactly what a coastline cutting the bbox into strips gives.
        # Snap first — parts split by the same coastline can end up a float hair
        # apart, which unary_union leaves alone but the mesh welds anyway. 1e-7
        # deg is ~1cm on the ground, ~0.0003mm in the model.
        snapped = [shapely.set_precision(poly, 1e-7) for poly in selected]
        merged = unary_union([poly for poly in snapped if not poly.is_empty])
        parts = [
            geom
            for geom in getattr(merged, "geoms", [merged])
            if isinstance(geom, Polygon) and not geom.is_empty
        ]

        # Two bays can still meet at a single point — a union rightly keeps
        # those apart (they share no area), but the two slabs would then share
        # the vertical edge at that point and pinch the surface. Pull only the
        # offenders back by a hair. 1e-6 deg is ~10cm on the ground and ~0.003mm
        # in the model: invisible next to a 0.4mm nozzle, an order of magnitude
        # above the weld tolerance. Mitred joins keep the shrink from spraying
        # micro-segments along the coast, which would mesh into slivers.
        SEPARATION_DEG = 1e-6
        selected = []
        for index, part in enumerate(parts):
            touches_another = any(
                other_index != index and part.touches(other)
                for other_index, other in enumerate(parts)
            )
            if not touches_another:
                selected.append(part)
                continue
            shrunk = part.buffer(-SEPARATION_DEG, join_style="mitre").simplify(SEPARATION_DEG / 2)
            if shrunk.is_empty:
                shrunk = part  # a sliver this thin has nothing left to shrink
            selected.extend(
                geom
                for geom in getattr(shrunk, "geoms", [shrunk])
                if isinstance(geom, Polygon) and not geom.is_empty
            )

        # Return full Shapely polygons (with interior holes = islands preserved)
        total_area = 0.0
        for i, poly in enumerate(selected):
            total_area += poly.area
            if len(list(poly.interiors)) > 0:
                print(
                    f"    \u2713 Ocean polygon {i} has {len(list(poly.interiors))} island hole(s) — will be cut out"
                )

        print(
            f"    \u2713 Created {len(selected)} ocean surface polygon(s), total area: {total_area:.6f} deg^2"
        )

        return selected

    def _create_flat_ocean_surface(
        self, feature: WaterFeature
    ) -> tuple[np.ndarray, np.ndarray, int] | None:
        """
        Create perfectly flat ocean/sea surface at constant Z height.

        Unlike terrain-following surfaces, ocean surfaces are always flat and smooth.
        No terrain elevation sampling is used - all vertices at identical height.

        Args:
            feature: WaterFeature object tagged with natural=sea

        Returns:
            Tuple of (vertices, faces, top_vertex_count) or None if error
        """
        if not feature.is_closed():
            return None

        try:
            # Convert polygon to mesh coordinates (with bounding box clipping)
            clipped_geo_coords = self.geo_converter.clip_polygon_to_bbox(feature.coords)

            # Skip if polygon is completely outside bounding box
            if not clipped_geo_coords or len(clipped_geo_coords) < 3:
                return None

            # Convert geographic coordinates to mesh coordinates (mm)
            mesh_coords = [self._geo_to_model(lon, lat) for lon, lat in clipped_geo_coords]

            if not mesh_coords or len(mesh_coords) < 3:
                return None

            # Create shapely polygon for processing
            polygon = Polygon(mesh_coords)

            if not polygon.is_valid:
                from shapely import make_valid

                polygon = make_valid(polygon)
                if not polygon.is_valid or polygon.is_empty:
                    return None

            # Ocean surface at CONSTANT Z height - no terrain sampling
            # This creates a perfectly smooth, flat surface
            water_surface_z = self.base_thickness_mm + self.water_offset_mm
            water_bottom_z = water_surface_z - self.water_thickness_mm

            # Generate flat mesh using water polygon method
            result = self._create_water_polygon_mesh(polygon, water_surface_z, water_bottom_z)

            if result is None:
                return None

            # Same shape as _create_terrain_following_surface, so the caller can
            # take either branch without re-checking which one it got.
            vertices, faces, top_vertex_count = result
            return np.array(vertices), faces, top_vertex_count

        except Exception as e:
            print(f"    ⚠ Warning: Failed to create flat ocean surface: {e}")
            return None

    def _create_terrain_following_surface(
        self, feature: WaterFeature
    ) -> tuple[np.ndarray, np.ndarray, int] | None:
        """
        Create 3D surface that follows terrain within water polygon.

        Args:
            feature: WaterFeature object

        Returns:
            Tuple of (vertices, faces) or None if error
        """
        if not feature.is_closed():
            # Skip non-closed features (rivers need different handling)
            return None

        try:
            # Convert polygon to mesh coordinates (with bounding box clipping)
            # This returns the clipped coordinates in mesh space
            clipped_geo_coords = self.geo_converter.clip_polygon_to_bbox(feature.coords)

            # Skip if polygon is completely outside bounding box
            if not clipped_geo_coords or len(clipped_geo_coords) < 3:
                return None

            # Convert the clipped geographic coordinates to mesh coordinates
            # Note: No reversal needed - OSM polygon orientation combined with Y/Z swap
            # during OBJ export produces correct upward-facing normals
            mesh_coords = [self._geo_to_model(lon, lat) for lon, lat in clipped_geo_coords]

            # Skip if conversion failed
            if not mesh_coords or len(mesh_coords) < 3:
                return None

            # Create shapely polygon for triangulation and flat surface calculation
            polygon = Polygon(mesh_coords)

            if not polygon.is_valid:
                from shapely import make_valid

                polygon = make_valid(polygon)
                if not polygon.is_valid or polygon.is_empty:
                    return None

            # Clip to footprint polygon for non-rectangular shapes. The earlier
            # geo-bbox clip (clip_polygon_to_bbox) keeps the polygon inside the
            # rectangular geographic bounds; we additionally trim it to the
            # actual model outline so lakes don't spill out past hex/circle.
            if (
                self.footprint is not None
                and getattr(self.footprint, "shape", "rectangle") != "rectangle"
            ):
                try:
                    polygon = _clip_to_footprint(polygon, self.footprint.polygon)
                except Exception as e:
                    print(f"  ⚠ Dropped one water feature: clip to the footprint failed ({e})")
                    return None
                if polygon.is_empty:
                    return None
                if polygon.geom_type == "MultiPolygon":
                    # Pick the largest piece — multiple lake pieces from one OSM
                    # feature would each need a separate mesh, which this code
                    # path does not support. The smaller crumbs are normally
                    # negligible.
                    polygon = max(polygon.geoms, key=lambda g: g.area)
                if polygon.geom_type != "Polygon" or polygon.is_empty:
                    return None
                if polygon.area < 1e-3:
                    return None

            # Flowing water (rivers, canals, riverbanks) follows terrain per-vertex.
            # Standing water (lakes, ponds, reservoirs) uses a single flat elevation.
            waterway_tag = feature.get_tag("waterway")
            is_flowing = waterway_tag is not None  # any waterway= tag = flowing water

            # --- Triangulate the (possibly non-convex) polygon ---
            # Exterior ring CCW — the top/bottom/wall windings below all assume
            # it, and OSM rings come in either orientation.
            polygon = orient(polygon, sign=1.0)

            tris = _polygon_triangles(polygon)
            if not tris:
                return None

            # Compute Z for a (x_mm, y_mm) point in mesh space.
            if is_flowing:

                def z_at(x_mm: float, y_mm: float) -> float:
                    lon, lat = self._model_to_geo(x_mm, y_mm)
                    return self._sample_elevation(lat, lon) + self.water_offset_mm
            else:
                # Flat lake surface at the lake's water level (computed from the
                # ORIGINAL pre-clip polygon coordinates so the level matches the
                # natural waterline rather than the clipped subset).
                lake_elevation = self._calculate_lake_surface_elevation(clipped_geo_coords)
                z_mm = lake_elevation + self.water_offset_mm

                def z_at(x_mm: float, y_mm: float) -> float:
                    return z_mm

            # Top surface: 3 fresh vertices per triangle (no index sharing — STL
            # output flattens vertices anyway).
            all_vertices: list[Vertex] = []
            faces: list[list[int]] = []
            for tri in tris:
                coords = _ccw(list(tri.exterior.coords[:3]))
                base = len(all_vertices)
                for x, y in coords:
                    all_vertices.append([x, y, z_at(x, y)])
                faces.append([base, base + 1, base + 2])

            num_top_tris = len(tris)

            # Bottom surface + walls (printable thickness).
            thickness_mm = max(0.0, self.water_thickness_mm)
            if thickness_mm > 0.0:
                top_vertices_count = len(all_vertices)

                # Bottom vertices (shift each top vertex down by thickness).
                bottom_start_idx = len(all_vertices)
                for v in all_vertices[:top_vertices_count]:
                    all_vertices.append([v[0], v[1], v[2] - thickness_mm])

                # Bottom triangles (reverse winding).
                for ti in range(num_top_tris):
                    b = ti * 3
                    bb = bottom_start_idx + b
                    faces.append([bb + 2, bb + 1, bb])

                # Side walls along every ring. Islands (interior rings) need
                # walls too, otherwise the triangulated cap has a hole the shell
                # never closes.
                for ring in [polygon.exterior, *polygon.interiors]:
                    ring_coords = list(ring.coords[:-1])
                    n = len(ring_coords)
                    if n < 3:
                        continue
                    wall_base = len(all_vertices)
                    for x, y in ring_coords:
                        all_vertices.append([x, y, z_at(x, y)])
                    for x, y in ring_coords:
                        all_vertices.append([x, y, z_at(x, y) - thickness_mm])
                    for i in range(n):
                        ni = (i + 1) % n
                        t_i = wall_base + i
                        t_ni = wall_base + ni
                        b_i = wall_base + n + i
                        b_ni = wall_base + n + ni
                        # Water on the left of the walk (orient() left the
                        # exterior CCW and holes CW), so the wall normal falls
                        # on the water's outside: top -> bottom -> next, never
                        # top -> next -> bottom (that faces into the water).
                        faces.append([t_i, b_ni, t_ni])
                        faces.append([t_i, b_i, b_ni])
            else:
                top_vertices_count = len(all_vertices)

            return np.array(all_vertices), np.array(faces), top_vertices_count

        except Exception as e:
            print(f"  ⚠ Dropped one water surface: {e}")
            return None

    def _create_river_surface(self, feature: WaterFeature) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Create a continuous ribbon mesh for linear water features (rivers/streams).

        Rivers follow terrain surface by sampling elevation at both left and right edges,
        ensuring they remain visible and properly aligned even when terrain slopes
        perpendicular to flow direction.
        """
        try:
            if len(feature.coords) < 2:
                return None

            # Clip polyline to bbox with margin from edges
            # Convert margin from mm to geographic degrees
            margin_lon = (RIVER_EDGE_MARGIN_MM / self.model_width_mm) * (
                self.lon_max - self.lon_min
            )
            margin_lat = (RIVER_EDGE_MARGIN_MM / self.model_height_mm) * (
                self.lat_max - self.lat_min
            )

            # Create clipping bbox with inset margin to keep rivers away from edges
            line = LineString(feature.coords)
            bbox_poly = box(
                self.lon_min + margin_lon,
                self.lat_min + margin_lat,
                self.lon_max - margin_lon,
                self.lat_max - margin_lat,
            )
            clipped = line.intersection(bbox_poly)
            if clipped.is_empty:
                return None

            if clipped.geom_type == "LineString":
                lines = [clipped]
            elif clipped.geom_type == "MultiLineString":
                lines = list(clipped.geoms)
            else:
                return None

            all_vertices: list[Vertex] = []
            all_faces: list[list[int]] = []
            vertex_offset = 0

            width_m = self._estimate_waterway_width_m(feature)
            width_mm = compute_waterway_width_mm(
                width_m * self.scale,
                min(self.model_width_mm, self.model_height_mm),
            )

            # Densify step: sample terrain at least once per terrain pixel of
            # the FINAL (upsampled) grid so the ribbon top follows the ground
            # and the river stays continuous instead of bridging — and getting
            # buried — over sparse OSM nodes. Use the real array shape so the
            # upsample factor is accounted for.
            try:
                rows, cols = self.terrain_elevation.shape
                px_mm = min(
                    self.model_width_mm / max(cols, 1),
                    self.model_height_mm / max(rows, 1),
                )
            except Exception:
                px_mm = min(self.model_width_mm, self.model_height_mm) / max(
                    int(self.terrain_resolution), 1
                )
            densify_step_mm = max(0.15, min(px_mm, 1.0))

            def elevation_sampler(x_mm: float, y_mm: float) -> float:
                """Sample smoothed terrain elevation at given model coordinates."""
                lon, lat = self._model_to_geo(x_mm, y_mm)
                return self._sample_elevation(lat, lon)

            for line_part in lines:
                coords = list(line_part.coords)
                if len(coords) < 2:
                    continue

                centerline_pts = [self._geo_to_model(lon, lat) for lon, lat in coords]

                # Clip to footprint polygon (mesh-space) for non-rectangular shapes.
                # A river that crosses the hex/circle boundary is split into the
                # pieces that lie inside the footprint; pieces outside are dropped.
                centerline_segments = [centerline_pts]
                if (
                    self.footprint is not None
                    and getattr(self.footprint, "shape", "rectangle") != "rectangle"
                ):
                    try:
                        mesh_line = LineString(centerline_pts)
                        # Inset by the ribbon's own half-width as well as the
                        # cosmetic margin: what has to stay inside the model is
                        # the ribbon, not the centerline it is swept along. A
                        # river running parallel to the outline used to be
                        # clipped at 1.5mm and then widened to half its width on
                        # each side, so anything wider than 3mm hung over the
                        # edge - and the exporter answered that by deleting the
                        # triangles that stuck out, which opened the shell.
                        inset = self.footprint.polygon.buffer(
                            -max(RIVER_EDGE_MARGIN_MM, width_mm / 2.0)
                        )
                        # Snapped like the polygon clips above, and for the same
                        # reason: a centerline vertex grazing the inset boundary
                        # comes back doubled, and the ribbon then builds two
                        # cross-sections at one place. Where the crossing turns
                        # sharply the two get different miters and survive, but a
                        # shallow graze leaves them parallel, so they weld into
                        # each other and take the ribbon's walls with them.
                        clip_geo = None if inset.is_empty else _clip_to_footprint(mesh_line, inset)
                    except Exception:
                        clip_geo = None

                    # An empty inset means the ribbon is wider than the model can
                    # hold. Dropping the river is the only answer that keeps it
                    # closed; widening it back over the edge is what broke it.
                    if clip_geo is None or clip_geo.is_empty:
                        continue
                    if clip_geo.geom_type == "LineString":
                        centerline_segments = [list(clip_geo.coords)]
                    elif clip_geo.geom_type == "MultiLineString":
                        centerline_segments = [list(g.coords) for g in clip_geo.geoms]
                    else:
                        continue

                for seg in centerline_segments:
                    if len(seg) < 2:
                        continue
                    centerline = densify_polyline(np.array(seg), densify_step_mm)

                    # Create ribbon geometry with terrain-following surfaces
                    # Both left and right edges sample elevation at their positions
                    vertices, faces = create_ribbon_geometry(
                        centerline,
                        width_mm,
                        self.water_offset_mm,
                        elevation_sampler,
                        thickness_mm=self.water_thickness_mm,
                    )

                    adjusted_faces = faces + vertex_offset
                    all_vertices.extend(vertices)
                    all_faces.extend(adjusted_faces)
                    vertex_offset += len(vertices)

            if not all_vertices:
                return None

            return np.array(all_vertices), np.array(all_faces)

        except Exception as e:
            print(f"  ⚠ Water layer dropped while assembling the mesh: {e}")
            return None

    def _ocean_from_terrain_or_none(self, reason: str) -> Mesh | None:
        """Fall back to sea-level detection from the DEM, but only at LOD 1.

        LOD 1 means "oceans and seas"; OSM coastlines are a shortcut to a nicer
        shoreline, not the source of truth, so an empty or fully-filtered-out OSM
        answer must still produce the sea the DEM shows. An unreachable Overpass
        never reaches here: the client raises before this function is called.
        Higher LODs describe inland water that the terrain cannot infer, so they
        return nothing rather than invent a coastline.
        """
        print(f"  \u26a0 {reason}.")
        if self.lod_level != 1:
            return None
        print("  Falling back to terrain-based ocean detection...")
        return self._generate_ocean_surface()

    def generate_water_mesh(self) -> Mesh | None:
        """
        Generate complete water layer mesh from OSM coastline data and water features.

        LOD 1: Oceans/seas from OSM coastlines (fills ocean polygons from coastline geometry)
        LOD 2-3: Large water bodies + oceans
        LOD 4-10: Includes rivers, streams, canals

        Returns:
            Mesh object or None if no water features
        """
        # Load elevation data
        if not self._load_elevation_data():
            return None

        # Fetch water features from OSM (including coastlines at all LOD levels)
        print(f"  Fetching OSM water data (LOD {self.lod_level})...")
        osm_client = OverpassClient(cache_dir=self.osm_cache_dir, use_cache=self.use_cache)

        raw_data = osm_client.query_water_features(
            self.lat_min, self.lon_min, self.lat_max, self.lon_max
        )

        if not raw_data:
            return self._ocean_from_terrain_or_none("No OSM water data returned")

        # Extract features
        all_features = extract_water_features(raw_data)
        print(f"  OSM water features extracted: {len(all_features)}")

        # Filter by LOD using physical size thresholds
        features = filter_by_lod(all_features, self.water_config)
        print(f"  OSM water features after LOD {self.lod_level} filtering: {len(features)}")

        if not features:
            return self._ocean_from_terrain_or_none(
                f"No water features survived LOD {self.lod_level} filtering"
            )

        # Separate features by type
        coastline_features = [f for f in features if f.get_tag("natural") == "coastline"]
        linear_features = [f for f in features if self._is_linear_water_feature(f)]
        polygon_features = [
            f for f in features if f.is_closed() and f.get_tag("natural") != "coastline"
        ]
        print(
            "  Water feature breakdown: "
            f"coastline={len(coastline_features)}, "
            f"linear={len(linear_features)}, "
            f"polygon={len(polygon_features)}"
        )

        # Validation: Confirm filtering behavior
        if self.lod_level == 1:
            if linear_features:
                print(
                    f"  ⚠ WARNING: {len(linear_features)} linear features found at LOD {self.lod_level} (should be 0)"
                )
            else:
                print(f"  ✓ LOD {self.lod_level}: Oceans/seas only (no inland water)")
        elif self.lod_level == 2:
            if linear_features:
                print(
                    f"  ⚠ WARNING: {len(linear_features)} linear features found at LOD {self.lod_level} (should be 0)"
                )
            else:
                print(f"  ✓ LOD {self.lod_level}: Large lakes + oceans/seas (no rivers/streams)")
        elif self.lod_level >= 3:
            print(f"  ✓ LOD {self.lod_level}: {len(linear_features)} linear waterways included")

        # Build ocean polygons from coastline data
        # This is the CRITICAL step: convert coastline LINES into filled ocean POLYGONS
        sea_polygons = []  # List[Shapely Polygon] — populated below if coastlines found
        if coastline_features:
            print(f"  Processing {len(coastline_features)} coastline segments...")
            sea_polygons = self._build_sea_polygons_from_coastline(coastline_features)
            print(f"  ✓ Created {len(sea_polygons)} ocean polygon(s) from coastlines")

            # Validate that we got actual polygons
            if not sea_polygons:
                print("  ⚠ WARNING: Coastlines found but no ocean polygons created")
                print("  This may indicate coastline topology issues")
            else:
                # sea_polygons are Shapely Polygon objects (with island holes).
                # Keep them separate so holes survive mesh generation.
                print(f"  ✓ Added {len(sea_polygons)} ocean surface(s) to mesh generation queue")
        elif self.lod_level == 1:
            # No coastline ways here, but a `natural=sea` polygon may still cover
            # the bay. Keep going and let the terrain fallback at the end catch
            # the case where nothing at all gets meshed.
            print("  ⚠ No coastlines found in OSM data")

        # Merge linear waterways to ensure continuity
        linear_features = self._merge_linear_waterways(linear_features)
        if linear_features:
            print(f"  Linear water features after merge: {len(linear_features)}")

        # Generate surfaces
        all_vertices = []
        all_faces = []
        vertex_offset = 0

        # Track feature info for verification
        feature_info = []
        ocean_feature_count = 0
        ocean_face_count = 0
        lake_feature_count = 0
        lake_face_count = 0

        # --- Process sea polygons (Shapely objects, holes = islands preserved) ---
        # Ocean is at sea level (0m normalized = base_thickness_mm).
        # Do NOT add water_offset here — the offset lifts water above terrain,
        # which is correct for inland features but would flood coastal land for ocean.
        water_surface_z = self.base_thickness_mm
        water_bottom_z = water_surface_z - self.water_thickness_mm

        if coastline_features and sea_polygons:
            print(
                f"  Generating meshes for {len(sea_polygons)} ocean polygon(s) (with island holes)..."
            )
            for i, geo_poly in enumerate(sea_polygons):
                try:
                    # Convert geographic polygon to model space, preserving holes
                    ext_mm = [
                        self._geo_to_model(lon, lat) for lon, lat in geo_poly.exterior.coords[:-1]
                    ]
                    holes_mm = [
                        [self._geo_to_model(lon, lat) for lon, lat in ring.coords[:-1]]
                        for ring in geo_poly.interiors
                        if len(ring.coords) >= 4
                    ]
                    if len(ext_mm) < 3:
                        continue

                    model_poly = Polygon(ext_mm, holes_mm)
                    if not model_poly.is_valid:
                        from shapely import make_valid

                        model_poly = make_valid(model_poly)
                    if not model_poly.is_valid or model_poly.is_empty:
                        continue

                    coastline_result = self._create_water_polygon_mesh(
                        model_poly, water_surface_z, water_bottom_z
                    )
                    if coastline_result is None:
                        print(f"    ⚠ Failed to create mesh for ocean polygon {i}")
                        continue

                    shore_vertices, shore_faces, _ = coastline_result

                    ocean_feature_count += 1
                    ocean_face_count += len(shore_faces)
                    print(
                        f"    ✓ Ocean surface 'ocean_{i}': {len(shore_vertices)} vertices, "
                        f"{len(shore_faces)} faces, {len(holes_mm)} island hole(s)"
                    )

                    adjusted_faces = np.array(shore_faces) + vertex_offset
                    all_vertices.extend(shore_vertices)
                    all_faces.extend(adjusted_faces.tolist())
                    vertex_offset += len(shore_vertices)

                except Exception as e:
                    print(f"    ⚠ Warning: Failed to process ocean polygon {i}: {e}")

        print(f"  Generating meshes for {len(polygon_features)} polygon feature(s)...")
        for feature in polygon_features:
            # Use different methods for ocean vs other water features
            # Oceans: flat surface at constant Z (smooth, no terrain variation)
            # Lakes/ponds: flat surface at average terrain elevation (realistic flat lakes)
            is_ocean = feature.get_tag("natural") == "sea"

            if is_ocean:
                result = self._create_flat_ocean_surface(feature)
            else:
                result = self._create_terrain_following_surface(feature)

            if result is None:
                feature_type = feature.get_tag("natural") or "unknown"
                print(f"    ⚠ Warning: Failed to create surface for {feature_type} feature")
                continue

            vertices, faces, top_vertex_count = result

            # Track ocean and lake features specifically
            if is_ocean:
                ocean_feature_count += 1
                ocean_face_count += len(faces)
                feature_name = feature.get_tag("name") or f"ocean_{ocean_feature_count}"
                print(
                    f"    ✓ Ocean surface '{feature_name}': {len(vertices)} vertices, {len(faces)} faces"
                )
                # Validation: Assert faces exist
                if len(faces) == 0:
                    print("    ⚠ ERROR: Ocean surface has ZERO faces - this is a BUG!")
                    continue
            else:
                lake_feature_count += 1
                lake_face_count += len(faces)
                feature_name = feature.get_tag("name") or f"lake_{lake_feature_count}"
                # Calculate approximate area
                if len(faces) > 0:
                    vertices_array = np.array(vertices)
                    area_mm2 = 0.0
                    for face_idx in range(len(faces)):
                        v0, v1, v2 = vertices_array[faces[face_idx]]
                        edge1 = v1 - v0
                        edge2 = v2 - v0
                        area_mm2 += 0.5 * abs(float(edge1[0] * edge2[1] - edge1[1] * edge2[0]))
                    print(
                        f"    ✓ Lake '{feature_name}': {len(vertices)} vertices, {len(faces)} faces, {area_mm2:.1f}mm² (FLAT)"
                    )
                if len(faces) == 0:
                    print(f"    ⚠ Warning: Lake '{feature_name}' has ZERO faces")
                    continue

            # Store feature info for verification
            feature_info.append(
                {
                    "feature": feature,
                    "vertex_start": vertex_offset,
                    "vertex_count": len(vertices),
                    "top_vertex_count": top_vertex_count,
                    "face_count": len(faces),
                }
            )

            # Adjust face indices and add to global lists
            adjusted_faces = faces + vertex_offset
            all_vertices.extend(vertices)
            all_faces.extend(adjusted_faces)

            vertex_offset += len(vertices)

        # Validation summary for ocean and lake surfaces
        if ocean_feature_count > 0:
            print(
                f"  ✓ Ocean mesh validation: {ocean_feature_count} surface(s), {ocean_face_count} total faces"
            )
            if ocean_face_count == 0:
                print("  ⚠ CRITICAL ERROR: Ocean polygons created but have ZERO faces!")
                print("  Ocean surfaces must have area (faces), not just edges (lines)")
        elif coastline_features:
            print("  ⚠ WARNING: Coastlines found but no ocean meshes created")
            print("  Check coastline polygon building and mesh generation")

        if lake_feature_count > 0:
            print(
                f"  ✓ Lake mesh validation: {lake_feature_count} lake(s), {lake_face_count} total faces (all FLAT)"
            )

        # Add linear water features (rivers/streams)
        river_count = 0
        river_vertex_count = 0
        river_face_count = 0
        for feature in linear_features:
            river_result = self._create_river_surface(feature)
            if river_result is None:
                continue

            vertices, faces = river_result
            river_count += 1
            river_vertex_count += len(vertices)
            river_face_count += len(faces)

            # Debug output for rivers
            waterway_type = feature.get_tag("waterway") or "unknown"
            feature_name = feature.get_tag("name") or f"{waterway_type}_{river_count}"
            width_m = self._estimate_waterway_width_m(feature)
            width_mm = compute_waterway_width_mm(
                width_m * self.scale,
                min(self.model_width_mm, self.model_height_mm),
            )

            # Calculate Z range of river vertices for verification
            vertices_arr = np.array(vertices)
            z_min = vertices_arr[:, 2].min()
            z_max = vertices_arr[:, 2].max()

            print(
                f"    ✓ River '{feature_name}' ({waterway_type}): {len(vertices)} vertices, {len(faces)} faces, "
                f"width={width_mm:.2f}mm, Z={z_min:.2f}-{z_max:.2f}mm (offset=+{self.water_offset_mm:.2f}mm)"
            )

            adjusted_faces = faces + vertex_offset
            all_vertices.extend(vertices)
            all_faces.extend(adjusted_faces)
            vertex_offset += len(vertices)

        if river_count > 0:
            print(
                f"  ✓ River mesh validation: {river_count} river(s), {river_face_count} total faces"
            )
        elif linear_features:
            print(
                f"  ⚠ WARNING: {len(linear_features)} linear features found but NO river meshes created"
            )
            print("  Check river mesh generation and buffering")

        if not all_vertices:
            return self._ocean_from_terrain_or_none("No water mesh vertices generated")

        if not all_faces:
            print("  \u26a0 ERROR: Water vertices exist but NO FACES generated!")
            print("  Water surfaces must have filled faces, not just edge vertices")
            return self._ocean_from_terrain_or_none("Water vertices carried no faces")

        vertices_array = np.array(all_vertices)

        # Verify water is above terrain surface
        self._verify_water_surface_heights(feature_info, vertices_array)

        # Create mesh in numpy-stl compatible format
        faces_array = np.array(all_faces)

        print(
            f"  \u2713 Final water mesh: {len(vertices_array)} vertices, {len(faces_array)} faces"
        )
        # Validation: Calculate mesh area
        if len(faces_array) > 0:
            total_area_mm2 = 0.0
            for face in faces_array:
                v0, v1, v2 = vertices_array[face]
                # Triangle area = 0.5 * |cross product|
                edge1 = v1 - v0
                edge2 = v2 - v0
                # 2D area in XY plane
                area = 0.5 * abs(float(edge1[0] * edge2[1] - edge1[1] * edge2[0]))
                total_area_mm2 += area
            print(f"  Total water surface area: {total_area_mm2:.2f} mm^2")
            if total_area_mm2 < 1.0:
                print(
                    "  \u26a0 WARNING: Water surface area is very small - may be degenerate geometry"
                )

        # Build structured array with face triangles
        mesh_data = np.zeros(len(faces_array), dtype=Mesh.dtype)
        for i, face in enumerate(faces_array):
            # Get three vertices of this triangle
            v0, v1, v2 = vertices_array[face]
            mesh_data["vectors"][i] = np.array([v0, v1, v2])
            # Calculate face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm_length = np.linalg.norm(normal)
            if norm_length > 0:
                normal = normal / norm_length
            mesh_data["normals"][i] = normal

        return Mesh(mesh_data)
