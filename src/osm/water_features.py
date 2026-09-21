# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Water feature extraction and LOD (Level of Detail) filtering.

Processes OSM water data and filters based on physical size thresholds.
"""

import numpy as np
from shapely.geometry import Polygon


class WaterFilterConfig:
    """Physical size-based water filtering configuration."""

    def __init__(
        self, bbox_area_km2: float, model_width_mm: float, bbox_width_km: float, lod_level: int
    ):
        """
        Initialize water filter configuration.

        Args:
            bbox_area_km2: Bounding box area in square kilometers
            model_width_mm: Model width in millimeters
            bbox_width_km: Bounding box width in kilometers
            lod_level: Level of detail (0-5)
        """
        self.bbox_area_km2 = bbox_area_km2
        self.model_width_mm = model_width_mm
        self.bbox_width_km = bbox_width_km
        self.lod_level = lod_level

        # Metres of ground per millimetre of model. Used to report scale; the
        # printable floor lives in `MIN_WATERWAY_WIDTH_MM`, which widens a thin
        # ribbon rather than dropping it, so a river network keeps its shape.
        self.meters_per_mm = (bbox_width_km * 1000) / model_width_mm if model_width_mm > 0 else 1.0

        # Level-specific thresholds
        self._calculate_thresholds()

    def _calculate_thresholds(self):
        """Calculate thresholds based on LOD level."""
        if self.lod_level == 0:
            # Disabled
            self.allow_linear = False
            self.min_lake_area_m2 = float("inf")
            self.min_river_width_m = float("inf")
            self.min_river_length_m = float("inf")
        elif self.lod_level == 1:
            # Oceans and seas only
            self.allow_linear = False
            self.min_lake_area_m2 = float("inf")  # No lakes
            self.min_river_width_m = float("inf")
            self.min_river_length_m = float("inf")
        elif self.lod_level == 2:
            # Level 1 + Large lakes
            self.allow_linear = False
            # Use min() to allow smaller lakes in small bboxes, cap at 5km² for large bboxes
            self.min_lake_area_m2 = min(
                5_000_000, 0.005 * self.bbox_area_km2 * 1_000_000
            )  # 5 km² OR 0.5% of bbox (whichever is smaller)
            self.min_river_width_m = float("inf")  # No rivers yet
            self.min_river_length_m = float("inf")
        elif self.lod_level == 3:
            # Level 2 + Major rivers
            self.allow_linear = True
            self.min_lake_area_m2 = min(
                5_000_000, 0.005 * self.bbox_area_km2 * 1_000_000
            )  # Same as level 2
            self.min_river_width_m = 40.0  # Major rivers: 40m+ (e.g., large rivers)
            self.min_river_length_m = 3000.0  # 3 km
        elif self.lod_level == 4:
            # Level 3 + Rivers and streams
            self.allow_linear = True
            self.min_lake_area_m2 = min(
                5_000_000, 0.005 * self.bbox_area_km2 * 1_000_000
            )  # Same as level 2
            self.min_river_width_m = 20.0  # Rivers: 20m+ (e.g., medium rivers)
            self.min_river_length_m = 0.0  # No length restriction
        elif self.lod_level == 5:
            # Level 4 + All waterways
            self.allow_linear = True
            self.min_lake_area_m2 = min(
                5_000_000, 0.005 * self.bbox_area_km2 * 1_000_000
            )  # Same as level 2
            self.min_river_width_m = 3.0  # Small streams: 3m+ (includes creeks, brooks)
            self.min_river_length_m = 500.0  # 500m minimum for small streams
        else:
            raise ValueError(f"Invalid LOD level: {self.lod_level}, must be 0-5")


class WaterFeature:
    """Represents a water feature from OSM."""

    def __init__(self, osm_id: int, tags: dict, coords: list[tuple[float, float]]):
        """
        Initialize water feature.

        Args:
            osm_id: OSM element ID
            tags: Dictionary of OSM tags
            coords: List of (lon, lat) tuples defining the polygon
        """
        self.osm_id = osm_id
        self.tags = tags
        self.coords = coords
        self.area_m2 = None
        self.length_m = None

        # Calculate area if polygon is closed
        if len(coords) > 2 and coords[0] == coords[-1]:
            self._calculate_area()
        else:
            # Calculate length for linear features
            self._calculate_length()

    def _calculate_area(self):
        """Calculate approximate area in square meters using spherical approximation."""
        if len(self.coords) < 3:
            self.area_m2 = 0.0
            return

        try:
            # Use shapely for area calculation
            # Convert to approximate local projection (meters)
            lat_avg = np.mean([lat for lon, lat in self.coords])
            meters_per_deg_lat = 111320
            meters_per_deg_lon = 111320 * np.cos(np.radians(lat_avg))

            # Convert coordinates to meters
            coords_m = [
                (
                    (lon - self.coords[0][0]) * meters_per_deg_lon,
                    (lat - self.coords[0][1]) * meters_per_deg_lat,
                )
                for lon, lat in self.coords
            ]

            # Calculate area using shapely
            from shapely import make_valid

            polygon = Polygon(coords_m)
            if not polygon.is_valid:
                polygon = make_valid(polygon)
            self.area_m2 = polygon.area

        except Exception:
            self.area_m2 = 0.0

    def _calculate_length(self):
        """Calculate approximate length in meters for linear features."""
        if len(self.coords) < 2:
            self.length_m = 0.0
            return

        try:
            # Convert to approximate local projection (meters)
            lat_avg = np.mean([lat for lon, lat in self.coords])
            meters_per_deg_lat = 111320
            meters_per_deg_lon = 111320 * np.cos(np.radians(lat_avg))

            # Calculate cumulative length
            total_length = 0.0
            for i in range(len(self.coords) - 1):
                lon1, lat1 = self.coords[i]
                lon2, lat2 = self.coords[i + 1]

                dx = (lon2 - lon1) * meters_per_deg_lon
                dy = (lat2 - lat1) * meters_per_deg_lat

                segment_length = np.sqrt(dx**2 + dy**2)
                total_length += segment_length

            self.length_m = total_length

        except Exception:
            self.length_m = 0.0

    def get_tag(self, key: str, default: str | None = None) -> str:
        """Get OSM tag value."""
        return self.tags.get(key, default)

    def is_closed(self) -> bool:
        """Check if polygon is closed."""
        return len(self.coords) > 2 and self.coords[0] == self.coords[-1]

    @property
    def name(self) -> str:
        """Get feature name from tags or generate a description."""
        if "name" in self.tags:
            return self.tags["name"]

        # Generate descriptive name
        if "natural" in self.tags:
            natural_type = self.tags["natural"]
            if "water" in self.tags:
                water_type = self.tags["water"]
                return f"{natural_type}:{water_type}"
            return natural_type

        if "waterway" in self.tags:
            return self.tags["waterway"]

        return f"water_{self.osm_id}"

    @property
    def feature_type(self) -> str:
        """Get feature type for display."""
        if "natural" in self.tags:
            if "water" in self.tags:
                return self.tags["water"]
            return self.tags["natural"]

        if "waterway" in self.tags:
            return self.tags["waterway"]

        return "water"

    @property
    def area(self) -> float:
        """Get area in square meters."""
        return self.area_m2 if self.area_m2 is not None else 0.0


def extract_water_features(osm_data: dict) -> list[WaterFeature]:
    """
    Extract water features from OSM Overpass API response.

    Args:
        osm_data: JSON response from Overpass API

    Returns:
        List of WaterFeature objects
    """
    if not osm_data or "elements" not in osm_data:
        return []

    # Build node lookup table (for queries using 'out body')
    nodes = {}
    for element in osm_data["elements"]:
        if element["type"] == "node":
            nodes[element["id"]] = (element["lon"], element["lat"])

    # Extract ways (polygons)
    features = []
    for element in osm_data["elements"]:
        if element["type"] == "way":
            tags = element.get("tags", {})

            # Check if it's a water feature
            if "natural" not in tags and "waterway" not in tags and "water" not in tags:
                continue

            coords = []

            # Method 1: 'out geom' format - geometry is inline
            if "geometry" in element:
                for point in element["geometry"]:
                    if point is not None:
                        coords.append((point["lon"], point["lat"]))

            # Method 2: 'out body' format - need to lookup nodes
            elif "nodes" in element:
                for node_id in element["nodes"]:
                    if node_id in nodes:
                        coords.append(nodes[node_id])
            else:
                # No coordinate data available
                continue

            if len(coords) < 3:
                continue

            feature = WaterFeature(element["id"], tags, coords)
            features.append(feature)

    # Handle relations (multipolygons) for complex water bodies
    for element in osm_data["elements"]:
        if element["type"] == "relation":
            tags = element.get("tags", {})

            # Check if it's a water feature
            if "natural" not in tags and "waterway" not in tags and "water" not in tags:
                continue

            # Extract outer way coordinates from relation members
            if "members" not in element:
                continue

            # Find outer ways (outer boundary of the water body)
            outer_coords = []
            for member in element["members"]:
                if member.get("role") == "outer" and member.get("type") == "way":
                    # Get way reference
                    way_ref = member.get("ref")
                    if not way_ref:
                        continue

                    # Find the corresponding way in elements
                    for way_elem in osm_data["elements"]:
                        if way_elem.get("type") == "way" and way_elem.get("id") == way_ref:
                            # Extract coordinates from this way
                            way_coords = []

                            # Method 1: geometry inline
                            if "geometry" in way_elem:
                                for point in way_elem["geometry"]:
                                    way_coords.append((point["lon"], point["lat"]))

                            # Method 2: node references
                            elif "nodes" in way_elem:
                                for node_id in way_elem["nodes"]:
                                    if node_id in nodes:
                                        way_coords.append(nodes[node_id])

                            # Add to outer coordinates (remove last point if duplicate to avoid gaps)
                            if way_coords:
                                if outer_coords and way_coords[0] == outer_coords[-1]:
                                    outer_coords.extend(way_coords[1:])
                                else:
                                    outer_coords.extend(way_coords)
                            break

            # Create feature from outer boundary
            if len(outer_coords) >= 3:
                # Ensure polygon is closed (first point == last point)
                if outer_coords[0] != outer_coords[-1]:
                    outer_coords.append(outer_coords[0])

                feature = WaterFeature(element["id"], tags, outer_coords)
                features.append(feature)

    return features


def filter_by_lod(features: list[WaterFeature], config: WaterFilterConfig) -> list[WaterFeature]:
    """
    Filter water features by level of detail using physical size thresholds.

    Args:
        features: List of water features
        config: Water filter configuration with physical thresholds

    Returns:
        Filtered list of features

    Filtering rules, as `LODConfig._calculate_thresholds` sets them:
        Level 1: oceans and seas only, no inland water
        Level 2: level 1 + lakes >= min(5 km², 0.5% of the bbox area)
        Level 3: level 2 + rivers >= 40 m wide and >= 3 km long
        Level 4: level 3 + rivers and streams >= 20 m wide, any length
        Level 5: level 4 + every waterway >= 3 m wide and >= 500 m long,
            minus the ones tagged intermittent, seasonal or ephemeral

    A lake threshold takes the smaller of the two bounds, not the larger, so a
    small model box still keeps the lakes that are large relative to it.
    """
    if config.lod_level == 0:
        return []

    # Define linear waterway tags
    linear_waterways = {
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

    # Intermittent waterway tags to exclude at level 5
    intermittent_tags = {"intermittent", "seasonal", "ephemeral"}

    filtered = []
    debug_stats = {
        "total": len(features),
        "oceans_seas": 0,
        "lakes_included": 0,
        "lakes_excluded_area": 0,
        "rivers_included": 0,
        "rivers_excluded_width": 0,
        "rivers_excluded_length": 0,
        "rivers_excluded_intermittent": 0,
    }

    # Sample features for detailed logging
    sample_lakes = []
    sample_rivers = []

    for feature in features:
        # Always include seas and oceans (regardless of LOD level >= 1)
        natural = feature.get_tag("natural")
        if natural in {"sea", "ocean", "coastline"}:
            filtered.append(feature)
            debug_stats["oceans_seas"] += 1
            continue

        # Check if this is a linear waterway
        waterway = feature.get_tag("waterway")
        is_linear = waterway in linear_waterways

        # Level 1: No inland water
        if config.lod_level == 1:
            continue  # Skip all non-ocean/sea features

        # For linear waterways (levels 3-5)
        if is_linear and config.allow_linear:
            # Estimate width from tags
            width_m = _estimate_waterway_width(feature)

            # Check width threshold
            if width_m < config.min_river_width_m:
                debug_stats["rivers_excluded_width"] += 1
                continue  # Too narrow

            # Check length threshold (if feature has length calculated)
            if (
                feature.length_m is not None
                and config.min_river_length_m > 0
                and feature.length_m < config.min_river_length_m
            ):
                debug_stats["rivers_excluded_length"] += 1
                continue  # Too short

            # Level 5: Exclude intermittent/seasonal waterways
            if config.lod_level == 5:
                if feature.get_tag("intermittent") in intermittent_tags:
                    debug_stats["rivers_excluded_intermittent"] += 1
                    continue
                if feature.get_tag("seasonal") == "yes":
                    debug_stats["rivers_excluded_intermittent"] += 1
                    continue

            filtered.append(feature)
            debug_stats["rivers_included"] += 1
            if len(sample_rivers) < 3:
                sample_rivers.append(
                    (feature.name, waterway, width_m, feature.length_m if feature.length_m else 0)
                )

        # For lakes and polygonal water bodies (levels 2+)
        elif feature.area_m2 is not None and feature.area_m2 > 0:
            if feature.area_m2 >= config.min_lake_area_m2:
                filtered.append(feature)
                debug_stats["lakes_included"] += 1
                if len(sample_lakes) < 3:
                    sample_lakes.append((feature.name, feature.area_m2 / 1_000_000))
            else:
                debug_stats["lakes_excluded_area"] += 1

        # Skip linear features if not allowed at this level
        elif is_linear and not config.allow_linear:
            continue

    # Print debug statistics
    print(f"  ✓ Water filtering stats (LOD {config.lod_level}):")
    print(f"    Total features: {debug_stats['total']}")
    print(f"    Oceans/seas: {debug_stats['oceans_seas']}")
    print(f"    Lakes included: {debug_stats['lakes_included']}")
    if sample_lakes:
        for name, area_km2 in sample_lakes:
            print(f"      - {name}: {area_km2:.3f}km²")
    print(
        f"    Lakes excluded (area < {config.min_lake_area_m2 / 1_000_000:.3f}km²): {debug_stats['lakes_excluded_area']}"
    )
    if config.allow_linear:
        print(f"    Rivers/streams included: {debug_stats['rivers_included']}")
        if sample_rivers:
            for name, rtype, width, length in sample_rivers:
                print(f"      - {name} ({rtype}): width={width:.1f}m, length={length:.0f}m")
        print(
            f"    Rivers excluded (width < {config.min_river_width_m:.1f}m): {debug_stats['rivers_excluded_width']}"
        )
        print(
            f"    Rivers excluded (length < {config.min_river_length_m:.0f}m): {debug_stats['rivers_excluded_length']}"
        )
        print(f"    Rivers excluded (intermittent): {debug_stats['rivers_excluded_intermittent']}")

    return filtered


def _estimate_waterway_width(feature: WaterFeature) -> float:
    """
    Estimate waterway width in meters from OSM tags.

    Args:
        feature: Water feature with tags

    Returns:
        Estimated width in meters
    """
    # Try explicit width tags
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
            numeric = "".join(ch for ch in width_str if (ch.isdigit() or ch == "." or ch == "-"))
            if numeric:
                return max(0.0, float(numeric))
        except Exception:
            continue

    # Fallback to type-based defaults
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
