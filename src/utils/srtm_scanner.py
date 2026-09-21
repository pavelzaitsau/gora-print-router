# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
SRTM/FABDEM TIF Scanner

Utilities for scanning SRTM and FABDEM TIF files and finding elevation data by coordinates.

Main functions:
- find_tif_for_coordinates: Find which TIF file contains given coordinates
- load_results_from_csv: Load cached scan results
- scan_srtm_tif_files: Scan all SRTM TIF files and update CSV file
- scan_fabdem_tiles: Scan FABDEM tiles from local STAC catalog
- find_fabdem_tile: Find FABDEM tile for given coordinates
"""

import csv
import json
import os
import sys
from typing import TypedDict

import rasterio


class TileBounds(TypedDict):
    """Geographic extent of one tile, in degrees. Keys match the CSV columns."""

    left: float
    right: float
    top: float
    bottom: float


class TileRecord(TypedDict):
    """One row of `data/bounds.csv`: a tile path plus the area it covers."""

    tif_file: str
    bounds: TileBounds


class FabdemTile(TypedDict):
    """A FABDEM tile name parsed into its south-west corner and 1x1 degree box."""

    lat: float
    lon: float
    bounds: TileBounds


def load_results_from_csv(csv_path: str) -> list[TileRecord]:
    """
    Load existing scan results from CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of previously scanned results
    """
    results: list[TileRecord] = []

    if not os.path.exists(csv_path):
        return results

    try:
        with open(csv_path, encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                results.append(
                    {
                        # The index stores a relative path, and whoever wrote
                        # it used their own separator. A CSV written on Windows
                        # holds `tiles\\N45E006.tif`, which names no file
                        # anywhere else, so every tile would read as missing.
                        "tif_file": row["tif_file"].replace("\\", "/"),
                        "bounds": {
                            "bottom": float(row["lat_min"]),
                            "top": float(row["lat_max"]),
                            "left": float(row["lon_min"]),
                            "right": float(row["lon_max"]),
                        },
                    }
                )
    except OSError as e:
        print(f"Warning: could not read {csv_path}: {e}")
        return []
    except Exception as e:
        # A row that will not parse is a damaged index, which is a different
        # problem from an index that simply does not cover the area. Returning
        # an empty list makes the two read the same, and the caller then tells
        # the user to run the scanner they already ran.
        raise ValueError(f"{csv_path} is damaged and cannot be read: {e}") from e

    return results


def find_tif_for_coordinates(lat: float, lon: float, csv_path: str) -> list[str]:
    """
    Find which TIF files contain the given coordinates.

    Args:
        lat: Latitude
        lon: Longitude
        csv_path: Path to CSV file with TIF bounds

    Returns:
        List of TIF file names that contain the coordinates
    """
    # Load results from CSV
    results = load_results_from_csv(csv_path)

    if not results:
        return []

    matching = []

    for result in results:
        bounds = result["bounds"]
        if bounds["bottom"] <= lat <= bounds["top"] and bounds["left"] <= lon <= bounds["right"]:
            matching.append(result["tif_file"])

    return matching


def find_tifs_for_bounds(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    csv_path: str,
) -> list[str]:
    """
    Find all TIF files that intersect with the given bounding box.

    Args:
        lat_min: Minimum latitude
        lat_max: Maximum latitude
        lon_min: Minimum longitude
        lon_max: Maximum longitude
        csv_path: Path to CSV file with TIF bounds

    Returns:
        List of TIF file names that intersect with the bounds
    """
    # Load results from CSV
    results = load_results_from_csv(csv_path)

    if not results:
        return []

    matching = []

    for result in results:
        bounds = result["bounds"]
        # Strict overlap, not touching. FABDEM tiles abut exactly, so a box
        # that stops on a whole degree shares a line with the next tile and no
        # area at all. Including it reads a window zero pixels high, the merge
        # rejects that tile as empty, and the run fails over a tile it never
        # needed. A box ending on a degree is the ordinary case.
        lat_overlap = bounds["top"] > lat_min and bounds["bottom"] < lat_max
        lon_overlap = bounds["right"] > lon_min and bounds["left"] < lon_max

        if lat_overlap and lon_overlap:
            matching.append(result["tif_file"])

    return matching


def get_tif_bounds(tif_path: str) -> TileBounds | None:
    """
    Get geographic bounds of a TIF file using rasterio.

    Args:
        tif_path: Path to the TIF file

    Returns:
        Dictionary with bounds {bottom, top, left, right} or None if failed
    """
    try:
        with rasterio.open(tif_path) as dataset:
            bounds = dataset.bounds
            return {
                "left": float(bounds.left),
                "right": float(bounds.right),
                "top": float(bounds.top),
                "bottom": float(bounds.bottom),
            }

    except Exception as e:
        print(f"Error reading TIF bounds from {tif_path}: {e}")
        return None


def scan_srtm_tif_files(
    tif_dir: str = "tif", output_csv: str | None = None, verbose: bool = True
) -> list[TileRecord]:
    """
    Scan all SRTM TIF files in a directory and extract their bounds.

    Args:
        tif_dir: Directory containing SRTM TIF files (default: "tif")
        output_csv: Path to output CSV file (default: data/bounds.csv)
        verbose: Print progress messages

    Returns:
        List of dictionaries containing TIF file info and bounds
    """
    if output_csv is None:
        # Default to bounds.csv in parent directory of tif_dir
        parent_dir = os.path.dirname(tif_dir)
        if parent_dir and os.path.exists(parent_dir):
            output_csv = os.path.join(parent_dir, "bounds.csv")
        else:
            # If no parent dir or it doesn't exist, use current directory
            output_csv = "bounds.csv"

    if not os.path.exists(tif_dir):
        print(f"Error: Directory not found: {tif_dir}")
        return []

    # Find all TIF files
    tif_files = [
        f for f in os.listdir(tif_dir) if f.lower().endswith(".tif") or f.lower().endswith(".tiff")
    ]
    tif_files.sort()

    if not tif_files:
        print(f"No TIF files found in {tif_dir}")
        return []

    if verbose:
        print(f"Found {len(tif_files)} TIF files in {tif_dir}")

    results: list[TileRecord] = []

    for i, tif_file in enumerate(tif_files, 1):
        if verbose:
            print(f"[{i}/{len(tif_files)}] Processing {tif_file}...")

        tif_path = os.path.join(tif_dir, tif_file)

        try:
            # Get bounds
            bounds = get_tif_bounds(tif_path)

            if bounds:
                results.append({"tif_file": tif_file, "bounds": bounds})
                if verbose:
                    print(
                        f"  ✓ lat [{bounds['bottom']:.6f}, {bounds['top']:.6f}], "
                        f"lon [{bounds['left']:.6f}, {bounds['right']:.6f}]"
                    )
            else:
                if verbose:
                    print(f"  Warning: Could not read bounds from {tif_file}")

        except Exception as e:
            if verbose:
                print(f"  Error processing {tif_file}: {e}")
            continue

    # Write results to CSV
    if results:
        try:
            with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["tif_file", "lat_min", "lat_max", "lon_min", "lon_max"])

                for result in results:
                    row_bounds = result["bounds"]
                    writer.writerow(
                        [
                            result["tif_file"],
                            f"{row_bounds['bottom']:.6f}",
                            f"{row_bounds['top']:.6f}",
                            f"{row_bounds['left']:.6f}",
                            f"{row_bounds['right']:.6f}",
                        ]
                    )

            if verbose:
                print(f"\n✓ Successfully scanned {len(results)} TIF files")
                print(f"✓ Results saved to: {output_csv}")

        except Exception as e:
            print(f"Error writing CSV file: {e}")
            return results
    else:
        if verbose:
            print("\nNo valid TIF files were scanned")

    return results


def parse_fabdem_tile_name(tile_name: str) -> FabdemTile | None:
    """
    Parse FABDEM tile name to extract coordinates.

    Format: N00E006_FABDEM_V1-2 or S23W041_FABDEM_V1-2
    Returns dictionary with lat, lon, and bounds.

    Args:
        tile_name: Name like 'N00E006_FABDEM_V1-2'

    Returns:
        Dictionary with lat, lon, and bounds or None if invalid
    """
    import re

    # Pattern: N79W106 or S23W041
    pattern = r"([NS])(\d+)([EW])(\d+)_FABDEM"
    match = re.match(pattern, tile_name)

    if not match:
        return None

    lat_dir, lat_val, lon_dir, lon_val = match.groups()

    # Convert to decimal degrees
    lat = float(lat_val) if lat_dir == "N" else -float(lat_val)
    lon = float(lon_val) if lon_dir == "E" else -float(lon_val)

    # FABDEM tiles are typically 1x1 degree tiles
    return {
        "lat": lat,
        "lon": lon,
        "bounds": {
            "left": lon,
            "right": lon + 1,
            "bottom": lat,
            "top": lat + 1,
        },
    }


def find_fabdem_tile(lat: float, lon: float, tiles_dir: str) -> str | None:
    """
    Find FABDEM tile for given coordinates.

    Args:
        lat: Latitude
        lon: Longitude
        tiles_dir: Directory containing FABDEM tiles

    Returns:
        Path to TIF file or None if not found
    """
    # Determine which tile we need based on coordinates
    # FABDEM tiles are 1x1 degree
    tile_lat = int(lat) if lat >= 0 else int(lat) - (1 if lat != int(lat) else 0)
    tile_lon = int(lon) if lon >= 0 else int(lon) - (1 if lon != int(lon) else 0)

    # Build tile name
    lat_dir = "N" if tile_lat >= 0 else "S"
    lon_dir = "E" if tile_lon >= 0 else "W"
    tile_name = f"{lat_dir}{abs(tile_lat):02d}{lon_dir}{abs(tile_lon):03d}_FABDEM_V1-2"

    if not os.path.exists(tiles_dir):
        return None

    # The tiles are organized in grouped folders
    # We need to search through the tile group directories
    for group_dir in os.listdir(tiles_dir):
        group_path = os.path.join(tiles_dir, group_dir)
        if not os.path.isdir(group_path):
            continue

        # Look for the specific tile file
        tile_file = f"{tile_name}.tif"
        tile_path = os.path.join(group_path, tile_file)

        if os.path.exists(tile_path):
            return tile_path

    return None


def scan_fabdem_tiles_direct(
    tiles_dir: str,
    output_csv: str,
    verbose: bool = True,
) -> list[TileRecord]:
    """
    Scan FABDEM tiles by directly scanning the tiles directory.

    Args:
        tiles_dir: Directory containing FABDEM tiles
        output_csv: Path to output CSV file
        verbose: Print progress messages

    Returns:
        List of dictionaries containing tile info and bounds
    """

    base_dir = os.path.dirname(os.path.normpath(tiles_dir)) or "."

    if not os.path.exists(tiles_dir):
        print(f"Error: Tiles directory not found: {tiles_dir}")
        return []

    if not os.path.exists(tiles_dir):
        print(f"Error: Tiles directory not found: {tiles_dir}")
        return []

    if verbose:
        print(f"Scanning tiles directory: {tiles_dir}")

    results: list[TileRecord] = []
    tile_count = 0

    # Walk through all subdirectories in tiles/
    for group_dir in sorted(os.listdir(tiles_dir)):
        group_path = os.path.join(tiles_dir, group_dir)
        if not os.path.isdir(group_path):
            continue

        # Find all TIF files in this group directory
        for tif_file in sorted(os.listdir(group_path)):
            if not (tif_file.lower().endswith(".tif") or tif_file.lower().endswith(".tiff")):
                continue

            tile_count += 1

            # Parse the tile name to get bounds
            tile_info = parse_fabdem_tile_name(tif_file)

            if tile_info:
                # Get relative path from base directory so we store tiles/...
                rel_path = os.path.relpath(os.path.join(group_path, tif_file), base_dir).replace(
                    os.sep, "/"
                )

                results.append({"tif_file": rel_path, "bounds": tile_info["bounds"]})

                if verbose and tile_count % 500 == 0:
                    print(f"  Processed {tile_count} tiles...")
            else:
                if verbose and tile_count <= 10:
                    print(f"  Warning: Could not parse tile name: {tif_file}")

    if verbose:
        print(f"\n✓ Successfully cataloged {len(results)} FABDEM tiles")

    # Write results to CSV
    if results:
        try:
            with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["tif_file", "lat_min", "lat_max", "lon_min", "lon_max"])

                for result in results:
                    row_bounds = result["bounds"]
                    writer.writerow(
                        [
                            result["tif_file"],
                            f"{row_bounds['bottom']:.6f}",
                            f"{row_bounds['top']:.6f}",
                            f"{row_bounds['left']:.6f}",
                            f"{row_bounds['right']:.6f}",
                        ]
                    )

            if verbose:
                print(f"✓ Results saved to: {output_csv}")

        except Exception as e:
            print(f"Error writing CSV file: {e}")
            return results

    return results


def scan_fabdem_tiles(
    tiles_dir: str,
    output_csv: str,
    geojson_path: str,
    verbose: bool = True,
) -> list[TileRecord]:
    """
    Scan FABDEM tiles from local STAC catalog or directory.

    Args:
        tiles_dir: Directory containing FABDEM tiles
        output_csv: Path to output CSV file
        geojson_path: Path to FABDEM GeoJSON
        verbose: Print progress messages

    Returns:
        List of dictionaries containing tile info and bounds
    """

    base_dir = os.path.dirname(os.path.normpath(tiles_dir)) or "."

    # If GeoJSON doesn't exist, fall back to direct directory scanning
    if not os.path.exists(geojson_path):
        if verbose:
            print("GeoJSON not found, scanning tiles directory directly...")
        return scan_fabdem_tiles_direct(tiles_dir, output_csv, verbose)

    if verbose:
        print(f"Reading FABDEM tile metadata from {geojson_path}")

    results: list[TileRecord] = []

    try:
        with open(geojson_path, encoding="utf-8") as f:
            data = json.load(f)

        features = data.get("features", [])

        if verbose:
            print(f"Found {len(features)} FABDEM tiles in catalog")

        for i, feature in enumerate(features, 1):
            properties = feature.get("properties", {})
            tile_name = properties.get("tile_name", "")
            file_name = properties.get("file_name_corrected", properties.get("file_name", ""))

            if not tile_name or not file_name:
                continue

            # Parse tile coordinates
            tile_info = parse_fabdem_tile_name(f"{tile_name}_FABDEM_V1-2")

            if not tile_info:
                if verbose:
                    print(f"  Warning: Could not parse tile name: {tile_name}")
                continue

            # Find the actual file in the tiles directory
            tile_found = False
            for group_dir in os.listdir(tiles_dir):
                group_path = os.path.join(tiles_dir, group_dir)
                if not os.path.isdir(group_path):
                    continue

                tile_path = os.path.join(group_path, file_name)
                if os.path.exists(tile_path):
                    # Get relative path from base directory so we store tiles/...
                    rel_path = os.path.relpath(tile_path, base_dir).replace(os.sep, "/")

                    results.append({"tif_file": rel_path, "bounds": tile_info["bounds"]})

                    tile_found = True

                    if verbose and i % 1000 == 0:
                        print(f"  Processed {i}/{len(features)} tiles...")

                    break

            if not tile_found and verbose and i <= 10:  # Only warn for first few
                print(f"  Warning: Tile file not found: {file_name}")

        if verbose:
            print(f"\n✓ Successfully cataloged {len(results)} FABDEM tiles")

        # Write results to CSV
        if results:
            try:
                with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["tif_file", "lat_min", "lat_max", "lon_min", "lon_max"])

                    for result in results:
                        bounds = result["bounds"]
                        writer.writerow(
                            [
                                result["tif_file"],
                                f"{bounds['bottom']:.6f}",
                                f"{bounds['top']:.6f}",
                                f"{bounds['left']:.6f}",
                                f"{bounds['right']:.6f}",
                            ]
                        )

                if verbose:
                    print(f"✓ Results saved to: {output_csv}")

            except Exception as e:
                print(f"Error writing CSV file: {e}")
                return results

    except Exception as e:
        print(f"Error reading GeoJSON file: {e}")
        return []

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scan SRTM or FABDEM elevation data tiles")
    parser.add_argument(
        "--mode",
        choices=["srtm", "fabdem"],
        default="fabdem",
        help="Scan mode: 'srtm' for traditional TIF files or 'fabdem' for FABDEM catalog (default: fabdem)",
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Directory to scan (default: 'tif' for SRTM, 'data/tiles' for FABDEM)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory (default: 'data')",
    )

    args = parser.parse_args()

    if args.mode == "fabdem":
        scan_tiles_dir = args.dir or os.path.join(args.data_dir, "tiles")
        scan_bounds_csv = os.path.join(args.data_dir, "bounds.csv")
        scan_geojson_path = os.path.join(args.data_dir, "FABDEM_v1-2_tiles.geojson")

        print("=" * 60)
        print("FABDEM Tile Scanner")
        print("=" * 60)
        print()

        scan_results = scan_fabdem_tiles(
            scan_tiles_dir, scan_bounds_csv, scan_geojson_path, verbose=True
        )

        if scan_results:
            print()
            print("=" * 60)
            print(f"Scan complete! Found {len(scan_results)} FABDEM tiles")
            print("=" * 60)
        else:
            print()
            print("=" * 60)
            print("Scan failed or no FABDEM tiles found")
            print("=" * 60)
            sys.exit(1)

    else:  # srtm mode
        tif_directory = args.dir or "tif"

        print("=" * 60)
        print("SRTM TIF Scanner")
        print("=" * 60)
        print()

        scan_results = scan_srtm_tif_files(tif_directory, verbose=True)

        if scan_results:
            print()
            print("=" * 60)
            print(f"Scan complete! Found {len(scan_results)} valid TIF files")
            print("=" * 60)
        else:
            print()
            print("=" * 60)
            print("Scan failed or no TIF files found")
            print("=" * 60)
            sys.exit(1)
