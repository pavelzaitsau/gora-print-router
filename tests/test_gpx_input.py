# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What the GPX reader accepts.

A GPX file carries a path as a track (`<trk>`), a route (`<rte>`) or a list of
waypoints. Planning tools mostly emit routes; recording devices mostly emit
tracks. Both name the same thing to the person who exported it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from src.utils.gpx_utils import parse_gpx_track

HEAD = (
    '<?xml version="1.0"?><gpx version="1.1" creator="t" xmlns="http://www.topografix.com/GPX/1/1">'
)


def _write(path: Path, body: str) -> str:
    path.write_text(f"{HEAD}{body}</gpx>", encoding="utf-8")
    return str(path)


def _points(*lat_lon: tuple[float, float]) -> str:
    return "".join(f'<trkpt lat="{a}" lon="{b}"/>' for a, b in lat_lon)


def _route(*lat_lon: tuple[float, float]) -> str:
    return "<rte>" + "".join(f'<rtept lat="{a}" lon="{b}"/>' for a, b in lat_lon) + "</rte>"


def test_a_track_is_read(tmp_path):
    path = _write(
        tmp_path / "t.gpx", f"<trk><trkseg>{_points((46.0, 7.0), (46.1, 7.1))}</trkseg></trk>"
    )
    assert parse_gpx_track(path) == [(46.0, 7.0), (46.1, 7.1)]


def test_a_route_is_read_like_a_track(tmp_path):
    """Komoot, Garmin Connect courses and most planners export a route.

    Rejecting it reports "no track points found" about a file that plainly has
    a path in it, which reads as "your file is empty" and is not true.
    """
    path = _write(tmp_path / "r.gpx", _route((46.0, 7.0), (46.1, 7.1)))
    assert parse_gpx_track(path) == [(46.0, 7.0), (46.1, 7.1)]


def test_a_track_wins_over_a_route_in_the_same_file(tmp_path):
    """Some exporters write both. The recorded track is the better source."""
    body = f"<trk><trkseg>{_points((45.0, 6.0), (45.1, 6.1))}</trkseg></trk>" + _route(
        (46.0, 7.0), (46.1, 7.1)
    )
    path = _write(tmp_path / "both.gpx", body)
    assert parse_gpx_track(path) == [(45.0, 6.0), (45.1, 6.1)]


def test_a_file_with_no_path_at_all_is_still_rejected(tmp_path):
    path = _write(tmp_path / "empty.gpx", "<metadata><name>nothing</name></metadata>")
    with pytest.raises(ValueError, match=r"[Nn]o track or route points"):
        parse_gpx_track(path)
