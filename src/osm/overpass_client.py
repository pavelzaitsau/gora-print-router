# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
OpenStreetMap Overpass API client with local caching.

Overpass is a shared volunteer service, and every rule below exists because
ignoring it costs someone else.

* The request carries a User-Agent and an Accept header. overpass-api.de
  answers an anonymous, header-less POST with `406 Not Acceptable`, and an
  unidentifiable client is the one operators block first.
* The query goes out flat. Leading whitespace from f-string indentation adds
  bytes to every request and the stricter parsers reject it outright.
* `out geom;` inlines the geometry, so no `>; out skel qt;` follows it. The
  recursion would duplicate the payload and double the response size.
* TLS verification stays on.
* An empty answer is cached like any other. A bbox with no water is a fact,
  and re-asking the servers for it on every run is what gets a client
  rate-limited.
* "Every server errored" and "a server answered, with nothing in it" are
  different outcomes and are reported differently. The first raises; the
  second is a legitimate empty result.
"""

import hashlib
import json
import os
import time
from typing import Any

import requests

# Public so callers/tests can introspect.
OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]

DEFAULT_TIMEOUT = 180  # seconds, the Overpass query budget and the read wait
CONNECT_TIMEOUT = 10  # seconds; longer than any handshake, shorter than a dead host

# HTTP headers. The User-Agent identifies the client, and the URL in it has to
# reach a human: Overpass operators block an unidentifiable client, and the
# block lands on every user of this tool, not on the one who caused it.
# Accept-Encoding lets the server gzip the often huge JSON response, and some
# servers answer 406 without an explicit Accept.
HTTP_HEADERS = {
    "User-Agent": "gora-print-router/0.1 (+https://github.com/pavelzaitsau/gora-print-router)",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}


def _build_water_query(
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
    timeout: int,
) -> str:
    """Build a single-line Overpass QL query for water features."""
    bbox = f"{lat_min},{lon_min},{lat_max},{lon_max}"
    # One newline-free statement per line minimises whitespace bytes and
    # avoids parser edge cases on stricter Overpass instances.
    parts = [
        f"[out:json][timeout:{timeout}][bbox:{bbox}];",
        "(",
        "way[natural=water];",
        "way[natural=sea];",
        "way[natural=ocean];",
        "way[natural=coastline];",
        "way[waterway];",
        "way[water];",
        "rel[natural=water];",
        "rel[natural=sea];",
        "rel[natural=ocean];",
        "rel[waterway=riverbank];",
        "rel[water];",
        ");",
        # `out geom;` is sufficient — it inlines node coordinates for every
        # way/relation member. The previous query also did `>; out skel qt;`
        # which re-fetched the same nodes separately and roughly doubled the
        # payload (and tripped server-side rate limits).
        "out geom;",
    ]
    return "\n".join(parts)


class OverpassClient:
    """Overpass API client with on-disk JSON caching."""

    OVERPASS_SERVERS = OVERPASS_SERVERS
    DEFAULT_TIMEOUT = DEFAULT_TIMEOUT

    def __init__(self, cache_dir: str = "osm_cache", use_cache: bool = True):
        """
        Initialise the client.

        Args:
            cache_dir: Directory for caching query responses (JSON).
            use_cache: Disable on-disk caching when False.
        """
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        # Only when it will be used. Creating it regardless turns a read-only
        # working directory into a PermissionError out of a constructor, for a
        # directory the caller asked the tool not to keep.
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)

    # ---- cache plumbing ----------------------------------------------------

    def _get_cache_key(self, query: str, bbox: tuple[float, float, float, float]) -> str:
        cache_str = f"{bbox}_{query}"
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    # ---- query -------------------------------------------------------------

    def query_water_features(
        self,
        lat_min: float,
        lon_min: float,
        lat_max: float,
        lon_max: float,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> dict[str, Any] | None:
        """
        Fetch water-related OSM features inside a bbox.

        Returns the parsed Overpass JSON response (a dict with an "elements"
        list). Cached on disk by (bbox, query) hash.

        Raises:
            RuntimeError: All Overpass servers failed.
        """
        query = _build_water_query(lat_min, lon_min, lat_max, lon_max, timeout)

        bbox = (lat_min, lon_min, lat_max, lon_max)
        cache_key = self._get_cache_key(query, bbox)
        cache_path = self._get_cache_path(cache_key)

        # Cache hit
        if self.use_cache and os.path.exists(cache_path):
            print("  Loading water features from cache...")
            try:
                with open(cache_path, encoding="utf-8") as f:
                    cached: dict[str, Any] = json.load(f)
                return cached
            except (OSError, json.JSONDecodeError) as e:
                print(f"  ⚠ Cache read error, refetching: {e}")

        print("  Fetching water features from OpenStreetMap...")

        last_error: BaseException | None = None
        any_server_failed = False

        for server_idx, server_url in enumerate(self.OVERPASS_SERVERS, 1):
            if server_idx > 1:
                print(f"  Trying alternate server ({server_idx}/{len(self.OVERPASS_SERVERS)})...")

            try:
                response = requests.post(
                    server_url,
                    data={"data": query},
                    headers=HTTP_HEADERS,
                    # Connect and read are different waits. A server that
                    # refuses the connection is known bad in milliseconds;
                    # giving it the query's own budget means the run spends
                    # minutes per dead mirror before it gives up, with the
                    # terrain already built and about to be thrown away.
                    timeout=(CONNECT_TIMEOUT, timeout + 30),
                )
            except requests.Timeout as e:
                last_error = e
                any_server_failed = True
                print(f"  ⚠ Server timeout: {e}")
                continue
            except requests.RequestException as e:
                last_error = e
                any_server_failed = True
                print(f"  ⚠ Server error: {e}")
                continue

            # Distinguish recoverable HTTP errors (rate-limit / busy) from
            # genuine 4xx so we can try the next server only when it makes
            # sense to. 429/5xx → try next; 4xx that isn't a rate limit
            # usually means the request itself is malformed and trying
            # another server won't help, but we still try in case it's
            # implementation-specific.
            if response.status_code >= 400:
                last_error = requests.HTTPError(
                    f"{response.status_code} {response.reason} from {server_url}"
                )
                any_server_failed = True
                print(f"  ⚠ Server HTTP {response.status_code}: {response.reason}")
                # Quick backoff so we don't immediately re-hit a busy server.
                if response.status_code in (429, 503, 504):
                    time.sleep(2.0)
                continue

            # 2xx — try to parse.
            try:
                data: dict[str, Any] = response.json()
            except json.JSONDecodeError as e:
                last_error = e
                any_server_failed = True
                print(f"  ⚠ Invalid JSON from {server_url}: {e}")
                continue

            # An empty elements list is a legitimate "no water in this area"
            # response. But if previous servers errored, warn so the user can
            # tell the difference between a real empty bbox and a degraded
            # query that just happened to round-trip without errors.
            if not data.get("elements"):
                if any_server_failed:
                    print(
                        "  ⚠ Server returned empty result after previous server "
                        "errors — could be a transient outage, not actually a "
                        "water-free area. Delete the cache file and retry "
                        "later if the model is missing water you expect."
                    )
                else:
                    print("  ℹ No water features found in this area.")
                # Still cache to avoid hammering on re-runs.
                self._write_cache(cache_path, data)
                return data

            # Got real data.
            self._write_cache(cache_path, data)
            return data

        # All servers failed
        raise RuntimeError(
            f"All Overpass API servers are unavailable. "
            f"Tried {len(self.OVERPASS_SERVERS)} servers. "
            f"Last error: {last_error}"
        )

    # ---- helpers -----------------------------------------------------------

    def _write_cache(self, cache_path: str, data: dict[str, Any]) -> None:
        if not self.use_cache:
            return
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print("  ✓ Cached water features")
        except OSError as e:
            print(f"  ⚠ Cache write error: {e}")
