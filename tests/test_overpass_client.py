# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for src.osm.overpass_client.

Network is not contacted — `requests.post` is monkey-patched to fake server
responses. These tests pin the new behaviours added on top of the original
client:

* Proper HTTP headers (User-Agent + Accept) are sent — Overpass servers
  sometimes refuse anonymous header-less POSTs with `406 Not Acceptable`.
* Server failover continues on connection errors, timeouts, and 4xx/5xx
  responses.
* "Empty response after some servers errored" emits a distinct warning so
  the user can tell a real empty bbox from a degraded query.
* On-disk caching round-trips correctly and is keyed by (bbox, query).
* All-servers-failed raises RuntimeError.
"""

from __future__ import annotations

import json
import os

import pytest
import requests
from src.osm import overpass_client as oc
from src.osm.overpass_client import OverpassClient, _build_water_query

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Stand-in for `requests.Response` used by the fake POST."""

    def __init__(self, *, status_code: int = 200, json_data=None, raise_exc=None):
        self.status_code = status_code
        self.reason = "OK" if status_code < 400 else "Error"
        self._json = json_data if json_data is not None else {"elements": []}
        self._raise_exc = raise_exc

    def json(self):
        if isinstance(self._json, Exception):
            raise self._json
        return self._json


def _install_fake_post(monkeypatch, responses):
    """
    Replace `requests.post` so the n-th call returns the n-th item from
    `responses`. Items may be:
      * `_FakeResponse` — returned as-is
      * `Exception` instance — raised
    """
    calls = []

    def fake_post(url, data=None, headers=None, timeout=None, **kwargs):
        calls.append(
            {
                "url": url,
                "data": data,
                "headers": headers,
                "timeout": timeout,
            }
        )
        idx = len(calls) - 1
        result = responses[min(idx, len(responses) - 1)]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(oc.requests, "post", fake_post)
    return calls


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------


class TestBuildWaterQuery:
    def test_contains_bbox(self):
        q = _build_water_query(51.8, 5.7, 51.95, 5.92, 60)
        assert "[bbox:51.8,5.7,51.95,5.92]" in q

    def test_contains_timeout(self):
        q = _build_water_query(0, 0, 1, 1, 240)
        assert "[timeout:240]" in q

    def test_no_recursion_after_geom(self):
        # The old client emitted `>; out skel qt;` after `out geom;` — that
        # doubled the payload. Make sure we don't accidentally bring it back.
        q = _build_water_query(0, 0, 1, 1, 60)
        assert "out geom;" in q
        assert ">;" not in q
        assert "out skel" not in q

    def test_minimal_whitespace(self):
        # Each statement should be on its own line with no leading indentation.
        q = _build_water_query(0, 0, 1, 1, 60)
        for line in q.splitlines():
            assert line == line.lstrip(), f"leading whitespace: {line!r}"


# ---------------------------------------------------------------------------
# OverpassClient.query_water_features
# ---------------------------------------------------------------------------


class TestQueryWaterFeatures:
    def test_success_first_server(self, monkeypatch, tmp_path):
        calls = _install_fake_post(
            monkeypatch,
            [_FakeResponse(json_data={"elements": [{"type": "way", "id": 1}]})],
        )
        client = OverpassClient(cache_dir=str(tmp_path))
        data = client.query_water_features(0, 0, 1, 1, timeout=60)

        assert data == {"elements": [{"type": "way", "id": 1}]}
        assert len(calls) == 1
        assert calls[0]["url"] == OverpassClient.OVERPASS_SERVERS[0]

    def test_sends_user_agent_and_accept_headers(self, monkeypatch, tmp_path):
        calls = _install_fake_post(
            monkeypatch,
            [_FakeResponse(json_data={"elements": []})],
        )
        client = OverpassClient(cache_dir=str(tmp_path), use_cache=False)
        client.query_water_features(0, 0, 1, 1, timeout=60)

        headers = calls[0]["headers"]
        assert headers.get("User-Agent")
        assert headers.get("Accept") == "application/json"

    def test_failover_on_connection_error(self, monkeypatch, tmp_path):
        calls = _install_fake_post(
            monkeypatch,
            [
                requests.ConnectionError("boom"),
                _FakeResponse(json_data={"elements": [{"id": 99}]}),
            ],
        )
        client = OverpassClient(cache_dir=str(tmp_path), use_cache=False)
        data = client.query_water_features(0, 0, 1, 1, timeout=60)

        assert data is not None
        assert data["elements"][0]["id"] == 99
        assert len(calls) == 2

    def test_failover_on_http_406(self, monkeypatch, tmp_path):
        calls = _install_fake_post(
            monkeypatch,
            [
                _FakeResponse(status_code=406),
                _FakeResponse(json_data={"elements": [{"id": 7}]}),
            ],
        )
        client = OverpassClient(cache_dir=str(tmp_path), use_cache=False)
        data = client.query_water_features(0, 0, 1, 1, timeout=60)
        assert data is not None
        assert data["elements"][0]["id"] == 7
        assert len(calls) == 2

    def test_all_servers_fail_raises(self, monkeypatch, tmp_path):
        _install_fake_post(
            monkeypatch,
            [requests.ConnectionError("down")] * len(OverpassClient.OVERPASS_SERVERS),
        )
        client = OverpassClient(cache_dir=str(tmp_path), use_cache=False)
        with pytest.raises(RuntimeError, match="All Overpass API servers"):
            client.query_water_features(0, 0, 1, 1, timeout=60)

    def test_empty_after_failures_is_warned(self, monkeypatch, tmp_path, capsys):
        _install_fake_post(
            monkeypatch,
            [
                requests.ConnectionError("flaky"),
                _FakeResponse(json_data={"elements": []}),
            ],
        )
        client = OverpassClient(cache_dir=str(tmp_path), use_cache=False)
        data = client.query_water_features(0, 0, 1, 1, timeout=60)
        out = capsys.readouterr().out
        assert data == {"elements": []}
        assert "after previous server errors" in out

    def test_empty_clean_is_not_warned(self, monkeypatch, tmp_path, capsys):
        _install_fake_post(
            monkeypatch,
            [_FakeResponse(json_data={"elements": []})],
        )
        client = OverpassClient(cache_dir=str(tmp_path), use_cache=False)
        client.query_water_features(0, 0, 1, 1, timeout=60)
        out = capsys.readouterr().out
        assert "No water features found" in out
        assert "after previous server errors" not in out

    def test_cache_hit_skips_network(self, monkeypatch, tmp_path):
        # First call populates the cache.
        _install_fake_post(
            monkeypatch,
            [_FakeResponse(json_data={"elements": [{"id": 5}]})],
        )
        client = OverpassClient(cache_dir=str(tmp_path), use_cache=True)
        first = client.query_water_features(0, 0, 1, 1, timeout=60)

        # Second call must NOT touch the network — replace post with a
        # function that fails the test if invoked.
        def boom(*a, **kw):  # pragma: no cover — should not run
            raise AssertionError("network hit despite cache")

        monkeypatch.setattr(oc.requests, "post", boom)

        second = client.query_water_features(0, 0, 1, 1, timeout=60)
        assert first == second

    def test_empty_response_is_cached(self, monkeypatch, tmp_path):
        # Avoids hammering Overpass on re-runs for a genuinely empty bbox.
        _install_fake_post(
            monkeypatch,
            [_FakeResponse(json_data={"elements": []})],
        )
        client = OverpassClient(cache_dir=str(tmp_path), use_cache=True)
        client.query_water_features(0, 0, 1, 1, timeout=60)
        # Exactly one .json file in the cache dir.
        files = [p for p in os.listdir(tmp_path) if p.endswith(".json")]
        assert len(files) == 1
        with open(os.path.join(tmp_path, files[0])) as f:
            cached = json.load(f)
        assert cached == {"elements": []}

    def test_use_cache_false_writes_nothing(self, monkeypatch, tmp_path):
        _install_fake_post(
            monkeypatch,
            [_FakeResponse(json_data={"elements": [{"id": 1}]})],
        )
        client = OverpassClient(cache_dir=str(tmp_path), use_cache=False)
        client.query_water_features(0, 0, 1, 1, timeout=60)
        # No .json files written.
        files = [p for p in os.listdir(tmp_path) if p.endswith(".json")]
        assert files == []
