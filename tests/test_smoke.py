# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Smoke tests: the package imports and its CLI wiring is intact.

These catch the failures no unit test does — a broken import chain, a missing
dependency in pyproject.toml, an argparse definition that raises on build.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODULES = [
    "src.core.footprint",
    "src.core.mesh_generator",
    "src.core.router",
    "src.core.water_generator",
    "src.osm.geometry_converter",
    "src.osm.osm_utils",
    "src.osm.overpass_client",
    "src.osm.water_features",
    "src.utils.gpx_utils",
    "src.utils.obj_exporter",
    "src.utils.srtm_scanner",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    """Every module imports without side effects blowing up."""
    assert importlib.import_module(module_name) is not None


def test_entry_point_help_runs():
    """`python main.py --help` exits 0 — the CLI wiring is intact end to end."""
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "--model-shape" in result.stdout
