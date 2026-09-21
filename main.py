#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Main entry point for the 3D Terrain Router application.
This script launches the router from the project root directory.
"""

# Import via the `src.` package path that every module inside src/ uses for its
# own imports. Putting src/ on sys.path and importing `core.router` instead
# would load router.py a second time under a different module name, giving two
# copies of its module-level state and breaking isinstance across the pair.
from src.core.router import main

if __name__ == "__main__":
    main()
