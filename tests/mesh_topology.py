# SPDX-FileCopyrightText: 2025-2026 Pavel Zaitsau
# SPDX-FileCopyrightText: 2025-2026 Góra Print
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Topology checks for printable meshes: watertightness, manifoldness, winding.

A model is printable only when its surface is closed and orientable: every edge
is shared by exactly two triangles, the two triangles traverse that edge in
opposite directions, and the enclosed volume is positive (normals point out).

The mesh pipeline emits triangle soup — three fresh vertex records per face, no
index sharing — so topology is invisible until identical coordinates are welded
back into one vertex id. Everything here welds first, then counts edges.

Not a test module (no `test_` prefix): imported by tests/test_watertight.py.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Triangles = FloatArray
"""(N, 3, 3): N triangles, three corners each, three coordinates per corner."""
Faces = npt.NDArray[np.int64]
"""(N, 3): the same triangles after welding, as vertex ids."""

# Weld tolerance in decimal places on millimetres. 1e-4 mm is far below any
# print resolution yet far above float32 round-off on ~100 mm coordinates
# (~1e-5 mm), so coincident vertices collapse and distinct ones never do.
WELD_DECIMALS = 4


def triangles_of(mesh: Any) -> Triangles:
    """Extract an (N, 3, 3) float array of triangle corners from a `Mesh`."""
    return np.asarray(mesh.data["vectors"], dtype=np.float64)


def weld(triangles: Triangles, decimals: int = WELD_DECIMALS) -> Faces:
    """Collapse coincident corners into shared ids; return (N, 3) index faces."""
    corners = np.round(np.asarray(triangles, dtype=np.float64).reshape(-1, 3), decimals)
    # -0.0 and 0.0 round-trip to different byte patterns and would not unify.
    corners = corners + 0.0
    _, inverse = np.unique(corners, axis=0, return_inverse=True)
    return inverse.reshape(-1, 3).astype(np.int64)


@dataclass(frozen=True)
class Topology:
    """Edge/face bookkeeping for one welded triangle set."""

    face_count: int
    vertex_count: int
    edge_count: int
    degenerate_faces: int
    """Triangles with a repeated corner — zero area, no well-defined normal."""
    boundary_edges: int
    """Edges used by one face only. Any of these is a hole in the surface."""
    nonmanifold_edges: int
    """Edges used by three or more faces — surfaces meeting along a seam."""
    flipped_edges: int
    """Directed edges emitted twice: the two faces disagree on which side is out."""
    euler_characteristic: int
    """V - E + F. A closed genus-0 solid gives 2; a torus 0; a hole lowers it."""
    volume_mm3: float
    """Signed divergence volume. Positive means outward-facing normals."""

    @property
    def is_watertight(self) -> bool:
        return (
            self.degenerate_faces == 0
            and self.boundary_edges == 0
            and self.nonmanifold_edges == 0
            and self.flipped_edges == 0
        )

    def describe(self, name: str = "mesh") -> str:
        return (
            f"{name}: faces={self.face_count} vertices={self.vertex_count} "
            f"edges={self.edge_count} euler={self.euler_characteristic} "
            f"degenerate={self.degenerate_faces} holes(boundary edges)={self.boundary_edges} "
            f"non-manifold={self.nonmanifold_edges} flipped={self.flipped_edges} "
            f"volume={self.volume_mm3:.2f}mm^3"
        )


def analyze(triangles: Triangles, decimals: int = WELD_DECIMALS) -> Topology:
    """Weld `triangles` and count every way the surface can fail to be a solid."""
    tris = np.asarray(triangles, dtype=np.float64)
    faces = weld(tris, decimals)

    degenerate = int(
        np.sum(
            (faces[:, 0] == faces[:, 1])
            | (faces[:, 1] == faces[:, 2])
            | (faces[:, 2] == faces[:, 0])
        )
    )

    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, directed_counts = np.unique(directed, axis=0, return_counts=True)
    undirected = np.sort(directed, axis=1)
    unique_edges, edge_counts = np.unique(undirected, axis=0, return_counts=True)

    # Divergence theorem on the origin-based tetrahedra of every triangle.
    volume = float(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6.0)

    vertex_count = len(np.unique(faces))
    edge_count = len(unique_edges)
    return Topology(
        face_count=len(faces),
        vertex_count=vertex_count,
        edge_count=edge_count,
        degenerate_faces=degenerate,
        boundary_edges=int(np.sum(edge_counts == 1)),
        nonmanifold_edges=int(np.sum(edge_counts > 2)),
        flipped_edges=int(np.sum(directed_counts > 1)),
        euler_characteristic=vertex_count - edge_count + len(faces),
        volume_mm3=volume,
    )


def assert_watertight(triangles: Triangles, name: str = "mesh") -> Topology:
    """Fail with a full topology dump unless `triangles` form a closed solid."""
    topo = analyze(triangles)
    assert topo.is_watertight, f"not watertight — {topo.describe(name)}"
    assert topo.euler_characteristic == 2, (
        f"closed but not genus 0 (internal void or tunnel) — {topo.describe(name)}"
    )
    assert topo.volume_mm3 > 0.0, f"inward-facing normals — {topo.describe(name)}"
    return topo


# A crease this sharp is not a feature. The track's own hard edges are right
# angles at worst (flat bottom into the wall, wall into an end cap), which read
# as a dot of 0; anything past -0.5 means the two faces sharing that edge face
# each other, i.e. the surface has folded back over itself.
#
# Calibrated on the synthetic fixtures, which sit on gentle ground. A ribbon
# draped over genuinely steep terrain tilts its own base towards its wall, so a
# real DEM can produce dihedrals this acute honestly; judging one of those wants
# a limit nearer -0.85.
FOLD_DOT_LIMIT = -0.5
# Folds below this area are smaller than one extrusion can render, so they are
# reported but not failed: a sub-resolution hairpin tip pinches to a needle.
MIN_FOLD_AREA_MM2 = 1e-3


@dataclass(frozen=True)
class Folds:
    """Inverted geometry found by walking edge-adjacent face pairs.

    A swept ribbon whose offset exceeds its turn radius folds inside out: the
    surface stays closed, so every check in `Topology` still passes, but the
    slicer meets inward-facing normals and prints a void. This is the check
    that sees it.
    """

    count: int
    """Edge-adjacent face pairs facing each other, above `MIN_FOLD_AREA_MM2`."""
    sub_resolution_count: int
    """The same, but on faces too small to print — reported, not failed."""
    worst_dot: float
    """Most negative normal dot product found (1.0 when nothing is adjacent)."""
    worst_xyz: tuple[float, float, float] | None
    """Centroid of the worst pair, for pointing at the offending corner."""

    def describe(self, name: str = "mesh") -> str:
        where = (
            "none"
            if self.worst_xyz is None
            else f"({self.worst_xyz[0]:.2f}, {self.worst_xyz[1]:.2f}, {self.worst_xyz[2]:.2f})"
        )
        return (
            f"{name}: folded face pairs={self.count} "
            f"(sub-resolution={self.sub_resolution_count}) "
            f"worst normal dot={self.worst_dot:.3f} at {where}"
        )


def find_folds(
    triangles: Triangles,
    decimals: int = WELD_DECIMALS,
    dot_limit: float = FOLD_DOT_LIMIT,
    min_area_mm2: float = MIN_FOLD_AREA_MM2,
) -> Folds:
    """Find edge-adjacent triangle pairs whose normals point at each other."""
    tris = np.asarray(triangles, dtype=np.float64)
    faces = weld(tris, decimals)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    areas = np.linalg.norm(normals, axis=1) / 2.0
    unit = normals / np.where(areas[:, None] == 0.0, 1.0, 2.0 * areas[:, None])

    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    owners = np.tile(np.arange(len(faces)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owners = edges[order], owners[order]
    # Manifold edges arrive as adjacent duplicate rows after the sort.
    shared = np.flatnonzero(np.all(edges[:-1] == edges[1:], axis=1))
    left, right = owners[shared], owners[shared + 1]

    dots = np.einsum("ij,ij->i", unit[left], unit[right])
    folded = dots < dot_limit
    printable = folded & (np.minimum(areas[left], areas[right]) >= min_area_mm2)

    worst_dot = float(dots.min()) if len(dots) else 1.0
    worst_xyz: tuple[float, float, float] | None = None
    if printable.any():
        worst = int(np.flatnonzero(printable)[np.argmin(dots[printable])])
        centroid = tris[left[worst]].mean(axis=0)
        worst_xyz = (float(centroid[0]), float(centroid[1]), float(centroid[2]))

    return Folds(
        count=int(printable.sum()),
        sub_resolution_count=int(folded.sum() - printable.sum()),
        worst_dot=worst_dot,
        worst_xyz=worst_xyz,
    )


def assert_no_folds(triangles: Triangles, name: str = "mesh") -> Folds:
    """Fail unless the surface is free of printable inside-out geometry."""
    folds = find_folds(triangles)
    assert folds.count == 0, f"surface folds back on itself — {folds.describe(name)}"
    return folds


def split_shells(triangles: Triangles, decimals: int = WELD_DECIMALS) -> list[Triangles]:
    """Split a triangle set into connected components (separate printed bodies).

    A water layer holds one body per lake, so "closed solid" has to be checked
    per shell — the whole group's Euler number is just the sum over its shells.
    """
    faces = weld(triangles, decimals)
    parent = list(range(int(faces.max()) + 1)) if len(faces) else []

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b, c in faces:
        for x, y in ((int(a), int(b)), (int(b), int(c))):
            root_x, root_y = find(x), find(y)
            if root_x != root_y:
                parent[root_x] = root_y

    groups: dict[int, list[int]] = {}
    for index, face in enumerate(faces):
        groups.setdefault(find(int(face[0])), []).append(index)
    tris = np.asarray(triangles, dtype=np.float64)
    return [tris[np.asarray(indices)] for indices in groups.values()]


def assert_watertight_shells(triangles: Triangles, name: str = "mesh") -> list[Topology]:
    """Assert every connected body in `triangles` is a closed genus-0 solid.

    Use for layers that legitimately consist of several bodies (lakes, sea
    pockets); `assert_watertight` is the stricter single-body check.
    """
    shells = split_shells(triangles)
    assert shells, f"{name}: no geometry"
    return [assert_watertight(shell, f"{name}[shell {i}]") for i, shell in enumerate(shells)]


def parse_obj_groups(path: Path | str) -> dict[str, Triangles]:
    """Read an exported OBJ back into {group name: (N, 3, 3) triangles} in model space.

    The exporter writes `v x z y` (OBJ is Y-up, the model frame is Z-up) and
    reverses each face's corner order to go with it, because swapping two axes
    mirrors the frame. Both are undone here, so what comes back is exactly the
    mesh the generator built and its volume sign means the same thing as for an
    in-memory `Mesh`. Use `parse_obj_groups_raw` to judge the file as a slicer
    sees it.
    """
    groups = {}
    for name, tris in parse_obj_groups_raw(path).items():
        model_space = tris[:, :, [0, 2, 1]]  # x, y (depth), z (up)
        groups[name] = model_space[:, ::-1, :]  # undo the mirrored winding
    return groups


def parse_obj_groups_raw(path: Path | str) -> dict[str, Triangles]:
    """Read an exported OBJ verbatim: OBJ coordinates, OBJ winding.

    This is what a slicer or viewer actually loads — Y up, Z pointing south,
    right-handed — so a positive volume here means the normals really do face
    out in the delivered file.
    """
    vertices: list[tuple[float, float, float]] = []
    groups: dict[str, list[list[int]]] = {}
    current = "default"
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append(tuple(float(v) for v in line.split()[1:4]))  # type: ignore[arg-type]
        elif line.startswith("g "):
            current = line.split(maxsplit=1)[1].strip()
        elif line.startswith("f "):
            corners = [int(token.split("/")[0]) - 1 for token in line.split()[1:4]]
            groups.setdefault(current, []).append(corners)

    vertex_array = np.array(vertices, dtype=np.float64)
    return {name: vertex_array[np.array(faces, dtype=np.int64)] for name, faces in groups.items()}


def unit_cube_triangles(missing: Iterable[int] = ()) -> Triangles:
    """A closed unit cube as 12 triangles, minus the face indices in `missing`.

    Used to prove the checkers above actually fire — a watertightness assert
    that passes on a broken mesh is worse than no assert at all.
    """
    corners = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    quads = [
        (0, 3, 2, 1),  # bottom (normal -Z)
        (4, 5, 6, 7),  # top
        (0, 1, 5, 4),  # front
        (1, 2, 6, 5),  # right
        (2, 3, 7, 6),  # back
        (3, 0, 4, 7),  # left
    ]
    skip = set(missing)
    faces = []
    for index, (a, b, c, d) in enumerate(quads):
        if index in skip:
            continue
        faces.append([corners[a], corners[b], corners[c]])
        faces.append([corners[a], corners[c], corners[d]])
    return np.array(faces, dtype=np.float64)
