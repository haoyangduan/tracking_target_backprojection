"""Car-mesh loading, unit scaling, decimation/caching, and raycasting.

The real CAD mesh (e.g. projection_tracking_targets' BIP_Flat.stl) is
authored in meters and can be tens of millions of triangles -- far too
heavy to raycast/view interactively on every run. This module scales the
mesh into the same millimeter frame used by the tracking-target points3d
file and the PnP-solved camera pose, then decimates and caches the result
next to the source file so repeat runs are fast.

Vertex normals use crease-angle-aware smoothing (`_compute_crease_aware_
normals`), not `mesh.compute_vertex_normals()`, to avoid the faceted/
"broken-looking" body this project's Open3D viewer used to show. Ported
verbatim from `projection_tracking_targets - Copy/prepare_web_mesh.py`,
where this was already solved and verified (see that file's own docstring
for the debugging history -- decimation, skipping decimation, and Taubin
smoothing were all tried and rejected first): Open3D's
`compute_vertex_normals()` unconditionally averages every face touching a
vertex, including across a real design crease/panel fold, which blends
two genuinely different surfaces into one wrong normal and shows up as
visible banding. Professional viewers (Babylon Sandbox, Microsoft 3D
Viewer, Blender's "shade auto smooth") only average faces within a small
angle of each other and keep a real crease's two sides as separate
normals -- exactly what `_compute_crease_aware_normals` does, at the cost
of splitting a vertex wherever a real crease meets it (more vertices, but
the mesh's actual triangle geometry -- and therefore raycasting/backproject
results -- is unaffected; only vertex-normal-based shading changes).
"""

import hashlib
import os

import numpy as np
import open3d as o3d

CREASE_ANGLE_DEG = 40.0  # faces within this angle of each other are smoothed together

# Bump this whenever _fix_triangle_winding/_compute_crease_aware_normals's
# algorithm changes in a way that changes its output for the same mesh/
# angle -- it's folded into the mesh cache key so an algorithm change can't
# silently keep serving an old, differently-computed cached mesh.
_NORMALS_ALGO_VERSION = 7


def _fix_triangle_winding(tris: np.ndarray, max_repair_passes: int = 30) -> np.ndarray:
    """Makes triangle winding consistent (a shared edge is traversed in
    opposite directions by the triangles on either side of it, as a properly
    wound manifold mesh requires), and returns a new `tris` array (vertex
    positions and triangle count are unchanged -- only a triangle's own
    3 indices are sometimes reordered).

    Why this exists: `mesh.simplify_quadric_decimation()` leaves this
    project's decimated BIP_Flat.stl with widespread inconsistent triangle
    winding -- measured directly, ~12% of shared edges (25,640 of 214,068)
    had their two triangles wound in agreeing, not opposite, directions. A
    naive per-face-normal crease-angle comparison (see
    `_compute_crease_aware_normals`) cannot tell such a flipped-winding
    duplicate apart from a genuine near-180-degree design crease -- both
    show up as two adjacent face normals pointing in near-opposite
    directions. Only a topological check (this function; unrelated to any
    angle) can tell them apart, which is why this runs as its own,
    angle-independent repair pass before crease detection, rather than
    inside it. `Open3D`'s own `orient_triangles()` cannot substitute for
    this: it requires the whole mesh to be globally orientable, and this
    decimated mesh is not (`is_orientable()` is False even after
    `remove_non_manifold_edges()`).

    This repair is deliberately scoped to edges with *exactly* 2 adjacent
    triangles: a proper 2-manifold edge has no legitimate reason for its two
    triangles to agree in direction, so "same direction" there is always a
    genuine bug. An edge shared by 3+ triangles is NOT necessarily a bug,
    though -- measured directly, 11% of this project's real decimated
    BIW-sheet-metal edges (55,854 of 510,854) touch more than 2 triangles,
    which is far too many to be decimation artifacts; they're real
    non-manifold seams (spot-welded flanges, hems, a rib meeting the outer
    skin), where it is entirely normal for two of the three-plus triangles
    to share a direction (e.g. two co-planar-ish triangles continuing the
    outer surface, one interior flange folding the other way). An earlier
    version of this function tried to force ALL same-direction pairs to
    disagree, everywhere; measured directly, that left 41,451 pairs it
    still couldn't resolve (proof by counterexample that the assumption was
    wrong) and, worse, would have been happy to flip perfectly correct
    triangles at a genuine junction just because they legitimately share a
    neighbor's direction. Leave 3+-triangle edges alone.

    Two passes over 2-triangle edges only:

    1. A flood fill, which fixes every edge it traverses. Fast, and
       correctly resolves the vast majority of the mesh in one pass -- but
       a BFS/DFS only walks a *spanning tree* of each connected component
       and, by construction, never re-examines an edge between two
       already-visited triangles. Any bolt hole, wiring grommet, or other
       topological handle in the sheet metal closes a loop in that
       component's dual graph (using only its 2-triangle edges), and the
       edge that closes the loop is exactly such an edge: silently never
       checked, so it can be left inconsistent even after the flood fill.

    2. An iterative repair pass: each round flips one side of every
       still-inconsistent 2-triangle edge, which converges to 0 in a few
       rounds for any even-length loop (the vast majority in practice) --
       only a genuinely non-orientable pocket, not expected from decimating
       a real manifold-ish car body, could leave a single edge irreducibly
       bad. Measured directly on this project's real decimated mesh: pass 1
       alone left 642 of 205,882 (0.3%) 2-triangle edges inconsistent, each
       one an isolated wrong-winding (and so wrong-normal) triangle --
       invisible-ish under Open3D's flat fixed-function shading, but a
       stark dark/black patch under a real PBR renderer's image-based
       lighting, which reflects it from the wrong side. Pass 2 drives this
       to exactly 0.

    Performance note: on the full (undecimated, ~6.2M-triangle) mesh this
    used to take ~70s, almost entirely pure-Python dict/loop overhead (one
    dict entry appended per of the 3*n_tris directed edges, then a
    Python-level `_consistent()` call -- itself rebuilding 2 tuples via 6
    `int()` conversions -- for every one of the ~7.2M 2-triangle edges, on
    *every* repair pass even though only pass 1 usually finds anything).
    None of that per-edge work depends on control flow, so it's replaced
    below with vectorized numpy: `key`/argsort/`np.unique` group the 3*n_tris
    directed edges into 2-triangle groups in one pass (no dict), and each
    triangle's current orientation is tracked as a single `flipped` bool
    rather than by re-deriving its edges from a mutated `tris` array --
    letting the repair-pass "which edges are still bad" check run as one
    array comparison over all edges instead of a Python loop calling
    `_consistent()` per edge. Only the BFS itself stays a Python loop (graph
    traversal doesn't vectorize), but each step is now an integer/bool
    comparison instead of tuple-rebuilding. Output is unchanged: both stages
    still only ever look at edges with exactly 2 triangles, and the result
    is still verified to reach 0 inconsistent among them.
    """
    n_tris = len(tris)
    tris = tris.copy()

    # Every triangle's 3 directed edges, flattened to (3*n_tris,) arrays,
    # in the same (face 0's 3 edges, then face 1's, ...) order the old
    # per-triangle loop produced.
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    edge_u = np.empty(3 * n_tris, dtype=np.int64)
    edge_v = np.empty(3 * n_tris, dtype=np.int64)
    edge_u[0::3], edge_v[0::3] = a, b
    edge_u[1::3], edge_v[1::3] = b, c
    edge_u[2::3], edge_v[2::3] = c, a
    face_of_edge = np.repeat(np.arange(n_tris), 3)

    lo = np.minimum(edge_u, edge_v)
    hi = np.maximum(edge_u, edge_v)
    # This triangle's own directed edge, relative to the (lo, hi) convention:
    # +1 if it already runs lo->hi, -1 if hi->lo (never 0 -- no degenerate
    # triangles at this point, so lo != hi always).
    orig_sign = np.where(edge_u == lo, np.int8(1), np.int8(-1))

    n_verts = int(tris.max()) + 1
    key = lo * np.int64(n_verts) + hi

    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    face_sorted = face_of_edge[order]
    sign_sorted = orig_sign[order]

    group_start = np.flatnonzero(np.concatenate(([True], key_sorted[1:] != key_sorted[:-1])))
    group_size = np.diff(np.append(group_start, len(order)))

    two_start = group_start[group_size == 2]
    f1_arr = face_sorted[two_start]
    f2_arr = face_sorted[two_start + 1]
    s1_arr = sign_sorted[two_start]
    s2_arr = sign_sorted[two_start + 1]

    # `flipped[f]`: whether triangle f's current winding is reversed from
    # the `tris` snapshot `orig_sign` was computed from. Flipping a
    # triangle reverses its direction on all 3 of its edges at once, so
    # this one bool per face is all the bookkeeping needed -- nothing below
    # re-derives edges from `tris` again until the final materialization.
    flipped = np.zeros(n_tris, dtype=bool)

    # ---- Phase 1: flood fill over strict 2-triangle edges ----
    # Per-face adjacency as (neighbor, this_face's_sign, neighbor's_sign) so
    # a BFS step is pure integer comparison, no edge-pair reconstruction.
    adjacency = [[] for _ in range(n_tris)]
    for f1, f2, s1, s2 in zip(f1_arr.tolist(), f2_arr.tolist(), s1_arr.tolist(), s2_arr.tolist()):
        adjacency[f1].append((f2, s1, s2))
        adjacency[f2].append((f1, s2, s1))

    visited = np.zeros(n_tris, dtype=bool)
    queue = []
    for start in range(n_tris):
        if visited[start]:
            continue
        visited[start] = True
        queue.append(start)
        while queue:
            f = queue.pop()
            f_sign = -1 if flipped[f] else 1
            for nbr, s_self, s_nbr in adjacency[f]:
                if visited[nbr]:
                    continue
                if (s_self * f_sign) == s_nbr:  # same effective direction = inconsistent
                    flipped[nbr] = True
                visited[nbr] = True
                queue.append(nbr)

    # ---- Phase 2: iterative repair over every 2-triangle edge ----
    for _ in range(max_repair_passes):
        eff1 = np.where(flipped[f1_arr], -s1_arr, s1_arr)
        eff2 = np.where(flipped[f2_arr], -s2_arr, s2_arr)
        bad_mask = eff1 == eff2
        if not bad_mask.any():
            break
        flipped_this_pass = set()
        for f1, f2 in zip(f1_arr[bad_mask].tolist(), f2_arr[bad_mask].tolist()):
            target = f2 if f2 not in flipped_this_pass else (f1 if f1 not in flipped_this_pass else None)
            if target is not None:
                flipped[target] = not flipped[target]
                flipped_this_pass.add(target)

    flip_idx = np.flatnonzero(flipped)
    tris[flip_idx] = tris[flip_idx][:, [0, 2, 1]]
    return tris


def _compute_crease_aware_normals(verts: np.ndarray, tris: np.ndarray):
    """Returns (out_positions, out_normals, out_triangles) where vertices
    have been split wherever a real crease (angle >= CREASE_ANGLE_DEG)
    exists, so curved regions stay smooth while genuine hard edges (panel
    folds, sharp character lines) stay crisp instead of being blended
    across. See this module's docstring for why this replaces
    `mesh.compute_vertex_normals()`. Expects `tris` to already have
    consistent winding (see `_fix_triangle_winding`) -- run on this
    project's decimated mesh without that repair, a flipped-winding
    decimation artifact is indistinguishable from a genuine near-180-degree
    crease by angle alone, and gets (wrongly) treated as one.
    """
    n_tris = len(tris)
    crease_cos = np.cos(np.radians(CREASE_ANGLE_DEG))

    # Per-face normals.
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_areas = np.linalg.norm(face_normals, axis=1)
    face_normals = face_normals / np.maximum(face_areas[:, None], 1e-12)

    # Flatten to (vertex, triangle, corner-column) triples, sorted by vertex
    # so each vertex's adjacent corners are contiguous.
    tris_flat = tris.reshape(-1)
    tri_idx_flat = np.repeat(np.arange(n_tris), 3)
    col_flat = np.tile(np.arange(3), n_tris)
    order = np.argsort(tris_flat, kind="stable")
    sorted_verts = tris_flat[order]
    sorted_tri = tri_idx_flat[order]
    sorted_col = col_flat[order]
    unique_verts, group_start = np.unique(sorted_verts, return_index=True)
    group_start = np.append(group_start, len(sorted_verts))

    # Per-corner smoothed normal: average only the adjacent faces within
    # CREASE_ANGLE_DEG of this corner's own face (a small dense matmul per
    # vertex group; groups are small -- mean valence ~6-7 -- so this is
    # fast; the earlier, much slower approach called np.unique once per
    # vertex instead of once globally here).
    n_corners = len(sorted_tri)
    corner_normals = np.zeros((n_corners, 3), dtype=np.float64)
    for gi in range(len(unique_verts)):
        s, e = group_start[gi], group_start[gi + 1]
        tri_ids = sorted_tri[s:e]
        fn = face_normals[tri_ids]
        if len(tri_ids) == 1:
            corner_normals[s] = fn[0]
            continue
        sims = fn @ fn.T
        mask = sims >= crease_cos
        summed = mask.astype(np.float64) @ fn
        lens = np.linalg.norm(summed, axis=1, keepdims=True)
        lens[lens < 1e-12] = 1.0
        corner_normals[s:e] = summed / lens

    # Merge corners that ended up with (effectively) the same normal at the
    # same original vertex back into one output vertex, so smooth regions
    # keep sharing vertices and only real creases cause a split.
    quant = np.round(corner_normals * 500).astype(np.int64)  # ~0.002 precision
    combined = np.column_stack([sorted_verts.astype(np.int64), quant])
    _, inverse, counts = np.unique(combined, axis=0, return_inverse=True, return_counts=True)
    n_out = len(counts)

    sums = np.zeros((n_out, 3), dtype=np.float64)
    np.add.at(sums, inverse, corner_normals)
    out_normals = sums / counts[:, None]
    out_lens = np.linalg.norm(out_normals, axis=1, keepdims=True)
    out_lens[out_lens < 1e-12] = 1.0
    out_normals = (out_normals / out_lens).astype(np.float64)

    # out_positions: each output vertex's source position is the position of
    # ANY corner that maps to it -- take the first occurrence per group.
    order_by_out = np.argsort(inverse, kind="stable")
    inv_sorted = inverse[order_by_out]
    first_positions = np.searchsorted(inv_sorted, np.arange(n_out))
    first_idx = sorted_verts[order_by_out[first_positions]]
    out_positions = verts[first_idx].astype(np.float64)

    new_tris = np.zeros((n_tris, 3), dtype=np.int64)
    new_tris[sorted_tri, sorted_col] = inverse

    return out_positions, out_normals, new_tris


def _cache_key(mesh_path: str, target_triangles: int, scale: float) -> str:
    stat = os.stat(mesh_path)
    raw = (f"{os.path.abspath(mesh_path)}|{stat.st_size}|{stat.st_mtime_ns}|"
           f"{target_triangles}|{scale}|crease{CREASE_ANGLE_DEG:.0f}|"
           f"normalsv{_NORMALS_ALGO_VERSION}")
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def load_car_mesh(
    mesh_path: str,
    cache_dir: str = None,
    target_triangles: int = 300_000,
    scale: float = 1000.0,
) -> o3d.geometry.TriangleMesh:
    """Load a car mesh, scale it into millimeters, and decimate it to
    `target_triangles` (skipped if the mesh is already smaller). The
    decimated, scaled result is cached as a binary PLY next to the source
    file (or under `cache_dir`) so subsequent loads are near-instant.

    Args:
        mesh_path: path to the source mesh (STL/PLY/OBJ/...).
        cache_dir: directory for the cache file; defaults to a `cache/`
            folder next to `mesh_path`.
        target_triangles: decimation target triangle count. Pass 0 to
            disable decimation (load at full resolution).
        scale: uniform scale applied to the source mesh's vertices to
            convert its native units into millimeters (e.g. 1000.0 for a
            mesh authored in meters; 1.0 if already in millimeters).
    """
    mesh_path = str(mesh_path)
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(mesh_path)), "cache")
    os.makedirs(cache_dir, exist_ok=True)

    key = _cache_key(mesh_path, target_triangles, scale)
    cache_path = os.path.join(cache_dir, f"{os.path.splitext(os.path.basename(mesh_path))[0]}_{key}.ply")

    if os.path.isfile(cache_path):
        mesh = o3d.io.read_triangle_mesh(cache_path)
        if mesh is not None and len(mesh.vertices) > 0:
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            return mesh

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh is None or len(mesh.vertices) == 0:
        raise RuntimeError(f"Could not read mesh: {mesh_path}")

    if scale != 1.0:
        mesh.scale(scale, center=(0.0, 0.0, 0.0))

    n_tri = len(mesh.triangles)
    if target_triangles and n_tri > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    # Crease-aware normals split some vertices on purpose (see module
    # docstring) -- do NOT run any further dedup/cleanup after this, or
    # it would merge those deliberately-separate corners back together.
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    tris = _fix_triangle_winding(tris)
    out_positions, out_normals, out_tris = _compute_crease_aware_normals(verts, tris)

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(out_positions),
        o3d.utility.Vector3iVector(out_tris),
    )
    mesh.vertex_normals = o3d.utility.Vector3dVector(out_normals)

    o3d.io.write_triangle_mesh(cache_path, mesh)
    return mesh


def apply_normal_colors(mesh: o3d.geometry.TriangleMesh) -> None:
    normals = np.asarray(mesh.vertex_normals)
    colors = np.clip((normals + 1.0) * 0.5, 0.0, 1.0)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)


def build_raycasting_scene(mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def cast_rays(scene: o3d.t.geometry.RaycastingScene, origins: np.ndarray, directions: np.ndarray):
    """Batch ray cast. origins/directions: (N, 3) float arrays.

    Returns (t_hit, hit_points): t_hit is (N,) float32 (inf if no hit),
    hit_points is (N, 3) float64 (NaN rows where there was no hit).
    """
    origins = np.asarray(origins, dtype=np.float32).reshape(-1, 3)
    directions = np.asarray(directions, dtype=np.float32).reshape(-1, 3)
    rays = np.hstack((origins, directions)).astype(np.float32)
    ray_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(ray_tensor)
    t_hit = ans["t_hit"].numpy()
    hit_points = origins.astype(np.float64) + t_hit[:, None].astype(np.float64) * directions.astype(np.float64)
    hit_points[~np.isfinite(t_hit)] = np.nan
    return t_hit, hit_points
