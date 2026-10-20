"""Build automatic ground-truth labels for ALL VO-covered frames using the
COMPLETE 161-point tracking model (with doors and hood), paired with the matching
FULL mesh (0_NE_70_SIMPLIFICATION.stl, also with doors/hood) -- the natural
next step after build_oracle_labels.py's door-free 90-point version.

Key simplification over build_oracle_labels.py: that script needed a
separate "door safe-gate" (reproject through a SECOND, full mesh) because
its primary occlusion check ran against the door-FREE mesh, which has no
door geometry to block anything. Here, points3d and mesh are both already
door/hood-INCLUSIVE and mutually consistent (new_car_300926_whood.txt
was fit directly onto 0_NE_70_SIMPLIFICATION.stl's surface), so a single
occlusion raycast against this one mesh already accounts for occlusion
correctly: a point hidden behind a closed door or hood has its
line-of-sight blocked by the mesh geometry, which IS present in
this mesh, so the ray's first hit lands closer than the known point's own
distance and the standard occlusion check already flags it -- no second
mesh or reprojection trick needed.

points3d rows 1-90 are legacy-aligned (see reindex_new_car_to_legacy_
order.py) -- point_line 1-90 here names the exact same physical sticker
as the door-free model's point_line 1-90, so results are directly
comparable/validatable against the same known frame_000052/000060 ground
truth the door-free version was checked against. Rows 91-138 are the
48 door-region-and-beyond points, and rows 139-161 are the 23 hood
targets.

Output: oracle_labels_full.json -- {frame_name: [{point_line, u, v}, ...]}
"""
import json
import os

import cv2
import numpy as np

from live_backprojection import _pathsetup  # noqa: F401

from calib_target.pnp_io import load_camera_intrinsics_any
from single_frame_backprojection.vo_pose import compute_camera_pose_from_vo_track
from single_frame_backprojection.mesh_utils import load_car_mesh, build_raycasting_scene, cast_rays

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_THIS_DIR, "out")
_DATA_DIR = os.path.join(_pathsetup.PROJECT_ROOT, "data")

VO_TRACK = os.path.join(_DATA_DIR, "track_cam_test_cad-NE_pair-both_cam-both_mode-vo_det-aliked_frames-0-180_ref-0-511_sc-0.25_st-2_hc71c45f4b1.json")
POINTS3D = os.path.join(_DATA_DIR, "new_car_300926_whood.txt")  # final tracking model (with hood)
MESH = os.path.join(_DATA_DIR, "mesh", "0_NE_70_SIMPLIFICATION.stl")  # full mesh, WITH doors -- matches points3d above
CAMERA_NAME = "hik_DA0179014"
IMAGE_SIZE = (4096, 3000)
OCCLUSION_TOL_MM = 15.0
OUT_PATH = os.path.join(_OUT_DIR, "oracle_labels_full.json")

intr = load_camera_intrinsics_any(os.path.join(_DATA_DIR, "intrinsics_sensors_0922.json"), CAMERA_NAME)
raw = np.loadtxt(POINTS3D)
points_mm = raw[:, :3]

print(f"Loaded {len(points_mm)} points from {POINTS3D}")
print("Loading full mesh (with doors) + building raycasting scene (full resolution)...")
mesh = load_car_mesh(MESH, target_triangles=0, scale=1.0)
scene = build_raycasting_scene(mesh)


def expected_visible_pixels(frame_name, return_debug=False):
    pose = compute_camera_pose_from_vo_track(VO_TRACK, frame_name, camera_name=CAMERA_NAME)
    cam_pos = pose.camera_position_world
    T_cam_world = pose.T_cam_world

    rvec, _ = cv2.Rodrigues(T_cam_world[:3, :3])
    tvec = T_cam_world[:3, 3]
    img_pts, _ = cv2.projectPoints(points_mm.reshape(-1, 1, 3), rvec, tvec, intr.camera_matrix, intr.dist_coeffs)
    img_pts = img_pts.reshape(-1, 2)

    view_dirs = points_mm - cam_pos
    dists = np.linalg.norm(view_dirs, axis=1)
    view_dirs_n = view_dirs / dists[:, None]

    origins = np.tile(cam_pos, (len(points_mm), 1))
    t_hit, _ = cast_rays(scene, origins, view_dirs_n)
    hit_dist_diff = np.abs(t_hit - dists)

    w, h = IMAGE_SIZE
    results, debug_rows = [], []
    for i in range(len(points_mm)):
        u, v = img_pts[i]
        in_bounds = (0 <= u < w) and (0 <= v < h)
        unoccluded = np.isfinite(t_hit[i]) and hit_dist_diff[i] < OCCLUSION_TOL_MM
        visible = in_bounds and unoccluded
        if visible:
            results.append({"point_line": i + 1, "u": float(u), "v": float(v)})
        if return_debug:
            debug_rows.append({"point_line": i + 1, "u": float(u), "v": float(v),
                                "in_bounds": in_bounds, "unoccluded": unoccluded, "visible": visible,
                                "hit_dist_diff_mm": float(hit_dist_diff[i]) if np.isfinite(t_hit[i]) else None})
    if return_debug:
        return results, debug_rows
    return results


def main():
    labels = {}
    skipped = []
    for idx in range(181):
        frame_name = f"frame_{idx + 1:06d}.png"
        img_path = os.path.join(_DATA_DIR, "hik_DA0179014", frame_name)
        if not os.path.isfile(img_path):
            skipped.append(frame_name)
            continue
        try:
            labels[frame_name] = expected_visible_pixels(frame_name)
        except KeyError:
            skipped.append(frame_name)

    with open(OUT_PATH, "w") as f:
        json.dump(labels, f, indent=1)

    counts = [len(v) for v in labels.values()]
    legacy_counts = [sum(1 for e in v if e["point_line"] <= 90) for v in labels.values()]
    door_counts = [sum(1 for e in v if 90 < e["point_line"] <= 138) for v in labels.values()]
    hood_counts = [sum(1 for e in v if e["point_line"] > 138) for v in labels.values()]
    print(f"\nBuilt oracle labels (final 161-pt model) for {len(labels)} frames -> {OUT_PATH}")
    print(f"Skipped: {len(skipped)}")
    print(f"Visible points per frame (all 161): min={min(counts)} max={max(counts)} mean={np.mean(counts):.1f} median={np.median(counts):.1f}")
    print(f"  of which legacy BIW (1-90): mean={np.mean(legacy_counts):.1f}")
    print(f"  of which doors/sides (91-138): mean={np.mean(door_counts):.1f}")
    print(f"  of which hood (139-161): mean={np.mean(hood_counts):.1f}")


if __name__ == "__main__":
    main()
