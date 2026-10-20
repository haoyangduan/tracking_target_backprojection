"""Build automatic ground-truth labels for ALL VO-covered frames (0-180,
i.e. frame_000001.png..frame_000181.png) by projecting the door-free
90-point tracking model through each frame's VO pose and culling to only
the points actually visible (in image bounds + unoccluded via raycast
against the door-free mesh).

This is a TUNING/DEVELOPMENT tool, not part of the live detector: it lets
Claude evaluate and improve the threshold-based detector's parameters
against ~181 frames' worth of real, varied-lighting data instead of the 8
frames that were manually hand-labeled -- without requiring any new manual
labeling. The deployed detector itself still never references the 3D
model or any pose at runtime; this script's output is only ever compared
against the detector's output for grading, never fed into the detector.

Visibility test: in-bounds + occlusion raycast only -- NOT a surface-normal
back-face cull, which was tried and found unreliable (verified: the
normals in stickers_on_car_surface_mm.txt are inconsistently oriented,
a known triangle-winding issue with the underlying mesh -- only ~33% point
outward from even a crude centroid heuristic, and sanity-checking against
real known-visible points showed most getting the wrong front-facing
verdict). Occlusion raycasting alone doesn't depend on normal direction
and was verified instead to get 21/21 known-real points right on
frame_000052/frame_000060, with the same ~1-24px positional accuracy VO's
own validation already established.

DOOR SAFE-GATE: the occlusion raycast above runs against BIP_Flat.stl,
which is door-free -- it structurally cannot detect a point being hidden
behind a closed door, because there's no door geometry there to block
anything; the ray just sails through empty space where the door would be
and hits the correct body point behind it, wrongly reporting "visible."
A real photo shows the assembled car with doors on, so a door-free
occlusion check alone is not trustworthy for any point near a door.

Fix (per direct instruction): for every point, after projecting it to its
predicted pixel (u, v), REPROJECT that same pixel back into 3D -- this
time against the FULL mesh WITH doors (0_NE_70_SIMPLIFICATION.stl) -- via
`pixel_to_world_ray` + a raycast. If that hit point lands more than
DOOR_TOL_MM away from the original known 3D point, the full mesh's actual
visible surface at that pixel is something else entirely (almost always
the door's outer skin, which sits in front of the real target once the
door is shut) -- the target is blocked by the door and must not be
trusted as visible, regardless of what the door-free check said. A point
only counts as visible if it passes BOTH checks.

Output: oracle_labels.json -- {frame_name: [{point_line, u, v}, ...]}
(only entries with visible=True are kept).
"""
import json
import os

import cv2
import numpy as np

from live_backprojection import _pathsetup  # noqa: F401

from calib_target.pnp_io import load_camera_intrinsics_any
from single_frame_backprojection.vo_pose import compute_camera_pose_from_vo_track
from single_frame_backprojection.camera_model import pixel_to_world_ray
from single_frame_backprojection.mesh_utils import load_car_mesh, build_raycasting_scene, cast_rays

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_THIS_DIR, "out")
_DATA_DIR = os.path.join(_pathsetup.PROJECT_ROOT, "data")

VO_TRACK = os.path.join(_DATA_DIR, "track_cam_test_cad-NE_pair-both_cam-both_mode-vo_det-aliked_frames-0-180_ref-0-511_sc-0.25_st-2_hc71c45f4b1.json")
POINTS3D = os.path.join(_DATA_DIR, "points3d", "stickers_on_car_surface_mm.txt")  # door-free, per user's request
MESH_DOOR_FREE = os.path.join(_DATA_DIR, "mesh", "BIP_Flat.stl")
MESH_FULL = os.path.join(_DATA_DIR, "mesh", "0_NE_70_SIMPLIFICATION.stl")  # WITH doors -- for the door safe-gate only
CAMERA_NAME = "hik_DA0179014"
IMAGE_SIZE = (4096, 3000)
OCCLUSION_TOL_MM = 15.0
DOOR_TOL_MM = 20.0
OUT_PATH = os.path.join(_OUT_DIR, "oracle_labels.json")

intr = load_camera_intrinsics_any(os.path.join(_DATA_DIR, "intrinsics_sensors_0922.json"), CAMERA_NAME)
raw = np.loadtxt(POINTS3D)
points_mm = raw[:, :3]

print("Loading door-free mesh + building raycasting scene (full resolution)...")
mesh_door_free = load_car_mesh(MESH_DOOR_FREE, target_triangles=0, scale=1000.0)
scene_door_free = build_raycasting_scene(mesh_door_free)

print("Loading FULL mesh (with doors) + building raycasting scene, for the door safe-gate...")
mesh_full = load_car_mesh(MESH_FULL, target_triangles=0, scale=1.0)
scene_full = build_raycasting_scene(mesh_full)


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
    t_hit, _ = cast_rays(scene_door_free, origins, view_dirs_n)
    hit_dist_diff = np.abs(t_hit - dists)

    w, h = IMAGE_SIZE
    results = []
    debug_rows = []
    for i in range(len(points_mm)):
        u, v = img_pts[i]
        in_bounds = (0 <= u < w) and (0 <= v < h)
        unoccluded_door_free = np.isfinite(t_hit[i]) and hit_dist_diff[i] < OCCLUSION_TOL_MM

        # Door safe-gate: reproject the predicted pixel through the FULL
        # (with-door) mesh and compare to the known 3D point. Only
        # evaluated when in-bounds (no point projecting off-frame anyway).
        door_blocked = False
        door_hit_dist = None
        if in_bounds:
            o2, d2 = pixel_to_world_ray(u, v, intr, pose)
            t_hit2, hit_pts2 = cast_rays(scene_full, o2, d2)
            if np.isfinite(t_hit2[0]):
                door_hit_dist = float(np.linalg.norm(hit_pts2[0] - points_mm[i]))
                door_blocked = door_hit_dist > DOOR_TOL_MM
            else:
                door_blocked = True  # no hit at all against the full mesh -- don't trust it either

        visible = in_bounds and unoccluded_door_free and not door_blocked
        if visible:
            results.append({"point_line": i + 1, "u": float(u), "v": float(v)})
        if return_debug:
            debug_rows.append({
                "point_line": i + 1, "u": float(u), "v": float(v), "in_bounds": in_bounds,
                "unoccluded_door_free": unoccluded_door_free, "door_blocked": door_blocked,
                "door_hit_dist_mm": door_hit_dist, "visible": visible,
            })
    if return_debug:
        return results, debug_rows
    return results


def main():
    labels = {}
    skipped = []
    total_in_bounds_and_door_free_unocc = 0
    total_door_blocked = 0
    for idx in range(181):  # VO covers 0-180
        frame_name = f"frame_{idx + 1:06d}.png"
        img_path = os.path.join(_DATA_DIR, "hik_DA0179014", frame_name)
        if not os.path.isfile(img_path):
            skipped.append(frame_name)
            continue
        try:
            visible, debug = expected_visible_pixels(frame_name, return_debug=True)
        except KeyError:
            skipped.append(frame_name)
            continue
        labels[frame_name] = visible
        for d in debug:
            if d["in_bounds"] and d["unoccluded_door_free"]:
                total_in_bounds_and_door_free_unocc += 1
                if d["door_blocked"]:
                    total_door_blocked += 1

    with open(OUT_PATH, "w") as f:
        json.dump(labels, f, indent=1)

    counts = [len(v) for v in labels.values()]
    print(f"Built oracle labels for {len(labels)} frames -> {OUT_PATH}")
    print(f"Skipped (no image or no VO pose): {len(skipped)}")
    print(f"Visible points per frame: min={min(counts)} max={max(counts)} mean={np.mean(counts):.1f} median={np.median(counts):.1f}")
    print(f"\nDoor safe-gate impact: of {total_in_bounds_and_door_free_unocc} point-instances the door-free check "
          f"alone called 'visible', {total_door_blocked} ({100*total_door_blocked/max(total_in_bounds_and_door_free_unocc,1):.1f}%) "
          f"were caught as actually door-blocked and excluded.")


if __name__ == "__main__":
    main()
