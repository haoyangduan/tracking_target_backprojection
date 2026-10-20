"""Core workflow: treat surveyed tracking-target stickers as known-GT
"defects" in a real photo, back-project each detected target through a
given (externally-solved) camera pose onto the real car mesh, and measure
the 3D error against its known position.

This is defect_visualization's validation idea (compare an estimated
back-projected point against a known ground truth, in 3D, in millimeters)
ported to real photos:

  - GT source: projection_tracking_targets' surveyed sticker positions
    (points3d file), instead of a single Isaac-Sim-clicked point.
  - Detection: close_track_gui's tracking-target circle detector (or its
    already-computed circle_debug.json), instead of one HSV red blob.
  - Correspondence: close_track_gui's circle_index -> point_line JSON,
    instead of a single manual/auto pixel.
  - Camera pose: an external input (already solved by another pipeline,
    e.g. close_track_gui's PnP), not solved here.
  - Back-projection + raycasting + 3D error: the same ray/mesh-
    intersection method as defect_visualization's red_mark_to_world.py,
    adapted to the real (distorted) OpenCV camera model.

One photo at a time, matching close_track_gui's own single-image PnP/
detection workflow (see calib/extrinsics_pnp.py upstream). Usable both as
a library (`run()`) and as a CLI.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict

import cv2
import numpy as np

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_PIPELINE_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from calib_target.blob_detect import BlobDetectorParams, detect_circles_multi_roi
from calib_target.points3d import load_object_points, get_point_by_line
from calib_target.pnp_io import load_camera_intrinsics_any, load_camera_pose

from single_frame_backprojection.camera_model import pixel_to_world_ray, zero_distortion_intrinsics, RayHit
from single_frame_backprojection.mesh_utils import load_car_mesh, apply_normal_colors, build_raycasting_scene, cast_rays
from single_frame_backprojection.run_logger import RunLogger
from single_frame_backprojection.frames import load_transform, transform_points, transform_pose


# ---------------------------------------------------------------------------
# Loading detected circles / correspondences
# ---------------------------------------------------------------------------

def load_circles_from_debug_json(path: str):
    """Load a close_track_gui circle_debug.json (`{"circles": [{index,x,y,size}, ...]}`).

    Returns ({circle_index: (x, y)}, meta) where `meta["image_undistorted"]`
    is the file's own `image_undistorted` flag (False if absent -- i.e. a
    circle_debug.json from before pipeline/circle_tuner_qt.py's "Undistort
    before detection" checkbox existed, or one saved with it unchecked),
    saying whether `pipeline/circle_tuner_qt.py` ran detection against an
    already-undistorted image (see pipeline/camera_model.undistort_image).
    `run()` uses this to auto-select zero-distortion ray math -- see its
    `image_already_undistorted` parameter.
    """
    with open(path, "r") as f:
        data = json.load(f)
    circles_by_index = {c["index"]: (float(c["x"]), float(c["y"])) for c in data["circles"]}
    meta = {"image_undistorted": bool(data.get("image_undistorted", False))}
    return circles_by_index, meta


def detect_circles_fresh(image_path: str, rois_and_params) -> dict:
    """Re-run circle detection on the image. Returns {circle_index: (x, y)}."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    circles = detect_circles_multi_roi(image, rois_and_params)
    return {c.index: (c.x, c.y) for c in circles}


def load_correspondences(path: str) -> list:
    """Load a close_track_gui correspondences JSON. Returns [(circle_index, point_line), ...]."""
    with open(path, "r") as f:
        data = json.load(f)
    correspondences = data.get("correspondences", [])
    if not correspondences:
        raise ValueError(f"No correspondences found in {path!r}")
    return [(c["circle_index"], c["point_line"]) for c in correspondences]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def backproject_and_compare(intr, pose, object_points, correspondences, circles_by_index, scene,
                             raw_object_points=None) -> list:
    """For each (circle_index, point_line) correspondence, back-project the
    detected pixel through the given pose and raycast onto the mesh;
    compare against the known 3D GT point. Returns list[RayHit].

    `raw_object_points`: optional, same `point_line` indexing as
    `object_points` (line i of one corresponds to line i of the other) --
    e.g. the raw pre-surface-fit Creaform/C-Track measurement for the same
    target. If given, each hit's `raw_target_point` is filled in purely for
    display (it plays no part in the error computation, which is always
    against `object_points`/`gt_point`).
    """
    origins = []
    directions = []
    meta = []  # (circle_index, point_line, u, v, gt_point, raw_point)

    for circle_index, point_line in correspondences:
        if circle_index not in circles_by_index:
            print(f"  WARNING: circle_index={circle_index} not found in detected circles; skipping")
            continue
        u, v = circles_by_index[circle_index]
        gt_point = get_point_by_line(object_points, point_line)
        raw_point = get_point_by_line(raw_object_points, point_line) if raw_object_points is not None else None
        origin, direction = pixel_to_world_ray(u, v, intr, pose)
        origins.append(origin)
        directions.append(direction)
        meta.append((circle_index, point_line, u, v, gt_point, raw_point))

    if not meta:
        return []

    t_hit, hit_points = cast_rays(scene, np.array(origins), np.array(directions))

    hits = []
    for i, (circle_index, point_line, u, v, gt_point, raw_point) in enumerate(meta):
        found = np.isfinite(t_hit[i])
        estimated = hit_points[i] if found else np.array([np.nan, np.nan, np.nan])
        error_mm = float(np.linalg.norm(estimated - gt_point)) if found else float("nan")
        hits.append(RayHit(
            circle_index=circle_index,
            point_line=point_line,
            pixel_u=float(u),
            pixel_v=float(v),
            ray_origin=origins[i],
            ray_direction=directions[i],
            hit=bool(found),
            t_hit=float(t_hit[i]),
            estimated_point=estimated,
            gt_point=gt_point,
            error_mm=error_mm,
            raw_target_point=raw_point,
        ))
    return hits


def summarize(hits: list) -> dict:
    errors = np.array([h.error_mm for h in hits if h.hit], dtype=np.float64)
    n_total = len(hits)
    n_hit = len(errors)
    summary = {
        "num_targets": n_total,
        "num_hit": n_hit,
        "num_missed": n_total - n_hit,
    }
    if n_hit > 0:
        summary.update({
            "error_mm_mean": float(np.mean(errors)),
            "error_mm_rms": float(np.sqrt(np.mean(errors ** 2))),
            "error_mm_median": float(np.median(errors)),
            "error_mm_std": float(np.std(errors)),
            "error_mm_max": float(np.max(errors)),
            "error_mm_min": float(np.min(errors)),
        })
    return summary


def draw_debug_image(image_path: str, hits: list) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    out = image.copy()
    for h in hits:
        center = (int(round(h.pixel_u)), int(round(h.pixel_v)))
        if h.hit:
            t = float(np.clip((h.error_mm - 1.0) / 7.0, 0.0, 1.0))
            color = (0, int(255 * (1 - t)), int(255 * t))  # BGR: green->red
        else:
            color = (128, 128, 128)
        cv2.circle(out, center, 14, color, 2)
        label = f"L{h.point_line}:{h.error_mm:.1f}mm" if h.hit else f"L{h.point_line}:MISS"
        cv2.putText(out, label, (center[0] + 16, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def hit_to_dict(h: RayHit) -> dict:
    d = asdict(h)
    for k in ("ray_origin", "ray_direction", "estimated_point", "gt_point"):
        d[k] = np.asarray(d[k]).tolist()
    d["raw_target_point"] = np.asarray(d["raw_target_point"]).tolist() if d["raw_target_point"] is not None else None
    return d


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(
    intrinsics_path,
    extrinsics_path,
    points3d_path,
    correspondences_path,
    mesh_path,
    image_path=None,
    circles_debug_path=None,
    rois_and_params=None,
    mesh_cache_dir=None,
    mesh_target_triangles=0,
    mesh_scale=1000.0,
    body_to_cad=None,
    raw_points3d_path=None,
    raw_body_to_cad=None,
    camera_name=None,
    image_already_undistorted=None,
    out_json=None,
    out_debug_image=None,
    log_run=True,
):
    """Run the full back-projection-vs-GT comparison. Returns the report dict.

    Circle pixel locations come from `circles_debug_path` (a close_track_gui
    circle_debug.json) if given, otherwise are freshly detected from
    `image_path` using `rois_and_params` (list of (roi, BlobDetectorParams)).

    `body_to_cad`: optional rigid transform (4x4 array, or path to a JSON
    file with a "matrix" key) that re-expresses the points3d file and the
    solved camera pose -- both assumed to share one "body"/raw-measurement
    frame -- into the mesh's CAD frame. Omit if points3d/pose are already
    in the mesh's frame. See pipeline/frames.py.

    `raw_points3d_path`: optional, purely for display -- the same targets'
    raw Creaform/C-Track measurement BEFORE they were nearest-surface-
    projected onto the mesh to produce `points3d_path` (e.g.
    data/points3d/biw_plus_frontrail_90pt_raw_ctrack_mm.txt versus
    data/points3d/stickers_on_car_surface_mm.txt -- same line order/count,
    so `point_line` indexes both identically). Plays no part in the error
    computation; only populates each hit's `raw_target_point` for the 3D
    viewer. `raw_body_to_cad` is the rigid transform (same shape as
    `body_to_cad`) needed to bring it into the mesh's CAD frame -- almost
    always required, since a raw measurement is not already CAD-frame
    (defaults to data/points3d/cad_T_body_mm.json's matrix when omitted but
    `raw_points3d_path` is given).

    `camera_name`: required only if `intrinsics_path` is a multi-camera
    JSON file (e.g. intrinsics_sensors_0922.json) rather than a
    single-camera cv2.FileStorage YAML -- selects which camera's
    intrinsics to load. See calib_target/pnp_io.py.

    `image_already_undistorted`: whether the circle pixel coordinates are
    already in undistorted space (e.g. detected on an image that went
    through `pipeline/circle_tuner_qt.py`'s "Undistort before detection"
    checkbox, or `pipeline/undistort_image.py`). Defaults to `None`, which
    auto-detects from `circles_debug_path`'s own saved `image_undistorted`
    flag (False for older circle_debug.json files that predate that
    checkbox, or ones saved with it unchecked) -- pass `True`/`False`
    explicitly to override, which is the only option when using fresh
    detection (`image_path`/`rois_and_params`) instead of a saved
    circle_debug.json, since there's no file to read the flag from. When
    True, swaps in `camera_model.zero_distortion_intrinsics(intr)` for the
    ray math instead of `intr` itself, so `pixel_to_world_ray` doesn't
    undistort an already-undistorted point a second time.

    `mesh_target_triangles`: defaults to 0 (full resolution, no decimation).
    `simplify_quadric_decimation()` -- previously the default here at
    300,000 -- was measured to leave small gaps/cracks in the outer-skin
    surface at some locations (see mesh_utils.py's own docstring on the
    non-manifold topology it introduces); a back-projection ray that
    happens to pass through one of those gaps sails past the intended
    surface and hits an inner structural panel instead, inflating
    `error_mm` by tens of millimeters for a hit that is actually accurate
    (measured directly on a real run: one target's reported 87.7mm error
    dropped to 2.4mm at full resolution). Raycasting itself is not the
    reason this used to be decimated -- casting rays against the full
    ~6M-triangle mesh takes a fraction of a second once loaded; only the
    one-time load+cleanup+crease-normal pass is slow (a minute or two), and
    that result is cached to disk (see `mesh_utils.load_car_mesh`) so it
    only costs anything the first time a given mesh file is used. Pass a
    positive triangle count here only if you specifically need faster
    *first-time* loads and can tolerate this gap-induced error inflation.
    """
    intr = load_camera_intrinsics_any(intrinsics_path, camera_name)
    pose = load_camera_pose(extrinsics_path)
    object_points = load_object_points(points3d_path)
    correspondences = load_correspondences(correspondences_path)

    if body_to_cad is not None:
        T = load_transform(body_to_cad)
        pose = transform_pose(T, pose)
        object_points = transform_points(T, object_points)

    raw_object_points = None
    if raw_points3d_path is not None:
        raw_object_points = load_object_points(raw_points3d_path)
        default_raw_transform = os.path.join(_PROJECT_ROOT, "data", "points3d", "cad_T_body_mm.json")
        T_raw = load_transform(raw_body_to_cad if raw_body_to_cad is not None else default_raw_transform)
        raw_object_points = transform_points(T_raw, raw_object_points)

    if circles_debug_path:
        circles_by_index, circles_meta = load_circles_from_debug_json(circles_debug_path)
    elif image_path and rois_and_params:
        circles_by_index = detect_circles_fresh(image_path, rois_and_params)
        circles_meta = {}
    else:
        raise ValueError("Must supply either circles_debug_path or (image_path + rois_and_params)")

    if image_already_undistorted is None:
        image_already_undistorted = circles_meta.get("image_undistorted", False)
    if image_already_undistorted:
        intr = zero_distortion_intrinsics(intr)

    mesh = load_car_mesh(mesh_path, cache_dir=mesh_cache_dir,
                          target_triangles=mesh_target_triangles, scale=mesh_scale)
    apply_normal_colors(mesh)
    scene = build_raycasting_scene(mesh)

    hits = backproject_and_compare(intr, pose, object_points, correspondences, circles_by_index, scene,
                                    raw_object_points=raw_object_points)
    summary = summarize(hits)

    print(f"\n=== Tracking-target back-projection: {summary['num_hit']}/{summary['num_targets']} hit ===")
    if summary["num_hit"] > 0:
        print(f"  RMS error    : {summary['error_mm_rms']:.3f} mm")
        print(f"  Mean error   : {summary['error_mm_mean']:.3f} mm")
        print(f"  Median error : {summary['error_mm_median']:.3f} mm")
        print(f"  Max error    : {summary['error_mm_max']:.3f} mm  (worst target below)")
        worst = max((h for h in hits if h.hit), key=lambda h: h.error_mm)
        print(f"    -> circle_index={worst.circle_index} point_line={worst.point_line} "
              f"pixel=({worst.pixel_u:.1f},{worst.pixel_v:.1f}) error={worst.error_mm:.3f}mm")

    report = {
        "inputs": {
            "intrinsics_path": str(intrinsics_path),
            "camera_name": camera_name,
            "extrinsics_path": str(extrinsics_path),
            "points3d_path": str(points3d_path),
            "correspondences_path": str(correspondences_path),
            "mesh_path": str(mesh_path),
            "image_path": str(image_path) if image_path else None,
            "circles_debug_path": str(circles_debug_path) if circles_debug_path else None,
            "image_already_undistorted": bool(image_already_undistorted),
            "body_to_cad": str(body_to_cad) if isinstance(body_to_cad, str) else (
                "inline_matrix" if body_to_cad is not None else None),
        },
        "camera_pose": {
            "camera_name": pose.camera_name,
            "camera_position_world_mm": pose.camera_position_world.tolist(),
            "pnp_rms_reprojection_error_px": pose.rms_reproj_error_px,
            "pnp_num_inliers": pose.num_inliers,
            "pnp_num_points": pose.num_points,
        },
        "summary": summary,
        "targets": [hit_to_dict(h) for h in hits],
    }

    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {out_json}")

    if out_debug_image and image_path:
        debug_img = draw_debug_image(image_path, hits)
        os.makedirs(os.path.dirname(os.path.abspath(out_debug_image)) or ".", exist_ok=True)
        cv2.imwrite(out_debug_image, debug_img)
        print(f"Debug image saved to: {out_debug_image}")

    if log_run:
        logger = RunLogger(script_name="backproject_targets", suffix="target_backprojection")
        logger.log_many(report)
        if image_path:
            logger.set_image(image_path)
        logger.save()

    return report, mesh, hits, pose


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_roi_args(parser):
    g = parser.add_argument_group(
        "fresh detection (only used if --circles-debug is not given)"
    )
    g.add_argument("--roi", nargs=4, type=int, action="append", metavar=("X", "Y", "W", "H"),
                   help="ROI to detect circles in (repeatable for multiple ROIs)")
    g.add_argument("--dark-threshold", type=float, default=100.0)
    g.add_argument("--min-area", type=float, default=10.0)
    g.add_argument("--max-area", type=float, default=20.0)
    g.add_argument("--min-circularity", type=float, default=0.6)
    g.add_argument("--min-aspect-ratio", type=float, default=0.6)
    g.add_argument("--merge-dist-px", type=float, default=0.1)


def main():
    parser = argparse.ArgumentParser(
        description="Back-project surveyed tracking targets through a given camera pose "
                    "and measure 3D error against their known positions (single photo)."
    )
    parser.add_argument("--intrinsics", required=True,
                         help="camera intrinsics: a single-camera cv2.FileStorage YAML, or a "
                              "multi-camera JSON (e.g. intrinsics_sensors_0922.json; requires --camera-name)")
    parser.add_argument("--camera-name", default=None,
                         help="camera name to select within a multi-camera JSON --intrinsics file")
    parser.add_argument("--extrinsics", required=True,
                         help="solved camera pose YAML, from an external PnP pipeline "
                              "(cv2.FileStorage, T_world_cam/T_cam_world)")
    parser.add_argument("--points3d", required=True, help="GT tracking-target points file (mm)")
    parser.add_argument("--correspondences", required=True,
                         help="circle_index -> point_line correspondences JSON")
    parser.add_argument("--mesh", required=True, help="car mesh (e.g. BIP_Flat.stl)")
    parser.add_argument("--mesh-scale", type=float, default=1000.0,
                         help="scale to convert the mesh's native units to millimeters (default 1000 = meters->mm)")
    parser.add_argument("--mesh-target-triangles", type=int, default=0,
                         help="decimation target for raycasting (0 = full resolution, the default -- "
                              "decimating was measured to introduce mesh gaps that inflate error_mm "
                              "for rays that happen to pass through them; see run()'s docstring)")
    parser.add_argument("--mesh-cache-dir", default=None)
    parser.add_argument("--body-to-cad", default=None,
                         help="optional 4x4 rigid-transform JSON aligning the points3d/pose frame "
                              "to the mesh's CAD frame (see data/points3d/cad_T_body_mm.json)")
    parser.add_argument("--raw-points3d", default=None,
                         help="optional, display-only: same targets' raw pre-surface-fit Creaform/"
                              "C-Track measurement (e.g. data/points3d/biw_plus_frontrail_90pt_raw_ctrack_mm.txt); "
                              "shown in the 3D viewer alongside the surface-fit GT and the estimated hit")
    parser.add_argument("--raw-body-to-cad", default=None,
                         help="rigid transform for --raw-points3d into the mesh's CAD frame "
                              "(defaults to data/points3d/cad_T_body_mm.json)")
    parser.add_argument("--image", default=None, help="source photo (for debug image / re-detection / logging)")
    parser.add_argument("--circles-debug", default=None,
                         help="precomputed circle_debug.json (preferred; skips re-detection)")
    parser.add_argument("--image-undistorted", action="store_true", default=None,
                         help="force zero-distortion ray math, for circle coordinates from an "
                              "already-undistorted image (pipeline/circle_tuner_qt.py's checkbox, or "
                              "pipeline/undistort_image.py). Only needed with fresh detection "
                              "(--image/--roi); with --circles-debug this is auto-detected from the "
                              "file's own saved flag, so you normally don't need to pass it")
    parser.add_argument("--out", default=None, help="output report JSON path")
    parser.add_argument("--out-debug-image", default=None, help="output annotated image path")
    parser.add_argument("--view", action="store_true", help="open the Open3D result viewer")
    parser.add_argument("--no-log", action="store_true", help="skip writing a logs/ run folder")
    _add_roi_args(parser)
    args = parser.parse_args()

    rois_and_params = None
    if args.roi:
        params = BlobDetectorParams(
            dark_threshold=args.dark_threshold, min_area=args.min_area, max_area=args.max_area,
            min_circularity=args.min_circularity, min_aspect_ratio=args.min_aspect_ratio,
            merge_dist_px=args.merge_dist_px,
        )
        rois_and_params = [(tuple(r), params) for r in args.roi]

    report, mesh, hits, pose = run(
        intrinsics_path=args.intrinsics,
        extrinsics_path=args.extrinsics,
        points3d_path=args.points3d,
        correspondences_path=args.correspondences,
        mesh_path=args.mesh,
        image_path=args.image,
        circles_debug_path=args.circles_debug,
        rois_and_params=rois_and_params,
        mesh_cache_dir=args.mesh_cache_dir,
        mesh_target_triangles=args.mesh_target_triangles,
        mesh_scale=args.mesh_scale,
        body_to_cad=args.body_to_cad,
        raw_points3d_path=args.raw_points3d,
        raw_body_to_cad=args.raw_body_to_cad,
        camera_name=args.camera_name,
        image_already_undistorted=args.image_undistorted,
        out_json=args.out,
        out_debug_image=args.out_debug_image,
        log_run=not args.no_log,
    )

    if args.view:
        from single_frame_backprojection.web_viewer import show as show_web_viewer
        s = report["summary"]
        if s["num_hit"] > 0:
            summary_line = (
                f"{s['num_hit']}/{s['num_targets']} targets hit | "
                f"RMS={s['error_mm_rms']:.2f}mm mean={s['error_mm_mean']:.2f}mm "
                f"median={s['error_mm_median']:.2f}mm max={s['error_mm_max']:.2f}mm"
            )
        else:
            summary_line = f"0/{s['num_targets']} targets hit -- check pose/points3d/mesh alignment."

        # `mesh` above is already full resolution (mesh_target_triangles
        # defaults to 0) -- reuse it rather than loading a second copy.
        show_web_viewer(
            mesh, hits,
            camera_position=pose.camera_position_world,
            camera_rotation=pose.T_world_cam[:3, :3],
            summary_line=summary_line,
        )


if __name__ == "__main__":
    main()
