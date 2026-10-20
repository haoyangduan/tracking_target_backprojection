"""Stage 2: offline projection.

For each frame_id in Stage 1's detection output, looks up that frame's VO
camera pose, builds one ray per detected pixel, and batch-casts all of that
frame's rays against the car mesh in a single call -- using the exact
verified mesh-loading method (full resolution, no decimation) this project
established, not a reimplementation.

Unlike pipeline/backproject_targets.py's backproject_and_compare (which
requires a known 3D ground-truth point per detection via object_points/
correspondences), this has no ground truth to compare against -- it only
answers "where does this detected pixel's line of sight hit the car
surface," via the same pixel_to_world_ray + cast_rays mechanics.
"""
import argparse
import csv
import json
import os
import tempfile

from live_backprojection import _pathsetup  # noqa: F401

import numpy as np

from calib_target.pnp_io import load_camera_intrinsics_any
from single_frame_backprojection.camera_model import pixel_to_world_ray
from single_frame_backprojection.mesh_utils import build_raycasting_scene, cast_rays, load_car_mesh
from live_backprojection.pose_resolution import resolve_pose_fn

_DATA = os.path.join(_pathsetup.PROJECT_ROOT, "data")
VO_TRACK = os.path.join(
    _DATA, "track_cam_test_cad-NE_pair-both_cam-both_mode-vo_det-aliked_frames-0-180_ref-0-511_sc-0.25_st-2_hc71c45f4b1.json"
)
MESH_PATH = os.path.join(_DATA, "mesh", "0_NE_70_SIMPLIFICATION.stl")
MESH_SCALE = 1.0  # 0_NE_70_SIMPLIFICATION.stl is already authored in millimeters; a meters-authored
                   # mesh (e.g. BIP_Flat.stl) needs 1000.0 instead -- see mesh_utils.load_car_mesh's
                   # `scale` docstring. Kept as a real parameter (not hardcoded) so a GUI/caller that
                   # points mesh_path at a different mesh isn't silently given the wrong scale.
INTRINSICS_PATH = os.path.join(_DATA, "intrinsics_sensors_0922.json")
CAMERA_NAME = "hik_DA0179014"


def build_mesh_scene(mesh_path: str = MESH_PATH, cache_dir: str = None, scale: float = MESH_SCALE):
    """The non-negotiable verified mesh load: full resolution (target_
    triangles=0 -- decimation was proven elsewhere in this project to leave
    surface gaps that let rays skip through to inner panels). `scale` IS a
    real parameter (unlike target_triangles) since it depends on which mesh
    file is loaded, not on raycasting accuracy. Build the RaycastingScene
    once; the caller reuses it for every frame."""
    mesh = load_car_mesh(mesh_path, cache_dir=cache_dir, target_triangles=0, scale=scale)
    scene = build_raycasting_scene(mesh)
    return mesh, scene


from live_backprojection.live_detector import load_capture_timestamps
from live_backprojection.live_detector import load_detection_csv

def project_frame(frame_rec: dict, intr, scene, pose_fn):
    """Project one frame's detections onto the mesh. `pose_fn` is
    frame_name -> CameraPose (see pose_resolution.resolve_pose_fn, which
    both direct-VO-covers-this-camera and rig-composed-from-a-different-
    camera cases). Returns the per-frame hit record (see module docstring /
    README schema), or None if this frame has no resolvable pose (caught
    KeyError) -- caller should skip such frames rather than aborting the
    whole run, exactly as build_oracle_labels_full.py's main() does for
    frames outside VO's coverage."""
    try:
        pose = pose_fn(frame_rec["frame_name"])
    except KeyError:
        return None

    cam_pos = pose.camera_position_world.tolist()
    cam_rot = pose.T_world_cam[:3, :3].tolist()
    capture_ts_ns = frame_rec.get("capture_ts_ns")
    dets = frame_rec["detections"]

    if not dets:
        return {"frame_id": frame_rec["frame_id"], "frame_name": frame_rec["frame_name"],
                "timestamp": frame_rec.get("timestamp"),
                "capture_ts_ns": capture_ts_ns,
                "camera_position_world_mm": cam_pos, "camera_rotation_world_cam": cam_rot, "hits": []}

    origins, directions = [], []
    for d in dets:
        origin, direction = pixel_to_world_ray(d["u"], d["v"], intr, pose)
        origins.append(origin)
        directions.append(direction)

    # ONE cast_rays call for this frame's entire detection set -- never
    # per-detection (see mesh_utils.cast_rays' batching contract).
    t_hit, hit_points = cast_rays(scene, np.array(origins), np.array(directions))

    hits = []
    for i, d in enumerate(dets):
        found = bool(np.isfinite(t_hit[i]))
        hits.append({
            "pixel_u": float(d["u"]), "pixel_v": float(d["v"]), "size": float(d["size"]),
            "hit": found,
            "t_hit_mm": float(t_hit[i]) if found else None,
            "point_mm": hit_points[i].tolist() if found else None,
        })

    return {"frame_id": frame_rec["frame_id"], "frame_name": frame_rec["frame_name"],
            "timestamp": frame_rec.get("timestamp"),
            "capture_ts_ns": capture_ts_ns,
            "camera_position_world_mm": cam_pos, "camera_rotation_world_cam": cam_rot, "hits": hits}


def project_all_frames(detector_output: dict, intrinsics_path: str = INTRINSICS_PATH,
                        camera_name: str = None, vo_track_path: str = VO_TRACK,
                        mesh_path: str = MESH_PATH, mesh_scale: float = MESH_SCALE,
                        rig_extrinsics_path: str = None, mesh_cache_dir: str = None,
                        capture_log_path: str = None,
                        detection_scale: float = None) -> dict:
    """Loads intrinsics once, builds the mesh+raycasting scene once (the
    expensive, cached step), then projects every frame. Frames with no
    resolvable pose are counted in skipped_frame_ids and omitted from
    frames.

    camera_name's pose comes directly from vo_track_path if that track
    covers camera_name; otherwise, if rig_extrinsics_path is given, it's
    derived per frame via the rig's fixed camera-to-camera extrinsics (see
    pose_resolution.resolve_pose_fn) -- lets the live dataset run on a
    DIFFERENT rigidly-mounted camera than whichever one the VO track
    directly covers.

    detection_scale: the detector's image-resize scale factor (0.5 = downsampled,
    1.0 = full resolution). If not passed explicitly, it is read from
    detector_output.meta.detector.scale (set by LiveDetectionRun.to_json_dict)
    so the projection.json always reflects the actual resolution mode used."""
    if not camera_name:
        camera_name = (detector_output.get("meta", {}).get("camera_name")
                       or detector_output.get("source", {}).get("camera_name")
                       or CAMERA_NAME)

    # Resolve detection scale: explicit kwarg wins, then the in-memory detector
    # representation's meta,
    # then None (omitted from output rather than guessing a wrong default).
    if detection_scale is None:
        detection_scale = (
            detector_output.get("meta", {}).get("detector", {}).get("scale")
        )

    intr = load_camera_intrinsics_any(intrinsics_path, camera_name)
    mesh, scene = build_mesh_scene(mesh_path, cache_dir=mesh_cache_dir, scale=mesh_scale)
    pose_fn = resolve_pose_fn(vo_track_path, camera_name, rig_extrinsics_path)

    # Fallback timestamp lookup if detector_output frames don't carry capture_ts_ns
    timestamps = None
    first_frame = detector_output.get("frames", [{}])[0] if detector_output.get("frames") else {}
    if first_frame and "capture_ts_ns" not in first_frame:
        timestamps = load_capture_timestamps(capture_log_path, camera_name=camera_name)

    out_frames, skipped = [], []
    for frame_rec in detector_output["frames"]:
        if timestamps and frame_rec.get("capture_ts_ns") is None:
            frame_rec["capture_ts_ns"] = timestamps.get(
                frame_rec.get("frame_name"),
                timestamps.get(frame_rec.get("frame_id"))
            )
        result = project_frame(frame_rec, intr, scene, pose_fn)
        if result is None:
            skipped.append(frame_rec["frame_id"])
        else:
            out_frames.append(result)

    # Human-readable resolution mode label for quick inspection of the JSON
    # without needing to compare the float value directly.
    if detection_scale is not None:
        if abs(detection_scale - 1.0) < 1e-9:
            resolution_mode = "full_resolution_1x"
        elif abs(detection_scale - 0.5) < 1e-9:
            resolution_mode = "downsampled_0.5x"
        else:
            resolution_mode = f"custom_{detection_scale}x"
    else:
        resolution_mode = None

    meta = {
        "source": "live_backprojection.project_to_mesh",
        "camera_name": camera_name,
        "intrinsics_path": intrinsics_path,
        "vo_track_path": vo_track_path,
        "mesh_path": mesh_path,
        "mesh_scale": mesh_scale,
        "mesh_target_triangles": 0,
        "rig_extrinsics_path": rig_extrinsics_path,
        "detection_resolution_mode": resolution_mode,
    }

    return {
        "meta": meta,
        "source": meta,
        "frames": out_frames,
        "skipped_frame_ids": skipped,
    }


def project_detection_csv_by_camera(detections_csv: str, intrinsics_path: str = INTRINSICS_PATH,
                                    vo_track_path: str = VO_TRACK, mesh_path: str = MESH_PATH,
                                    mesh_scale: float = MESH_SCALE,
                                    rig_extrinsics_path: str = None,
                                    capture_log_path: str = None,
                                    camera_names=None) -> dict:
    """Project each camera in a merged ``detect.csv`` independently.

    The return value maps each camera name to its own Stage 2 output. No frame
    records are combined across cameras, which keeps camera-specific intrinsics,
    pose resolution, and timeline viewers unambiguous.
    """
    grouped = load_detection_csv(detections_csv)
    selected = set(camera_names) if camera_names else None
    outputs = {}
    for camera_name, detector_output in grouped.items():
        if selected is not None and camera_name not in selected:
            continue
        outputs[camera_name] = project_all_frames(
            detector_output,
            intrinsics_path=intrinsics_path,
            camera_name=camera_name,
            vo_track_path=vo_track_path,
            mesh_path=mesh_path,
            mesh_scale=mesh_scale,
            rig_extrinsics_path=rig_extrinsics_path,
            capture_log_path=capture_log_path,
        )
    return outputs



PROJECTION_CSV_FIELDS = (
    "camera_name", "frame_id", "frame_name", "timestamp", "capture_ts_ns",
    "pixel_u", "pixel_v", "size", "hit", "t_hit_mm",
    "point_x_mm", "point_y_mm", "point_z_mm",
    "camera_x_mm", "camera_y_mm", "camera_z_mm",
    "source", "intrinsics_path", "vo_track_path", "mesh_path",
    "mesh_scale", "rig_extrinsics_path", "detection_resolution_mode",
)

def projection_csv_rows(outputs: dict) -> list[dict]:
    """Flatten per-camera projection results into combined CSV rows."""
    rows = []
    for camera_name, result in outputs.items():
        meta = result.get("meta", {}) or result.get("source", {}) or {}
        metadata = {
            "source": meta.get("source", "live_backprojection.project_to_mesh"),
            "intrinsics_path": meta.get("intrinsics_path"),
            "vo_track_path": meta.get("vo_track_path"),
            "mesh_path": meta.get("mesh_path"),
            "mesh_scale": meta.get("mesh_scale"),
            "rig_extrinsics_path": meta.get("rig_extrinsics_path"),
            "detection_resolution_mode": meta.get("detection_resolution_mode"),
        }
        for frame in result.get("frames", []):
            camera_position = frame.get("camera_position_world_mm") or [None, None, None]
            for hit in frame.get("hits", []):
                point = hit.get("point_mm") or [None, None, None]
                rows.append({
                    "camera_name": camera_name,
                    "frame_id": frame.get("frame_id"),
                    "frame_name": frame.get("frame_name", ""),
                    "timestamp": frame.get("timestamp"),
                    "capture_ts_ns": frame.get("capture_ts_ns"),
                    "pixel_u": hit.get("pixel_u"),
                    "pixel_v": hit.get("pixel_v"),
                    "size": hit.get("size"),
                    "hit": hit.get("hit", False),
                    "t_hit_mm": hit.get("t_hit_mm"),
                    "point_x_mm": point[0], "point_y_mm": point[1], "point_z_mm": point[2],
                    "camera_x_mm": camera_position[0], "camera_y_mm": camera_position[1], "camera_z_mm": camera_position[2],
                    **metadata,
                })
    return rows

def save_projection_csv(path: str, outputs: dict) -> str:
    """Write one combined projection CSV with rows from all cameras."""
    rows = projection_csv_rows(outputs)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".projection_", suffix=".csv", dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PROJECTION_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", required=True, help="Stage-1 detector JSON or merged detect.csv")
    parser.add_argument("--out", required=True, help="Output path for projection.json")
    parser.add_argument("--camera-name", default=None, help="Project only this camera")
    parser.add_argument("--intrinsics", default=INTRINSICS_PATH)
    parser.add_argument("--vo-track", default=VO_TRACK)
    parser.add_argument("--mesh", default=MESH_PATH)
    parser.add_argument("--mesh-scale", type=float, default=MESH_SCALE)
    parser.add_argument("--rig-extrinsics", default=None)
    parser.add_argument("--capture-log", default=None)
    parser.add_argument("--timeline-dir", default=None, help="Also write one interactive HTML timeline per camera")
    parser.add_argument("--projection-csv", default=None, help="Also write one combined projection CSV for all cameras")
    parser.add_argument("--cluster-dir", default=None, help="Also write per-camera cluster JSON and a combined cluster CSV")
    parser.add_argument("--cluster-merge-dist-mm", type=float, default=25.0, help="Cluster merge distance in millimetres")
    parser.add_argument("--tracking-model", default=None, help="Tracking-model points (.txt) for cluster error analysis")
    parser.add_argument("--false-detect-threshold-mm", type=float, default=20.0)
    args = parser.parse_args()

    common = dict(intrinsics_path=args.intrinsics, vo_track_path=args.vo_track,
                  mesh_path=args.mesh, mesh_scale=args.mesh_scale,
                  rig_extrinsics_path=args.rig_extrinsics,
                  capture_log_path=args.capture_log)
    mesh = None
    if args.timeline_dir:
        from live_backprojection.project_to_mesh import build_mesh_scene
        mesh, _ = build_mesh_scene(args.mesh, scale=args.mesh_scale)
    if args.detections.lower().endswith(".csv"):
        selected = [args.camera_name] if args.camera_name else None
        outputs = project_detection_csv_by_camera(args.detections, camera_names=selected, **common)
        if not outputs:
            raise SystemExit("No matching cameras found in detection CSV")
        out_dir = os.path.dirname(os.path.abspath(args.out))
        for cam, result in outputs.items():
            out_path = args.out if len(outputs) == 1 else os.path.join(out_dir, f"projection_{cam}.json")
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Projected {cam}: {len(result['frames'])} frames -> {out_path}")
            print(f"Skipped (no VO coverage): {result['skipped_frame_ids']}")
            if args.timeline_dir:
                from live_backprojection.web_viewer_timeline import build_timeline_viewer_html
                os.makedirs(args.timeline_dir, exist_ok=True)
                timeline_path = os.path.join(args.timeline_dir, f"timeline_{cam}.html")
                html_doc = build_timeline_viewer_html(mesh, result, window_name=f"Projection timeline - {cam}")
                with open(timeline_path, "w", encoding="utf-8") as tf:
                    tf.write(html_doc)
                print(f"Timeline {cam}: {timeline_path}")
        if args.timeline_dir:
            launcher = os.path.join(args.timeline_dir, "timelines_all.html")
            options = "".join(f"<option value='timeline_{cam}.html'>{cam}</option>" for cam in outputs)
            html_doc = ("<!doctype html><html><body style='margin:0;background:#11141a;color:#eee;font:14px Segoe UI'><div style='height:40px;padding:10px'>Camera <select onchange=\"document.getElementById('v').src=this.value\">" + options + "</select></div><iframe id='v' src='timeline_" + next(iter(outputs)) + ".html' style='border:0;width:100%;height:calc(100vh - 60px)'></iframe></body></html>")
            with open(launcher, "w", encoding="utf-8") as lf: lf.write(html_doc)
            print(f"Combined timeline viewer: {launcher}")
        if args.projection_csv:
            save_projection_csv(args.projection_csv, outputs)
            print(f"Combined projection CSV: {args.projection_csv}")
        if args.cluster_dir:
            from live_backprojection.cluster_results import build_final_results
            os.makedirs(args.cluster_dir, exist_ok=True)
            cluster_rows = []
            clustered_outputs = {}
            for cam, result in outputs.items():
                clustered = build_final_results(result, camera_name=cam, points3d_path=args.tracking_model, merge_dist_mm=args.cluster_merge_dist_mm, error_threshold_mm=args.false_detect_threshold_mm)
                clustered_outputs[cam] = clustered
                with open(os.path.join(args.cluster_dir, f"cluster_{cam}.json"), "w", encoding="utf-8") as cf:
                    json.dump(clustered, cf, indent=2)
                for cluster in clustered.get("clusters", []):
                    cluster_rows.append({
                        "camera_name": cam, "merge_dist_mm": args.cluster_merge_dist_mm, "cluster_id": cluster.get("cluster_id"),
                        "point_x_mm": cluster.get("point_mm", [None, None, None])[0],
                        "point_y_mm": cluster.get("point_mm", [None, None, None])[1],
                        "point_z_mm": cluster.get("point_mm", [None, None, None])[2],
                        "n_observations": cluster.get("n_observations"),
                        "spread_mm": cluster.get("spread_mm"),
                    })
            cluster_csv = os.path.join(args.cluster_dir, "clusters.csv")
            with open(cluster_csv, "w", newline="", encoding="utf-8") as cf:
                fields = ("camera_name", "merge_dist_mm", "cluster_id", "point_x_mm", "point_y_mm", "point_z_mm", "n_observations", "spread_mm")
                writer = csv.DictWriter(cf, fieldnames=fields)
                writer.writeheader(); writer.writerows(cluster_rows)
            print(f"Combined cluster CSV: {cluster_csv}")
            from live_backprojection.web_viewer_final_results import build_final_results_viewer_html
            mesh_for_clusters, _ = build_mesh_scene(args.mesh, scale=args.mesh_scale)
            for cam, clustered in clustered_outputs.items():
                cluster_html = os.path.join(args.cluster_dir, f"cluster_{cam}.html")
                with open(cluster_html, "w", encoding="utf-8") as vf:
                    vf.write(build_final_results_viewer_html(mesh_for_clusters, clustered, window_name=f"Clustered defects - {cam}"))
                print(f"Cluster viewer {cam}: {cluster_html}")
            first = next(iter(clustered_outputs.values()))
            combined = dict(first)
            combined["clusters"] = [c for result in clustered_outputs.values() for c in result.get("clusters", [])]
            combined["pairs"] = [pair for result in clustered_outputs.values() for pair in result.get("pairs", [])]
            combined["n_clusters"] = len(combined["clusters"])
            combined["source"] = {"camera_name": "All cameras"}
            combined_html = os.path.join(args.cluster_dir, "clusters_all.html")
            with open(combined_html, "w", encoding="utf-8") as vf:
                vf.write(build_final_results_viewer_html(mesh_for_clusters, combined, window_name="Clustered defects - All cameras"))
            print(f"Combined cluster viewer: {combined_html}")
            index_html = os.path.join(args.cluster_dir, "clusters_viewer.html")
            options = ['<option value="clusters_all.html">All cameras</option>']
            options.extend(f'<option value="cluster_{cam}.html">{cam}</option>' for cam in clustered_outputs)
            option_html = "".join(options)
            viewer_shell = f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>Cluster result viewer</title>
<style>html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:#111827;color:#e5e7eb;font:14px Arial,sans-serif}}
#bar{{height:54px;display:flex;align-items:center;gap:14px;padding:0 18px;box-sizing:border-box;background:#1f2937;border-bottom:1px solid #374151}}
#bar strong{{font-size:16px}} select{{background:#111827;color:#f9fafb;border:1px solid #4b5563;border-radius:5px;padding:7px 32px 7px 10px;font-size:14px}}
iframe{{display:block;width:100%;height:calc(100% - 54px);border:0;background:#111827}}</style></head>
<body><div id=\"bar\"><strong>Cluster results</strong><label for=\"camera\">Camera</label><select id=\"camera\" onchange=\"document.getElementById('viewer').src=this.value\">{option_html}</select></div>
<iframe id=\"viewer\" src=\"clusters_all.html\" title=\"Cluster result viewer\"></iframe></body></html>"""
            with open(index_html, "w", encoding="utf-8") as vf:
                vf.write(viewer_shell)
            print(f"Cluster viewer index: {index_html}")
    else:
        with open(args.detections) as f:
            detector_output = json.load(f)
        cam = args.camera_name or detector_output.get("meta", {}).get("camera_name")
        result = project_all_frames(detector_output, camera_name=cam, **common)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Projected {len(result['frames'])} frames -> {args.out}")
        print(f"Camera name: {result['meta']['camera_name']}")
        print(f"Skipped (no VO coverage): {result['skipped_frame_ids']}")


if __name__ == "__main__":
    main()
