"""End-to-end demo: live detection (Stage 1) -> VO-pose mesh projection
(Stage 2) -> per-frame web viewer (Stage 3), run back to back over a
directory of saved frames (replaying them as a stand-in for a live camera
feed, per live_detector.py's frame_source_from_directory).

    python -m live_backprojection.run_demo --view
"""
import argparse
import json
import os

from live_backprojection import _pathsetup  # noqa: F401

from live_backprojection.cluster_results import build_final_results, export_cluster_json
from live_backprojection.live_detector import DATA_DIR, LiveDetectionRun, frame_source_from_directory, save_detection_csv
from live_backprojection.project_to_mesh import build_mesh_scene, project_all_frames
from live_backprojection.web_viewer_timeline import TRACKING_MODEL_PATH, show_timeline

_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--camera-name", default=None)
    parser.add_argument("--out-detections", default=os.path.join(_OUT_DIR, "detect.csv"))
    parser.add_argument("--out-projection", default=os.path.join(_OUT_DIR, "projection.json"))
    parser.add_argument("--out-cluster", default=os.path.join(_OUT_DIR, "cluster.json"))
    parser.add_argument("--tracking-model", default=TRACKING_MODEL_PATH)
    parser.add_argument("--target-hz", type=float, default=None,
                         help="Throttle Stage 1 to this rate (e.g. 10.0); default: run as fast as possible")
    view_group = parser.add_mutually_exclusive_group()
    view_group.add_argument("--view", dest="view", action="store_true", default=True)
    view_group.add_argument("--no-view", dest="view", action="store_false")
    args = parser.parse_args()

    print(f"=== Stage 1: live detection over {args.data_dir} ===")
    run = LiveDetectionRun(data_dir=args.data_dir, camera_name=args.camera_name)

    def progress(rec):
        print(f"  frame {rec.frame_name}: {len(rec.detections)} detections")

    run.run(frame_source_from_directory(args.data_dir), target_hz=args.target_hz, progress_cb=progress,
            camera_name=args.camera_name, data_dir=args.data_dir)
    save_detection_csv(run, args.out_detections, data_dir=args.data_dir, camera_name=args.camera_name)
    print(f"Merged {len(run.frames)} frames' detections into {args.out_detections}")

    print("\n=== Stage 2: projecting detections onto the car mesh ===")
    # Projection consumes the complete in-memory frame representation. The
    # shared CSV is the external multi-camera handoff and is deliberately not
    # reparsed here, because this run's camera must be projected with its own
    # intrinsics and pose track.
    detector_output = run.to_json_dict(data_dir=args.data_dir, camera_name=args.camera_name)
    cam = args.camera_name or detector_output.get("meta", {}).get("camera_name")
    stage2 = project_all_frames(detector_output, camera_name=cam)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_projection)), exist_ok=True)
    with open(args.out_projection, "w") as f:
        json.dump(stage2, f, indent=2)
    print(f"Projected {len(stage2['frames'])} frames, skipped {stage2['skipped_frame_ids']} (no VO coverage)")
    print(f"Saved -> {args.out_projection}")

    print("\n=== Stage 3: clustering detections and exporting cluster.json ===")
    final_results = build_final_results(
        stage2,
        points3d_path=args.tracking_model if os.path.isfile(args.tracking_model) else None,
        camera_name=cam,
    )
    export_cluster_json(final_results, args.out_cluster, camera_name=cam)
    print(f"Clustered {final_results['n_clusters']} points ({final_results['n_false_detects']} false detects)")
    print(f"Saved -> {args.out_cluster}")

    if args.view:
        print("\n=== Stage 4: opening timeline viewer ===")
        mesh, _scene = build_mesh_scene()  # cache hit (Stage 2 already built it once) -- fast reload
        out_html = os.path.join(_OUT_DIR, "timeline_viewer.html")
        show_timeline(mesh, stage2, tracking_model_path=args.tracking_model, out_path=out_html)
        print(f"Viewer written -> {out_html}")


if __name__ == "__main__":
    main()
