"""Stage 4: consolidate per-frame detections into a final 3D "detected
tracking model", and validate it against the known Creaform-surveyed
tracking model.

The same physical sticker gets independently back-projected to a slightly
different 3D point on nearly every frame it's visible in (pixel-detection
noise + per-frame pose noise). This module:

  1. Pools every hit point_mm across ALL frames of a projection.json
     (project_to_mesh.project_all_frames' output) and hierarchically
     clusters them with a distance cutoff, so repeated observations of the
     same physical sticker collapse into one consolidated point (the
     cluster's median -- robust to the occasional outlier observation,
     unlike a mean).
  2. Matches each consolidated point to its nearest point in the known
     ground-truth tracking-model points3d file (the same Creaform-surveyed-
     and-mesh-fit file used elsewhere in this project, e.g.
     data/new_car_300926_whood.txt) via a KD-tree nearest-
     neighbor query. A match whose distance exceeds error_threshold_mm is
     flagged as a likely false detect: a real, correctly-detected sticker
     should land close to SOME known survey point; one that doesn't is more
     likely a bolt/hole/noise artifact that was consistently false-
     positive'd across frames (e.g. from a plausible but wrong candidate
     recurring in roughly the same physical spot), not a different problem
     from this project's long-running single-frame false-positive work --
     just visible here in 3D instead of per-frame pixel coordinates.
  3. Exports cluster.json with metadata including camera_name, cluster
     statistics, and ground-truth pairing comparisons.
"""
import argparse
import json
import os
from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree

DEFAULT_MERGE_DIST_MM = 25.0
DEFAULT_ERROR_THRESHOLD_MM = 20.0


def cluster_detected_points(stage2_output: dict, merge_dist_mm: float = DEFAULT_MERGE_DIST_MM) -> list:
    """Pools every hit point_mm across all frames, then hierarchically
    clusters them (average-linkage, flat clusters cut at merge_dist_mm) so
    repeated observations of the same physical sticker collapse into one
    point. Returns a list of {"cluster_id": int, "point_mm": [x,y,z] (cluster median),
    "n_observations": int, "spread_mm": float (max distance from any member
    to the cluster median -- a rough per-cluster quality/consistency
    indicator; large spread means the "same" cluster disagrees a lot,
    possibly two real points merged together if merge_dist_mm is too large)}."""
    all_points = [h["point_mm"] for f in stage2_output.get("frames", []) for h in f.get("hits", []) if h.get("hit")]
    if not all_points:
        return []

    pts = np.array(all_points)
    if len(pts) == 1:
        return [{"cluster_id": 1, "point_mm": pts[0].tolist(), "n_observations": 1, "spread_mm": 0.0}]

    Z = linkage(pts, method="average")
    labels = fcluster(Z, t=merge_dist_mm, criterion="distance")

    clusters = []
    for idx, label in enumerate(np.unique(labels), start=1):
        member_pts = pts[labels == label]
        center = np.median(member_pts, axis=0)
        spread = float(np.max(np.linalg.norm(member_pts - center, axis=1))) if len(member_pts) > 1 else 0.0
        clusters.append({
            "cluster_id": idx,
            "point_mm": center.tolist(),
            "n_observations": int(len(member_pts)),
            "spread_mm": spread,
        })
    return clusters


def match_to_ground_truth(clusters: list, points3d_path: str,
                           error_threshold_mm: float = DEFAULT_ERROR_THRESHOLD_MM) -> list:
    """For each clustered detected point, finds its nearest ground-truth
    point (3D Euclidean distance) in the Creaform tracking-model points3d
    file. Returns a list of {"point_line": int (1-indexed line in
    points3d_path), "gt_point_mm": [x,y,z], "detected_point_mm": [x,y,z],
    "error_mm": float, "n_observations": int, "spread_mm": float,
    "is_false_detect": bool}. is_false_detect is True when error_mm exceeds
    error_threshold_mm. Multiple detected clusters CAN match the same
    ground-truth point_line (not deduplicated here) -- that's itself a
    useful signal (e.g. a sticker whose detections split into two clusters
    because of a large pose jump) worth seeing rather than silently
    discarding."""
    if not points3d_path or not os.path.isfile(points3d_path) or not clusters:
        return []

    raw = np.loadtxt(points3d_path)
    gt_points = raw[:, :3]
    tree = cKDTree(gt_points)

    pairs = []
    for c in clusters:
        d, idx = tree.query(c["point_mm"])
        pairs.append({
            "point_line": int(idx) + 1,
            "gt_point_mm": gt_points[idx].tolist(),
            "detected_point_mm": c["point_mm"],
            "error_mm": float(d),
            "n_observations": c["n_observations"],
            "spread_mm": c["spread_mm"],
            "is_false_detect": bool(d > error_threshold_mm),
        })
    return pairs


def build_final_results(stage2_output: dict, points3d_path: Optional[str] = None,
                         merge_dist_mm: float = DEFAULT_MERGE_DIST_MM,
                         error_threshold_mm: float = DEFAULT_ERROR_THRESHOLD_MM,
                         camera_name: Optional[str] = None) -> dict:
    """Convenience wrapper: cluster_detected_points() + optional match_to_ground_truth().
    Returns a complete results dictionary with metadata header including camera_name,
    consolidated 3D clusters, matched ground truth pairs, and error statistics."""
    clusters = cluster_detected_points(stage2_output, merge_dist_mm)
    pairs = match_to_ground_truth(clusters, points3d_path, error_threshold_mm) if points3d_path else []

    cam_name = (
        camera_name
        or stage2_output.get("meta", {}).get("camera_name")
        or stage2_output.get("source", {}).get("camera_name")
        or "hik_DA0179014"
    )

    inliers = [p for p in pairs if not p["is_false_detect"]]
    errors = [p["error_mm"] for p in inliers]
    summary = {
        "n_clusters": len(clusters),
        "n_matched_pairs": len(pairs),
        "n_inliers": len(inliers),
        "n_false_detects": sum(1 for p in pairs if p["is_false_detect"]) if pairs else 0,
        "rms_error_mm": float(np.sqrt(np.mean(np.array(errors) ** 2))) if errors else 0.0,
        "mean_error_mm": float(np.mean(errors)) if errors else 0.0,
        "median_error_mm": float(np.median(errors)) if errors else 0.0,
        "max_error_mm": float(np.max(errors)) if errors else 0.0,
    }

    meta = {
        "source": "live_backprojection.cluster_results",
        "camera_name": cam_name,
        "merge_dist_mm": float(merge_dist_mm),
        "error_threshold_mm": float(error_threshold_mm),
        "points3d_path": points3d_path,
        "n_clusters": len(clusters),
        "n_false_detects": summary["n_false_detects"],
    }

    return {
        "meta": meta,
        "clusters": clusters,
        "pairs": pairs,
        "summary": summary,
        "merge_dist_mm": merge_dist_mm,
        "error_threshold_mm": error_threshold_mm,
        "n_clusters": len(clusters),
        "n_false_detects": summary["n_false_detects"],
        "source": stage2_output.get("source", stage2_output.get("meta", {})),
    }


def export_cluster_json(
    final_results_or_stage2: dict,
    out_path: str,
    camera_name: Optional[str] = None,
    points3d_path: Optional[str] = None,
    merge_dist_mm: float = DEFAULT_MERGE_DIST_MM,
    error_threshold_mm: float = DEFAULT_ERROR_THRESHOLD_MM,
) -> str:
    """Exports clustered 3D detection results to cluster.json with metadata including camera_name.
    
    Accepts either:
      - final_results dict (from build_final_results)
      - stage2_output dict (from project_all_frames) -- in which case clustering and optional
        ground-truth matching are performed.
    """
    if "frames" in final_results_or_stage2:
        # Received Stage 2 projection output directly
        results = build_final_results(
            final_results_or_stage2,
            points3d_path=points3d_path,
            merge_dist_mm=merge_dist_mm,
            error_threshold_mm=error_threshold_mm,
            camera_name=camera_name,
        )
    else:
        results = dict(final_results_or_stage2)
        cam = (
            camera_name
            or results.get("meta", {}).get("camera_name")
            or results.get("source", {}).get("camera_name")
            or "hik_DA0179014"
        )
        if "meta" not in results:
            results["meta"] = {
                "source": "live_backprojection.cluster_results",
                "camera_name": cam,
                "merge_dist_mm": float(results.get("merge_dist_mm", merge_dist_mm)),
                "error_threshold_mm": float(results.get("error_threshold_mm", error_threshold_mm)),
                "points3d_path": points3d_path or results.get("meta", {}).get("points3d_path"),
                "n_clusters": results.get("n_clusters", len(results.get("pairs", []))),
                "n_false_detects": results.get("n_false_detects", 0),
            }
        else:
            results["meta"]["camera_name"] = cam

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", required=True, help="Path to Stage 2 projection.json")
    parser.add_argument("--points3d", default=None, help="Creaform ground-truth points3d (.txt)")
    parser.add_argument("--out", default="live_backprojection/out/cluster.json", help="Output path for cluster.json")
    parser.add_argument("--camera-name", default=None, help="Camera name override")
    parser.add_argument("--merge-dist", type=float, default=DEFAULT_MERGE_DIST_MM, help="Cluster merge distance (mm)")
    parser.add_argument("--error-thresh", type=float, default=DEFAULT_ERROR_THRESHOLD_MM, help="False detect threshold (mm)")
    args = parser.parse_args()

    stage2 = json.load(open(args.projection, "r", encoding="utf-8"))
    out_path = export_cluster_json(
        stage2,
        out_path=args.out,
        camera_name=args.camera_name,
        points3d_path=args.points3d,
        merge_dist_mm=args.merge_dist,
        error_threshold_mm=args.error_thresh,
    )
    print(f"Exported cluster.json -> {out_path}")


if __name__ == "__main__":
    main()
