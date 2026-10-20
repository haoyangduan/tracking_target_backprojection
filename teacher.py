"""Automatic adaptive-threshold parameter derivation ("teacher" pipeline).

Builds an oracle -- known 3D tracking-model points + camera pose +
occlusion raycasting against the car mesh, the exact method
build_oracle_labels_full.py uses, generalized to arbitrary paths instead of
this project's specific files -- for a TEACHER dataset, then automatically
searches the production detector's parameters (block_size, c_offset,
center_ring_min; `scale` is a fixed architectural choice, not searched)
against that oracle. No threshold is ever hand-entered and no human
judgment call decides the final pick -- both search stages use an explicit,
reproducible selection rule (see derive_detector_params' docstring).

The oracle's camera pose comes from CREAFORM tracking (pipeline/
creaform_pose.py + live_backprojection/pose_resolution.resolve_pose_fn_
creaform), not VO: VO is computed from the image sequence itself (so any
systematic image-based detector/pose artifact could in principle leak into
both), whereas Creaform is an independent external motion-capture ground
truth -- the correct choice for a TEACHER oracle that detector parameters
are judged against. See resolve_pose_fn_creaform's docstring for how poorly
time-synced Creaform samples (creaform_max_time_diff_ms) are excluded
rather than silently used.

The teacher dataset is INDEPENDENT of whatever live dataset the derived
config later runs detection on: today they happen to be the same frames
(hik_DA0179014 + its Creaform track), but every path here (data_dir,
creaform calibration/track, mesh, points3d, intrinsics, camera_name) is a
plain parameter, not a hardcoded assumption -- a future teacher capture
(e.g. a dedicated calibration-rig run with denser Creaform coverage) can
differ from the live production dataset entirely.
"""
import os
import random

import cv2
import numpy as np

from live_backprojection import _pathsetup  # noqa: F401

from calib_target.pnp_io import load_camera_intrinsics_any
from single_frame_backprojection.mesh_utils import build_raycasting_scene, cast_rays, load_car_mesh
from live_backprojection.detector.evaluate_against_oracle import MERGE_FRAC, auto_gain
from live_backprojection.detector.evaluate_downsampled_full import AREA_HI as _FALLBACK_AREA_HI
from live_backprojection.detector.evaluate_downsampled_full import AREA_LO as _FALLBACK_AREA_LO
from live_backprojection.detector.evaluate_downsampled_full import compute_mask, pre_ring_candidates_and_scores
from calib_target.blob_detect import _merge_concentric_candidates_relative
from live_backprojection.live_detector import _FRAME_NAME_RE
from live_backprojection.pose_resolution import resolve_pose_fn, resolve_pose_fn_creaform

# See resolve_pose_fn_creaform's docstring: this project's own Creaform track
# has a clean, data-confirmed split -- 19 frames (a startup sync-lock-in
# transient) exceed 100ms, every one of the other 163 is under 20ms -- so a
# frame whose Creaform sample misses sync by more than this is excluded from
# the oracle rather than silently poisoning it with a wrong position.
DEFAULT_CREAFORM_MAX_TIME_DIFF_MS = 100.0

DEFAULT_BLOCK_SIZES = [21, 31, 45, 61, 81, 101, 131]
DEFAULT_C_OFFSETS = [2, 4, 6, 8, 10, 14]
DEFAULT_RING_MIN_CANDIDATES = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 35, 40]
DEFAULT_N_SAMPLE_FRAMES = 35
DEFAULT_MATCH_TOL_PX = 35.0
DEFAULT_EDGE_MARGIN_PX = 100.0
DEFAULT_OCCLUSION_TOL_MM = 15.0
# Percentage points of combined recall willing to give up, relative to the
# most lenient center_ring_min candidate, in exchange for fewer false
# positives. This is the automatic "stop just before the cliff" rule --
# replacing a human eyeballing a recall/extra table (which is what this
# project's own center_ring_min=25 was originally picked by) with a fixed,
# reproducible criterion: accept the LARGEST threshold whose recall hasn't
# dropped by more than this many percentage points from the most lenient
# (lowest-threshold) candidate's recall.
DEFAULT_MAX_RECALL_DROP_PP = 2.0
# Fractional margin applied beyond the observed [p1, p99] true-positive area
# percentiles when deriving area_lo/area_hi (Stage C) -- area_lo/area_hi used
# to be a hardcoded constant (838.0-4818.0 px^2, from an early full-resolution
# investigation) that was never re-derived as the rest of the pipeline evolved
# (downsampling, multiple cameras, the teacher-derivation system itself) and
# wasn't even exposed as a real function parameter until now. Unlike
# circularity/aspect-ratio (already scale-invariant ratios) or center_ring_min
# (already derived in Stage B), a pixel-area bound is inherently tied to how
# large the marker appears, which depends on camera distance/resolution --
# exactly the kind of thing that should come from this dataset's own observed
# data, not a frozen historical number from a different investigation.
DEFAULT_AREA_MARGIN_FRAC = 0.25


def _near_edge(u, v, w, h, margin):
    return u < margin or u > w - margin or v < margin or v > h - margin


def build_teacher_oracle(data_dir, mesh_path, mesh_scale, points3d_path, intrinsics_path, camera_name,
                          creaform_calibration_path=None, creaform_track_csv_path=None,
                          vo_track_path=None, pose_source="creaform", rig_extrinsics_path=None,
                          creaform_max_time_diff_ms=DEFAULT_CREAFORM_MAX_TIME_DIFF_MS,
                          occlusion_tol_mm=DEFAULT_OCCLUSION_TOL_MM,
                          image_size=(4096, 3000), name_re=_FRAME_NAME_RE, progress_cb=None):
    """Builds {frame_name: [{point_line, u, v}, ...]} for every frame in
    data_dir that also has a resolvable pose. Same method as
    build_oracle_labels_full.py: project the known 3D tracking-model points
    into pixel space via the solved camera pose + intrinsics, then an
    occlusion raycast against the SAME mesh those points were fit to (a
    single consistent check suffices when mesh and points3d are already a
    matched, door-inclusive pair -- see that script's docstring for why a
    door-free mesh would need a second "door safe-gate" raycast instead).

    camera_name's pose comes from Creaform tracking data (creaform_
    calibration_path + creaform_track_csv_path), not VO -- see this module's
    own docstring for why. A frame whose Creaform sample is poorly synced
    (beyond creaform_max_time_diff_ms) or simply missing is skipped, same as
    a frame with no VO coverage used to be."""
    pose_src = (pose_source or "").lower()
    if pose_src == "vo" or (vo_track_path and not (creaform_calibration_path and creaform_track_csv_path)):
        if not vo_track_path:
            raise ValueError("vo_track_path must be provided when using VO pose source for teacher oracle.")
        pose_fn = resolve_pose_fn(vo_track_path, camera_name, rig_extrinsics_path=rig_extrinsics_path)
    else:
        if not creaform_calibration_path or not creaform_track_csv_path:
            raise ValueError("creaform_calibration_path and creaform_track_csv_path must be provided when using Creaform pose source.")
        pose_fn = resolve_pose_fn_creaform(creaform_calibration_path, creaform_track_csv_path, camera_name,
                                            max_time_diff_ms=creaform_max_time_diff_ms,
                                            rig_extrinsics_path=rig_extrinsics_path)
    intr = load_camera_intrinsics_any(intrinsics_path, camera_name)
    raw = np.loadtxt(points3d_path)
    points_mm = raw[:, :3]

    if progress_cb:
        progress_cb(f"Loading teacher mesh ({mesh_path}, scale={mesh_scale}) + building raycasting scene "
                     f"(full resolution, no decimation -- first load of a given mesh can take 1-2 min)...")
    mesh = load_car_mesh(mesh_path, target_triangles=0, scale=mesh_scale)
    scene = build_raycasting_scene(mesh)

    names = sorted(f for f in os.listdir(data_dir) if name_re.match(f))
    w, h = image_size
    oracle = {}
    skipped = []
    for name in names:
        try:
            pose = pose_fn(name)
        except KeyError:
            skipped.append(name)
            continue

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

        entries = []
        for i in range(len(points_mm)):
            u, v = img_pts[i]
            in_bounds = (0 <= u < w) and (0 <= v < h)
            unoccluded = np.isfinite(t_hit[i]) and hit_dist_diff[i] < occlusion_tol_mm
            if in_bounds and unoccluded:
                entries.append({"point_line": i + 1, "u": float(u), "v": float(v)})
        oracle[name] = entries

        if progress_cb and len(oracle) % 20 == 0:
            progress_cb(f"  teacher oracle: {len(oracle)}/{len(names)} frames processed")

    if progress_cb:
        progress_cb(f"Teacher oracle built: {len(oracle)} frames with Creaform pose, "
                     f"{len(skipped)} skipped (missing/poorly-synced Creaform sample)")
    return oracle


def _match_recall_extra(det_xy_per_frame, expected_per_frame, match_tol_px, edge_margin_px, image_size):
    """Shared greedy nearest-neighbor matching: for each frame, match oracle
    points (excluding ones within edge_margin_px of the frame border --
    likely partially clipped from the actual sensor view even when their
    projected center is technically in-bounds) to detector candidates
    within match_tol_px. Returns (matched, total_oracle, total_extra)."""
    w, h = image_size
    matched = total_oracle = total_extra = 0
    for det_xy, expected_all in zip(det_xy_per_frame, expected_per_frame):
        expected = [e for e in expected_all if not _near_edge(e["u"], e["v"], w, h, edge_margin_px)]
        used = set()
        for e in expected:
            best, best_d = None, match_tol_px
            for i, (dx, dy) in enumerate(det_xy):
                if i in used:
                    continue
                d = np.hypot(e["u"] - dx, e["v"] - dy)
                if d < best_d:
                    best, best_d = i, d
            if best is not None:
                used.add(best)
                matched += 1
        total_oracle += len(expected)
        total_extra += len(det_xy) - len(used)
    return matched, total_oracle, total_extra


def _collect_matched_areas(loaded_full, center_ring_min, match_tol_px, edge_margin_px, image_size):
    """For the FINAL center_ring_min, re-applies the candidate filter and
    greedily matches against the oracle (same logic as _match_recall_extra),
    but collects the full-resolution px^2 AREA (pi*minor*major/4) of every
    candidate that matches a real oracle point -- the true-positive size
    distribution Stage C derives area_lo/area_hi from, instead of guessing."""
    w, h = image_size
    areas = []
    for pre_ring, scores, expected_all in loaded_full:
        cands = [(cx, cy, mn, mx) for (cx, cy, mn, mx, ang, circ), s in zip(pre_ring, scores) if s >= center_ring_min]
        expected = [e for e in expected_all if not _near_edge(e["u"], e["v"], w, h, edge_margin_px)]
        used = set()
        for e in expected:
            best, best_d = None, match_tol_px
            for i, (cx, cy, mn, mx) in enumerate(cands):
                if i in used:
                    continue
                d = np.hypot(e["u"] - cx, e["v"] - cy)
                if d < best_d:
                    best, best_d = i, d
            if best is not None:
                used.add(best)
                _cx, _cy, mn, mx = cands[best]
                areas.append(np.pi * mn * mx / 4.0)
    return areas


def derive_detector_params(oracle: dict, data_dir: str, scale: float = 0.5,
                            block_sizes=DEFAULT_BLOCK_SIZES, c_offsets=DEFAULT_C_OFFSETS,
                            ring_min_candidates=DEFAULT_RING_MIN_CANDIDATES,
                            n_sample_frames: int = DEFAULT_N_SAMPLE_FRAMES,
                            match_tol_px: float = DEFAULT_MATCH_TOL_PX,
                            edge_margin_px: float = DEFAULT_EDGE_MARGIN_PX,
                            image_size=(4096, 3000), max_recall_drop_pp: float = DEFAULT_MAX_RECALL_DROP_PP,
                            area_margin_frac: float = DEFAULT_AREA_MARGIN_FRAC,
                            seed: int = 3, name_re=_FRAME_NAME_RE, progress_cb=None) -> dict:
    """Three-stage automatic search -- the same process this project validated
    by hand earlier (search_downsampled_params2.py then
    tune_ring_threshold_downsampled_full.py), generalized to arbitrary
    teacher data and made fully automatic: no threshold is ever typed in,
    and the final pick is an explicit rule, not a judgment call on a table.

    Stage A: grid-search (block_size, c_offset) on a random SAMPLE of
    n_sample_frames teacher frames (for speed), each evaluated WITH the
    ring filter applied at the most lenient candidate in ring_min_candidates
    (representative of realistic final behavior, not raw unfiltered noise).
    Picks the combo with the highest recall, breaking ties by fewest extras.

    Stage B: using that (block_size, c_offset), sweeps every ring_min
    candidate over the FULL teacher oracle (all frames, not just the
    sample -- cheap here since pre_ring_candidates_and_scores is computed
    ONCE per frame and each candidate threshold just re-filters the
    already-computed scores). Picks the LARGEST threshold whose combined
    recall is within max_recall_drop_pp percentage points of the most
    lenient candidate's recall -- see DEFAULT_MAX_RECALL_DROP_PP's comment.

    Stage C: using the final (block_size, c_offset, center_ring_min),
    collects the full-resolution px^2 area of every candidate that matched
    a real oracle point across the full teacher set, then sets
    area_lo/area_hi from the [1st, 99th] percentile of that distribution
    with a +/-area_margin_frac safety margin -- replacing a historical fixed
    area window (838-4818 px^2, from an early full-resolution-only
    investigation, never revisited since) with one derived from this
    dataset's own real target sizes. See DEFAULT_AREA_MARGIN_FRAC's comment.

    Returns {"block_size", "c_offset", "scale", "center_ring_min",
    "area_lo", "area_hi", "diagnostics": {...}} -- diagnostics carries all
    three sweep tables for
    display/logging.
    """
    oracle_frame_names = sorted(n for n, v in oracle.items() if v)
    if not oracle_frame_names:
        raise ValueError("Teacher oracle has no frames with any visible points -- cannot derive parameters.")

    # ---- Stage A: block_size / c_offset on a sample ----
    rng = random.Random(seed)
    sample_names = oracle_frame_names[:]
    rng.shuffle(sample_names)
    sample_names = sample_names[:n_sample_frames]

    loaded_sample = []
    for name in sample_names:
        raw_gray = cv2.imread(os.path.join(data_dir, name), cv2.IMREAD_GRAYSCALE)
        if raw_gray is None:
            continue
        gain = auto_gain(raw_gray)
        gray_full = cv2.convertScaleAbs(raw_gray, alpha=gain, beta=0)
        loaded_sample.append((gray_full, oracle[name]))

    provisional_ring_min = min(ring_min_candidates)
    if progress_cb:
        progress_cb(f"Stage A: grid-searching block_size x C on {len(loaded_sample)} sample frames "
                     f"(provisional ring cutoff={provisional_ring_min})...")

    block_c_results = []
    for block_size in block_sizes:
        for c_offset in c_offsets:
            det_xy_list, expected_list = [], []
            for gray_full, expected in loaded_sample:
                pre_ring, scores = pre_ring_candidates_and_scores(gray_full, block_size=block_size,
                                                                   c_offset=c_offset, scale=scale)
                det_xy = [(cx, cy) for (cx, cy, mn, mx, ang, circ), s in zip(pre_ring, scores)
                          if s >= provisional_ring_min]
                det_xy_list.append(det_xy)
                expected_list.append(expected)
            matched, total_oracle, total_extra = _match_recall_extra(
                det_xy_list, expected_list, match_tol_px, edge_margin_px, image_size)
            recall = matched / total_oracle if total_oracle else 0.0
            block_c_results.append({"block_size": block_size, "c_offset": c_offset, "recall": recall,
                                     "matched": matched, "total_oracle": total_oracle, "extra": total_extra})
            if progress_cb:
                progress_cb(f"  block={block_size:3d} C={c_offset:2d}  recall={100*recall:5.1f}%  extra={total_extra}")

    best_bc = max(block_c_results, key=lambda r: (r["recall"], -r["extra"]))
    block_size, c_offset = best_bc["block_size"], best_bc["c_offset"]
    if progress_cb:
        progress_cb(f"Stage A result: block_size={block_size} C={c_offset} "
                     f"(recall={100*best_bc['recall']:.1f}%, extra={best_bc['extra']})")

    # ---- Stage B: center_ring_min on the FULL teacher oracle ----
    if progress_cb:
        progress_cb(f"Stage B: sweeping center_ring_min on all {len(oracle_frame_names)} teacher frames...")

    loaded_full = []
    for name in oracle_frame_names:
        raw_gray = cv2.imread(os.path.join(data_dir, name), cv2.IMREAD_GRAYSCALE)
        if raw_gray is None:
            continue
        gain = auto_gain(raw_gray)
        gray_full = cv2.convertScaleAbs(raw_gray, alpha=gain, beta=0)
        pre_ring, scores = pre_ring_candidates_and_scores(gray_full, block_size=block_size,
                                                           c_offset=c_offset, scale=scale)
        loaded_full.append((pre_ring, scores, oracle[name]))

    sorted_candidates = sorted(ring_min_candidates)
    ring_results = []
    for ring_min in sorted_candidates:
        det_xy_list, expected_list = [], []
        for pre_ring, scores, expected in loaded_full:
            det_xy = [(cx, cy) for (cx, cy, mn, mx, ang, circ), s in zip(pre_ring, scores) if s >= ring_min]
            det_xy_list.append(det_xy)
            expected_list.append(expected)
        matched, total_oracle, total_extra = _match_recall_extra(
            det_xy_list, expected_list, match_tol_px, edge_margin_px, image_size)
        recall = matched / total_oracle if total_oracle else 0.0
        ring_results.append({"center_ring_min": ring_min, "recall": recall, "matched": matched,
                              "total_oracle": total_oracle, "extra": total_extra})
        if progress_cb:
            progress_cb(f"  ring_min={ring_min:5.1f}  recall={100*recall:5.1f}%  extra={total_extra}")

    recall_max = ring_results[0]["recall"]  # lowest (most lenient) candidate
    qualifying = [r for r in ring_results if r["recall"] >= recall_max - max_recall_drop_pp / 100.0]
    best_ring = max(qualifying, key=lambda r: r["center_ring_min"])
    center_ring_min = best_ring["center_ring_min"]

    if progress_cb:
        progress_cb(f"Stage B result: center_ring_min={center_ring_min} "
                     f"(recall={100*best_ring['recall']:.1f}%, extra={best_ring['extra']}, "
                     f"vs most-lenient recall={100*recall_max:.1f}%, "
                     f"allowed drop={max_recall_drop_pp}pp)")

    # ---- Stage C: area_lo/area_hi from the matched true-positive size distribution ----
    if progress_cb:
        progress_cb("Stage C: deriving area bounds from matched true-positive sizes...")

    tp_areas = _collect_matched_areas(loaded_full, center_ring_min, match_tol_px, edge_margin_px, image_size)
    if tp_areas:
        tp_areas_arr = np.array(tp_areas)
        p_lo, p_hi = np.percentile(tp_areas_arr, 1), np.percentile(tp_areas_arr, 99)
        area_lo = float(p_lo * (1.0 - area_margin_frac))
        area_hi = float(p_hi * (1.0 + area_margin_frac))
    else:
        # No matched candidates at all (shouldn't normally happen if Stage A/B found
        # anything) -- fall back to the historical constants rather than an empty/
        # degenerate [0, 0] window that would reject every future detection.
        area_lo, area_hi = _FALLBACK_AREA_LO, _FALLBACK_AREA_HI

    if progress_cb:
        diam_lo = 2 * np.sqrt(max(area_lo, 0.0) / np.pi)
        diam_hi = 2 * np.sqrt(max(area_hi, 0.0) / np.pi)
        progress_cb(f"Stage C result: area_lo={area_lo:.0f} area_hi={area_hi:.0f} "
                     f"(diameter {diam_lo:.1f}-{diam_hi:.1f}px, from {len(tp_areas)} matched candidates, "
                     f"p1-p99 +/-{area_margin_frac*100:.0f}% margin)")

    return {
        "block_size": block_size, "c_offset": c_offset, "scale": scale, "center_ring_min": center_ring_min,
        "area_lo": area_lo, "area_hi": area_hi,
        "diagnostics": {
            "block_c_sweep": block_c_results, "ring_min_sweep": ring_results,
            "n_sample_frames": len(loaded_sample), "n_full_frames": len(loaded_full),
            "max_recall_drop_pp": max_recall_drop_pp, "n_matched_areas": len(tp_areas),
            "area_margin_frac": area_margin_frac,
        },
    }


def export_review_images(data_dir: str, oracle: dict, derived_params: dict, out_dir: str,
                          match_tol_px: float = DEFAULT_MATCH_TOL_PX, edge_margin_px: float = DEFAULT_EDGE_MARGIN_PX,
                          image_size=(4096, 3000), progress_cb=None) -> str:
    """Writes two folders under out_dir, one pair of images per teacher
    frame, so a person can manually spot-check the automatically-derived
    thresholding without a Python REPL:

      out_dir/binary_mask/<frame_name>.jpg            -- the binary mask actually thresholded
                                                          against (post adaptiveThreshold + morph
                                                          open, at the downsampled resolution the
                                                          detector runs at -- see evaluate_
                                                          downsampled_full.compute_mask), not just
                                                          the raw grayscale frame, so a reviewer sees
                                                          what the detector's shape filter actually
                                                          operates on
      out_dir/annotated_downsampled/<frame_name>.jpg  -- raw gained frame showing only the detected
                                                          blobs (green circle contours at detected position
                                                          and size); all other visualization overlays
                                                          (miss crosses, extra markers, text legends)
                                                          are omitted.

    Re-runs detection with derived_params rather than reusing anything
    cached from derive_detector_params, so this is also callable
    independently (e.g. to re-review a config someone hand-adjusted)."""
    mask_dir = os.path.join(out_dir, "binary_mask")
    annotated_dir = os.path.join(out_dir, "annotated_downsampled")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)

    block_size = derived_params["block_size"]
    c_offset = derived_params["c_offset"]
    scale = derived_params["scale"]
    center_ring_min = derived_params["center_ring_min"]
    area_lo = derived_params.get("area_lo", _FALLBACK_AREA_LO)
    area_hi = derived_params.get("area_hi", _FALLBACK_AREA_HI)
    w, h = image_size

    n_written = 0
    frame_names = sorted(oracle.keys())
    for frame_name in frame_names:
        raw_gray = cv2.imread(os.path.join(data_dir, frame_name), cv2.IMREAD_GRAYSCALE)
        if raw_gray is None:
            continue
        gain = auto_gain(raw_gray)
        gray_full = cv2.convertScaleAbs(raw_gray, alpha=gain, beta=0)

        mask = compute_mask(gray_full, block_size=block_size, c_offset=c_offset, scale=scale)
        cv2.imwrite(os.path.join(mask_dir, frame_name.replace(".png", ".jpg")), mask,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

        pre_ring, scores = pre_ring_candidates_and_scores(gray_full, block_size=block_size,
                                                           c_offset=c_offset, scale=scale,
                                                           area_lo=area_lo, area_hi=area_hi)
        candidates = [(cx, cy, (mn + mx) / 2.0, circ) for (cx, cy, mn, mx, angle, circ), score
                      in zip(pre_ring, scores) if score >= center_ring_min]
        det = _merge_concentric_candidates_relative(candidates, MERGE_FRAC)
        det_xy = [(cx, cy, s) for cx, cy, s in det]

        out = cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR)
        for dx, dy, ds in det_xy:
            cv2.circle(out, (int(round(dx)), int(round(dy))), max(4, int(round(ds / 2.0))), (0, 255, 0), 2)

        cv2.imwrite(os.path.join(annotated_dir, frame_name.replace(".png", ".jpg")), out,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

        n_written += 1
        if progress_cb and n_written % 20 == 0:
            progress_cb(f"  exported {n_written}/{len(frame_names)} review images")

    if progress_cb:
        progress_cb(f"Exported {n_written} frames' binary_mask + annotated_downsampled images -> {out_dir}")
    return out_dir


def derive_from_teacher_config(teacher_config: dict, progress_cb=None, export_dir: str = None) -> dict:
    """Convenience wrapper: build_teacher_oracle() + derive_detector_params()
    from one config dict, the shape the GUI collects from its teacher-dataset
    fields: {data_dir, creaform_calibration_path, creaform_track_csv_path,
    mesh_path, mesh_scale, points3d_path, intrinsics_path, camera_name,
    rig_extrinsics_path (optional, default None), creaform_max_time_diff_ms
    (optional, default DEFAULT_CREAFORM_MAX_TIME_DIFF_MS),
    scale (optional, default 0.5 -- 0.5 = downsampled, 1.0 = full resolution;
    controls both the mask computation scale in Stage A/B and the scale value
    stored in the returned params dict so the detection stage uses the same
    resolution the threshold was derived at) -- see build_teacher_oracle and
    derive_detector_params}. Returns derive_detector_params()'s result dict
    plus the oracle itself (under "oracle") in case a caller wants to
    inspect/reuse it (e.g. to sanity-check the derived config visually).

    If export_dir is given, also calls export_review_images() so the
    resulting binary_mask + annotated_downsampled folders are ready to
    manually inspect as soon as derivation finishes, letting a person
    intervene (e.g. spot a systematic miss/extra pattern) without first
    writing their own export script."""
    oracle = build_teacher_oracle(
        data_dir=teacher_config["data_dir"],
        mesh_path=teacher_config["mesh_path"],
        mesh_scale=teacher_config["mesh_scale"],
        points3d_path=teacher_config["points3d_path"],
        intrinsics_path=teacher_config["intrinsics_path"],
        camera_name=teacher_config["camera_name"],
        creaform_calibration_path=teacher_config.get("creaform_calibration_path"),
        creaform_track_csv_path=teacher_config.get("creaform_track_csv_path"),
        vo_track_path=teacher_config.get("vo_track_path"),
        pose_source=teacher_config.get("pose_source", "creaform"),
        rig_extrinsics_path=teacher_config.get("rig_extrinsics_path"),
        creaform_max_time_diff_ms=teacher_config.get("creaform_max_time_diff_ms",
                                                       DEFAULT_CREAFORM_MAX_TIME_DIFF_MS),
        progress_cb=progress_cb,
    )
    # scale: forwarded from teacher_config so the GUI's resolution-mode
    # choice (downsampled 0.5x vs full-resolution 1.0x) propagates into
    # both the threshold search and the detection stage that uses the result.
    scale = teacher_config.get("scale", 0.5)
    result = derive_detector_params(oracle, teacher_config["data_dir"], scale=scale, progress_cb=progress_cb)
    result["oracle"] = oracle

    if export_dir:
        if progress_cb:
            progress_cb(f"Exporting binary_mask + annotated_downsampled review images -> {export_dir} ...")
        export_review_images(teacher_config["data_dir"], oracle, result, export_dir, progress_cb=progress_cb)
        result["export_dir"] = export_dir

    return result
