"""Evaluate the current detector (auto-gain + FULL-RESOLUTION adaptive
threshold + morph cleanup + fixed shape filter, NO stereo/ring/contrast
extra layers) against oracle_labels_full.json across all 181 VO-covered
frames.

Current validated result (RETR_LIST+CHAIN_APPROX_NONE, 3x3 morph open):
legacy recall 87.2% (455/522), new door-region recall 29.4% (40/136),
combined 75.2% (495/658), extra 2444 total (13.5/frame). This is BETTER
than the original no-morph baseline (84.7%/27.2%/72.8%, extra 5.29/frame)
on recall, at the cost of ~2.5x more extras -- see CONTOUR MODE BUG below
for why intermediate results during this investigation were much worse.

CONTOUR MODE BUG (found and fixed): for a while, this file's detect()
used cv2.RETR_EXTERNAL instead of cv2.RETR_LIST, which silently collapsed
recall to ~37% legacy / ~29% combined -- NOT a real regression from morph
or downsampling, but from RETR_EXTERNAL dropping real sticker-ring
contours that are frequently NESTED inside larger adaptive-threshold
blobs (local background bleed). Confirmed directly on 8 hand-labeled
frames: RETR_LIST+NONE matched 60/63 known circles with the SAME mask
that RETR_EXTERNAL+SIMPLE matched only 10/63 on, even with morph OFF for
both. This means every morph/downsampling accuracy comparison made while
RETR_EXTERNAL was in place (the "<1px shift, 0 lost targets" morph
validation, the downsampled-param grid search) was comparing two
already-crippled variants against each other, not against the true
original. Fixed by reverting to RETR_LIST+CHAIN_APPROX_NONE here.

Downsampling (0.5x) was tried as a speed optimization and abandoned per
user instruction ("give up downsampling") after it could not close the
30fps gap and introduced a separate resolution-dependent noise problem
at high gain -- that conclusion still stands independent of the
RETR_EXTERNAL bug above (downsampling was evaluated as its own, self-
contained comparison in search_downsampled_params.py / export_eval_
annotated_compare.py, not against this file's detect()).

Uses the COMPLETE 138-point oracle (build_oracle_labels_full.py: full
tracking model + full with-door mesh, single consistent occlusion check)
rather than the door-free 90-point version -- the most complete automatic
ground truth available, so an "extra" (unmatched) detection is far more
likely to be a genuine false positive now than a real-but-uncovered
sticker. Not guaranteed, though: the full model only has 138 of however
many physical targets actually exist, and this is still a DOOR-CLOSED,
static-mesh assumption -- if a real captured frame had a door open, or a
sticker not in this tracking model at all, an unmatched detection there
would still not be a true error. Spot-check before concluding anything
from "extra" counts alone.
"""
import json
import os

import cv2
import numpy as np

from live_backprojection import _pathsetup  # noqa: F401

from calib_target.blob_detect import _to_gray, _merge_concentric_candidates_relative

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_THIS_DIR, "out")

DATA_DIR = os.path.join(_pathsetup.PROJECT_ROOT, "data", "hik_DA0179014")
ORACLE_PATH = os.path.join(_OUT_DIR, "oracle_labels_full.json")
TARGET_MEAN = 87.0
GAIN_MIN, GAIN_MAX = 1.0, 8.0
BLOCK_SIZE, C_OFFSET = 151, 5
AREA_LO, AREA_HI, CIRC_LO, ASPECT_LO, MERGE_DIST_PX = 838.0, 4818.0, 0.301, 0.607, 2.0
# Size-RELATIVE merge fraction (replaces the flat MERGE_DIST_PX above for the actual merge step --
# MERGE_DIST_PX is kept only for signature/back-compat, see detect()'s merge_dist_px param): RETR_LIST
# returns a ring's inner AND outer contour as separate candidates for the SAME physical sticker, and
# measured on real data their center offset scales with marker size (up to ~9% of avg size), not a
# fixed pixel count -- a flat 2.0px missed ~20 same-sticker duplicate pairs with 2-5px offsets on
# 40-90px markers that were clearly visible as two overlapping circles in annotated review images.
# The closest two DIFFERENT real stickers ever get in this project's frames is ~108px, so this
# fraction has three orders of magnitude of headroom before it could risk merging distinct targets.
MERGE_FRAC = 0.15
MATCH_TOL_PX = 35.0  # generous: VO's own observed accuracy ranged up to ~24px
IMAGE_SIZE = (4096, 3000)
EDGE_MARGIN_PX = 100.0  # oracle points this close to the frame boundary are excluded --
                         # likely partially clipped out of the actual sensor view even
                         # when their projected center technically counts as "in bounds"
                         # (confirmed real cases: point_line 73/76 at u~4060-4090, against
                         # a 4096px-wide frame)


def near_edge(u, v, w=IMAGE_SIZE[0], h=IMAGE_SIZE[1], margin=EDGE_MARGIN_PX):
    return u < margin or u > w - margin or v < margin or v > h - margin


def auto_gain(raw_gray):
    raw_mean = float(raw_gray.mean())
    if raw_mean <= 0:
        return GAIN_MAX
    return float(np.clip(TARGET_MEAN / raw_mean, GAIN_MIN, GAIN_MAX))


MORPH_KERNEL = np.ones((3, 3), np.uint8)  # validated on the full 181-frame oracle (RETR_LIST+NONE,
                                           # the correct contour mode -- see module docstring for the
                                           # RETR_EXTERNAL bug this superseded): removes most noise
                                           # contours while INCREASING recall vs no-morph (87.2% vs
                                           # 84.7% legacy) at the cost of ~2.5x more extras (13.5 vs
                                           # 5.29/frame) -- morph cleans up irregular real-target
                                           # blobs into cleaner ellipses that pass the shape filter
                                           # more often, not a pure noise-removal step as first assumed.
                                           # Larger kernels (9x9+) start eroding real ring strokes
                                           # with no further speed benefit; 0.5x downsampling was
                                           # tried and abandoned (best achievable recall after a
                                           # dedicated grid search on downsampled images: 51.6%,
                                           # vs full-res's 87.2% legacy recall -- a real information-
                                           # loss ceiling from the resize, not a tunable parameter
                                           # problem). This detect() is FULL RESOLUTION, morph-only:
                                           # ~175-350ms/frame measured (higher gain = more contours
                                           # = slower), not 30fps-capable on its own -- ROI restriction (not
                                           # downsampling) is the next lever to close that gap,
                                           # since it reduces pixel area without shrinking/
                                           # distorting the targets themselves.


CENTER_RING_MIN = 20.0  # minimum (center_brightness - ring_brightness), sampled from the
                         # ORIGINAL grayscale (not the binary mask). Real targets are a dark
                         # ring around a BRIGHT interior; solid holes/bolts are dark all the
                         # way through (the known FP taxonomy). Validated on a 40-frame random
                         # sample: TP scores are ALL >=23.7 (10th percentile), FP scores are
                         # much lower on average and often negative -- at thr=20, keeps 100%
                         # of true positives while removing 77.6% of false positives (631->141
                         # candidates in that sample). A simple inner/outer edge-contrast
                         # metric (tried first) did NOT separate TP/FP at all; this
                         # brightness-polarity metric is the one that actually works.


def _ellipse_ring_sample(gray, cx, cy, minor, major, angle_deg, r_scale, n_samples):
    h, w = gray.shape
    a = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    ts = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    rmaj, rmin = major / 2.0 * r_scale, minor / 2.0 * r_scale
    xl, yl = rmaj * np.cos(ts), rmin * np.sin(ts)
    xr = np.clip(cx + xl * cos_a - yl * sin_a, 0, w - 1).astype(np.float32).reshape(-1, 1)
    yr = np.clip(cy + xl * sin_a + yl * cos_a, 0, h - 1).astype(np.float32).reshape(-1, 1)
    patch = cv2.remap(gray, xr, yr, interpolation=cv2.INTER_LINEAR)
    return patch.reshape(-1).astype(np.float32)


def center_ring_contrast(gray, cx, cy, minor, major, angle_deg):
    """center_brightness (sampled over several interior radii) minus
    ring_brightness (sampled right at the fitted ellipse boundary)."""
    if (minor + major) <= 2.0:
        return 0.0
    ring_b = float(_ellipse_ring_sample(gray, cx, cy, minor, major, angle_deg, 1.0, 48).mean())
    h, w = gray.shape
    xi, yi = int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1))
    inner_vals = [float(gray[yi, xi])]
    for rs in (0.15, 0.3, 0.45):
        inner_vals.extend(_ellipse_ring_sample(gray, cx, cy, minor, major, angle_deg, rs, 16).tolist())
    center_b = float(np.mean(inner_vals))
    return center_b - ring_b


_INNER_SCALES = (0.15, 0.3, 0.45)
_RING_N_SAMPLES = 48
_INNER_N_SAMPLES = 16


def center_ring_contrast_batch(gray, cands):
    """Vectorized center_ring_contrast() for ALL candidates in a frame at once.
    Mathematically identical to calling center_ring_contrast() per candidate
    (same sample points, same formula) -- but issues only 4 cv2.remap calls
    total (1 ring radius + 3 inner radii), instead of 4 calls PER CANDIDATE,
    since most of the per-call cost is fixed overhead, not sample-point count.
    `cands` is a list of (cx, cy, minor, major, angle_deg). Returns a numpy
    array of scores, same order as `cands`."""
    n = len(cands)
    if n == 0:
        return np.zeros(0)
    h, w = gray.shape
    cx = np.array([c[0] for c in cands])
    cy = np.array([c[1] for c in cands])
    minor = np.array([c[2] for c in cands])
    major = np.array([c[3] for c in cands])
    angle = np.deg2rad(np.array([c[4] for c in cands]))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rmaj, rmin = major / 2.0, minor / 2.0
    valid = (minor + major) > 2.0

    def sample_batch(r_scale, n_samples):
        ts = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        xl = (rmaj * r_scale)[:, None] * np.cos(ts)[None, :]
        yl = (rmin * r_scale)[:, None] * np.sin(ts)[None, :]
        xr = np.clip(cx[:, None] + xl * cos_a[:, None] - yl * sin_a[:, None], 0, w - 1).astype(np.float32)
        yr = np.clip(cy[:, None] + xl * sin_a[:, None] + yl * cos_a[:, None], 0, h - 1).astype(np.float32)
        return cv2.remap(gray, xr, yr, interpolation=cv2.INTER_LINEAR).astype(np.float32)  # (n, n_samples)

    ring_vals = sample_batch(1.0, _RING_N_SAMPLES)
    ring_b = ring_vals.mean(axis=1)

    xi = np.clip(cx, 0, w - 1).astype(int)
    yi = np.clip(cy, 0, h - 1).astype(int)
    inner_sum = gray[yi, xi].astype(np.float64)
    inner_count = np.ones(n)
    for rs in _INNER_SCALES:
        vals = sample_batch(rs, _INNER_N_SAMPLES)
        inner_sum += vals.sum(axis=1)
        inner_count += _INNER_N_SAMPLES
    center_b = inner_sum / inner_count

    scores = center_b - ring_b
    scores[~valid] = 0.0
    return scores


def detect(gray, block_size=BLOCK_SIZE, c_offset=C_OFFSET, area_lo=AREA_LO, area_hi=AREA_HI,
           circ_lo=CIRC_LO, aspect_lo=ASPECT_LO, merge_frac=MERGE_FRAC, center_ring_min=CENTER_RING_MIN):
    """Full-resolution, morph-cleaned detection -- see MORPH_KERNEL docstring
    for why downsampling was tried and abandoned in favor of this."""
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_offset)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    # RETR_LIST (not RETR_EXTERNAL) is required: real sticker-ring contours are
    # frequently NESTED inside larger adaptive-threshold blobs (local background
    # bleed) -- i.e. the real target is often a HOLE inside a bigger blob, not a
    # separate foreground island -- so RETR_EXTERNAL (top-level only) or
    # connectedComponents (foreground-blobs only, same blind spot) silently drop
    # most real targets. Confirmed twice: on 8 hand-labeled frames, RETR_LIST+NONE
    # matched 60/63 vs RETR_EXTERNAL+SIMPLE's 10/63 on the SAME mask; on the full
    # 181-frame oracle, swapping in connectedComponentsWithStats dropped legacy
    # recall from 74.3% to 47.3%. CHAIN_APPROX_SIMPLE (vs NONE) was separately
    # isolated and found to be a pure win (slightly BETTER recall, fewer extras,
    # ~25% faster) since it only changes point storage, not retrieval semantics.
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    pre_ring = []
    for c in contours:
        if len(c) < 5:
            continue
        area = cv2.contourArea(c)
        if area < area_lo or area > area_hi:
            continue
        perim = cv2.arcLength(c, True)
        if perim <= 0:
            continue
        circularity = 4 * np.pi * area / (perim * perim)
        if circularity < circ_lo:
            continue
        (cx, cy), (minor, major), angle = cv2.fitEllipse(c)
        if minor > major:
            minor, major = major, minor
        if major <= 0 or minor / major < aspect_lo:
            continue
        pre_ring.append((cx, cy, minor, major, angle, circularity))

    # Vectorized ring-filter check: one batch of cv2.remap calls for ALL
    # surviving candidates instead of one call-set per candidate -- same
    # scores (verified byte-identical to the per-candidate version), just
    # amortizes cv2.remap's fixed per-call overhead across the whole frame.
    candidates = []
    if center_ring_min is not None and pre_ring:
        scores = center_ring_contrast_batch(gray, [(cx, cy, mn, mx, ang) for cx, cy, mn, mx, ang, _ in pre_ring])
        for (cx, cy, minor, major, angle, circularity), score in zip(pre_ring, scores):
            if score < center_ring_min:
                continue
            candidates.append((cx, cy, (minor + major) / 2.0, circularity))
    else:
        for cx, cy, minor, major, angle, circularity in pre_ring:
            candidates.append((cx, cy, (minor + major) / 2.0, circularity))
    return _merge_concentric_candidates_relative(candidates, merge_frac)


def evaluate(detect_kwargs=None, verbose=True, exclude_edge=True):
    detect_kwargs = detect_kwargs or {}
    oracle = json.load(open(ORACLE_PATH))

    total_extra = 0
    n_excluded_edge = 0
    # Separate legacy (1-90, extensively validated) vs new door-region (91-138,
    # added from new_car_220926.txt, only aggregate-validated) counters --
    # confirmed earlier that some new-door-region "misses" show no visible
    # marker at all, i.e. may not be reliable ground truth; reporting them
    # separately avoids that contaminating the legacy recall number.
    stats = {"legacy": {"total": 0, "matched": 0}, "new": {"total": 0, "matched": 0}}
    per_frame = []
    miss_records, extra_records = [], []

    for frame_name, expected in oracle.items():
        # These PNGs are natively single-channel grayscale on disk -- IMREAD_GRAYSCALE
        # gives byte-identical pixels to _to_gray(IMREAD_COLOR) (verified: max abs
        # diff 0) while skipping the wasted decode-to-3-channel-then-convert-back
        # round trip. Measured: ~53ms/frame saved (~19% of total pipeline time),
        # zero accuracy cost.
        raw_gray = cv2.imread(f"{DATA_DIR}/{frame_name}", cv2.IMREAD_GRAYSCALE)
        gain = auto_gain(raw_gray)
        gray = cv2.convertScaleAbs(raw_gray, alpha=gain, beta=0)

        det = detect(gray, **detect_kwargs)
        det_xy = [(cx, cy, s) for cx, cy, s in det]

        if exclude_edge:
            kept, excluded = [], 0
            for e in expected:
                if near_edge(e["u"], e["v"]):
                    excluded += 1
                else:
                    kept.append(e)
            n_excluded_edge += excluded
            expected = kept

        used = set()
        for pl_entry in expected:
            pu, pv = pl_entry["u"], pl_entry["v"]
            group = "legacy" if pl_entry["point_line"] <= 90 else "new"
            stats[group]["total"] += 1
            best, best_d = None, MATCH_TOL_PX
            for i, (dx, dy, ds) in enumerate(det_xy):
                if i in used:
                    continue
                d = np.hypot(pu - dx, pv - dy)
                if d < best_d:
                    best, best_d = i, d
            if best is not None:
                used.add(best)
                stats[group]["matched"] += 1
            else:
                miss_records.append({"frame": frame_name, "point_line": pl_entry["point_line"], "u": pu, "v": pv, "gain": gain})

        n_matched_total = len(used)
        n_extra = len(det_xy) - n_matched_total
        for i, (dx, dy, ds) in enumerate(det_xy):
            if i not in used:
                extra_records.append({"frame": frame_name, "u": dx, "v": dy, "size": ds, "gain": gain})

        total_extra += n_extra
        per_frame.append((frame_name, len(expected), n_matched_total, n_extra))

    total_oracle = stats["legacy"]["total"] + stats["new"]["total"]
    total_matched = stats["legacy"]["matched"] + stats["new"]["matched"]

    if verbose:
        print(f"Edge margin applied: {EDGE_MARGIN_PX:.0f}px (excluded {n_excluded_edge} oracle point-instances)")
        print(f"\n=== LEGACY (1-90, extensively validated) ===")
        lt, lm = stats["legacy"]["total"], stats["legacy"]["matched"]
        print(f"  Matched (recall): {lm}/{lt} ({100*lm/lt:.1f}%)" if lt else "  (none)")
        print(f"\n=== NEW door-region (91-138, aggregate-validated only) ===")
        nt, nm = stats["new"]["total"], stats["new"]["matched"]
        print(f"  Matched (recall): {nm}/{nt} ({100*nm/nt:.1f}%)" if nt else "  (none)")
        print(f"\n=== COMBINED ===")
        print(f"  Oracle points total: {total_oracle}")
        print(f"  Matched (recall)   : {total_matched}/{total_oracle} ({100*total_matched/total_oracle:.1f}%)")
        print(f"  Extra (unmatched)  : {total_extra} total, {total_extra/len(oracle):.2f}/frame average")
        print(f"  Missed             : {total_oracle - total_matched}")

    return {
        "total_oracle": total_oracle, "total_matched": total_matched, "total_extra": total_extra,
        "stats_by_group": stats, "n_excluded_edge": n_excluded_edge,
        "per_frame": per_frame, "miss_records": miss_records, "extra_records": extra_records,
    }


if __name__ == "__main__":
    result = evaluate()
    _out_path = os.path.join(_OUT_DIR, "eval_full_oracle.json")
    with open(_out_path, "w") as f:
        json.dump({k: v for k, v in result.items() if k != "per_frame"}, f, indent=1)
    print(f"\nSaved detailed miss/extra records to {_out_path}")
