"""Full 181-frame oracle evaluation of the 0.5x-downsampled detector
(block_size=81, C=4, grid-searched in search_downsampled_params2.py) with
the center_ring_contrast filter verified against full-resolution grayscale.
Directly comparable to evaluate_against_oracle.py's full-res numbers
(legacy/new split, same edge margin, same matching tolerance).
"""
import json

import cv2
import numpy as np

from live_backprojection import _pathsetup  # noqa: F401

from calib_target.blob_detect import _to_gray, _merge_concentric_candidates_relative
from live_backprojection.detector.evaluate_against_oracle import (
    auto_gain, near_edge, MATCH_TOL_PX, ORACLE_PATH, DATA_DIR,
    CIRC_LO, ASPECT_LO, CENTER_RING_MIN, MERGE_FRAC, center_ring_contrast_batch,
)

SCALE = 0.5
BLOCK_SIZE, C_OFFSET = 81, 4
AREA_LO, AREA_HI = 838.0, 4818.0


def compute_mask(gray_full, block_size=BLOCK_SIZE, c_offset=C_OFFSET, scale=SCALE):
    """The binary mask detect_half_ring()/pre_ring_candidates_and_scores()
    threshold against: resize -> adaptiveThreshold -> morph open. Exposed
    separately so a caller that wants to inspect/export the mask itself
    (e.g. live_backprojection/teacher.py's review-image export) doesn't
    need to duplicate this exact sequence -- single source of truth for
    what the detector actually thresholds against."""
    gray_small = cv2.resize(gray_full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = cv2.adaptiveThreshold(gray_small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_offset)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def pre_ring_candidates_and_scores(gray_full, block_size=BLOCK_SIZE, c_offset=C_OFFSET, scale=SCALE,
                                    area_lo=AREA_LO, area_hi=AREA_HI):
    """Everything detect_half_ring() does up to (but not including) the
    center_ring_min cutoff + merge: mask, findContours, shape filter, then
    center_ring_contrast_batch scores for the survivors. Exposed separately
    (not just inlined in detect_half_ring) so a multi-threshold sweep --
    e.g. live_backprojection/teacher.py's automatic parameter derivation --
    can do this work ONCE per frame instead of once per candidate
    center_ring_min value, since none of it (mask/contours/shape filter/
    scores) depends on center_ring_min at all.

    area_lo/area_hi are in FULL-RESOLUTION px^2 units (scaled internally by
    scale^2 to match the downsampled mask) and are real parameters -- not
    hardcoded -- so they can be derived from real target-size data (see
    live_backprojection/teacher.py) instead of left as a historical fixed
    constant that never adapts to a different camera/marker-size setup.

    Returns (pre_ring, scores): pre_ring is a list of
    (cx, cy, minor, major, angle, circularity) in FULL-RESOLUTION pixel
    coords; scores[i] is pre_ring[i]'s center_ring_contrast value. Empty
    scores is a zero-length array, not a Python list, matching
    center_ring_contrast_batch's return type."""
    mask = compute_mask(gray_full, block_size, c_offset, scale)
    # CHAIN_APPROX_SIMPLE: validated as a pure win on the full oracle (slightly
    # better recall, fewer extras, ~25% faster than NONE) -- see detect()'s
    # comment in evaluate_against_oracle.py.
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area_lo_scaled, area_hi_scaled = area_lo * scale * scale, area_hi * scale * scale
    pre_ring = []
    for c in contours:
        if len(c) < 5:
            continue
        area = cv2.contourArea(c)
        if area < area_lo_scaled or area > area_hi_scaled:
            continue
        perim = cv2.arcLength(c, True)
        if perim <= 0:
            continue
        circ = 4 * np.pi * area / (perim * perim)
        if circ < CIRC_LO:
            continue
        (cx, cy), (mn, mx), angle = cv2.fitEllipse(c)
        if mn > mx:
            mn, mx = mx, mn
        if mx <= 0 or mn / mx < ASPECT_LO:
            continue
        pre_ring.append((cx / scale, cy / scale, mn / scale, mx / scale, angle, circ))

    if pre_ring:
        scores = center_ring_contrast_batch(gray_full, [(cx, cy, mn, mx, ang) for cx, cy, mn, mx, ang, _ in pre_ring])
    else:
        scores = np.zeros(0)
    return pre_ring, scores


def detect_half_ring(gray_full, block_size=BLOCK_SIZE, c_offset=C_OFFSET, scale=SCALE, center_ring_min=CENTER_RING_MIN,
                      area_lo=AREA_LO, area_hi=AREA_HI):
    pre_ring, scores = pre_ring_candidates_and_scores(gray_full, block_size, c_offset, scale, area_lo, area_hi)
    candidates = []
    for (cx, cy, mn, mx, angle, circ), score in zip(pre_ring, scores):
        if center_ring_min is not None and score < center_ring_min:
            continue
        candidates.append((cx, cy, (mn + mx) / 2.0, circ))
    return _merge_concentric_candidates_relative(candidates, MERGE_FRAC)


def main():
    oracle = json.load(open(ORACLE_PATH))
    stats = {"legacy": {"total": 0, "matched": 0}, "new": {"total": 0, "matched": 0}}
    total_extra = 0
    n_excluded_edge = 0

    for frame_name, expected_all in oracle.items():
        # Natively single-channel grayscale PNGs -- read directly, skipping the
        # wasted decode-to-BGR-then-convert-back round trip (see evaluate_against_
        # oracle.py for the byte-identical-pixels verification). ~19% total pipeline
        # time saved, zero accuracy cost.
        raw_gray = cv2.imread(f"{DATA_DIR}/{frame_name}", cv2.IMREAD_GRAYSCALE)
        if raw_gray is None:
            continue
        gain = auto_gain(raw_gray)
        gray_full = cv2.convertScaleAbs(raw_gray, alpha=gain, beta=0)

        det = detect_half_ring(gray_full)
        det_xy = [(cx, cy, s) for cx, cy, s in det]

        kept, excluded = [], 0
        for e in expected_all:
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
        total_extra += len(det_xy) - len(used)

    total_oracle = stats["legacy"]["total"] + stats["new"]["total"]
    total_matched = stats["legacy"]["matched"] + stats["new"]["matched"]

    print(f"Downsampled 0.5x + ring filter (block_size={BLOCK_SIZE}, C={C_OFFSET})")
    print(f"Edge margin applied (excluded {n_excluded_edge} oracle point-instances)")
    lt, lm = stats["legacy"]["total"], stats["legacy"]["matched"]
    nt, nm = stats["new"]["total"], stats["new"]["matched"]
    print(f"\n=== LEGACY ===\n  Matched: {lm}/{lt} ({100*lm/lt:.1f}%)")
    print(f"\n=== NEW door-region ===\n  Matched: {nm}/{nt} ({100*nm/nt:.1f}%)")
    print(f"\n=== COMBINED ===")
    print(f"  Matched: {total_matched}/{total_oracle} ({100*total_matched/total_oracle:.1f}%)")
    print(f"  Extra: {total_extra} total, {total_extra/len(oracle):.2f}/frame")


if __name__ == "__main__":
    main()
