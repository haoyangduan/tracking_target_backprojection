"""Circular ring-marker detection within a user-defined ROI.

Used by the single-image PnP extrinsic-calibration workflow. The
calibration fixture has circular ring-style dot markers (a white/light
sticker with a dark circular outline, bright center) at known 3D
positions. Empirically, `cv2.SimpleBlobDetector` on the bright interior of
these markers gave imprecise, noisy centers, because the bright center is
small, low-contrast, and often distorted by lighting/reflections.

Instead this module thresholds for the *dark ring outline* (which spans
many more pixels and has a much cleaner, higher-contrast edge against the
mid-gray board background), finds its contour, and fits an ellipse
(`cv2.fitEllipse`) to get a sub-pixel-accurate center. Each physical ring
marker typically yields two concentric contours (the ring's inner and
outer edge); these are deduplicated by merging ellipse centers that are
within a small distance of each other, keeping the better-conditioned
(higher-circularity) of the pair.

Detection order is made deterministic (sorted top-to-bottom, then
left-to-right) so re-running detection on the same image/ROI/params always
yields the same `circle_index` for the same physical dot -- this is what
lets a correspondences JSON authored against one run's debug image remain
valid for subsequent runs.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BlobDetectorParams:
    # Grayscale threshold below which pixels are considered part of the
    # dark ring outline. Board background is typically well above this;
    # the ring outline is typically well below it.
    dark_threshold: float = 100.0

    # Contour area filters (pixels^2, in ROI-crop coordinates) -- reject
    # tiny noise specks and large non-ring contours (board edges, text).
    min_area: float = 10.0
    max_area: float = 20.0

    # Roundness filters on the fitted contour: circularity = 4*pi*area /
    # perimeter^2 (1.0 = perfect circle); aspect = minor_axis/major_axis of
    # the fitted ellipse (1.0 = perfect circle, closer to 0 = elongated).
    min_circularity: float = 0.6
    min_aspect_ratio: float = 0.6

    # Two contours (inner/outer ring edge) belonging to the same physical
    # marker are merged if their fitted-ellipse centers are within this
    # many pixels of each other.
    merge_dist_px: float = 0.1


    # Retained for backwards-compatibility with older callers; unused by
    # the ellipse-fitting detector but harmless if passed.
    blob_color: int = 0
    min_dist_between_blobs: float = 5.0


@dataclass
class DetectedCircle:
    index: int
    x: float  # full-image pixel coords
    y: float
    size: float  # mean ellipse-axis diameter estimate


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _fit_ring_ellipses(gray: np.ndarray, params: BlobDetectorParams):
    """Threshold for the dark ring outline and fit an ellipse to every
    contour that plausibly looks like a ring marker. Returns a list of
    (cx, cy, mean_diameter, circularity) tuples in crop-local coordinates.
    """
    _, dark_mask = cv2.threshold(
        gray, params.dark_threshold, 255, cv2.THRESH_BINARY_INV
    )
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    candidates = []
    for c in contours:
        if len(c) < 5:  # fitEllipse requires >= 5 points
            continue
        area = cv2.contourArea(c)
        if area < params.min_area or area > params.max_area:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < params.min_circularity:
            continue

        (cx, cy), (minor_axis, major_axis), _angle = cv2.fitEllipse(c)
        if minor_axis > major_axis:
            minor_axis, major_axis = major_axis, minor_axis
        if major_axis <= 0:
            continue
        aspect = minor_axis / major_axis
        if aspect < params.min_aspect_ratio:
            continue

        mean_diameter = (minor_axis + major_axis) / 2.0
        candidates.append((cx, cy, mean_diameter, circularity))

    return candidates


def _merge_concentric_candidates(candidates, merge_dist_px):
    """Merge candidate ellipse centers that are within merge_dist_px of
    each other (the inner/outer edge of the same physical ring), keeping
    the higher-circularity candidate of each cluster."""
    merged = []
    used = [False] * len(candidates)

    for i, (cx_i, cy_i, size_i, circ_i) in enumerate(candidates):
        if used[i]:
            continue
        cluster = [i]
        for j in range(i + 1, len(candidates)):
            if used[j]:
                continue
            cx_j, cy_j, _size_j, _circ_j = candidates[j]
            dist = np.hypot(cx_i - cx_j, cy_i - cy_j)
            if dist <= merge_dist_px:
                cluster.append(j)

        for idx in cluster:
            used[idx] = True

        # Keep the candidate with the best circularity in this cluster.
        best_idx = max(cluster, key=lambda k: candidates[k][3])
        best_cx, best_cy, best_size, _best_circ = candidates[best_idx]

        # Average centers across the cluster for a slightly more stable
        # estimate (inner/outer edge centers of a true concentric ring
        # should coincide almost exactly; averaging reduces noise).
        avg_cx = float(np.mean([candidates[k][0] for k in cluster]))
        avg_cy = float(np.mean([candidates[k][1] for k in cluster]))

        merged.append((avg_cx, avg_cy, best_size))

    return merged


def _merge_concentric_candidates_relative(candidates, merge_frac):
    """Like _merge_concentric_candidates, but merges candidates whose
    center distance is within merge_frac of their PAIR-AVERAGE size, rather
    than a fixed absolute pixel distance -- scales correctly for both near
    (large apparent marker) and far (small apparent marker) stickers,
    unlike a flat px threshold tuned for one marker size.

    Needed because RETR_LIST returns both a ring's inner and outer contour
    as separate candidates for the SAME physical sticker (see evaluate_
    against_oracle.detect()'s docstring on why RETR_LIST, not RETR_EXTERNAL,
    is required), and the two contours' center offset scales with the
    ring's own apparent size, not a fixed pixel count: measured on real
    production data, true same-sticker duplicate pairs have center offsets
    up to ~9% of their average size (e.g. a 45px and a 76px candidate --
    inner vs. outer ring edge -- 2-5px apart), while the closest any two
    DIFFERENT real stickers ever get in this project's frames is ~108px --
    three orders of magnitude more headroom than needed, so a generous
    merge_frac here cannot accidentally conflate distinct targets."""
    merged = []
    used = [False] * len(candidates)

    for i, (cx_i, cy_i, size_i, circ_i) in enumerate(candidates):
        if used[i]:
            continue
        cluster = [i]
        for j in range(i + 1, len(candidates)):
            if used[j]:
                continue
            cx_j, cy_j, size_j, _circ_j = candidates[j]
            dist = np.hypot(cx_i - cx_j, cy_i - cy_j)
            avg_size = (size_i + size_j) / 2.0
            if avg_size > 0 and dist <= merge_frac * avg_size:
                cluster.append(j)

        for idx in cluster:
            used[idx] = True

        best_idx = max(cluster, key=lambda k: candidates[k][3])
        _best_cx, _best_cy, best_size, _best_circ = candidates[best_idx]

        avg_cx = float(np.mean([candidates[k][0] for k in cluster]))
        avg_cy = float(np.mean([candidates[k][1] for k in cluster]))

        merged.append((avg_cx, avg_cy, best_size))

    return merged


def detect_circles(
    image: np.ndarray,
    roi: tuple,
    params: BlobDetectorParams = None,
    refine_subpixel: bool = True,
    subpixel_win: tuple = (5, 5),
) -> list:
    """Detect circular ring markers inside `roi` of `image`.

    Args:
        image: full BGR or grayscale image.
        roi: (x, y, w, h) in full-image pixel coordinates.
        params: BlobDetectorParams (defaults used if None).
        refine_subpixel: kept for API compatibility; the ellipse-fitting
            approach already yields sub-pixel centers directly from the
            contour, so this flag has no additional effect unless a future
            revision adds a secondary refinement pass.
        subpixel_win: unused by the ellipse-fitting detector; retained for
            API compatibility.

    Returns:
        List[DetectedCircle], sorted top-to-bottom then left-to-right,
        with `index` assigned in that sorted order and (x, y) in
        full-image pixel coordinates.
    """
    if params is None:
        params = BlobDetectorParams()

    x, y, w, h = roi
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid ROI {roi!r}: width and height must be positive")

    ih, iw = image.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(iw, int(x + w)), min(ih, int(y + h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI {roi!r} does not intersect image of size {(iw, ih)}")

    crop = image[y0:y1, x0:x1]
    gray = _to_gray(crop)

    candidates = _fit_ring_ellipses(gray, params)
    merged = _merge_concentric_candidates(candidates, params.merge_dist_px)

    # Sort deterministically: top-to-bottom, then left-to-right.
    merged = sorted(merged, key=lambda t: (round(t[1], 1), round(t[0], 1)))

    circles = []
    for i, (cx, cy, size) in enumerate(merged):
        circles.append(DetectedCircle(index=i, x=cx + x0, y=cy + y0, size=size))

    return circles


def detect_circles_multi_roi(
    image: np.ndarray,
    rois_and_params: list,
    refine_subpixel: bool = True,
    subpixel_win: tuple = (5, 5),
) -> list:
    """Detect circular ring markers across multiple ROIs, each with its own
    detector params, and return one flat, sequentially-indexed circle list.

    Args:
        image: full BGR or grayscale image.
        rois_and_params: list of (roi, params) tuples, where `roi` is
            (x, y, w, h) in full-image pixel coordinates and `params` is a
            BlobDetectorParams instance for that ROI. Detection is run
            independently per (roi, params) pair, in the order given, and
            results are concatenated.
        refine_subpixel: see `detect_circles`.
        subpixel_win: see `detect_circles`.

    Returns:
        List[DetectedCircle] with `index` assigned sequentially across all
        ROIs in the order given (e.g. if ROI 1 yields 5 circles indexed
        0..4, ROI 2's first circle is index 5), so a flat correspondences
        JSON referencing `circle_index` remains unambiguous regardless of
        how many ROIs were used.
    """
    all_circles = []
    next_index = 0
    for roi, params in rois_and_params:
        circles = detect_circles(
            image, roi, params,
            refine_subpixel=refine_subpixel, subpixel_win=subpixel_win,
        )
        for c in circles:
            all_circles.append(
                DetectedCircle(index=next_index, x=c.x, y=c.y, size=c.size)
            )
            next_index += 1
    return all_circles


def draw_circles(image: np.ndarray, circles: list, roi=None) -> np.ndarray:
    """Return a copy of `image` with detected circles + index labels drawn,
    and optionally ROI rectangle(s).

    `roi` may be a single (x, y, w, h) tuple, a list of such tuples (to draw
    multiple ROI rectangles), or None.
    """
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    if roi is not None:
        rois = roi if isinstance(roi, list) else [roi]
        for r in rois:
            x, y, w, h = r
            cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)

    for c in circles:
        center = (int(round(c.x)), int(round(c.y)))
        radius = max(3, int(round(c.size / 2)))

        cv2.circle(out, center, radius, (0, 255, 0), 2)
        cv2.drawMarker(out, center, (0, 255, 255), cv2.MARKER_CROSS, 8, 1)
        cv2.putText(
            out,
            str(c.index),
            (center[0] + radius + 2, center[1] - radius - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    return out
