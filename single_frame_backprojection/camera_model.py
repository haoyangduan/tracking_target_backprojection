"""Pixel -> 3D-ray back-projection using a real (distorted) OpenCV camera
model and an externally-supplied camera pose.

This is the real-camera counterpart of defect_visualization's
`build_ray_to_world()`. Two differences from that Isaac-Sim version:

  1. Lens distortion is undone via `cv2.undistortPoints` before building
     the ray (Isaac Sim's simulated pinhole camera has no distortion).
  2. The ray is built directly in OpenCV camera convention (+Z forward,
     +Y down, +X right) since the pose comes from `cv2.solvePnP`, instead
     of Isaac Sim's OpenGL convention (-Z forward, +Y up) which needed an
     explicit axis sign-flip.

Because the pose (`CameraPose.T_world_cam`) and the object points already
share one frame (the CAD/points3d millimeter frame -- see
calib_target/pnp_io.py), there is no separate car-pose/CAD-offset
composition step here: the ray is built straight into that frame.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from calib_target.pnp_io import CameraIntrinsics, CameraPose


def undistort_image(image: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    """Undistort a full image using its own camera_matrix/dist_coeffs, as a
    preprocessing step BEFORE circle detection (see pipeline/undistort_image.py)
    -- not used anywhere in this module's own ray math, which already
    undoes distortion per-point via `cv2.undistortPoints` instead.

    Deliberately omits `newCameraMatrix` (cv2.undistort then reuses
    `intr.camera_matrix` for the output), so a pixel (u, v) in the result
    is the same ideal pinhole projection `pixel_to_normalized_ray_cam`
    would compute for that scene point -- pair the result with
    `zero_distortion_intrinsics(intr)`, not `intr` itself, for any
    downstream ray math against pixel coordinates measured on this
    undistorted image (e.g. circles detected on it).
    """
    return cv2.undistort(image, intr.camera_matrix, intr.dist_coeffs)


def zero_distortion_intrinsics(intr: CameraIntrinsics) -> CameraIntrinsics:
    """Same camera_matrix/image_size as `intr`, with distortion zeroed out.

    Use this instead of the original `intr` for ray math against pixel
    coordinates detected on an `undistort_image()`-preprocessed image --
    passing the original (non-zero) dist_coeffs there would run
    `cv2.undistortPoints` a second time on an already-undistorted point,
    silently warping it in the wrong direction instead of leaving it alone.
    """
    return CameraIntrinsics(
        camera_matrix=intr.camera_matrix.copy(),
        dist_coeffs=np.zeros_like(intr.dist_coeffs),
        image_size=intr.image_size,
    )


def pixel_to_normalized_ray_cam(u: float, v: float, intr: CameraIntrinsics) -> np.ndarray:
    """Undistort pixel (u, v) and return a unit ray in the CAMERA frame."""
    pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
    undistorted = cv2.undistortPoints(pts, intr.camera_matrix, intr.dist_coeffs)
    x, y = undistorted[0, 0]
    ray_cam = np.array([x, y, 1.0], dtype=np.float64)
    return ray_cam / np.linalg.norm(ray_cam)


def pixel_to_world_ray(u: float, v: float, intr: CameraIntrinsics, pose: CameraPose):
    """Back-project pixel (u, v) into a ray in the world/CAD frame.

    Returns (origin_world, direction_world), both (3,) float64, direction
    normalized.
    """
    ray_cam = pixel_to_normalized_ray_cam(u, v, intr)
    R_world_cam = pose.T_world_cam[:3, :3]
    origin_world = pose.camera_position_world
    ray_world = R_world_cam @ ray_cam
    ray_world = ray_world / np.linalg.norm(ray_world)
    return origin_world, ray_world


@dataclass
class RayHit:
    circle_index: int
    point_line: int
    pixel_u: float
    pixel_v: float
    ray_origin: np.ndarray
    ray_direction: np.ndarray
    hit: bool
    t_hit: float
    estimated_point: np.ndarray  # (3,) mm, car/CAD-local frame; NaNs if no hit
    gt_point: np.ndarray         # (3,) mm, car/CAD-local frame
    error_mm: float              # NaN if no hit
    raw_target_point: np.ndarray = None  # (3,) mm, car/CAD-local frame; the
    # target's raw Creaform/C-Track measurement, BEFORE nearest-surface
    # projection onto the mesh (unlike gt_point) -- None if not supplied
