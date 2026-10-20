"""Undistort a photo using its camera's calibrated intrinsics, as a
preprocessing step to run BEFORE circle detection (pipeline/circle_tuner_qt.py
/ calib_target/blob_detect.py) -- a one-shot file conversion, not a live/
runtime change to the detection or back-projection pipeline.

Why: calib_target/blob_detect.py currently detects ring markers directly in
the raw (distorted) photo; lens distortion warps a physically circular
marker into a slightly elliptical shape in raw pixels (more so toward the
frame edges), which could bias its circularity/aspect-ratio filters and the
fitted ellipse center. The only undistortion that currently happens is
per-point, late, in pipeline/camera_model.py's `pixel_to_world_ray`
(`cv2.undistortPoints` on the already-detected circle center) -- it never
touches the pixels the detector itself looks at. This script tests the
alternative: undistort the whole image first, then run the EXACT SAME
detection pipeline (e.g. via pipeline/circle_tuner_qt.py, opened on this
script's output instead of the raw photo) to see whether detection quality
improves.

Output image note: deliberately does NOT use `cv2.getOptimalNewCameraMatrix`
-- the output keeps the SAME camera_matrix as the input (cv2.undistort's
default when no `newCameraMatrix` is given). This means pixel (u, v) in the
output is the exact ideal-pinhole projection `pixel_to_normalized_ray_cam`
would already compute for that scene point. Downstream, feed circles
detected on this output through `pipeline.camera_model.
zero_distortion_intrinsics(intr)` instead of `intr` itself (or pass
`--image-undistorted` to pipeline/backproject_targets.py's CLI, which does
exactly that) -- NOT the original distorted intrinsics, which would
incorrectly undistort an already-undistorted point a second time.
"""

import argparse
import os
import sys

import cv2

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_PIPELINE_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from calib_target.pnp_io import load_camera_intrinsics_any
from single_frame_backprojection.camera_model import undistort_image


def main():
    parser = argparse.ArgumentParser(
        description="Undistort a photo using its camera's intrinsics, for testing whether "
                    "circle detection improves when run on the undistorted image instead of "
                    "the raw one."
    )
    parser.add_argument("--image", required=True, help="source photo")
    parser.add_argument("--intrinsics", required=True,
                         help="camera intrinsics: a single-camera cv2.FileStorage YAML, or a "
                              "multi-camera JSON (e.g. intrinsics_sensors_0922.json; requires --camera-name)")
    parser.add_argument("--camera-name", default=None,
                         help="camera name to select within a multi-camera JSON --intrinsics file")
    parser.add_argument("--out", required=True, help="output undistorted image path")
    args = parser.parse_args()

    intr = load_camera_intrinsics_any(args.intrinsics, args.camera_name)
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    undistorted = undistort_image(image, intr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    cv2.imwrite(args.out, undistorted)
    print(f"Wrote undistorted image: {args.out}")
    print("Detect circles on THIS file, then re-run backproject_targets.py with "
          "--image-undistorted so the ray math doesn't undistort the already-undistorted "
          "pixel coordinates a second time.")


if __name__ == "__main__":
    main()
