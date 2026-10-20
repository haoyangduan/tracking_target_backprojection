"""Vendored, trimmed subset of close_track_gui's `calib` package.

Only the pieces needed to (a) load tracking-target circle detections and
(b) load a camera pose already solved by the upstream close_track_gui / PnP
pipeline are kept here. Trimmed from close_track_gui (commit a2d8c92,
"hy/close_track_gui") on 2026-09-24:

  - blob_detect.py  -- verbatim: circular ring-marker detection.
  - points3d.py     -- verbatim: GT 3D reference-point file loader.
  - pnp_io.py        -- trimmed: PnPResult dataclass + cv2.FileStorage
                        loaders for extrinsics.yml / intrinsics.yml,
                        without close_track_gui's calib.config /
                        calib.intrinsics / calib.stereo dependencies
                        (this project never re-solves PnP -- the camera
                        pose is an external input from another pipeline).
"""
