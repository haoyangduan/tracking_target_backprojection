"""Rigid-transform helpers for aligning a "raw measurement" frame (e.g. a
C-Track/Creaform "body" frame) into the CAD frame the car mesh is
authored in.

Why this exists: a points3d file and the camera pose solved against it are
only meaningful together with whatever mesh shares their frame. Not every
points3d file is already in the mesh's CAD frame -- e.g.
projection_tracking_targets' fit_stickers_to_mesh.py documents a fixed
`cad_T_body` matrix (from an independent jig calibration) that maps raw
C-Track "body"-frame points into BIP_Flat.stl's CAD frame. If a given
session's points3d/pose pair is in that raw body frame, it must be passed
through the matching transform before raycasting against a CAD-frame mesh.

If a session's points3d/pose is already CAD-frame (e.g. solved directly
against a fitted points file like stickers_on_car_surface_mm.txt), no
transform is needed -- pass `None` / identity.
"""

import json
import os

import numpy as np

from calib_target.pnp_io import CameraPose


def load_transform(path_or_matrix, yaml_key: str = "cad_T_body") -> np.ndarray:
    """Accepts a (4,4) array/list, a path to a JSON file containing either
    a flat/nested "matrix" key or a bare 4x4 list, or a path to a plain
    (PyYAML, not cv2.FileStorage) YAML file containing one or more named
    4x4 matrices -- e.g. a `creaform_calibration_*.yml` jig-calibration
    file, which is NOT the cv2.FileStorage format `calib_target/pnp_io.py`
    otherwise reads. `yaml_key` picks which top-level key to read from such
    a file; defaults to "cad_T_body" since that's this project's actual
    use case for one of these files (body-frame -> CAD-frame alignment).
    """
    if path_or_matrix is None:
        return np.eye(4, dtype=np.float64)
    if isinstance(path_or_matrix, (np.ndarray, list, tuple)):
        T = np.asarray(path_or_matrix, dtype=np.float64)
    else:
        path_str = str(path_or_matrix)
        ext = os.path.splitext(path_str)[1].lower()
        if ext in (".yml", ".yaml"):
            import yaml
            with open(path_str, "r") as f:
                data = yaml.safe_load(f)
            if yaml_key not in data:
                matrix_keys = sorted(
                    k for k, v in data.items()
                    if isinstance(v, list) and len(v) == 4 and all(isinstance(row, list) and len(row) == 4 for row in v)
                )
                raise KeyError(f"{path_str}: key {yaml_key!r} not found; available 4x4 matrices: {matrix_keys}")
            T = np.asarray(data[yaml_key], dtype=np.float64)
        else:
            with open(path_str, "r") as f:
                data = json.load(f)
            T = np.asarray(data["matrix"] if isinstance(data, dict) else data, dtype=np.float64)
    T = T.reshape(4, 4)
    return T


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homog = np.hstack([points, np.ones((len(points), 1))])
    return (homog @ T.T)[:, :3]


def transform_pose(T: np.ndarray, pose: CameraPose) -> CameraPose:
    """Re-express a CameraPose (world = body frame) in a new "world" frame
    related to the old one by `new_from_old = T` (rigid, no scale)."""
    T_world_cam_new = T @ pose.T_world_cam
    T_cam_world_new = np.linalg.inv(T_world_cam_new)
    return CameraPose(
        R=T_cam_world_new[:3, :3],
        T_cam_world=T_cam_world_new,
        T_world_cam=T_world_cam_new,
        camera_position_world=T_world_cam_new[:3, 3].copy(),
        rms_reproj_error_px=pose.rms_reproj_error_px,
        num_points=pose.num_points,
        num_inliers=pose.num_inliers,
        coplanarity_ratio=pose.coplanarity_ratio,
        is_coplanar=pose.is_coplanar,
        camera_name=pose.camera_name,
    )
