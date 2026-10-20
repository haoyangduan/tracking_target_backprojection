"""Loaders for camera intrinsics and a solved camera pose (PnP result).

Both file formats are cv2.FileStorage YAML, written by close_track_gui's
`calib.io_yaml` (see calib/intrinsics.py's IntrinsicsResult and
calib/pnp.py's PnPResult upstream). This project never solves PnP itself --
the camera pose is a given input, already computed by another pipeline
(e.g. close_track_gui's `extrinsics_pnp.py solve-pnp`, or any equivalent
tool that writes the same T_world_cam / T_cam_world convention). Only the
read side is reproduced here, trimmed to drop the calib.config /
calib.intrinsics / calib.stereo dataclass dependencies.
"""

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np


def _open_for_read(path: str) -> cv2.FileStorage:
    """Open a cv2.FileStorage YAML for reading, with a clear error instead
    of what a missing/unreadable file actually does in cv2's Python
    bindings -- which varies by OpenCV build: some versions just log a C++
    error and hand back a closed FileStorage (`isOpened() == False`),
    others raise an opaque `SystemError: <class 'cv2.FileStorage'>
    returned a result with an exception set` straight out of the
    constructor. Checking existence first, then `isOpened()`, catches both
    without depending on which behavior the caller's OpenCV build has."""
    path_str = str(path)
    if not os.path.isfile(path_str):
        raise FileNotFoundError(f"No such file: {path_str}")
    try:
        fs = cv2.FileStorage(path_str, cv2.FILE_STORAGE_READ)
    except Exception as exc:
        # Some OpenCV builds throw straight out of the constructor for a
        # malformed file (sometimes as an opaque `SystemError`) instead of
        # just handing back a closed FileStorage -- normalize either way.
        raise RuntimeError(
            f"Could not open as a cv2.FileStorage YAML (corrupt or wrong format?): {path_str}"
        ) from exc
    if not fs.isOpened():
        raise RuntimeError(f"Could not open as a cv2.FileStorage YAML (corrupt or wrong format?): {path_str}")
    return fs


@dataclass
class CameraIntrinsics:
    camera_matrix: np.ndarray  # (3, 3)
    dist_coeffs: np.ndarray    # (1, N) or (N, 1)
    image_size: tuple          # (width, height)


@dataclass
class CameraPose:
    """A solved camera pose, in the same convention as close_track_gui's
    calib.pnp.PnPResult.

    T_cam_world: transforms a world-frame point into the camera frame
                 (p_cam = T_cam_world @ [p_world; 1]).
    T_world_cam: the camera's own pose in the world frame (inverse of the
                 above); its translation column is the camera position.
    """
    R: np.ndarray                # (3, 3)
    T_cam_world: np.ndarray      # (4, 4)
    T_world_cam: np.ndarray      # (4, 4)
    camera_position_world: np.ndarray  # (3,)
    rms_reproj_error_px: float = None
    num_points: int = None
    num_inliers: int = None
    coplanarity_ratio: float = None
    is_coplanar: bool = None
    camera_name: str = ""


def load_camera_intrinsics(path: str) -> CameraIntrinsics:
    """Load camera_matrix + dist_coeffs (+ image size, if present) from a
    cv2.FileStorage YAML (either a full intrinsics-calibration result, or
    any file that writes at least `camera_matrix` and `dist_coeffs`)."""
    fs = _open_for_read(path)
    try:
        camera_matrix = fs.getNode("camera_matrix").mat()
        dist_coeffs = fs.getNode("dist_coeffs").mat()
        if camera_matrix is None or dist_coeffs is None:
            raise ValueError(f"{path}: missing camera_matrix/dist_coeffs")

        width_node = fs.getNode("image_width")
        height_node = fs.getNode("image_height")
        image_size = None
        if not width_node.empty() and not height_node.empty():
            image_size = (int(width_node.real()), int(height_node.real()))

        return CameraIntrinsics(
            camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
            dist_coeffs=np.asarray(dist_coeffs, dtype=np.float64),
            image_size=image_size,
        )
    finally:
        fs.release()


def list_intrinsics_camera_names(path: str) -> list:
    """List camera names available in a multi-camera intrinsics JSON
    (e.g. intrinsics_sensors_0922.json's "cameras" dict)."""
    with open(path, "r") as f:
        data = json.load(f)
    return sorted(data.get("cameras", {}).keys())


def load_camera_intrinsics_json(path: str, camera_name: str) -> CameraIntrinsics:
    """Load one camera's intrinsics from a multi-camera JSON file of the
    form `{"cameras": {"<name>": {"camera_matrix": 3x3, "dist_coeffs": [...],
    "image_size": [w, h]}, ...}}` (e.g. close_track_gui's
    intrinsics_sensors_*.json). Distinct from `load_camera_intrinsics`,
    which reads the single-camera cv2.FileStorage YAML format."""
    with open(path, "r") as f:
        data = json.load(f)
    cams = data.get("cameras", {})
    if camera_name not in cams:
        raise KeyError(f"Camera {camera_name!r} not found in {path}; available: {sorted(cams)}")
    c = cams[camera_name]
    image_size = tuple(c["image_size"]) if "image_size" in c else None
    return CameraIntrinsics(
        camera_matrix=np.asarray(c["camera_matrix"], dtype=np.float64),
        dist_coeffs=np.asarray(c["dist_coeffs"], dtype=np.float64),
        image_size=image_size,
    )


def load_camera_intrinsics_any(path: str, camera_name: str = None) -> CameraIntrinsics:
    """Load camera intrinsics from either format this project encounters:
    a single-camera cv2.FileStorage YAML (`.yml`/`.yaml`), or a
    multi-camera JSON (`.json`, requires `camera_name`). Picks the loader
    by file extension.
    """
    ext = os.path.splitext(str(path))[1].lower()
    if ext in (".yml", ".yaml"):
        return load_camera_intrinsics(path)
    if ext == ".json":
        if not camera_name:
            raise ValueError(
                f"{path}: a JSON multi-camera intrinsics file requires --camera-name "
                f"(available: {list_intrinsics_camera_names(path)})"
            )
        return load_camera_intrinsics_json(path, camera_name)
    raise ValueError(f"Unrecognized intrinsics file extension: {path}")


def save_camera_pose(path: str, pose: CameraPose) -> None:
    """Persist a CameraPose in the same cv2.FileStorage YAML schema
    `load_camera_pose` reads (mirrors close_track_gui's save_pnp_result,
    trimmed to the fields this project actually uses)."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    try:
        fs.write("camera_name", pose.camera_name or "")
        fs.write("R", pose.R)
        fs.write("T_cam_world", pose.T_cam_world)
        fs.write("T_world_cam", pose.T_world_cam)
        fs.write("camera_position_world", pose.camera_position_world)
        if pose.rms_reproj_error_px is not None:
            fs.write("rms_reprojection_error_px", pose.rms_reproj_error_px)
        if pose.num_points is not None:
            fs.write("num_points", pose.num_points)
        if pose.num_inliers is not None:
            fs.write("num_inliers", pose.num_inliers)
        if pose.coplanarity_ratio is not None:
            fs.write("coplanarity_ratio", float(pose.coplanarity_ratio))
        if pose.is_coplanar is not None:
            fs.write("is_coplanar", int(pose.is_coplanar))
    finally:
        fs.release()


def load_camera_pose(path: str) -> CameraPose:
    """Load a solved camera pose (PnP result YAML) written by
    close_track_gui's `save_pnp_result` (or any file using the same
    T_cam_world / T_world_cam field names)."""
    fs = _open_for_read(path)
    try:
        T_cam_world = fs.getNode("T_cam_world").mat()
        T_world_cam = fs.getNode("T_world_cam").mat()
        R = fs.getNode("R").mat()
        if T_world_cam is None:
            if T_cam_world is None:
                raise ValueError(f"{path}: missing T_cam_world/T_world_cam")
            T_world_cam = np.linalg.inv(T_cam_world)
        if T_cam_world is None:
            T_cam_world = np.linalg.inv(T_world_cam)
        if R is None:
            R = T_cam_world[:3, :3]

        cam_pos_node = fs.getNode("camera_position_world")
        if not cam_pos_node.empty():
            camera_position_world = np.asarray(cam_pos_node.mat(), dtype=np.float64).flatten()
        else:
            camera_position_world = T_world_cam[:3, 3].copy()

        def _opt_real(name):
            node = fs.getNode(name)
            return None if node.empty() else float(node.real())

        def _opt_bool(name):
            node = fs.getNode(name)
            return None if node.empty() else bool(int(node.real()))

        name_node = fs.getNode("camera_name")
        camera_name = name_node.string() if not name_node.empty() else ""

        return CameraPose(
            R=np.asarray(R, dtype=np.float64),
            T_cam_world=np.asarray(T_cam_world, dtype=np.float64),
            T_world_cam=np.asarray(T_world_cam, dtype=np.float64),
            camera_position_world=camera_position_world,
            rms_reproj_error_px=_opt_real("rms_reprojection_error_px"),
            num_points=int(fs.getNode("num_points").real()) if not fs.getNode("num_points").empty() else None,
            num_inliers=int(fs.getNode("num_inliers").real()) if not fs.getNode("num_inliers").empty() else None,
            coplanarity_ratio=_opt_real("coplanarity_ratio"),
            is_coplanar=_opt_bool("is_coplanar"),
            camera_name=camera_name or "",
        )
    finally:
        fs.release()
