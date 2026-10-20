"""Loader/solver for a multi-camera rig's INTERNAL extrinsics (camera-to-
camera), as produced by close_track_gui's rig-extrinsic calibration
(e.g. extrinsics_on_gui_0922.json).

Important: this file does NOT give any camera's pose in the car/CAD
frame -- it only encodes how the rig's cameras sit relative to each other
(baseline + relative rotation between them, from board-based cross-camera
calibration). Each camera entry's `T_cam_ref` maps a point from the
reference camera's frame into that camera's OWN frame:

    p_camX = T_cam_ref[X] @ p_ref

(verified directly against this file's own redundant data: `T_cam_ref[X]`
is numerically IDENTICAL to its `relative_extrinsics` entry
`{"from": X, "to": <reference_camera>}`, and a transitivity check through
a third camera closes to ~2-3mm, consistent with ordinary multi-session
calibration noise -- not a sign/convention error).

That makes `T_cam_ref[X]` play exactly the role of our own
`CameraPose.T_cam_world`, but with "world" temporarily standing in for the
reference camera's frame. So once the REFERENCE camera's true pose in the
car/CAD frame is known (e.g. solved via PnP against visible tracking
targets in one frame, same as any other camera in this project), every
other rig camera's true world pose follows by composition:

    T_cam_world[X] = T_cam_ref[X] @ T_cam_world[reference]

Since the rig is physically rigid, this only needs to be solved once (from
whichever single photo has visible targets for the reference camera):
every other rig camera's extrinsics.yml can then be derived and reused for
any other single photo taken from this same fixed rig, without re-running
PnP for each camera individually.
"""

import json
import os
import sys

import numpy as np

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_PIPELINE_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from calib_target.pnp_io import CameraPose


def _transform_from_rot_trans(rotation_matrix, translation_m) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(rotation_matrix, dtype=np.float64)
    T[:3, 3] = np.asarray(translation_m, dtype=np.float64) * 1000.0  # metres -> mm
    return T


def load_rig_extrinsics(path: str):
    """Returns (reference_camera_name, {camera_name: T_cam_ref (4x4, mm)})."""
    with open(path, "r") as f:
        data = json.load(f)
    reference = data["reference_camera"]
    transforms = {}
    for cam in data["cameras"]:
        t_cam_ref = cam["T_cam_ref"]
        transforms[cam["name"]] = _transform_from_rot_trans(
            t_cam_ref["rotation_matrix"], t_cam_ref["translation"]
        )
    return reference, transforms


def compute_camera_world_poses(rig_extrinsics_path: str, reference_pose: CameraPose) -> dict:
    """Given the reference camera's already-solved world pose (a normal
    CameraPose, e.g. loaded via calib_target.pnp_io.load_camera_pose from a
    PnP solve against visible tracking targets), derive every other rig
    camera's world pose. Returns {camera_name: CameraPose}, including the
    reference camera itself (returned unchanged) for convenience.
    """
    reference_name, transforms = load_rig_extrinsics(rig_extrinsics_path)
    if reference_pose.camera_name and reference_pose.camera_name != reference_name:
        print(f"[rig_extrinsics] WARNING: reference_pose.camera_name="
              f"{reference_pose.camera_name!r} does not match this rig file's "
              f"reference_camera={reference_name!r}; proceeding anyway "
              f"(assuming reference_pose really is {reference_name}'s pose).")

    poses = {reference_name: reference_pose}
    T_cam_world_ref = reference_pose.T_cam_world

    for name, T_cam_ref in transforms.items():
        if name == reference_name:
            continue
        T_cam_world_X = T_cam_ref @ T_cam_world_ref
        T_world_cam_X = np.linalg.inv(T_cam_world_X)
        poses[name] = CameraPose(
            R=T_cam_world_X[:3, :3],
            T_cam_world=T_cam_world_X,
            T_world_cam=T_world_cam_X,
            camera_position_world=T_world_cam_X[:3, 3].copy(),
            rms_reproj_error_px=reference_pose.rms_reproj_error_px,
            num_points=reference_pose.num_points,
            num_inliers=reference_pose.num_inliers,
            camera_name=name,
        )
    return poses


def main():
    import argparse
    from calib_target.pnp_io import load_camera_pose, save_camera_pose

    parser = argparse.ArgumentParser(
        description="Derive every rig camera's world pose from one solved reference-camera "
                    "pose + a rig extrinsics JSON (camera-to-camera calibration)."
    )
    parser.add_argument("--rig-extrinsics", required=True, help="e.g. extrinsics_on_gui_0922.json")
    parser.add_argument("--reference-pose", required=True,
                         help="solved extrinsics.yml for the rig's reference camera")
    parser.add_argument("--out-dir", required=True, help="output dir; writes <camera>_extrinsics.yml per camera")
    args = parser.parse_args()

    reference_pose = load_camera_pose(args.reference_pose)
    poses = compute_camera_world_poses(args.rig_extrinsics, reference_pose)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, pose in poses.items():
        out_path = os.path.join(args.out_dir, f"{name}_extrinsics.yml")
        save_camera_pose(out_path, pose)
        print(f"{name}: camera_position_world = {pose.camera_position_world} mm  ->  {out_path}")


if __name__ == "__main__":
    main()
