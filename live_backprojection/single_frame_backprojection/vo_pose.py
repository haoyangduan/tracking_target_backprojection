"""Load a per-frame camera pose directly from a VO (visual-odometry) track
JSON -- e.g. `data/track_cam_test_cad-NE_pair-both_cam-both_mode-vo_det-
aliked_frames-0-180_ref-0-511_sc-0.25_st-2_hc71c45f4b1.json` -- an
alternative external pose source to pipeline/creaform_pose.py's
Creaform-jig chain.

File shape: `{"camera": "<zed_x_* name>", "frames": {"<0-indexed frame
number as a string>": <4x4 row-major matrix>, ...}}`. Each per-frame
matrix is the tracked car body's pose **in the camera frame** -- i.e.
already `p_cam = M @ [p_car; 1]`, exactly this project's `T_cam_world`
convention with "world" standing in for the car/CAD frame, the same role
as creaform_pose.py's derived `left_left_T_cad`. So, unlike the Creaform
CSV, there is no separate calibration chain to apply here: each frame's
matrix *is* the `CameraPose.T_cam_world` directly. It only needs a units
fix (translation is in meters here; this project works in mm throughout)
and a frame-name -> track-index lookup.

Frame indexing confirmed 0-indexed and matching the Creaform CSV's own
`frame_index` convention (the same `frame_name = f"frame_{index+1:06d}.png"`
mapping `pipeline/creaform_pose.py`'s `load_creaform_track_csv` uses): this
file's index 51 and the independently-validated `frame_000052.png` pose
(see `data/frame_000052_creaform_cam_pose_cad.yml`) agree to ~1-3mm
translation and a fraction of a degree rotation -- cross-validating both
this indexing convention and the units fix above against a source that
has nothing to do with VO.

`camera` names the ZED-naming-scheme camera the track was computed for
(e.g. "zed_x_left_left" == "hik_DA0179014" -- see README.md's "Real
calibration file formats" section for the full 4-camera name mapping);
`compute_camera_pose_from_vo_track`'s `camera_name` defaults to this raw
name, same as creaform_pose.py defaults to the calibration's own
`creaform_base_cam` -- pass the corresponding `hik_*` name explicitly if
you want the output CameraPose to carry that instead.
"""

import json
import os
import re
import sys

import numpy as np

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_PIPELINE_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from calib_target.pnp_io import CameraPose

_FRAME_NUM_RE = re.compile(r"(\d+)")


def load_vo_track_json(path: str) -> dict:
    """Load a VO per-frame track JSON. Returns
    {"camera": str, "frames": {int: (4,4) ndarray}} (translation left as-is,
    in the file's own units -- meters; see this module's docstring)."""
    with open(path, "r") as f:
        data = json.load(f)
    frames = {int(k): np.asarray(v, dtype=np.float64).reshape(4, 4) for k, v in data["frames"].items()}
    return {"camera": data.get("camera", ""), "frames": frames}


def vo_frame_index_from_name(frame_name: str) -> int:
    """`frame_000052.png` -> 51 -- the track file's frames are 0-indexed,
    one below the 1-indexed number in the filename (same convention as
    pipeline/creaform_pose.py's track CSV; see this module's docstring)."""
    m = _FRAME_NUM_RE.search(os.path.basename(frame_name))
    if not m:
        raise ValueError(f"Could not find a frame number in {frame_name!r}")
    return int(m.group(1)) - 1


def compute_camera_pose_from_vo_track(
    track_json_path: str,
    frame_name: str,
    camera_name: str = None,
    translation_scale_to_mm: float = 1000.0,
) -> CameraPose:
    """Look up one frame's camera pose (already in the car/CAD frame) from
    a VO track JSON -- no calibration chain needed, see this module's
    docstring for why.

    `translation_scale_to_mm` converts the file's own translation units
    (meters) to this project's mm convention; leave it at the default
    unless a future track file turns out to already be in mm.
    """
    track = load_vo_track_json(track_json_path)
    idx = vo_frame_index_from_name(frame_name)
    if idx not in track["frames"]:
        available = sorted(track["frames"])
        raise KeyError(
            f"frame index {idx} (from {frame_name!r}) not found in {track_json_path}; "
            f"available indices: {available[0]}..{available[-1]}"
        )
    M = track["frames"][idx].copy()
    M[:3, 3] *= translation_scale_to_mm

    T_cam_world = M
    T_world_cam = np.linalg.inv(T_cam_world)

    return CameraPose(
        R=T_cam_world[:3, :3],
        T_cam_world=T_cam_world,
        T_world_cam=T_world_cam,
        camera_position_world=T_world_cam[:3, 3].copy(),
        camera_name=camera_name or track["camera"],
    )


def main():
    import argparse
    from calib_target.pnp_io import save_camera_pose

    parser = argparse.ArgumentParser(
        description="Look up one frame's camera pose (in the car/CAD frame) directly from a "
                    "VO per-frame track JSON -- no calibration chain needed."
    )
    parser.add_argument("--track", required=True,
                         help="e.g. data/track_cam_test_cad-NE_..._hc71c45f4b1.json")
    parser.add_argument("--frame", required=True, help="frame image basename, e.g. frame_000052.png")
    parser.add_argument("--camera-name", default=None, help="defaults to the track file's own \"camera\" field")
    parser.add_argument("--out", required=True,
                         help="output pose YAML path, e.g. frame_000052_vo_cam_pose_cad.yml")
    args = parser.parse_args()

    pose = compute_camera_pose_from_vo_track(args.track, args.frame, args.camera_name)
    print(f"camera_position_world (CAD frame, mm): {pose.camera_position_world}")
    save_camera_pose(args.out, pose)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
