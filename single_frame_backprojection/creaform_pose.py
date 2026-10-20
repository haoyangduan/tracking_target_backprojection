"""Compute a real camera's pose in the CAD frame from live Creaform
tracking data, following the transform chain documented at the top of
`creaform_calibration_*.yml`:

    body_T_cad       = inv(cad_T_body)
    left_left_T_cad  = inv(creaform_T_camera) @ creaform_T_body @ body_T_cad

`cad_T_body` and `creaform_T_camera` are fixed jig-calibration constants
(from the calibration YAML, see pipeline/frames.py). `creaform_T_body` is
NOT fixed -- it's whatever the Creaform tracking system measured for that
specific camera frame, recorded in a per-frame CSV track (e.g.
`20260922_171111_fixed_synced.csv`, synced by timestamp to each camera
frame -- see its own `time_diff_ms` column for how close that match was).

`left_left_T_cad` plays the role of this project's `CameraPose.T_cam_world`
for the calibration's base camera (`creaform_base_cam`, e.g.
`zed_x_left_left` == `hik_DA0179014`) directly in the CAD frame -- no
`--body-to-cad` step needed downstream, since this chain already produces
a CAD-frame pose. The other 3 rig cameras' poses follow from this one via
pipeline/rig_extrinsics.py's rig-relative composition.

**Status: working, verified against real correspondences.** (RX, RY, RZ)
is an Euler angle triple in DEGREES, confirmed directly: "TX TY TZ是mm,
RX RY RZ是欧拉角角度制". Getting here took two false starts, both
instructive enough to leave a record of:

  1. First test used `frame_000001.png` (only 4 correspondences, coplanarity
     ratio 0.02 -- itself a confound) and an Euler construction that turned
     out to have a real bug (see #2). Result: ~2300-2660px reprojection
     error against a fresh solvePnP fit's ~1-5px. A search over ~10,000+
     Euler order/sign/scale/inversion combinations found nothing better
     than ~444px -- but this was on `frame_000001.png`, whose Creaform
     sample turned out to be stale by ~2.6s (the worst-synced frame in the
     whole 182-frame burst -- see `time_diff_ms` in the track CSV), so no
     rotation convention could have reproduced it correctly regardless.
  2. Re-tested on `frame_000052.png` (10 correspondences, coplanarity ratio
     0.18, sync error 0.32ms -- ruling out both confounds from #1). A
     sibling script in this project, `close_track_gui/scripts/
     align_board_to_creaform.py` (found via this machine's local Creaform
     install + a `close_track_gui` checkout), claims RX/RY/RZ is a
     Rodrigues vector, not Euler angles, for the raw Creaform "Model1"
     export it reads. Trying that dropped RMS from ~1870px to ~702px --
     real progress, but still far from working. This turned out to be
     the wrong lead: that script reads a *different*, raw Creaform export
     format, not this project's derived `*_fixed_synced.csv`.
  3. Told directly that RX/RY/RZ *is* Euler degrees after all, re-checked
     the actual bug: this module's own hand-rolled rotation matrix
     construction (`Rx @ Ry @ Rz`, labeled "extrinsic XYZ") turned out to
     compute scipy's *extrinsic* `'XYZ'` convention, but the correct one
     for this data is *intrinsic* `'xyz'` -- equal to `Rz @ Ry @ Rx`, the
     REVERSE multiplication order, not the same formula under a different
     name. Verified directly: `scipy.spatial.transform.Rotation.from_euler
     ('xyz', [rx, ry, rz], degrees=True)` != `Rx @ Ry @ Rz` for the same
     elementary Rx/Ry/Rz matrices, but does equal `Rz @ Ry @ Rx`. With this
     fixed, `frame_000052.png`'s reprojection RMS is 11.9px (individual
     points 2-29px), and running the full back-projection pipeline
     (`pipeline/backproject_targets.py`) against the real mesh gives
     10/10 hits, **median error 2.3mm** -- in line with the sticker-to-
     mesh fitting tolerance already documented in `projection_tracking_
     targets`. One point (line 80) is an 87.7mm outlier; investigate that
     specific correspondence/detection before trusting it individually,
     but the pose itself is now validated.

`data/frame_000052_creaform_cam_pose_cad.yml` is this validated pose,
kept as a working reference example. Deliberately NOT named
`*_extrinsics_creaform.yml` (an earlier, misleading name this project used
briefly): that name reads as "the camera's pose in the world,
recomputed" -- as if the camera itself might be moving between frames.
It isn't; the car is (see the module-level derivation above: this chain's
one per-frame-varying input, `creaform_T_body`, tracks the *car body's*
motion, not the camera's). What changes frame to frame is the camera's
pose *relative to the car's own CAD frame* -- which is real, physical,
and unavoidable precisely because the car moves under a stationary
camera. `cam_pose_cad` names that specific, per-frame-varying quantity
instead of implying a generic "extrinsics" recompute.
"""

import csv
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_PIPELINE_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from calib_target.pnp_io import CameraPose
from single_frame_backprojection.frames import load_transform


def creaform_rot_from_row(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """(RX, RY, RZ) is an intrinsic 'xyz' Euler angle triple, in degrees --
    equal to `Rz(rz) @ Ry(ry) @ Rx(rx)` (note the REVERSED multiplication
    order relative to the angle order; this is not the same as the
    extrinsic 'XYZ' convention `Rx @ Ry @ Rz` this project's OTHER 6-DOF
    pose helper, defect_visualization's build_transform, uses -- see the
    module docstring for how conflating the two cost real debugging time).
    """
    return Rotation.from_euler("xyz", [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix()


def load_creaform_track_csv(csv_path: str) -> dict:
    """Load a per-frame Creaform track CSV. Returns
    {frame_basename: {"tx","ty","tz","rx","ry","rz", ...raw row...}}.

    Expects columns frame_index, TX, TY, TZ, RX, RY, RZ (case-sensitive,
    matching 20260922_171111_fixed_synced.csv); extra columns (including
    the CSV's own `image_path`) are kept verbatim in each row's dict for
    reference/debugging.

    Keyed by a reconstructed `frame_{frame_index+1:06d}.png` name, NOT the
    CSV's own `image_path` column -- confirmed those use different
    numbering/naming than the actual files on disk (e.g. this CSV's row
    `frame_index=0` names itself "000000.png", but the real, 1-indexed
    capture on disk is `frame_000001.png`; row count (182) matches the
    real folder's frame count exactly, and this mapping was independently
    confirmed against `data/capture_log.csv`'s own per-camera `frame_id`
    counter).
    """
    rows = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_name = f"frame_{int(row['frame_index']) + 1:06d}.png"
            rows[frame_name] = {
                "tx": float(row["TX"]), "ty": float(row["TY"]), "tz": float(row["TZ"]),
                "rx": float(row["RX"]), "ry": float(row["RY"]), "rz": float(row["RZ"]),
                **row,
            }
    return rows


def creaform_T_body_from_row(row: dict) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = creaform_rot_from_row(row["rx"], row["ry"], row["rz"])
    T[:3, 3] = [row["tx"], row["ty"], row["tz"]]
    return T


def compute_camera_pose_from_creaform(
    calibration_yml_path: str,
    track_csv_path: str,
    frame_name: str,
    camera_name: str = None,
) -> CameraPose:
    """Compute the calibration's base camera's pose in the CAD frame for
    one specific camera frame, via the documented transform chain.

    `camera_name` defaults to the calibration file's own
    `creaform_base_cam` entry (e.g. "zed_x_left_left"); pass the
    corresponding hik_* name if you'd rather the output CameraPose carry
    that name instead (they're the same physical camera under two naming
    schemes -- see the "Real calibration file formats" section of
    README.md).

    Only reliable for frames with good Creaform sync (see the track CSV's
    `time_diff_ms` column) -- a stale match (seconds off) will silently
    give a wrong pose no matter how correct this function's math is, since
    it's measuring the wrong instant in time. `frame_000052.png` had
    `time_diff_ms=0.32` and is the validated reference case; the first few
    frames of that particular burst (`frame_000001.png`..`frame_000005.png`)
    had multi-second sync errors and should not be used to judge this
    function's correctness.
    """
    import yaml
    with open(calibration_yml_path, "r") as f:
        calib = yaml.safe_load(f)

    cad_T_body = load_transform(calibration_yml_path, yaml_key="cad_T_body")
    creaform_T_camera = load_transform(calibration_yml_path, yaml_key="creaform_T_camera")
    body_T_cad = np.linalg.inv(cad_T_body)

    track = load_creaform_track_csv(track_csv_path)
    if frame_name not in track:
        raise KeyError(
            f"{frame_name!r} not found in {track_csv_path}; "
            f"available frames include: {sorted(track)[:5]}..."
        )
    creaform_T_body = creaform_T_body_from_row(track[frame_name])

    T_cam_world = np.linalg.inv(creaform_T_camera) @ creaform_T_body @ body_T_cad
    T_world_cam = np.linalg.inv(T_cam_world)

    return CameraPose(
        R=T_cam_world[:3, :3],
        T_cam_world=T_cam_world,
        T_world_cam=T_world_cam,
        camera_position_world=T_world_cam[:3, 3].copy(),
        camera_name=camera_name or calib.get("creaform_base_cam", ""),
    )


def main():
    import argparse
    from calib_target.pnp_io import save_camera_pose

    parser = argparse.ArgumentParser(
        description="Compute a camera's CAD-frame pose for one frame from a Creaform "
                    "jig calibration + per-frame tracking CSV."
    )
    parser.add_argument("--calibration", required=True, help="e.g. data/creaform_calibration_220926.yml")
    parser.add_argument("--track", required=True, help="e.g. data/20260922_171111_fixed_synced.csv")
    parser.add_argument("--frame", required=True, help="frame image basename, e.g. frame_000052.png")
    parser.add_argument("--camera-name", default=None, help="defaults to the calibration's creaform_base_cam")
    parser.add_argument("--out", required=True,
                         help="output pose YAML path, e.g. frame_000052_creaform_cam_pose_cad.yml -- "
                              "avoid naming it '*_extrinsics_creaform.yml': this is the camera's pose "
                              "relative to the car's (moving) CAD frame, not a recomputed absolute "
                              "camera pose -- see this module's docstring")
    args = parser.parse_args()

    pose = compute_camera_pose_from_creaform(args.calibration, args.track, args.frame, args.camera_name)
    print(f"camera_position_world (CAD frame, mm): {pose.camera_position_world}")
    save_camera_pose(args.out, pose)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
