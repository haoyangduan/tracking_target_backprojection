"""Shared camera-pose resolution: a per-frame pose for ANY rig camera, not
just whichever one a given VO track file directly covers.

compute_camera_pose_from_vo_track's camera_name argument is COSMETIC ONLY
(see pipeline/vo_pose.py's docstring) -- it labels the returned CameraPose,
it does NOT select or validate which camera's poses are actually in the
track file. A VO track built for one camera will happily return THAT
camera's real pose, mislabeled with whatever camera_name you pass, for any
frame_name that matches an index in the file -- no error, just a physically
wrong extrinsic silently used downstream (confirmed as the root cause of a
real 0%-recall teacher-derivation report when this project's only VO track,
which covers hik_DA0179014 only, was used unmodified for a hik_DA0883721
dataset).

resolve_pose_fn() below is the one place in this project that guards
against that: it only returns poses it can actually justify -- either a
direct VO-track lookup (camera matches) or a rig-composed derivation
(camera is a DIFFERENT, rigidly-mounted camera on the same physical rig,
given that rig's own fixed camera-to-camera extrinsics) -- and raises
immediately, with a clear explanation, if neither holds. Used by both
live_backprojection/teacher.py (building a teacher oracle) and
live_backprojection/project_to_mesh.py (the live projection stage), so
both get this protection from one shared implementation.

Confirmed on this rig: composing hik_DA0179014's VO pose with
data/extrinsics_on_gui_0922.json's rig extrinsics reproduces the
hik_DA0179014<->hik_DA0883721 baseline at exactly 119.40mm across every
frame tested, consistent with this project's independently measured
~119.7mm stereo baseline.
"""
from live_backprojection import _pathsetup  # noqa: F401

import yaml

from single_frame_backprojection.creaform_pose import compute_camera_pose_from_creaform, load_creaform_track_csv
from single_frame_backprojection.rig_extrinsics import compute_camera_world_poses
from single_frame_backprojection.vo_pose import compute_camera_pose_from_vo_track, load_vo_track_json

# Maps this rig's ZED-naming-scheme camera names (used inside VO track JSON
# files) to the corresponding hik_* names used everywhere else in this
# project -- see README.md's "Real calibration file formats" section.
ZED_TO_HIK = {
    "zed_x_left_left": "hik_DA0179014",
    "zed_x_left_right": "hik_DA0883721",
    "zed_x_right_left": "hik_ED51079AAK00091",
    "zed_x_right_right": "hik_ED51079AAK00095",
}


def resolve_pose_fn(vo_track_path: str, camera_name: str, rig_extrinsics_path: str = None):
    """Returns a function frame_name -> CameraPose for camera_name. Uses a
    direct VO track lookup if the track actually covers camera_name;
    otherwise, if rig_extrinsics_path is given, derives camera_name's pose
    per frame by composing the track's own camera's VO pose with the rig's
    fixed extrinsics (T_cam_world[X] = T_cam_ref[X] @ T_cam_world[reference]
    -- re-solved every frame since the CAR moves relative to the stationary
    rig even though the rig's own internal geometry never changes). Raises
    ValueError immediately (not buried in a per-frame KeyError) if neither
    is possible."""
    track = load_vo_track_json(vo_track_path)
    track_camera_raw = track.get("camera", "")
    vo_covers = ZED_TO_HIK.get(track_camera_raw, track_camera_raw)

    if vo_covers == camera_name:
        print(f"    [Pose Resolution] Direct VO lookup for base camera '{camera_name}' (from {vo_track_path})")
        def pose_fn(frame_name):
            return compute_camera_pose_from_vo_track(vo_track_path, frame_name, camera_name=camera_name)
        return pose_fn

    if rig_extrinsics_path:
        print(f"    [Pose Resolution] Camera '{camera_name}' differs from VO base '{vo_covers}'. "
              f"Composing pose via rig extrinsics: {rig_extrinsics_path}")
        def pose_fn(frame_name):
            reference_pose = compute_camera_pose_from_vo_track(vo_track_path, frame_name, camera_name=vo_covers)
            poses = compute_camera_world_poses(rig_extrinsics_path, reference_pose)
            if camera_name not in poses:
                raise KeyError(f"{camera_name!r} not found in rig extrinsics {rig_extrinsics_path!r}; "
                                f"available cameras: {sorted(poses)}")
            return poses[camera_name]
        return pose_fn

    raise ValueError(
        f"VO track / camera mismatch: {vo_track_path!r} contains pose data for "
        f"{track_camera_raw!r} ({vo_covers}), not {camera_name!r}, and no rig extrinsics file was "
        f"given to derive {camera_name}'s pose from {vo_covers}'s VO pose via the fixed rig "
        f"geometry instead. compute_camera_pose_from_vo_track's camera_name argument only LABELS "
        f"its output -- it would otherwise silently return {vo_covers}'s real pose mislabeled as "
        f"{camera_name!r}, producing a physically wrong result. Either point the VO track path at "
        f"one that actually covers {camera_name}, set camera_name to {vo_covers!r}, or supply the "
        f"rig extrinsics file (e.g. data/extrinsics_on_gui_0922.json)."
    )


def resolve_pose_fn_creaform(calibration_path: str, track_csv_path: str, camera_name: str,
                              max_time_diff_ms: float = None, rig_extrinsics_path: str = None):
    """Returns a function frame_name -> CameraPose using Creaform tracking
    data (pipeline/creaform_pose.py) instead of VO.

    Unlike VO -- computed directly from the image sequence itself, so
    inherently frame-synced by construction -- Creaform is an EXTERNAL
    motion-capture system whose per-frame pose sample is matched to each
    camera frame by TIMESTAMP. The track CSV's own `time_diff_ms` column
    records how close that match actually was, and real, severe cases
    exist: this project's own track has a clean, data-confirmed split --
    19 frames (a startup sync-lock-in transient, decaying smoothly from
    2.6s down to ~1.4s over the first ~10 frames) exceed 100ms, while every
    one of the other 163 frames is under 20ms. If max_time_diff_ms is
    given, a frame whose |time_diff_ms| exceeds it raises KeyError --
    treated the same as "no pose available for this frame" by callers
    (skipped, not silently used with a wrong position) -- so a badly-
    synced Creaform sample can't poison the oracle the way an unfiltered
    one would.

    camera_name's pose comes directly from the Creaform chain if it matches
    the calibration's own `creaform_base_cam`; otherwise, if
    rig_extrinsics_path is given, it's derived per frame via the rig's
    fixed camera-to-camera extrinsics, same composition pattern as
    resolve_pose_fn() above."""
    with open(calibration_path, "r") as f:
        calib = yaml.safe_load(f)
    base_cam_raw = calib.get("creaform_base_cam", "")
    base_cam = ZED_TO_HIK.get(base_cam_raw, base_cam_raw)

    track = load_creaform_track_csv(track_csv_path)

    def base_pose_fn(frame_name):
        if frame_name not in track:
            raise KeyError(f"{frame_name!r} not found in Creaform track {track_csv_path!r}")
        if max_time_diff_ms is not None:
            td = track[frame_name].get("time_diff_ms")
            if td is not None and abs(float(td)) > max_time_diff_ms:
                raise KeyError(f"{frame_name!r} Creaform sync too poor: |time_diff_ms|="
                                f"{abs(float(td)):.1f} > {max_time_diff_ms} -- treated as no pose available")
        return compute_camera_pose_from_creaform(calibration_path, track_csv_path, frame_name, camera_name=base_cam)

    if base_cam == camera_name:
        print(f"    [Creaform Pose Resolution] Direct Creaform lookup for base camera '{camera_name}'.")
        return base_pose_fn

    if rig_extrinsics_path:
        print(f"    [Creaform Pose Resolution] Camera '{camera_name}' differs from Creaform base '{base_cam}'. "
              f"Composing pose via rig extrinsics: {rig_extrinsics_path}")
        def pose_fn(frame_name):
            reference_pose = base_pose_fn(frame_name)
            poses = compute_camera_world_poses(rig_extrinsics_path, reference_pose)
            if camera_name not in poses:
                raise KeyError(f"{camera_name!r} not found in rig extrinsics {rig_extrinsics_path!r}; "
                                f"available cameras: {sorted(poses)}")
            return poses[camera_name]
        return pose_fn

    raise ValueError(
        f"Creaform calibration mismatch: {calibration_path!r}'s creaform_base_cam is "
        f"{base_cam_raw!r} ({base_cam}), not {camera_name!r}, and no rig extrinsics file was given "
        f"to derive {camera_name}'s pose from {base_cam}'s Creaform pose via the fixed rig geometry "
        f"instead. Either set camera_name to {base_cam!r}, or supply the rig extrinsics file "
        f"(e.g. data/extrinsics_on_gui_0922.json)."
    )
