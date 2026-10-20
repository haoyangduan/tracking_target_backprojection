"""Loader for the 3D reference-point file used by single-image PnP extrinsic
calibration.

The file (e.g. cam2cam_in_exp_world_260731_full_extrin_calib.txt) is a plain
whitespace-separated text file, one point per line:

    X  Y  Z  nx  ny  nz  1.0000000000  400  0

Only columns 0:3 (X, Y, Z, millimeters) are used as PnP object points. The
remaining columns (a normal vector and two constant/metadata fields) are not
needed for this workflow and are ignored.

Line numbers are 1-indexed to match what a human reading the file with a
text editor / `read_file` line-numbering would see, and what the user is
expected to reference in a correspondences JSON's `point_line` field.
"""

from pathlib import Path

import numpy as np


def load_object_points(path) -> np.ndarray:
    """Parse the 3D point file and return an (N, 3) float64 array of X,Y,Z.

    `points[i]` corresponds to 1-indexed line number `i + 1` in the source
    file (blank lines, if any, are skipped and do NOT count as a line for
    this purpose was avoided by requiring every non-empty line to parse).
    """
    path = Path(path)
    points = []
    with path.open("r") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(
                    f"{path}:{line_no}: expected at least 3 columns (X Y Z), got {len(parts)!r}"
                )
            try:
                xyz = [float(parts[0]), float(parts[1]), float(parts[2])]
            except ValueError as e:
                raise ValueError(f"{path}:{line_no}: failed to parse X Y Z: {raw_line!r}") from e
            points.append(xyz)

    if not points:
        raise ValueError(f"No points parsed from {path}")

    return np.array(points, dtype=np.float64)


def get_point_by_line(points: np.ndarray, point_line: int) -> np.ndarray:
    """Fetch a single object point by its 1-indexed line number.

    Raises IndexError with a clear message if out of range.
    """
    idx = point_line - 1
    if idx < 0 or idx >= len(points):
        raise IndexError(
            f"point_line={point_line} out of range; file has {len(points)} points "
            f"(valid range: 1..{len(points)})"
        )
    return points[idx]
