"""Open3D 3D visualisation for the tracking-target back-projection result.

Fixed sphere convention: ground truth is always red, the back-projected
estimate is always green -- no error-magnitude color grading (an earlier
version graded the estimate green->red by its own error; this was
replaced with a plain fixed convention per instruction), plus:
  - thin gray lines connecting each GT/estimated pair (the error vector),
  - a small blue sphere + axis frame marking the camera position/orientation,
  - if a hit carries a `raw_target_point` (the target's raw Creaform/
    C-Track measurement, before nearest-surface projection onto the mesh --
    see backproject_targets.run()'s `raw_points3d_path`), a purple sphere
    for it plus a thin purple line to the surface-fit GT point, showing how
    far the surface-fit step moved that target.
"""

import threading

import numpy as np
import open3d as o3d

_GT_COLOR = (0.8, 0.0, 0.0)          # ground truth: always red
_ESTIMATE_COLOR = (0.0, 0.8, 0.0)    # back-projected estimate: always green


def _sphere(center, radius, color):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.compute_vertex_normals()
    s.paint_uniform_color(color)
    s.translate(center)
    return s


def build_result_geometries(
    mesh: o3d.geometry.TriangleMesh,
    hits,
    camera_position=None,
    camera_rotation=None,
    sphere_radius_mm: float = 3.0,
    camera_axis_len_mm: float = 100.0,
):
    """Build the list of Open3D geometries for one back-projection run.

    Args:
        mesh: car mesh, already scaled to millimeters.
        hits: list of camera_model.RayHit.
        camera_position: (3,) mm, world/CAD frame; draws a camera marker if given.
        camera_rotation: (3,3) world_from_cam rotation; draws an axis frame if given.
    """
    geoms = [mesh]

    for h in hits:
        geoms.append(_sphere(h.gt_point, sphere_radius_mm, _GT_COLOR))
        if h.hit:
            geoms.append(_sphere(h.estimated_point, sphere_radius_mm, _ESTIMATE_COLOR))
            line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([h.gt_point, h.estimated_point]),
                lines=o3d.utility.Vector2iVector([[0, 1]]),
            )
            line.colors = o3d.utility.Vector3dVector([[0.6, 0.6, 0.6]])
            geoms.append(line)

        raw_point = getattr(h, "raw_target_point", None)
        if raw_point is not None and np.all(np.isfinite(raw_point)):
            geoms.append(_sphere(raw_point, sphere_radius_mm, (0.6, 0.2, 0.9)))
            raw_line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([raw_point, h.gt_point]),
                lines=o3d.utility.Vector2iVector([[0, 1]]),
            )
            raw_line.colors = o3d.utility.Vector3dVector([[0.6, 0.2, 0.9]])
            geoms.append(raw_line)

    if camera_position is not None:
        geoms.append(_sphere(camera_position, sphere_radius_mm * 1.5, (0.1, 0.3, 1.0)))
        if camera_rotation is not None:
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=camera_axis_len_mm, origin=(0, 0, 0)
            )
            axes.rotate(camera_rotation, center=(0, 0, 0))
            axes.translate(camera_position)
            geoms.append(axes)

    return geoms


def show_blocking(geometries, window_name="Tracking-Target Back-Projection"):
    o3d.visualization.draw_geometries(
        geometries, window_name=window_name, width=1024, height=768,
    )


def show_async(geometries, window_name="Tracking-Target Back-Projection"):
    """Open the viewer in a background thread so a Tk GUI mainloop isn't blocked."""
    t = threading.Thread(target=show_blocking, args=(geometries, window_name), daemon=True)
    t.start()
    return t
