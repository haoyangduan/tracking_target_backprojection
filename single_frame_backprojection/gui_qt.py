"""PyQt6 front-end for the tracking-target back-projection workflow.

Single-photo, close_track_gui-style flow: a circle_debug.json +
correspondences.json pair, produced interactively via `circle_tuner_qt.py`
("Tune..." below), feeds a given camera pose + points3d + mesh to measure
each tracking target's back-projection error.

Uses Qt layouts + a resizable QSplitter throughout (replacing an earlier
Tkinter version that used fixed pixel sizes and clipped content on
smaller/higher-DPI screens). Matches close_track_gui's own stack (PyQt6 +
qtawesome).
"""

import os
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QLabel, QFileDialog, QMessageBox, QSplitter,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy,
)
import qtawesome as qta

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_PIPELINE_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from single_frame_backprojection.backproject_targets import run as run_backprojection
from single_frame_backprojection.web_viewer import show as show_web_viewer
from single_frame_backprojection.circle_tuner_qt import CircleTunerWindow
from single_frame_backprojection.creaform_pose import compute_camera_pose_from_creaform
from single_frame_backprojection.vo_pose import compute_camera_pose_from_vo_track
from calib_target.pnp_io import save_camera_pose

_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
_DEFAULT_MESH = os.path.join(_DATA_DIR, "mesh", "0_NE_70_SIMPLIFICATION.stl")
_DEFAULT_POINTS3D = os.path.join(_DATA_DIR, "new_car_300926_whood.txt")
_DEFAULT_INTRINSICS = os.path.join(_DATA_DIR, "intrinsics_sensors_0922.json")
_DEFAULT_CAMERA_NAME = "hik_DA0179014"
_DEFAULT_CREAFORM_CALIBRATION = os.path.join(_DATA_DIR, "creaform_calibration_220926.yml")
_DEFAULT_CREAFORM_TRACK_CSV = os.path.join(_DATA_DIR, "20260922_171111_fixed_synced.csv")
_DEFAULT_VO_TRACK_JSON = os.path.join(
    _DATA_DIR,
    "track_cam_test_cad-NE_pair-both_cam-both_mode-vo_det-aliked_frames-0-180_ref-0-511_sc-0.25_st-2_hc71c45f4b1.json",
)

# Two selectable mesh + tracking-model pairings (see the "Mesh preset"
# buttons below): the original door-free pair, and the newer full-mesh
# (with doors + hood, 161 points) pair built on top of it.
_MESH_PRESETS = {
    "door_free": {
        "label": "Door-free",
        "mesh": os.path.join(_DATA_DIR, "mesh", "BIP_Flat.stl"),
        "points3d": os.path.join(_DATA_DIR, "points3d", "stickers_on_car_surface_mm.txt"),
        "raw_points3d": os.path.join(_DATA_DIR, "points3d", "biw_plus_frontrail_90pt_raw_ctrack_mm.txt"),
        "mesh_scale": "1000.0",
    },
    "full_mesh": {
        "label": "Full mesh (with doors/hood)",
        "mesh": _DEFAULT_MESH,
        "points3d": _DEFAULT_POINTS3D,
        "raw_points3d": os.path.join(_DATA_DIR, "new_car_300926_whood_raw_ctrack_mm.txt"),
        "mesh_scale": "1.0",
    },
}

FIELDS = [
    ("image", "Photo (source image)", "Image files (*.png *.jpg *.jpeg *.bmp)"),
    ("intrinsics", "Camera intrinsics (.yml, or multi-camera .json)", "Intrinsics (*.yml *.yaml *.json)"),
    ("camera_name", "Camera name (only needed for a multi-camera .json above)", None),
    ("extrinsics", "Camera pose / extrinsics (.yml, from external pipeline)", "YAML (*.yml *.yaml)"),
    ("creaform_calibration", "  └ Creaform jig calibration (.yml, for \"Compute (Creaform)...\")", "YAML (*.yml *.yaml)"),
    ("creaform_track_csv", "  └ Creaform per-frame tracking CSV (for \"Compute (Creaform)...\")", "CSV (*.csv)"),
    ("vo_track_json", "  └ VO per-frame track JSON (for \"Compute (VO)...\")", "JSON (*.json)"),
    ("points3d", "GT tracking-target points3d (.txt)", "Text (*.txt)"),
    ("correspondences", "Correspondences (circle_index -> point_line, .json)", "JSON (*.json)"),
    ("circles_debug", "Detected circles (circle_debug.json)", "JSON (*.json)"),
    ("mesh", "Car mesh (.stl/.ply/.obj)", "Mesh files (*.stl *.ply *.obj)"),
    ("mesh_scale", "  └ Mesh scale -> mm (1000 if mesh is meters, 1 if already mm)", None),
    ("body_to_cad", "Body->CAD alignment transform (optional, .json)", "JSON (*.json)"),
    ("raw_points3d", "Raw (pre-surface-fit) Creaform target positions (optional, .txt)", "Text (*.txt)"),
    ("raw_body_to_cad", "Raw target body->CAD transform (optional, defaults to cad_T_body)", "JSON (*.json)"),
    ("tag_size_mm", "  └ Viewer tag size (mm)", None),
]
DEFAULTS = {
    "mesh": _DEFAULT_MESH,
    "mesh_scale": _MESH_PRESETS["full_mesh"]["mesh_scale"],
    "points3d": _DEFAULT_POINTS3D,
    "tag_size_mm": "3.0",
    # body_to_cad deliberately has NO default: the default `points3d` above
    # (stickers_on_car_surface_mm.txt) is already in the mesh's CAD frame,
    # same as any pose from "Compute..." (pipeline/creaform_pose.py) -- a
    # default body_to_cad here would apply cad_T_body a second time on top
    # of both, silently producing 0/N hits. Fill it in only if you point
    # points3d/extrinsics at a genuinely raw (non-CAD-frame) pair instead.
    "intrinsics": _DEFAULT_INTRINSICS,
    "camera_name": _DEFAULT_CAMERA_NAME,
    "creaform_calibration": _DEFAULT_CREAFORM_CALIBRATION,
    "creaform_track_csv": _DEFAULT_CREAFORM_TRACK_CSV,
    "vo_track_json": _DEFAULT_VO_TRACK_JSON,
}


class BackprojectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tracking-Target Back-Projection")
        self.resize(1200, 760)
        self.setMinimumSize(640, 480)

        self._last_result = None  # (mesh, hits, pose)
        self.entries = {}
        self._tuner_window = None

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        splitter.addWidget(self._build_form_panel())
        splitter.addWidget(self._build_results_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([560, 640])

    # ------------------------------------------------------------------
    def _build_form_panel(self) -> QWidget:
        form_container = QWidget()
        form_layout = QFormLayout(form_container)
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        for key, label, filter_str in FIELDS:
            if key == "mesh":
                preset_row = QWidget()
                preset_layout = QHBoxLayout(preset_row)
                preset_layout.setContentsMargins(0, 0, 0, 0)
                for preset_key, preset in _MESH_PRESETS.items():
                    btn = QPushButton(qta.icon("fa5s.car-side"), preset["label"])
                    btn.setToolTip(
                        f"Fill in mesh, points3d, raw_points3d, and mesh_scale together for the "
                        f"{preset['label']!r} pairing -- these four must move as a set (see "
                        f"README.md's \"Full mesh + all stickers pipeline\" section)."
                    )
                    btn.clicked.connect(lambda _=False, pk=preset_key: self._apply_mesh_preset(pk))
                    preset_layout.addWidget(btn)
                form_layout.addRow(QLabel("Mesh + tracking model preset"), preset_row)

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)

            entry = QLineEdit(DEFAULTS.get(key, ""))
            entry.setMinimumWidth(220)
            row_layout.addWidget(entry, stretch=1)

            if key not in ("camera_name", "mesh_scale"):  # plain text fields, nothing to browse for
                browse_btn = QPushButton(qta.icon("fa5s.folder-open"), "")
                browse_btn.setToolTip("Browse...")
                browse_btn.clicked.connect(lambda _=False, e=entry, f=filter_str: self._browse(e, f))
                row_layout.addWidget(browse_btn)

            if key == "circles_debug":
                tune_btn = QPushButton(qta.icon("fa5s.sliders-h"), "Tune...")
                tune_btn.clicked.connect(self._open_circle_tuner)
                row_layout.addWidget(tune_btn)

            if key == "extrinsics":
                compute_creaform_btn = QPushButton(qta.icon("fa5s.calculator"), "Compute (Creaform)...")
                compute_creaform_btn.setToolTip(
                    "Derive this frame's camera pose (relative to the car's CAD frame) from the "
                    "Creaform jig calibration + tracking CSV below, keyed by the photo's own "
                    "filename -- no need to run pipeline/creaform_pose.py by hand."
                )
                compute_creaform_btn.clicked.connect(self._compute_creaform_pose)
                row_layout.addWidget(compute_creaform_btn)

                compute_vo_btn = QPushButton(qta.icon("fa5s.route"), "Compute (VO)...")
                compute_vo_btn.setToolTip(
                    "Look up this frame's camera pose directly from the VO per-frame track JSON "
                    "below, keyed by the photo's own filename -- no calibration chain needed, see "
                    "pipeline/vo_pose.py's docstring."
                )
                compute_vo_btn.clicked.connect(self._compute_vo_pose)
                row_layout.addWidget(compute_vo_btn)

            self.entries[key] = entry
            form_layout.addRow(QLabel(label), row)

        run_btn = QPushButton(qta.icon("fa5s.play", color="white"), "  Run Back-Projection")
        run_btn.setStyleSheet("background-color:#2196F3; color:white; font-weight:bold; padding:8px;")
        run_btn.clicked.connect(self._run)
        form_layout.addRow(run_btn)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_container)
        scroll.setMinimumWidth(420)
        return scroll

    def _build_results_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Circle idx", "Point line", "Pixel (u, v)", "Hit?", "Error (mm)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.table, stretch=1)

        self.summary_label = QLabel("Fill in the fields on the left and click Run Back-Projection.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("font-family: Consolas, monospace; padding: 6px;")
        layout.addWidget(self.summary_label)

        self.view_btn = QPushButton(qta.icon("fa5s.cube"), "  Open 3D Viewer")
        self.view_btn.setEnabled(False)
        self.view_btn.clicked.connect(self._view)
        layout.addWidget(self.view_btn)

        return panel

    # ------------------------------------------------------------------
    @staticmethod
    def _start_dir(current_text: str) -> str:
        """Where a Browse dialog should start: wherever the field already
        points, otherwise this project's data/ subdirectory instead of the
        OS/last-used default -- most inputs live under data/."""
        return current_text.strip() or _DATA_DIR

    def _browse(self, entry: QLineEdit, filter_str: str):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Select file", self._start_dir(entry.text()), (filter_str or "") + ";;All files (*)")
        if fname:
            entry.setText(fname)

    def _get(self, key):
        val = self.entries[key].text().strip()
        return val or None

    def _apply_mesh_preset(self, preset_key):
        """Fill in mesh/points3d/raw_points3d/mesh_scale together for one of
        _MESH_PRESETS -- these four must move as a matched set (a mesh
        authored in meters vs. millimeters needs a different mesh_scale, and
        a points3d file's point_line numbering is only meaningful paired
        with the tracking model it was fit from). Does NOT touch body_to_cad
        or raw_body_to_cad: both points3d files are already CAD-frame
        surface fits (no body_to_cad needed), and raw_body_to_cad's default
        (data/points3d/cad_T_body_mm.json) is the same inherited transform
        for both pairings -- see README.md's "Full mesh + all stickers
        pipeline" section for why that's correct, not a coincidence.
        """
        preset = _MESH_PRESETS[preset_key]
        self.entries["mesh"].setText(preset["mesh"])
        self.entries["points3d"].setText(preset["points3d"])
        self.entries["raw_points3d"].setText(preset["raw_points3d"])
        self.entries["mesh_scale"].setText(preset["mesh_scale"])

    def _open_circle_tuner(self):
        image_path = self._get("image")
        if not image_path:
            QMessageBox.warning(self, "No image", "Pick a photo first, then open the circle tuner.")
            return
        camera_guess = os.path.basename(os.path.dirname(image_path))

        def _on_saved(circle_debug_path, correspondences_path):
            self.entries["circles_debug"].setText(circle_debug_path)
            if correspondences_path:
                self.entries["correspondences"].setText(correspondences_path)

        self._tuner_window = CircleTunerWindow(image_path=image_path, camera_name=camera_guess, on_saved=_on_saved)
        self._tuner_window.show()

    def _compute_creaform_pose(self):
        """Derive the current photo's camera-pose-in-CAD-frame from the
        Creaform jig calibration + per-frame tracking CSV, keyed by the
        photo's own filename -- e.g. picking `data/hik_DA0179014/
        frame_000052.png` above computes the pose for "frame_000052.png"
        automatically, instead of you running `pipeline/creaform_pose.py`
        by hand and pasting its --out path into the field below.

        Named `_creaform_cam_pose_cad.yml`, not `_extrinsics_creaform.yml`
        -- this is the camera's pose relative to the car's own (moving)
        CAD frame, which genuinely varies frame to frame because the car
        moves under a stationary camera; it is not a recomputed absolute
        camera pose. See pipeline/creaform_pose.py's module docstring.
        """
        image_path = self._get("image")
        if not image_path:
            QMessageBox.warning(self, "No image", "Pick a photo first -- its filename is the frame to compute.")
            return
        calibration_path = self._get("creaform_calibration")
        track_csv_path = self._get("creaform_track_csv")
        if not calibration_path or not track_csv_path:
            QMessageBox.warning(
                self, "Missing Creaform inputs",
                "Fill in both the Creaform jig calibration (.yml) and tracking CSV fields first."
            )
            return

        frame_name = os.path.basename(image_path)
        try:
            pose = compute_camera_pose_from_creaform(
                calibration_path, track_csv_path, frame_name, camera_name=self._get("camera_name"))
        except Exception as exc:
            QMessageBox.critical(self, "Creaform pose computation failed", str(exc))
            raise

        frame_stem = os.path.splitext(frame_name)[0]
        out_path = os.path.join(_DATA_DIR, f"{frame_stem}_creaform_cam_pose_cad.yml")
        save_camera_pose(out_path, pose)
        self.entries["extrinsics"].setText(out_path)
        QMessageBox.information(
            self, "Pose computed",
            f"{frame_name}: camera_position_world (CAD frame, mm) = "
            f"[{pose.camera_position_world[0]:.1f}, {pose.camera_position_world[1]:.1f}, "
            f"{pose.camera_position_world[2]:.1f}]\nSaved to: {out_path}"
        )

    def _compute_vo_pose(self):
        """Look up the current photo's camera-pose-in-CAD-frame directly
        from the VO per-frame track JSON, keyed by the photo's own
        filename. Unlike Creaform's, this pose needs no calibration chain
        -- the track file's per-frame matrix already *is* the camera pose
        in the car/CAD frame (see pipeline/vo_pose.py's docstring)."""
        image_path = self._get("image")
        if not image_path:
            QMessageBox.warning(self, "No image", "Pick a photo first -- its filename is the frame to look up.")
            return
        track_json_path = self._get("vo_track_json")
        if not track_json_path:
            QMessageBox.warning(self, "Missing VO track", "Fill in the VO per-frame track JSON field first.")
            return

        frame_name = os.path.basename(image_path)
        try:
            pose = compute_camera_pose_from_vo_track(
                track_json_path, frame_name, camera_name=self._get("camera_name"))
        except Exception as exc:
            QMessageBox.critical(self, "VO pose lookup failed", str(exc))
            raise

        frame_stem = os.path.splitext(frame_name)[0]
        out_path = os.path.join(_DATA_DIR, f"{frame_stem}_vo_cam_pose_cad.yml")
        save_camera_pose(out_path, pose)
        self.entries["extrinsics"].setText(out_path)
        QMessageBox.information(
            self, "Pose looked up",
            f"{frame_name}: camera_position_world (CAD frame, mm) = "
            f"[{pose.camera_position_world[0]:.1f}, {pose.camera_position_world[1]:.1f}, "
            f"{pose.camera_position_world[2]:.1f}]\nSaved to: {out_path}"
        )

    def _run(self):
        # run_backprojection's mesh now defaults to full resolution (see its
        # own docstring -- decimation was measured to introduce mesh gaps
        # that inflate error_mm for rays passing through them), which can
        # take a minute or two the first time a given mesh file is loaded
        # (cached to disk after, via mesh_utils.load_car_mesh) -- show a
        # wait cursor so the GUI doesn't look hung during that first run.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            mesh_scale_str = self._get("mesh_scale")
            try:
                mesh_scale = float(mesh_scale_str) if mesh_scale_str else 1000.0
            except ValueError:
                QMessageBox.critical(self, "Invalid mesh scale", f"{mesh_scale_str!r} is not a number.")
                return
            report, mesh, hits, pose = run_backprojection(
                intrinsics_path=self._get("intrinsics"),
                extrinsics_path=self._get("extrinsics"),
                points3d_path=self._get("points3d"),
                correspondences_path=self._get("correspondences"),
                mesh_path=self._get("mesh"),
                mesh_scale=mesh_scale,
                image_path=self._get("image"),
                circles_debug_path=self._get("circles_debug"),
                body_to_cad=self._get("body_to_cad"),
                raw_points3d_path=self._get("raw_points3d"),
                raw_body_to_cad=self._get("raw_body_to_cad"),
                camera_name=self._get("camera_name"),
                log_run=True,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Run failed", str(exc))
            raise
        finally:
            QApplication.restoreOverrideCursor()

        self._last_result = (mesh, hits, pose)
        self.view_btn.setEnabled(True)

        hits_sorted = sorted(hits, key=lambda h: (not h.hit, -(h.error_mm if h.hit else 0)))
        self.table.setRowCount(len(hits_sorted))
        for row, h in enumerate(hits_sorted):
            values = [
                str(h.circle_index), str(h.point_line), f"({h.pixel_u:.1f}, {h.pixel_v:.1f})",
                "yes" if h.hit else "MISS", f"{h.error_mm:.2f}" if h.hit else "-",
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, item)

        s = report["summary"]
        if s["num_hit"] > 0:
            self.summary_label.setText(
                f"{s['num_hit']}/{s['num_targets']} targets hit  |  "
                f"RMS={s['error_mm_rms']:.2f}mm  mean={s['error_mm_mean']:.2f}mm  "
                f"median={s['error_mm_median']:.2f}mm  max={s['error_mm_max']:.2f}mm"
            )
        else:
            self.summary_label.setText(f"0/{s['num_targets']} targets hit -- check pose/points3d/mesh alignment.")

    def _view(self):
        if not self._last_result:
            return
        # `mesh` here is already full resolution: run_backprojection's
        # mesh_target_triangles now defaults to 0 (see backproject_targets.py
        # run()'s docstring -- decimation was measured to introduce mesh
        # gaps that let some back-projection rays skip the outer skin and
        # hit an inner panel, inflating error_mm well beyond the real error).
        mesh, hits, pose = self._last_result
        try:
            tag_size_mm = self._parse_float(self._get("tag_size_mm"), "Viewer tag size")
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid field", str(exc))
            return
        show_web_viewer(
            mesh, hits,
            camera_position=pose.camera_position_world,
            camera_rotation=pose.T_world_cam[:3, :3],
            sphere_radius_mm=tag_size_mm,
            summary_line=self.summary_label.text(),
        )

    def _current_mesh_preset_key(self):
        """Which of _MESH_PRESETS the current "Car mesh" field matches, by
        path (not by remembering which preset button was last clicked --
        the field is a plain editable text box, so comparing its current
        value is what stays correct if it's hand-edited). Falls back to
        "door_free" if it matches neither (e.g. a custom mesh path) -- a
        custom mesh has no curated tracking-model viewer of its own either
        way, so that's the closest available fallback, not a real answer.
        """
        current = os.path.normcase(os.path.abspath(self._get("mesh") or ""))
        for key, preset in _MESH_PRESETS.items():
            if os.path.normcase(os.path.abspath(preset["mesh"])) == current:
                return key
        return "door_free"

def main():
    app = QApplication(sys.argv)
    win = BackprojectionGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
