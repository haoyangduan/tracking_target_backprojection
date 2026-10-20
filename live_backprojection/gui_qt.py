"""PyQt6 front-end for the live-detection -> VO back-projection -> timeline-
viewer pipeline, following the same form-panel / results-panel / Qt-layout
conventions as pipeline/gui_qt.py (the existing single-photo back-projection
GUI) -- a QFormLayout of editable path/parameter fields with Browse buttons
(nothing hardcoded), Run buttons, and a results panel.

Detector thresholding (block_size, C offset, center-ring cutoff) is NOT a
manually-entered field here and is never tuned by watching the live
dataset -- see live_backprojection/teacher.py. It's derived automatically
from a separate TEACHER dataset (its own data_dir/VO-track/mesh/points3d/
intrinsics/camera_name, all independently configurable -- today they
happen to equal the live dataset's own fields, but that's a coincidence of
what data exists right now, not an assumption baked into the code) via the
"Derive Thresholding..." button, which builds an oracle from the teacher's
VO poses + known tracking-model points + mesh raycasting (same method as
build_oracle_labels_full.py) and automatically grid-searches detector
parameters against it. "Run Pipeline" is disabled until that has produced
a result at least once.

Both the derivation and the live-detection+projection run execute on a
background QThread (not the GUI thread) since they take real time, and ALL
terminal output produced during either -- this module's own stage markers,
live_detector's per-frame progress, teacher.py's search-sweep tables, any
print() from deeper in mesh_utils/project_to_mesh -- is captured and
streamed live into an in-GUI log panel via a stdout redirect.

The "Open Web Viewer" button starts disabled and is only enabled after a
successful pipeline run, mirroring pipeline/gui_qt.py's "Open 3D Viewer"
button.
"""
import contextlib
import json
import os
import re
import sys
import traceback
import yaml

# Support both `python -m live_backprojection.gui_qt` from the parent
# directory and direct `python gui_qt.py` from a cloned repository.
if __package__ in (None, ""):
    _REPO_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO_PARENT not in sys.path:
        sys.path.insert(0, _REPO_PARENT)

from live_backprojection import _pathsetup  # noqa: F401
from live_backprojection.pose_resolution import ZED_TO_HIK
from single_frame_backprojection.vo_pose import load_vo_track_json

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QLabel, QFileDialog, QMessageBox, QSplitter,
    QScrollArea, QTextEdit, QSizePolicy, QFrame, QComboBox,
)
import qtawesome as qta

from live_backprojection.cluster_results import (
    DEFAULT_ERROR_THRESHOLD_MM as _DEFAULT_ERROR_THRESHOLD_MM,
    DEFAULT_MERGE_DIST_MM as _DEFAULT_MERGE_DIST_MM, build_final_results,
    export_cluster_json,
)
from live_backprojection.live_detector import (
    DATA_DIR,
    LiveDetectionRun,
    frame_source_from_directory,
    save_detection_csv,
)
from live_backprojection.project_to_mesh import (
    CAMERA_NAME as _DEFAULT_CAMERA_NAME, INTRINSICS_PATH as _DEFAULT_INTRINSICS,
    MESH_PATH as _DEFAULT_MESH, MESH_SCALE as _DEFAULT_MESH_SCALE, VO_TRACK as _DEFAULT_VO_TRACK,
    build_mesh_scene, project_all_frames,
)
from live_backprojection.teacher import (
    DEFAULT_CREAFORM_MAX_TIME_DIFF_MS as _DEFAULT_CREAFORM_MAX_TIME_DIFF_MS, derive_from_teacher_config,
)
from live_backprojection.web_viewer_final_results import show_final_results
from live_backprojection.web_viewer_timeline import (
    TRACKING_MODEL_PATH as _DEFAULT_TRACKING_MODEL, show_timeline,
)

_OUT_DIR = os.path.join(_pathsetup.PROJECT_ROOT, "live_backprojection", "out")
_DEFAULT_OUT_DETECTIONS = os.path.join(_OUT_DIR, "detect.csv")
_DEFAULT_OUT_PROJECTION = os.path.join(_OUT_DIR, "projection.json")
_DEFAULT_OUT_CLUSTER = os.path.join(_OUT_DIR, "cluster.json")

# Default rig extrinsics file defining fixed relative camera poses
_DEFAULT_RIG_EXTRINSICS = os.path.join(_pathsetup.PROJECT_ROOT, "data", "extrinsics_on_gui_0922.json")

# Resolution mode options: (label shown in combobox, scale value)
# The scale choice is made ONCE and applies to both threshold derivation
# (teacher oracle sweep) and live detection -- they must use the same
# resolution, since block_size/C/area bounds are pixel-scale-dependent.
# Full-res is ~4x slower per frame (4096x3000 instead of 2048x1500 mask)
# but gives more detection candidates to work with on the full sensor image.
RESOLUTION_OPTIONS = [
    ("Downsampled 0.5Ã—  (faster, default)", 0.5),
    ("Full resolution 1.0Ã—  (slower, ~4Ã— cost)", 1.0),
]

KNOWN_CAMERAS = [
    "hik_DA0179014",
    "hik_DA0883721",
    "hik_ED51079AAK00091",
    "hik_ED51079AAK00095",
]


def detect_camera_name_from_path(path_str: str) -> str | None:
    """Attempts to deduce a known camera name (hik_*) from a folder or file path.
    Handles exact matches, serial substrings, and relaxed casing/delimiters."""
    if not path_str:
        return None
    normalized = path_str.replace("\\", "/")
    # Exact known camera in path
    for cam in KNOWN_CAMERAS:
        if cam in normalized:
            return cam
    # Substring of serial: e.g. DA0883721, DA0179014, ED51079AAK00091, 0883721
    for cam in KNOWN_CAMERAS:
        parts = cam.split("_", 1)
        if len(parts) == 2:
            suffix = parts[1]
            if suffix in normalized:
                return cam
            clean_suffix = "".join(filter(str.isalnum, suffix)).lower()
            clean_norm = "".join(filter(str.isalnum, normalized)).lower()
            if clean_suffix and clean_suffix in clean_norm:
                return cam
            digits_suffix = "".join(filter(str.isdigit, suffix))
            digits_norm = "".join(filter(str.isdigit, normalized))
            if len(digits_suffix) >= 5 and digits_suffix in digits_norm:
                return cam
    m = re.search(r"(hik_[A-Za-z0-9]+)", normalized, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def get_base_camera_from_vo_track(vo_track_path: str) -> str:
    """Reads the camera name directly covered by the VO track file."""
    try:
        if vo_track_path and os.path.isfile(vo_track_path):
            track = load_vo_track_json(vo_track_path)
            raw = track.get("camera", "")
            return ZED_TO_HIK.get(raw, raw) or "hik_DA0179014"
    except Exception:
        pass
    return "hik_DA0179014"


def get_base_camera_from_creaform(calibration_path: str) -> str:
    """Reads the base camera name defined in Creaform calibration YAML."""
    try:
        if calibration_path and os.path.isfile(calibration_path):
            with open(calibration_path, "r") as f:
                calib = yaml.safe_load(f)
            raw = calib.get("creaform_base_cam", "")
            return ZED_TO_HIK.get(raw, raw) or "hik_DA0179014"
    except Exception:
        pass
    return "hik_DA0179014"

# Teacher-oracle pose source (see live_backprojection/teacher.py's module docstring for why the
# TEACHER oracle uses Creaform, an independent external ground truth, while the live pipeline above
# keeps using VO -- these are two different consumers of pose data, deliberately on two different
# sources).
_DEFAULT_CREAFORM_CALIBRATION = os.path.join(_pathsetup.PROJECT_ROOT, "data", "creaform_calibration_220926.yml")
_DEFAULT_CREAFORM_TRACK_CSV = os.path.join(_pathsetup.PROJECT_ROOT, "data", "20260922_171111_fixed_synced.csv")

# (key, short_label, kind, filter_str, tooltip) -- kind is "dir", "file", or None (plain text
# field, no browse button). Labels are kept short on purpose so the input/browse button doesn't
# get pushed far to the right -- longer explanations live in the tooltip instead.

# Shown once, above both the Live and Teacher sections below: this project's single multi-camera
# intrinsics file already has every ROI camera's calibration keyed by camera_name (see
# calib_target.pnp_io.load_camera_intrinsics_any) -- thresholding derivation and live detection
# never need their own separate copy of this path, just their own camera_name to select the right
# entry from it.
SHARED_FIELDS = [
    ("intrinsics_path", "Camera intrinsics (.json)", "file", "JSON (*.json)",
     "Shared across all ROI cameras -- the same file is used for both thresholding derivation and "
     "live detection, selecting the right calibration by camera_name below."),
]

LIVE_FIELDS = [
    ("data_dir", "Frame source directory", "dir", None,
     "Replayed as a live capture, in sorted filename order. Folder selection automatically detects camera name."),
    ("camera_name", "Camera name", None, None,
     "Target camera name. If different from base camera, rig extrinsics are required."),
    ("vo_track_path", "VO pose track (.json)", "file", "JSON (*.json)",
     "Per-frame camera pose track."),
    ("rig_extrinsics_path", "  â”” Rig extrinsics", "file", "JSON (*.json)",
     "Required if camera_name isn't the VO track's base camera -- derives its pose via the "
     "fixed rig geometry instead (default: data/extrinsics_on_gui_0922.json)."),
    ("mesh_path", "Car mesh", "file", "Mesh files (*.stl *.ply *.obj)", None),
    ("mesh_scale", "  â”” Mesh scale -> mm", None, None,
     "1000 if the mesh is authored in meters, 1 if already in millimeters."),
    ("tracking_model_path", "Ground truth points3d (.txt)", "file", "Text (*.txt)",
     "Creaform tracking-model points -- used for the timeline viewer's reference overlay and for "
     "final-results matching."),
    ("target_hz", "Throttle rate (fps)", None, None, "Blank = run as fast as possible."),
    ("tag_size_mm", "  â”” Viewer tag size (mm)", None, None,
     "Base radius (mm) for target spheres/tags in the 3D result viewers."),
    ("cluster_merge_dist_mm", "  â”” Cluster merge dist. (mm)", None, None,
     "Final results: detections closer than this, across all frames, are treated as the same "
     "physical sticker."),
    ("false_detect_threshold_mm", "  â”” False-detect thresh. (mm)", None, None,
     "Final results: a clustered point farther than this from its nearest ground-truth point is "
     "flagged as a likely false detect."),
]

# The teacher dataset that automatic thresholding is derived from -- a SEPARATE,
# independently-configurable set of paths from LIVE_FIELDS above (except intrinsics_path, shared
# via SHARED_FIELDS above). Defaults currently point at the same files as the live dataset (that's
# simply what data exists today, not a coupling the code assumes anywhere): a future teacher
# capture (e.g. a dedicated calibration-rig run with denser VO coverage) can point these at
# entirely different files.
TEACHER_FIELDS = [
    ("teacher_data_dir", "Teacher frame source dir.", "dir", None,
     "Teacher frame directory. Folder selection automatically detects camera name."),
    ("teacher_camera_name", "Teacher camera name", None, None,
     "Active teacher camera name. If different from base camera, rig extrinsics are required."),
    ("teacher_creaform_calibration_path", "Creaform calibration (.yml)", "file", "YAML (*.yml *.yaml)",
     "Maps the Creaform rig's own base camera to this project's hik_* camera names (creaform_base_cam)."),
    ("teacher_creaform_track_csv_path", "Creaform tracking data (.csv)", "file", "CSV (*.csv)",
     "Per-frame Creaform pose samples, matched to camera frames by timestamp (see time_diff_ms)."),
    ("teacher_creaform_max_time_diff_ms", "  â”” Max Creaform sync err. (ms)", None, None,
     "Frames whose Creaform sample is matched more than this many ms away from the camera "
     "timestamp are excluded from the oracle rather than used with a wrong position."),
    ("teacher_vo_track_path", "Teacher VO track (.json, optional)", "file", "JSON (*.json)",
     "Alternative pose source: per-frame VO track. If specified, VO is used instead of Creaform for teacher oracle."),
    ("teacher_mesh_path", "Teacher car mesh", "file", "Mesh files (*.stl *.ply *.obj)", None),
    ("teacher_mesh_scale", "  â”” Teacher mesh scale -> mm", None, None,
     "1000 if the mesh is authored in meters, 1 if already in millimeters."),
    ("teacher_points3d_path", "Teacher ground truth points3d", "file", "Text (*.txt)",
     "Ground truth for the oracle the detector parameters are searched against."),
    ("teacher_rig_extrinsics_path", "  â”” Rig extrinsics", "file", "JSON (*.json)",
     "Required if teacher_camera_name isn't the base camera -- "
     "derives its pose via the fixed rig geometry instead (default: data/extrinsics_on_gui_0922.json)."),
    ("teacher_export_dir", "Review images folder", "dir", None,
     "Binary-mask + annotated_downsampled images per frame, written after derivation finishes. "
     "Blank = skip."),
]

DEFAULTS = {
    "data_dir": DATA_DIR,
    "intrinsics_path": _DEFAULT_INTRINSICS,
    "camera_name": _DEFAULT_CAMERA_NAME,
    "vo_track_path": _DEFAULT_VO_TRACK,
    "rig_extrinsics_path": "",  # blank: not needed while camera_name matches the VO track's own
                                 # camera (hik_DA0179014 today); fill in to run the live pipeline on
                                 # a DIFFERENT rigidly-mounted camera instead.
    "mesh_path": _DEFAULT_MESH,
    "mesh_scale": str(_DEFAULT_MESH_SCALE),
    "tracking_model_path": _DEFAULT_TRACKING_MODEL,
    "target_hz": "",
    "tag_size_mm": "3.0",
    "cluster_merge_dist_mm": str(_DEFAULT_MERGE_DIST_MM),
    "false_detect_threshold_mm": str(_DEFAULT_ERROR_THRESHOLD_MM),
    "teacher_data_dir": DATA_DIR,
    "teacher_camera_name": _DEFAULT_CAMERA_NAME,
    "teacher_creaform_calibration_path": _DEFAULT_CREAFORM_CALIBRATION,
    "teacher_creaform_track_csv_path": _DEFAULT_CREAFORM_TRACK_CSV,
    "teacher_creaform_max_time_diff_ms": str(_DEFAULT_CREAFORM_MAX_TIME_DIFF_MS),
    "teacher_vo_track_path": "",
    "teacher_mesh_path": _DEFAULT_MESH,
    "teacher_mesh_scale": str(_DEFAULT_MESH_SCALE),
    "teacher_points3d_path": _DEFAULT_TRACKING_MODEL,
    "teacher_rig_extrinsics_path": "",
    "teacher_export_dir": os.path.join(_OUT_DIR, "teacher_review"),
}


class _QtLogStream:
    """File-like object that forwards complete lines to a Qt signal --
    redirected onto sys.stdout for the duration of a background run so
    every print() anywhere in the call chain reaches the in-GUI log panel,
    not just an external terminal."""

    def __init__(self, emit_fn):
        self._emit = emit_fn
        self._buf = ""

    def write(self, text):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line)

    def flush(self):
        pass

    def flush_remainder(self):
        if self._buf:
            self._emit(self._buf)
            self._buf = ""


class TeacherWorker(QThread):
    """Runs the automatic thresholding derivation (build_teacher_oracle +
    derive_detector_params, see teacher.py) off the GUI thread."""
    log_line = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, teacher_config: dict, export_dir: str = None):
        super().__init__()
        self.teacher_config = teacher_config
        self.export_dir = export_dir

    def run(self):
        stream = _QtLogStream(self.log_line.emit)
        try:
            with contextlib.redirect_stdout(stream):
                print(f"=== Deriving thresholding from teacher dataset: {self.teacher_config['data_dir']} ===")
                result = derive_from_teacher_config(self.teacher_config, progress_cb=print, export_dir=self.export_dir)
            stream.flush_remainder()
            self.finished_ok.emit(result)
        except Exception:
            stream.flush_remainder()
            self.failed.emit(traceback.format_exc())


class PipelineWorker(QThread):
    """Runs Stage 1 (live detection) + Stage 2 (mesh projection) off the GUI
    thread. Only communicates back via signals -- never touches a Qt widget
    directly, since this executes on a different thread."""
    log_line = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def run(self):
        stream = _QtLogStream(self.log_line.emit)
        try:
            with contextlib.redirect_stdout(stream):
                stage2 = self._do_run()
            stream.flush_remainder()
            self.finished_ok.emit(stage2)
        except Exception:
            stream.flush_remainder()
            self.failed.emit(traceback.format_exc())

    def _do_run(self) -> dict:
        cfg = self.config

        base_cam = get_base_camera_from_vo_track(cfg.get("vo_track_path", ""))
        print(f"=== Stage 1: live detection over {cfg['data_dir']} ===")
        print(f"    Target Camera: {cfg['camera_name']}")
        print(f"    VO Base Camera: {base_cam}")
        if cfg['camera_name'] != base_cam:
            print(f"    Rig Extrinsics: {cfg.get('rig_extrinsics_path')} (composing {cfg['camera_name']} from {base_cam})")
        else:
            print(f"    Rig Extrinsics: None needed (direct base camera VO lookup)")
        print(f"    detector params (from teacher derivation): block_size={cfg['block_size']} "
              f"c_offset={cfg['c_offset']} scale={cfg['scale']} center_ring_min={cfg['center_ring_min']} "
              f"area=[{cfg['area_lo']:.0f}, {cfg['area_hi']:.0f}]")
        run = LiveDetectionRun(block_size=cfg["block_size"], c_offset=cfg["c_offset"],
                                scale=cfg["scale"], center_ring_min=cfg["center_ring_min"],
                                area_lo=cfg["area_lo"], area_hi=cfg["area_hi"],
                                camera_name=cfg["camera_name"], data_dir=cfg["data_dir"])

        def progress(rec):
            print(f"  frame {rec.frame_name}: {len(rec.detections)} detections")

        run.run(frame_source_from_directory(cfg["data_dir"]), target_hz=cfg["target_hz"], progress_cb=progress,
                camera_name=cfg["camera_name"], data_dir=cfg["data_dir"])
        save_detection_csv(run, cfg["out_detections"], data_dir=cfg["data_dir"], camera_name=cfg["camera_name"])
        print(f"detect.csv merged for camera {cfg['camera_name']} -> {cfg['out_detections']}")

        print("\n=== Stage 2: projecting detections onto the car mesh "
              "(first load of a given mesh can take 1-2 min; cached after) ===")
        print(f"    Camera Intrinsics: {cfg['intrinsics_path']} (camera: {cfg['camera_name']})")
        print(f"    Camera Pose Track: {cfg['vo_track_path']}")
        if cfg['camera_name'] != base_cam:
            print(f"    Rig Composition: {cfg['camera_name']} poses derived via {cfg['rig_extrinsics_path']}")
        detector_output = run.to_json_dict(data_dir=cfg["data_dir"], camera_name=cfg["camera_name"])
        stage2 = project_all_frames(
            detector_output,
            intrinsics_path=cfg["intrinsics_path"],
            camera_name=cfg["camera_name"],
            vo_track_path=cfg["vo_track_path"],
            mesh_path=cfg["mesh_path"],
            mesh_scale=cfg["mesh_scale"],
            rig_extrinsics_path=cfg["rig_extrinsics_path"],
        )
        os.makedirs(os.path.dirname(os.path.abspath(cfg["out_projection"])), exist_ok=True)
        with open(cfg["out_projection"], "w") as f:
            json.dump(stage2, f, indent=2)
        print(f"Projected {len(stage2['frames'])} frames, skipped {stage2['skipped_frame_ids']} (no VO coverage)")
        if stage2["frames"]:
            f0 = stage2["frames"][0]
            print(f"    Verification: Frame {f0['frame_name']} camera position (world mm): {f0['camera_position_world_mm']}")
        print(f"projection.json saved to {cfg['out_projection']}")

        print("\n=== Stage 3: clustering detections into 3D tracking model ===")
        tracking_model = cfg.get("tracking_model_path")
        out_cluster = cfg.get("out_cluster", _DEFAULT_OUT_CLUSTER)
        final_results = build_final_results(
            stage2,
            points3d_path=tracking_model if (tracking_model and os.path.isfile(tracking_model)) else None,
            merge_dist_mm=cfg.get("cluster_merge_dist_mm", _DEFAULT_MERGE_DIST_MM),
            error_threshold_mm=cfg.get("false_detect_threshold_mm", _DEFAULT_ERROR_THRESHOLD_MM),
            camera_name=cfg["camera_name"],
        )
        export_cluster_json(final_results, out_cluster, camera_name=cfg["camera_name"])
        print(f"Clustered {final_results['n_clusters']} points ({final_results['n_false_detects']} false detects)")
        print(f"cluster.json saved to {out_cluster}")
        return stage2


class LiveBackprojectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Detection -> Back-Projection Timeline")
        self.resize(1400, 820)
        self.setMinimumSize(640, 480)

        self.entries = {}
        self._worker = None
        self._teacher_worker = None
        self._last_stage2 = None
        self._derived_params = None  # set once Derive Thresholding succeeds; Run Pipeline requires it

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        splitter.addWidget(self._build_form_panel())
        splitter.addWidget(self._build_results_panel())
        # The form panel (paths/config) is the one with real content; the
        # terminal/results panel is mostly empty until a run is in flight --
        # give the form the bulk of the space and stretch priority instead
        # of the previous even split, which squeezed every field's long
        # label/path text while leaving the results panel mostly blank.
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([980, 420])

    # ------------------------------------------------------------------
    def _add_fields(self, form_layout: QFormLayout, fields):
        for key, label, kind, filter_str, tooltip in fields:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)

            entry = QLineEdit(DEFAULTS.get(key, ""))
            entry.setMinimumWidth(340)
            if tooltip:
                entry.setToolTip(tooltip)
            row_layout.addWidget(entry, stretch=1)

            if kind == "dir":
                browse_btn = QPushButton(qta.icon("fa5s.folder-open"), "")
                browse_btn.setToolTip("Browse for a folder...")
                browse_btn.clicked.connect(lambda _=False, e=entry, k=key: self._browse_dir(e, k))
                row_layout.addWidget(browse_btn)
            elif kind == "file":
                browse_btn = QPushButton(qta.icon("fa5s.folder-open"), "")
                browse_btn.setToolTip("Browse...")
                browse_btn.clicked.connect(lambda _=False, e=entry, f=filter_str: self._browse_file(e, f))
                row_layout.addWidget(browse_btn)

            label_widget = QLabel(label)
            if tooltip:
                label_widget.setToolTip(tooltip)
            self.entries[key] = entry
            form_layout.addRow(label_widget, row)

            # Auto-detection and synchronization hooks
            if key in ("data_dir", "teacher_data_dir"):
                is_teacher = (key == "teacher_data_dir")
                entry.textChanged.connect(lambda txt, t=is_teacher: self._on_data_dir_text_changed(txt, is_teacher=t))
            elif key in ("camera_name", "teacher_camera_name"):
                is_teacher = (key == "teacher_camera_name")
                entry.textChanged.connect(lambda txt, t=is_teacher: self._on_camera_name_text_changed(txt, is_teacher=t))

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight: bold; padding-top: 10px;")
        return lbl

    @staticmethod
    def _hline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    def _build_form_panel(self) -> QWidget:
        form_container = QWidget()
        form_layout = QFormLayout(form_container)
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._add_fields(form_layout, SHARED_FIELDS)

        # --- Resolution mode: shared for both threshold derivation AND detection ---
        res_row = QWidget()
        res_layout = QHBoxLayout(res_row)
        res_layout.setContentsMargins(0, 0, 0, 0)
        self._resolution_combo = QComboBox()
        for label, _ in RESOLUTION_OPTIONS:
            self._resolution_combo.addItem(label)
        self._resolution_combo.setCurrentIndex(0)  # default: downsampled 0.5x
        self._resolution_combo.setToolTip(
            "Resolution used for BOTH threshold derivation (teacher sweep) and live detection. "
            "They must match: block_size, C offset, and area bounds are all pixel-scale-dependent.\n"
            "Full resolution runs the adaptive threshold on the original 4096Ã—3000 image -- "
            "~4Ã— more pixels than downsampled, so derivation and detection both take ~4Ã— longer."
        )
        res_layout.addWidget(self._resolution_combo, stretch=1)
        res_label = QLabel("Detection resolution")
        res_label.setToolTip(
            "Controls the scale factor applied to each frame before adaptive thresholding. "
            "0.5Ã— = half-resolution (faster). 1.0Ã— = full sensor resolution (slower, ~4Ã— cost)."
        )
        form_layout.addRow(res_label, res_row)

        form_layout.addRow(self._hline())
        form_layout.addRow(self._section_label("Live dataset"))
        self._add_fields(form_layout, LIVE_FIELDS)

        form_layout.addRow(self._hline())
        form_layout.addRow(self._section_label("Teacher dataset (automatic thresholding)"))
        self._add_fields(form_layout, TEACHER_FIELDS)

        self.derive_btn = QPushButton(qta.icon("fa5s.graduation-cap"), "  Derive Thresholding...")
        self.derive_btn.setToolTip(
            "Builds an oracle from the teacher dataset's VO poses + known tracking-model points + "
            "mesh raycasting, then automatically grid-searches detector parameters against it. "
            "No threshold is ever hand-entered -- see live_backprojection/teacher.py."
        )
        self.derive_btn.clicked.connect(self._derive_thresholding)
        form_layout.addRow(self.derive_btn)

        self.derived_label = QLabel("Not yet derived -- click \"Derive Thresholding...\" above.")
        self.derived_label.setWordWrap(True)
        self.derived_label.setStyleSheet("font-family: Consolas, monospace; color: #94a3b8; padding: 4px;")
        form_layout.addRow(self.derived_label)

        form_layout.addRow(self._hline())

        self.run_btn = QPushButton(qta.icon("fa5s.play", color="white"), "  Run Pipeline")
        self.run_btn.setStyleSheet("background-color:#2196F3; color:white; font-weight:bold; padding:8px;")
        self.run_btn.setEnabled(False)  # requires a successful Derive Thresholding first
        self.run_btn.clicked.connect(self._run)
        form_layout.addRow(self.run_btn)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_container)
        scroll.setMinimumWidth(680)
        return scroll

    def _build_results_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        layout.addWidget(QLabel("Terminal output"))
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; background:#11141a; color:#cbd5e1;")
        self.log_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.log_view, stretch=1)

        self.summary_label = QLabel(
            "1) Fill in the teacher dataset fields and click \"Derive Thresholding...\"\n"
            "2) Once derived, fill in the live dataset fields and click \"Run Pipeline\"."
        )
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("font-family: Consolas, monospace; padding: 6px;")
        layout.addWidget(self.summary_label)

        self.view_btn = QPushButton(qta.icon("fa5s.cube"), "  Open Web Viewer (per-frame timeline)")
        self.view_btn.setEnabled(False)
        self.view_btn.clicked.connect(self._view)
        layout.addWidget(self.view_btn)

        self.final_results_btn = QPushButton(qta.icon("fa5s.check-double"), "  Open Final Results Viewer")
        self.final_results_btn.setToolTip(
            "Clusters every frame's hit points into one consolidated 'detected tracking model', "
            "matches each cluster to its nearest Creaform ground-truth point, and flags matches "
            "whose error exceeds the false-detect threshold. See live_backprojection/cluster_results.py."
        )
        self.final_results_btn.setEnabled(False)
        self.final_results_btn.clicked.connect(self._view_final_results)
        layout.addWidget(self.final_results_btn)
        return panel

    # ------------------------------------------------------------------
    @staticmethod
    def _start_dir(current_text: str) -> str:
        return current_text.strip() or os.path.join(_pathsetup.PROJECT_ROOT, "data")

    def _browse_file(self, entry: QLineEdit, filter_str: str):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Select file", self._start_dir(entry.text()), (filter_str or "") + ";;All files (*)")
        if fname:
            entry.setText(fname)

    def _browse_dir(self, entry: QLineEdit, key: str = None):
        dname = QFileDialog.getExistingDirectory(self, "Select folder", self._start_dir(entry.text()))
        if dname:
            entry.setText(dname)
            if key == "data_dir":
                self._on_data_dir_selected(dname, is_teacher=False)
            elif key == "teacher_data_dir":
                self._on_data_dir_selected(dname, is_teacher=True)

    def _get_base_camera(self, is_teacher: bool = False) -> str:
        """Determines the reference base camera for VO or Creaform."""
        if is_teacher:
            teacher_vo = self._get("teacher_vo_track_path") if "teacher_vo_track_path" in self.entries else ""
            if teacher_vo and os.path.isfile(teacher_vo):
                return get_base_camera_from_vo_track(teacher_vo)
            creaform_calib = self._get("teacher_creaform_calibration_path")
            if creaform_calib and os.path.isfile(creaform_calib):
                return get_base_camera_from_creaform(creaform_calib)
            return "hik_DA0179014"
        else:
            vo_path = self._get("vo_track_path")
            if vo_path and os.path.isfile(vo_path):
                return get_base_camera_from_vo_track(vo_path)
            return "hik_DA0179014"

    def _on_data_dir_selected(self, dname: str, is_teacher: bool = False):
        """Called when a directory is chosen via Browse dialog."""
        cam = detect_camera_name_from_path(dname)
        target_cam_key = "teacher_camera_name" if is_teacher else "camera_name"
        target_rig_key = "teacher_rig_extrinsics_path" if is_teacher else "rig_extrinsics_path"

        if cam:
            self.entries[target_cam_key].setText(cam)
            base_cam = self._get_base_camera(is_teacher=is_teacher)
            if cam != base_cam:
                curr_rig = self._get(target_rig_key)
                if not curr_rig:
                    self.entries[target_rig_key].setText(_DEFAULT_RIG_EXTRINSICS)
                self._append_log(
                    f"[Camera Selected] Set {target_cam_key} to '{cam}'. This differs from base camera '{base_cam}'. "
                    f"Selected default rig extrinsics: {self._get(target_rig_key)}"
                )
                QMessageBox.information(
                    self,
                    "Different Base Camera Detected",
                    f"Selected folder corresponds to camera: <b>{cam}</b>.<br><br>"
                    f"This camera differs from the base camera (<b>{base_cam}</b>).<br>"
                    f"Rig extrinsics are required to project detections from this camera.<br><br>"
                    f"Automatically selected default rig extrinsics:<br>"
                    f"<code>{self._get(target_rig_key)}</code>",
                )
            else:
                self._append_log(f"[Camera Selected] Set {target_cam_key} to '{cam}' (matches base camera '{base_cam}').")

    def _on_data_dir_text_changed(self, txt: str, is_teacher: bool = False):
        """Called when data_dir or teacher_data_dir path text changes."""
        cam = detect_camera_name_from_path(txt)
        if not cam:
            return
        target_cam_key = "teacher_camera_name" if is_teacher else "camera_name"
        target_rig_key = "teacher_rig_extrinsics_path" if is_teacher else "rig_extrinsics_path"
        if target_cam_key in self.entries:
            curr = self._get(target_cam_key)
            if curr != cam:
                self.entries[target_cam_key].blockSignals(True)
                self.entries[target_cam_key].setText(cam)
                self.entries[target_cam_key].blockSignals(False)
                self._append_log(f"[Auto-detect] Updated {target_cam_key} to '{cam}' from folder path.")

        base_cam = self._get_base_camera(is_teacher=is_teacher)
        if cam != base_cam:
            curr_rig = self._get(target_rig_key)
            if not curr_rig and target_rig_key in self.entries:
                self.entries[target_rig_key].setText(_DEFAULT_RIG_EXTRINSICS)
                self._append_log(
                    f"[Auto-detect] Camera '{cam}' differs from base camera '{base_cam}'. "
                    f"Auto-selected default rig extrinsics: {_DEFAULT_RIG_EXTRINSICS}"
                )

    def _on_camera_name_text_changed(self, txt: str, is_teacher: bool = False):
        """Called when camera_name or teacher_camera_name text changes."""
        cam = txt.strip()
        if not cam:
            return
        base_cam = self._get_base_camera(is_teacher=is_teacher)
        target_rig_key = "teacher_rig_extrinsics_path" if is_teacher else "rig_extrinsics_path"
        if cam != base_cam and not self._get(target_rig_key) and target_rig_key in self.entries:
            self.entries[target_rig_key].setText(_DEFAULT_RIG_EXTRINSICS)
            self._append_log(
                f"[Notice] Camera '{cam}' is different from base camera '{base_cam}'. "
                f"Auto-selected default rig extrinsics: {_DEFAULT_RIG_EXTRINSICS}"
            )

    def _get(self, key: str) -> str:
        return self.entries[key].text().strip()

    def _append_log(self, line: str):
        self.log_view.append(line)

    @staticmethod
    def _parse_float(text: str, label: str) -> float:
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"{label!r} must be a number, got {text!r}")

    def _collect_config(self) -> dict:
        """Parses the live-dataset fields. Detector parameters are NOT
        collected here -- they come from self._derived_params, merged in by
        _run(). out_detections/out_projection are fixed defaults, not
        user-editable fields (see module-level _DEFAULT_OUT_DETECTIONS/
        _DEFAULT_OUT_PROJECTION) -- PipelineWorker prints where they land."""
        target_hz_text = self._get("target_hz")
        target_hz = self._parse_float(target_hz_text, "Throttle rate") if target_hz_text else None
        merge_dist_text = self._get("cluster_merge_dist_mm")
        merge_dist = self._parse_float(merge_dist_text, "Cluster merge distance") if merge_dist_text else _DEFAULT_MERGE_DIST_MM
        false_thresh_text = self._get("false_detect_threshold_mm")
        false_thresh = self._parse_float(false_thresh_text, "False-detect threshold") if false_thresh_text else _DEFAULT_ERROR_THRESHOLD_MM
        return {
            "data_dir": self._get("data_dir"),
            "intrinsics_path": self._get("intrinsics_path"),
            "camera_name": self._get("camera_name"),
            "vo_track_path": self._get("vo_track_path"),
            "rig_extrinsics_path": self._get("rig_extrinsics_path") or None,
            "mesh_path": self._get("mesh_path"),
            "mesh_scale": self._parse_float(self._get("mesh_scale"), "Mesh scale"),
            "tracking_model_path": self._get("tracking_model_path") or None,
            "cluster_merge_dist_mm": merge_dist,
            "false_detect_threshold_mm": false_thresh,
            "out_detections": _DEFAULT_OUT_DETECTIONS,
            "out_projection": _DEFAULT_OUT_PROJECTION,
            "out_cluster": _DEFAULT_OUT_CLUSTER,
            "target_hz": target_hz,
        }

    def _collect_teacher_config(self) -> dict:
        teacher_vo = self._get("teacher_vo_track_path") if "teacher_vo_track_path" in self.entries else ""
        vo_path = teacher_vo or self._get("vo_track_path")
        pose_source = "vo" if teacher_vo else "creaform"
        # Resolution mode: propagate the chosen scale into teacher derivation.
        # derive_from_teacher_config forwards this to derive_detector_params
        # so the search runs at the same resolution detection will use.
        scale = RESOLUTION_OPTIONS[self._resolution_combo.currentIndex()][1]
        return {
            "data_dir": self._get("teacher_data_dir"),
            "pose_source": pose_source,
            "creaform_calibration_path": self._get("teacher_creaform_calibration_path"),
            "creaform_track_csv_path": self._get("teacher_creaform_track_csv_path"),
            "vo_track_path": vo_path,
            "creaform_max_time_diff_ms": self._parse_float(
                self._get("teacher_creaform_max_time_diff_ms"), "Max Creaform sync error"),
            "mesh_path": self._get("teacher_mesh_path"),
            "mesh_scale": self._parse_float(self._get("teacher_mesh_scale"), "Teacher mesh scale"),
            "points3d_path": self._get("teacher_points3d_path"),
            "intrinsics_path": self._get("intrinsics_path"),  # shared field -- see SHARED_FIELDS
            "camera_name": self._get("teacher_camera_name"),
            "rig_extrinsics_path": self._get("teacher_rig_extrinsics_path") or None,
            "scale": scale,
        }

    def _collect_export_dir(self):
        return self._get("teacher_export_dir") or None

    # ------------------------------------------------------------------
    def _derive_thresholding(self):
        if not self._get("teacher_data_dir") or not os.path.isdir(self._get("teacher_data_dir")):
            QMessageBox.warning(self, "Invalid teacher frame source",
                                 "Pick a valid teacher frame source directory first.")
            return

        # Check 1: Teacher folder vs. Camera name consistency
        teacher_data_dir = self._get("teacher_data_dir")
        teacher_cam = self._get("teacher_camera_name")
        folder_cam = detect_camera_name_from_path(teacher_data_dir)
        if folder_cam and folder_cam != teacher_cam:
            reply = QMessageBox.question(
                self,
                "Teacher Camera / Folder Mismatch",
                f"The teacher frame source directory appears to be for camera <b>{folder_cam}</b>, "
                f"but Teacher Camera Name is currently set to <b>{teacher_cam}</b>.<br><br>"
                f"Did you mean to derive thresholding with camera <b>{folder_cam}</b>?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.entries["teacher_camera_name"].setText(folder_cam)
                teacher_cam = folder_cam
                self._append_log(f"[Corrected] Switched teacher_camera_name to '{folder_cam}'.")

        # Check 2: Rig extrinsics requirement for teacher non-base camera
        base_cam = self._get_base_camera(is_teacher=True)
        teacher_rig = self._get("teacher_rig_extrinsics_path")
        if teacher_cam != base_cam:
            if not teacher_rig or not os.path.isfile(teacher_rig):
                reply = QMessageBox.question(
                    self,
                    "Teacher Rig Extrinsics Required",
                    f"Teacher camera <b>{teacher_cam}</b> is different from the base camera (<b>{base_cam}</b>).<br><br>"
                    f"Rig extrinsics are required to compute teacher poses for {teacher_cam}.<br><br>"
                    f"Would you like to use the default rig extrinsics file?<br>"
                    f"<code>{_DEFAULT_RIG_EXTRINSICS}</code>",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.entries["teacher_rig_extrinsics_path"].setText(_DEFAULT_RIG_EXTRINSICS)
                    self._append_log(f"[Notice] Selected default teacher rig extrinsics: {_DEFAULT_RIG_EXTRINSICS}")
                else:
                    QMessageBox.warning(
                        self,
                        "Missing Rig Extrinsics",
                        f"Cannot derive thresholding with non-base camera '{teacher_cam}' without a valid rig extrinsics file."
                    )
                    return

        try:
            teacher_config = self._collect_teacher_config()
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid field", str(exc))
            return

        self.log_view.clear()
        self.summary_label.setText("Deriving thresholding from teacher dataset "
                                    "(mesh load can take 1-2 min the first time)...")
        self.derive_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self._resolution_combo.setEnabled(False)  # lock resolution during derivation

        self._teacher_worker = TeacherWorker(teacher_config, export_dir=self._collect_export_dir())
        self._teacher_worker.log_line.connect(self._append_log)
        self._teacher_worker.finished_ok.connect(self._on_derived)
        self._teacher_worker.failed.connect(self._on_derive_failed)
        self._teacher_worker.start()

    def _on_derived(self, result: dict):
        self.derive_btn.setEnabled(True)
        self._resolution_combo.setEnabled(True)
        self._derived_params = {k: result[k] for k in
                                 ("block_size", "c_offset", "scale", "center_ring_min", "area_lo", "area_hi")}
        self.run_btn.setEnabled(True)
        scale = self._derived_params['scale']
        res_label = "0.5Ã— downsampled" if scale == 0.5 else ("1.0Ã— full-res" if scale == 1.0 else f"{scale}Ã—")
        self.derived_label.setText(
            f"Derived ({res_label}): block_size={self._derived_params['block_size']}  "
            f"C={self._derived_params['c_offset']}  "
            f"ring_min={self._derived_params['center_ring_min']}  "
            f"area=[{self._derived_params['area_lo']:.0f}, {self._derived_params['area_hi']:.0f}]"
        )
        summary = "Thresholding derived -- ready to Run Pipeline."
        if result.get("export_dir"):
            summary += f"\nReview images (binary_mask + annotated_downsampled) -> {result['export_dir']}"
        self.summary_label.setText(summary)

    def _on_derive_failed(self, traceback_text: str):
        self.derive_btn.setEnabled(True)
        self._resolution_combo.setEnabled(True)
        self.run_btn.setEnabled(self._derived_params is not None)
        self.summary_label.setText("Thresholding derivation failed -- see terminal output / error dialog.")
        QMessageBox.critical(self, "Derivation failed", traceback_text)

    # ------------------------------------------------------------------
    def _run(self):
        if self._derived_params is None:
            QMessageBox.warning(
                self, "Thresholding not derived",
                "Click \"Derive Thresholding...\" first -- detector parameters are derived "
                "automatically from the teacher dataset, not entered manually."
            )
            return

        # Check: resolution mode must match what was derived -- block_size/C/area
        # bounds are scale-dependent, so running at a different scale than the
        # one they were derived at will silently use the wrong thresholds.
        selected_scale = RESOLUTION_OPTIONS[self._resolution_combo.currentIndex()][1]
        derived_scale = self._derived_params.get("scale", 0.5)
        if abs(selected_scale - derived_scale) > 1e-9:
            res_selected = "0.5Ã— downsampled" if selected_scale == 0.5 else "1.0Ã— full-res"
            res_derived = "0.5Ã— downsampled" if derived_scale == 0.5 else "1.0Ã— full-res"
            reply = QMessageBox.question(
                self,
                "Resolution Mismatch â€” Re-derive Required",
                f"The detection resolution is set to <b>{res_selected}</b>, but the last derivation "
                f"ran at <b>{res_derived}</b>.<br><br>"
                f"block_size, C offset, and area bounds are all pixel-scale-dependent â€” running at a "
                f"different resolution than they were derived at will give incorrect results.<br><br>"
                f"Click <b>Derive Thresholding...</b> first to re-derive at {res_selected}.",
                QMessageBox.StandardButton.Ok,
            )
            return
        if not self._get("data_dir") or not os.path.isdir(self._get("data_dir")):
            QMessageBox.warning(self, "Invalid frame source", "Pick a valid frame source directory first.")
            return

        # Check 1: Folder vs. Camera Name consistency
        data_dir = self._get("data_dir")
        cam_name = self._get("camera_name")
        folder_cam = detect_camera_name_from_path(data_dir)
        if folder_cam and folder_cam != cam_name:
            reply = QMessageBox.question(
                self,
                "Camera / Folder Mismatch",
                f"The frame source directory appears to be for camera <b>{folder_cam}</b>, "
                f"but Camera Name is currently set to <b>{cam_name}</b>.<br><br>"
                f"Did you mean to run detection with camera <b>{folder_cam}</b>?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.entries["camera_name"].setText(folder_cam)
                cam_name = folder_cam
                self._append_log(f"[Corrected] Switched camera_name to '{folder_cam}'.")

        # Check 2: Rig extrinsics requirement for non-base camera
        base_cam = self._get_base_camera(is_teacher=False)
        rig_path = self._get("rig_extrinsics_path")
        if cam_name != base_cam:
            if not rig_path or not os.path.isfile(rig_path):
                reply = QMessageBox.question(
                    self,
                    "Rig Extrinsics Required",
                    f"Camera <b>{cam_name}</b> is different from the base VO camera (<b>{base_cam}</b>).<br><br>"
                    f"Rig extrinsics are <b>strictly required</b> to compute poses and project detections for {cam_name}.<br><br>"
                    f"Would you like to use the default rig extrinsics file?<br>"
                    f"<code>{_DEFAULT_RIG_EXTRINSICS}</code>",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.entries["rig_extrinsics_path"].setText(_DEFAULT_RIG_EXTRINSICS)
                    rig_path = _DEFAULT_RIG_EXTRINSICS
                    self._append_log(f"[Notice] Selected default rig extrinsics: {_DEFAULT_RIG_EXTRINSICS}")
                else:
                    QMessageBox.warning(
                        self,
                        "Missing Rig Extrinsics",
                        f"Cannot run with non-base camera '{cam_name}' without a valid rig extrinsics file.\n"
                        f"Please select one in 'Rig extrinsics' before running."
                    )
                    return

        try:
            config = self._collect_config()
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid field", str(exc))
            return
        config.update(self._derived_params)

        self.log_view.clear()
        self.summary_label.setText("Running...")
        self.view_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.derive_btn.setEnabled(False)
        self._resolution_combo.setEnabled(False)  # lock resolution during pipeline run

        self._worker = PipelineWorker(config)
        self._worker.log_line.connect(self._append_log)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_finished(self, stage2: dict):
        self.run_btn.setEnabled(True)
        self.derive_btn.setEnabled(True)
        self._resolution_combo.setEnabled(True)
        self._last_stage2 = stage2
        self.view_btn.setEnabled(True)
        self.final_results_btn.setEnabled(True)

        n_frames = len(stage2["frames"])
        n_skipped = len(stage2.get("skipped_frame_ids", []))
        n_hits = sum(1 for f in stage2["frames"] for h in f["hits"] if h["hit"])
        n_total = sum(len(f["hits"]) for f in stage2["frames"])
        cam = stage2.get("source", {}).get("camera_name", self._get("camera_name"))
        rig = stage2.get("source", {}).get("rig_extrinsics_path")
        rig_str = " (via rig extrinsics)" if rig else " (base VO camera)"
        self.summary_label.setText(
            f"Active Camera: {cam}{rig_str}\n"
            f"{n_frames} frames projected"
            + (f", {n_skipped} skipped (no VO coverage)" if n_skipped else "")
            + f"  |  {n_hits}/{n_total} detections hit the mesh"
        )

    def _on_failed(self, traceback_text: str):
        self.run_btn.setEnabled(True)
        self.derive_btn.setEnabled(True)
        self._resolution_combo.setEnabled(True)
        self.summary_label.setText("Run failed -- see terminal output / error dialog.")
        QMessageBox.critical(self, "Run failed", traceback_text)

    def _view(self):
        if not self._last_stage2:
            return
        try:
            tag_size_mm = self._parse_float(self._get("tag_size_mm"), "Viewer tag size")
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid field", str(exc))
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            mesh, _scene = build_mesh_scene(self._get("mesh_path"), scale=float(self._get("mesh_scale") or 1.0))
            tracking_model_path = self._get("tracking_model_path") or None
            cam = self._last_stage2.get("source", {}).get("camera_name", self._get("camera_name"))
            window_title = f"Live Detection Timeline â€” Camera: {cam}"
            show_timeline(mesh, self._last_stage2, tracking_model_path=tracking_model_path,
                          sphere_radius_mm=tag_size_mm, window_name=window_title)
        finally:
            QApplication.restoreOverrideCursor()

    def _view_final_results(self):
        if not self._last_stage2:
            return
        tracking_model_path = self._get("tracking_model_path")
        if not tracking_model_path or not os.path.isfile(tracking_model_path):
            QMessageBox.warning(
                self, "No ground truth",
                "The final-results viewer matches clustered detections against the Creaform ground-"
                "truth tracking model -- fill in a valid 'Tracking model / Creaform ground truth "
                "points3d' path first."
            )
            return
        try:
            merge_dist_mm = self._parse_float(self._get("cluster_merge_dist_mm"), "Cluster merge distance")
            error_threshold_mm = self._parse_float(self._get("false_detect_threshold_mm"), "False-detect threshold")
            tag_size_mm = self._parse_float(self._get("tag_size_mm"), "Viewer tag size")
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid field", str(exc))
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            cam = self._last_stage2.get("source", {}).get("camera_name", self._get("camera_name"))
            final_results = build_final_results(self._last_stage2, tracking_model_path,
                                                 merge_dist_mm=merge_dist_mm, error_threshold_mm=error_threshold_mm,
                                                 camera_name=cam)
            export_cluster_json(final_results, _DEFAULT_OUT_CLUSTER, camera_name=cam)
            self._append_log(f"cluster.json exported to {_DEFAULT_OUT_CLUSTER}")
            mesh, _scene = build_mesh_scene(self._get("mesh_path"), scale=float(self._get("mesh_scale") or 1.0))
            window_title = f"Final Detected Tracking Model â€” Camera: {cam}"
            show_final_results(mesh, final_results, sphere_radius_mm=tag_size_mm, window_name=window_title)
        finally:
            QApplication.restoreOverrideCursor()


def main():
    app = QApplication(sys.argv)
    win = LiveBackprojectionGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
