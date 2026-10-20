"""Interactive tracking-target circle-detection tuner (PyQt6).

Replaces the old Tkinter version (pipeline/circle_tuner.py), which used
fixed-pixel Canvas widgets that clipped on smaller/higher-DPI screens.
This version uses QGraphicsView for both the ROI-picking thumbnail and
the zoomed preview -- both call `fitInView` on resize, so the image
always scales to fill whatever panel size the (freely resizable)
QSplitter gives it, and detected-circle markers are real scene items
(clickable to toggle active/deselected) instead of being baked into a
static image.

Output schema is unchanged from the Tkinter version: a `circle_debug.json`
matching close_track_gui's own sessions
(`image`, `camera`, `total_detected`, `active_count`, `active_indices`,
`deselected_indices`, `rois`, `circles`), so it plugs straight into
`pipeline/backproject_targets.py --circles-debug`.
"""

import json
import os
import sys

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor, QBrush
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QLabel, QFileDialog, QMessageBox, QSplitter,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsEllipseItem,
    QGraphicsPixmapItem, QGraphicsSimpleTextItem, QSlider, QDoubleSpinBox,
    QListWidget, QAbstractItemView,
)


def apply_gain(bgr_image: np.ndarray, gain: float) -> np.ndarray:
    """Multiply pixel values by `gain` (clipped to 0..255). Used only for
    display/detection input -- saved circle pixel coordinates are the
    same regardless of gain, only which pixels register as "dark enough"
    for the threshold changes."""
    if gain == 1.0:
        return bgr_image
    return cv2.convertScaleAbs(bgr_image, alpha=gain, beta=0)

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_PIPELINE_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
_DEFAULT_INTRINSICS = os.path.join(_DATA_DIR, "intrinsics_sensors_0922.json")

from calib_target.blob_detect import BlobDetectorParams, detect_circles_multi_roi
from calib_target.pnp_io import load_camera_intrinsics_any
from single_frame_backprojection.camera_model import undistort_image


def cv2_to_qpixmap(bgr_image: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())  # copy: detach from the numpy buffer


class RoiState:
    def __init__(self, roi, params: BlobDetectorParams = None):
        self.roi = tuple(roi)  # (x, y, w, h), full-image pixel coords
        self.params = params or BlobDetectorParams()


# ---------------------------------------------------------------------------
# Left panel: full-image view with drag-to-add-ROI
# ---------------------------------------------------------------------------

class RoiSelectView(QGraphicsView):
    roiDrawn = pyqtSignal(int, int, int, int)  # x, y, w, h in image pixel coords

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(self.renderHints())
        self._pixmap_item = None
        self._drag_start = None
        self._temp_rect_item = None
        self._roi_items = {}  # index -> QGraphicsRectItem

    def set_image(self, bgr_image: np.ndarray):
        self.scene().clear()
        self._roi_items = {}
        pixmap = cv2_to_qpixmap(bgr_image)
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene().addItem(self._pixmap_item)
        self.scene().setSceneRect(QRectF(pixmap.rect()))
        self._fit()

    def _fit(self):
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def redraw_rois(self, rois: list, selected_idx):
        for item in self._roi_items.values():
            self.scene().removeItem(item)
        self._roi_items = {}
        for i, rs in enumerate(rois):
            x, y, w, h = rs.roi
            color = QColor("lime") if i == selected_idx else QColor("cyan")
            pen = QPen(color, 3 if i == selected_idx else 2)
            item = self.scene().addRect(x, y, w, h, pen)
            self._roi_items[i] = item

    def mousePressEvent(self, event):
        if self._pixmap_item is not None and event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = self.mapToScene(event.pos())
            self._temp_rect_item = self.scene().addRect(
                QRectF(self._drag_start, self._drag_start), QPen(QColor("yellow"), 2))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None and self._temp_rect_item is not None:
            cur = self.mapToScene(event.pos())
            self._temp_rect_item.setRect(QRectF(self._drag_start, cur).normalized())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag_start is not None and self._temp_rect_item is not None:
            rect = self._temp_rect_item.rect()
            self.scene().removeItem(self._temp_rect_item)
            self._temp_rect_item = None
            self._drag_start = None
            if rect.width() > 5 and rect.height() > 5:
                self.roiDrawn.emit(int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()))
        super().mouseReleaseEvent(event)


# ---------------------------------------------------------------------------
# Middle panel: zoomed ROI crop with clickable circle markers
# ---------------------------------------------------------------------------

class _ClickableEllipse(QGraphicsEllipseItem):
    def __init__(self, rect, index, on_click):
        super().__init__(rect)
        self._index = index
        self._on_click = on_click
        self.setAcceptHoverEvents(True)

    def mousePressEvent(self, event):
        self._on_click(self._index)
        event.accept()


class ZoomPreviewView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self._bg_item = None

    def _fit(self):
        rect = self.scene().sceneRect()
        if not rect.isEmpty():
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def render_crop(self, bgr_or_mask_image: np.ndarray, circles, origin_xy, on_circle_click, deselected: set,
                     point_lines: dict = None):
        """circles: list of DetectedCircle already filtered to this ROI.
        origin_xy: (x0, y0) of the crop within the full image, for coordinate mapping.
        point_lines: optional {circle_index: point_line}, shown next to the index."""
        point_lines = point_lines or {}
        self.scene().clear()
        pixmap = cv2_to_qpixmap(bgr_or_mask_image)
        self._bg_item = QGraphicsPixmapItem(pixmap)
        self.scene().addItem(self._bg_item)
        self.scene().setSceneRect(QRectF(pixmap.rect()))

        x0, y0 = origin_xy
        for c in circles:
            active = c.index not in deselected
            color = QColor("#00FF00") if active else QColor("#FF3030")
            r = max(4.0, c.size / 2.0)
            cx, cy = c.x - x0, c.y - y0
            ellipse = _ClickableEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r), c.index, on_circle_click)
            ellipse.setPen(QPen(color, 2))
            ellipse.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            ellipse.setZValue(1)
            self.scene().addItem(ellipse)

            label_text = str(c.index)
            if c.index in point_lines:
                label_text += f"→L{point_lines[c.index]}"
            label = QGraphicsSimpleTextItem(label_text)
            label.setBrush(QBrush(color))
            label.setPos(cx + r + 2, cy - r - 12)
            label.setZValue(1)
            self.scene().addItem(label)

        self._fit()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class CircleTunerWindow(QMainWindow):
    def __init__(self, image_path: str = None, camera_name: str = "", on_saved=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Circle Detection Tuner")
        self.resize(1400, 850)
        self.setMinimumSize(800, 500)

        self.on_saved = on_saved
        self._raw_image = None     # raw loaded image, exactly as read from disk, never modified
        self.image = None          # working image detection runs against: _raw_image, or its
                                    # undistorted version if "Undistort before detection" is checked
        self.display_image = None  # brightness-gained version of self.image; what's shown + detected against
        self.image_path = None
        self.rois: list[RoiState] = []
        self.selected_roi_idx = None
        self.deselected_indices: set[int] = set()
        self.point_lines: dict[int, int] = {}  # circle_index -> points3d line number
        self.all_circles = []
        self._slider_widgets = {}

        self._build_ui()
        if image_path:
            self._load_image(image_path)
        self.camera_entry.setText(camera_name)

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        top = QHBoxLayout()
        open_btn = QPushButton("Open Image...")
        open_btn.clicked.connect(self._browse_image)
        top.addWidget(open_btn)
        self.image_label = QLabel("(no image loaded)")
        top.addWidget(self.image_label)
        top.addStretch(1)
        top.addWidget(QLabel("Brightness gain:"))
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(10, 100)  # -> 1.0x .. 10.0x
        self.gain_slider.setValue(10)
        self.gain_slider.setFixedWidth(140)
        self.gain_slider.valueChanged.connect(self._on_gain_change)
        top.addWidget(self.gain_slider)
        self.gain_label = QLabel("1.0x")
        self.gain_label.setFixedWidth(36)
        top.addWidget(self.gain_label)
        top.addWidget(QLabel("Camera name:"))
        self.camera_entry = QLineEdit()
        self.camera_entry.setFixedWidth(140)
        self.camera_entry.editingFinished.connect(self._on_undistort_inputs_changed)
        top.addWidget(self.camera_entry)
        save_btn = QPushButton("Save circle_debug.json")
        save_btn.setStyleSheet("background-color:#4CAF50; color:white; padding:4px 10px;")
        save_btn.clicked.connect(self._save)
        top.addWidget(save_btn)
        outer.addLayout(top)

        # Lens undistortion, BEFORE detection -- not just the per-point
        # correction pipeline/camera_model.pixel_to_world_ray already does
        # later for ray math. Rationale: calib_target/blob_detect.py fits
        # an ellipse to whatever shape the (possibly distortion-warped)
        # ring marker has in the raw pixels; a physically circular marker
        # that distortion has skewed into an ellipse can bias that fit and
        # the circularity/aspect-ratio filters, in a way a later per-point
        # correction of just the fitted center cannot undo. Off by default
        # (matches all this project's existing circle_debug.json files,
        # which were produced without it -- see load_circles_from_debug_
        # json's `image_undistorted` flag).
        top2 = QHBoxLayout()
        self.undistort_check = QCheckBox("Undistort before detection")
        self.undistort_check.stateChanged.connect(self._on_undistort_toggled)
        top2.addWidget(self.undistort_check)
        top2.addWidget(QLabel("Intrinsics:"))
        self.intrinsics_entry = QLineEdit(_DEFAULT_INTRINSICS)
        self.intrinsics_entry.setMinimumWidth(260)
        self.intrinsics_entry.editingFinished.connect(self._on_undistort_inputs_changed)
        top2.addWidget(self.intrinsics_entry, stretch=1)
        intrinsics_browse_btn = QPushButton("...")
        intrinsics_browse_btn.setFixedWidth(30)
        intrinsics_browse_btn.clicked.connect(self._browse_intrinsics)
        top2.addWidget(intrinsics_browse_btn)
        top2.addStretch(2)
        outer.addLayout(top2)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, stretch=1)

        # --- Left: thumbnail + ROI list ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Drag on the image to add an ROI"))
        self.thumb_view = RoiSelectView()
        self.thumb_view.roiDrawn.connect(self._on_roi_drawn)
        left_layout.addWidget(self.thumb_view, stretch=1)

        self.roi_list = QListWidget()
        self.roi_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.roi_list.currentRowChanged.connect(self._on_roi_select)
        left_layout.addWidget(self.roi_list)
        remove_btn = QPushButton("Remove Selected ROI")
        remove_btn.clicked.connect(self._remove_roi)
        left_layout.addWidget(remove_btn)
        splitter.addWidget(left)

        # --- Middle: params + zoomed preview ---
        mid = QWidget()
        mid_layout = QVBoxLayout(mid)

        params_box = QWidget()
        params_form = QFormLayout(params_box)
        self._add_slider(params_form, "dark_threshold", 0, 255, 100.0)
        self._add_slider(params_form, "min_area", 0, 2000, 10.0)
        self._add_slider(params_form, "max_area", 0, 50000, 20.0)
        self._add_slider(params_form, "min_circularity", 0, 1, 0.6, step=0.01)
        self._add_slider(params_form, "min_aspect_ratio", 0, 1, 0.6, step=0.01)
        self._add_slider(params_form, "merge_dist_px", 0, 20, 0.1, step=0.1)
        mid_layout.addWidget(params_box)

        self.show_mask_check = QCheckBox("Show dark-threshold mask instead of detections")
        self.show_mask_check.stateChanged.connect(self._refresh_zoom)
        mid_layout.addWidget(self.show_mask_check)

        mid_layout.addWidget(QLabel("Zoomed ROI preview (click a circle to toggle active/deselected)"))
        self.zoom_view = ZoomPreviewView()
        mid_layout.addWidget(self.zoom_view, stretch=1)
        splitter.addWidget(mid)

        # --- Right: circle table ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel(
            "Detected circles (double-click #/x/y/size/active to toggle active; "
            "double-click \"point line\" to type its known points3d line number)"
        ))
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["#", "x", "y", "size", "active", "point line"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.cellDoubleClicked.connect(self._on_table_toggle)
        self.table.itemChanged.connect(self._on_point_line_edited)
        right_layout.addWidget(self.table, stretch=1)
        splitter.addWidget(right)

        splitter.setSizes([500, 500, 400])

        self.status_label = QLabel("")
        outer.addWidget(self.status_label)

    def _add_slider(self, form: QFormLayout, name, lo, hi, default, step=1.0):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        n_steps = int(round((hi - lo) / step))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, n_steps)
        slider.setValue(int(round((default - lo) / step)))

        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setDecimals(2 if step < 1 else 0)
        spin.setValue(default)

        def _slider_to_spin(v):
            spin.blockSignals(True)
            spin.setValue(lo + v * step)
            spin.blockSignals(False)
            self._on_param_change()

        def _spin_to_slider(v):
            slider.blockSignals(True)
            slider.setValue(int(round((v - lo) / step)))
            slider.blockSignals(False)
            self._on_param_change()

        slider.valueChanged.connect(_slider_to_spin)
        spin.valueChanged.connect(_spin_to_slider)

        row_layout.addWidget(slider, stretch=1)
        row_layout.addWidget(spin)
        form.addRow(QLabel(name), row)
        self._slider_widgets[name] = spin

    def _current_params(self) -> BlobDetectorParams:
        return BlobDetectorParams(**{name: spin.value() for name, spin in self._slider_widgets.items()})

    def _sync_sliders_from(self, params: BlobDetectorParams):
        for name, spin in self._slider_widgets.items():
            spin.blockSignals(True)
            spin.setValue(getattr(params, name))
            spin.blockSignals(False)

    # ------------------------------------------------------------------
    # Lens undistortion (before detection)
    # ------------------------------------------------------------------

    def _browse_intrinsics(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Select intrinsics file", _DATA_DIR, "Intrinsics (*.yml *.yaml *.json);;All files (*)")
        if fname:
            self.intrinsics_entry.setText(fname)
            self._on_undistort_inputs_changed()

    def _on_undistort_toggled(self):
        self._rebuild_working_image()
        self._update_display_image()
        self._detect_all()

    def _on_undistort_inputs_changed(self):
        """Camera name / intrinsics path edited -- only matters while the
        checkbox is on, but cheap enough to just always recompute."""
        if self.undistort_check.isChecked():
            self._rebuild_working_image()
            self._update_display_image()
            self._detect_all()

    def _rebuild_working_image(self):
        """Recompute `self.image` (what gain/detection operate on) from
        `self._raw_image`: undistorted if the checkbox is on, the raw pixels
        otherwise. Falls back to the raw image (and unchecks the box) if
        undistortion fails, e.g. a bad intrinsics path or unknown camera
        name -- detection should still work on *something* rather than
        silently freezing on stale data."""
        if self._raw_image is None:
            self.image = None
            return
        if not self.undistort_check.isChecked():
            self.image = self._raw_image
            return
        try:
            intr = load_camera_intrinsics_any(
                self.intrinsics_entry.text().strip(), self.camera_entry.text().strip() or None)
            self.image = undistort_image(self._raw_image, intr)
        except Exception as exc:
            QMessageBox.critical(
                self, "Undistortion failed",
                f"Could not undistort using intrinsics {self.intrinsics_entry.text()!r} / "
                f"camera {self.camera_entry.text()!r}:\n{exc}\n\nFalling back to the raw (distorted) image."
            )
            self.undistort_check.blockSignals(True)
            self.undistort_check.setChecked(False)
            self.undistort_check.blockSignals(False)
            self.image = self._raw_image

    # ------------------------------------------------------------------
    # Image / ROI handling
    # ------------------------------------------------------------------

    def _browse_image(self):
        start_dir = os.path.dirname(self.image_path) if self.image_path else _DATA_DIR
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open image", start_dir, "Image files (*.png *.jpg *.jpeg *.bmp)")
        if fname:
            self._load_image(fname)

    def _load_image(self, path):
        image = cv2.imread(path)
        if image is None:
            QMessageBox.critical(self, "Error", f"Could not read image: {path}")
            return
        self._raw_image = image
        self.image_path = path
        self.image_label.setText(os.path.basename(path))
        self._rebuild_working_image()
        self._update_display_image()

        self.rois = []
        self.selected_roi_idx = None
        self.deselected_indices = set()
        self.point_lines = {}
        self._update_roi_list()
        self._detect_all()

    def _on_gain_change(self, value):
        gain = value / 10.0
        self.gain_label.setText(f"{gain:.1f}x")
        if self.image is None:
            return
        self._update_display_image()
        self._detect_all()

    def _update_display_image(self):
        gain = self.gain_slider.value() / 10.0
        self.display_image = apply_gain(self.image, gain)
        self.thumb_view.set_image(self.display_image)

    def _on_roi_drawn(self, x, y, w, h):
        self.rois.append(RoiState((x, y, w, h)))
        self.selected_roi_idx = len(self.rois) - 1
        self._sync_sliders_from(self.rois[self.selected_roi_idx].params)
        self._detect_all()

    def _remove_roi(self):
        if self.selected_roi_idx is None:
            return
        del self.rois[self.selected_roi_idx]
        self.selected_roi_idx = None
        self._detect_all()

    def _update_roi_list(self):
        self.roi_list.blockSignals(True)
        self.roi_list.clear()
        for i, rs in enumerate(self.rois):
            n = sum(1 for c in self.all_circles if self._circle_in_roi(c, rs.roi))
            self.roi_list.addItem(f"ROI {i}: {rs.roi}  ({n} circles)")
        if self.selected_roi_idx is not None:
            self.roi_list.setCurrentRow(self.selected_roi_idx)
        self.roi_list.blockSignals(False)
        self.thumb_view.redraw_rois(self.rois, self.selected_roi_idx)

    def _on_roi_select(self, row):
        if row < 0 or row >= len(self.rois):
            return
        self.selected_roi_idx = row
        self._sync_sliders_from(self.rois[row].params)
        self.thumb_view.redraw_rois(self.rois, self.selected_roi_idx)
        self._refresh_zoom()

    @staticmethod
    def _circle_in_roi(circle, roi):
        x, y, w, h = roi
        return x <= circle.x <= x + w and y <= circle.y <= y + h

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _on_param_change(self):
        if self.selected_roi_idx is None or self.image is None:
            return
        self.rois[self.selected_roi_idx].params = self._current_params()
        self._detect_all()

    def _detect_all(self):
        if self.display_image is None or not self.rois:
            self.all_circles = []
        else:
            rois_and_params = [(rs.roi, rs.params) for rs in self.rois]
            self.all_circles = detect_circles_multi_roi(self.display_image, rois_and_params)
            valid = {c.index for c in self.all_circles}
            self.deselected_indices &= valid
            self.point_lines = {idx: pl for idx, pl in self.point_lines.items() if idx in valid}

        self._update_roi_list()
        self._refresh_zoom()
        self._update_table()
        n_active = len(self.all_circles) - len(self.deselected_indices)
        self.status_label.setText(
            f"{len(self.all_circles)} detected total, {n_active} active, "
            f"{len(self.deselected_indices)} deselected"
        )

    def _update_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.all_circles))
        for row, c in enumerate(self.all_circles):
            active = c.index not in self.deselected_indices
            values = [str(c.index), f"{c.x:.1f}", f"{c.y:.1f}", f"{c.size:.1f}", "yes" if active else "no"]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, col, item)

            pl_item = QTableWidgetItem(str(self.point_lines[c.index]) if c.index in self.point_lines else "")
            pl_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 5, pl_item)
        self.table.blockSignals(False)

    def _on_table_toggle(self, row, col):
        if col == 5:
            return  # point-line column: double-click opens the editor instead of toggling active
        item = self.table.item(row, 0)
        if item:
            self._toggle_circle(int(item.text()))

    def _toggle_circle(self, idx):
        if idx in self.deselected_indices:
            self.deselected_indices.discard(idx)
        else:
            self.deselected_indices.add(idx)
        self._update_table()
        self._refresh_zoom()

    def _on_point_line_edited(self, item):
        if item.column() != 5:
            return
        idx_item = self.table.item(item.row(), 0)
        if idx_item is None:
            return
        circle_index = int(idx_item.text())
        text = item.text().strip()
        if not text:
            self.point_lines.pop(circle_index, None)
            self.status_label.setText(f"Circle {circle_index}: point line cleared")
            return
        try:
            point_line = int(text)
            if point_line < 1:
                raise ValueError
        except ValueError:
            self.status_label.setText(f"Circle {circle_index}: \"{text}\" is not a valid point line -- ignored")
            self.table.blockSignals(True)
            item.setText(str(self.point_lines[circle_index]) if circle_index in self.point_lines else "")
            self.table.blockSignals(False)
            return
        self.point_lines[circle_index] = point_line
        self.status_label.setText(f"Circle {circle_index} -> point line {point_line}")
        self._refresh_zoom()

    # ------------------------------------------------------------------
    # Zoomed preview
    # ------------------------------------------------------------------

    def _refresh_zoom(self):
        if self.display_image is None or self.selected_roi_idx is None:
            self.zoom_view.scene().clear()
            return
        rs = self.rois[self.selected_roi_idx]
        x, y, w, h = rs.roi
        ih, iw = self.display_image.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(iw, x + w), min(ih, y + h)
        crop = self.display_image[y0:y1, x0:x1]
        if crop.size == 0:
            return

        if self.show_mask_check.isChecked():
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            _, mask = cv2.threshold(gray, rs.params.dark_threshold, 255, cv2.THRESH_BINARY_INV)
            display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            display = crop

        circles_in_roi = [c for c in self.all_circles if self._circle_in_roi(c, rs.roi)]
        self.zoom_view.render_crop(display, circles_in_roi, (x0, y0), self._toggle_circle,
                                    self.deselected_indices, self.point_lines)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save(self):
        if self.image_path is None or not self.all_circles:
            QMessageBox.warning(self, "Nothing to save", "Load an image and detect at least one circle first.")
            return
        default_name = os.path.splitext(os.path.basename(self.image_path))[0] + "_circle_debug.json"
        default_path = os.path.join(os.path.dirname(self.image_path), default_name)
        out_path, _ = QFileDialog.getSaveFileName(self, "Save circle_debug.json", default_path, "JSON (*.json)")
        if not out_path:
            return

        active_indices = sorted(c.index for c in self.all_circles if c.index not in self.deselected_indices)
        data = {
            "image": self.image_path,
            "camera": self.camera_entry.text().strip(),
            "brightness_gain": self.gain_slider.value() / 10.0,
            "image_undistorted": self.undistort_check.isChecked(),
            "undistort_intrinsics_path": self.intrinsics_entry.text().strip() if self.undistort_check.isChecked() else None,
            "total_detected": len(self.all_circles),
            "active_count": len(active_indices),
            "active_indices": active_indices,
            "deselected_indices": sorted(self.deselected_indices),
            "rois": [list(rs.roi) for rs in self.rois],
            "roi_params": [
                {k: getattr(rs.params, k) for k in
                 ("dark_threshold", "min_area", "max_area", "min_circularity",
                  "min_aspect_ratio", "merge_dist_px")}
                for rs in self.rois
            ],
            "circles": [
                {"index": c.index, "x": c.x, "y": c.y, "size": c.size,
                 **({"point_line": self.point_lines[c.index]} if c.index in self.point_lines else {})}
                for c in self.all_circles
            ],
        }
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

        # Also write correspondences.json right alongside it, from whatever
        # point lines were filled in for active circles -- no separate
        # hand-authoring step needed.
        correspondences = [
            {"circle_index": idx, "point_line": pl}
            for idx, pl in sorted(self.point_lines.items())
            if idx in set(active_indices)
        ]
        corr_path = None
        status = f"Saved to {out_path}"
        if correspondences:
            base, _ext = os.path.splitext(out_path)
            if base.endswith("_circle_debug"):
                base = base[: -len("_circle_debug")]
            corr_path = base + "_correspondences.json"
            corr_data = {
                "camera": self.camera_entry.text().strip(),
                "image": self.image_path,
                "correspondences": correspondences,
            }
            with open(corr_path, "w") as f:
                json.dump(corr_data, f, indent=2)
            status += f"  |  correspondences ({len(correspondences)}) saved to {corr_path}"

        missing = len(active_indices) - len(correspondences)
        if missing > 0:
            status += f"  |  {missing} active circle(s) still have no point line assigned"
        self.status_label.setText(status)

        if self.on_saved:
            self.on_saved(out_path, corr_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive tracking-target circle detection tuner")
    parser.add_argument("--image", default=None)
    parser.add_argument("--camera", default="")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = CircleTunerWindow(image_path=args.image, camera_name=args.camera)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
