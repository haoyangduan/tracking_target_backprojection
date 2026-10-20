"""Stage 1: live detection.

Simulates the live capture -> detect pipeline by replaying saved frames in
order, running the production detector (evaluate_downsampled_full.
detect_half_ring, ~47-71ms/frame + ~20ms gain -- comfortably inside the
settled 10fps/100ms budget) on each one, and holding the results in an
in-memory, append-only list of {frame_id, pixel coords} records.

The frame source is a plain generator yielding (frame_id, frame_name,
raw_gray) -- the only place that knows frames come from disk today. A real
camera callback would just be a different source of that same triple
(or calls to LiveDetectionRun.ingest() directly); nothing downstream of
process_frame() needs to change.

Checked the actual camera capture rate against this (see data/capture_log.csv):
host-to-host elapsed time across all 182 logged frames is 26.17s (~6.92fps,
144.6ms/frame), comfortably slower than this detector's ~90-100ms/frame --
no backpressure problem with this dataset today. (The log's own cam_ts_ns
column implies ~69fps if taken at face value, but that's a 10x mismatch
against the host-side elapsed time for the exact same frames with zero
frame_id gaps -- far more consistent with cam_ts_ns's tick unit actually
being 10ns rather than the 1ns its name implies, than with a real 69fps
sensor whose host only drains it at 1-in-10.) Still, a push-based live
source (a real camera SDK callback, unlike the pull-based generator below)
has no inherent backpressure handling at all -- LatestFrameSlot/run_live
below add a "keep only the newest frame, drop stale backlog" policy as a
safety net for a transient stall or a different deployment's trigger rate,
not because this specific dataset needs it.
"""
import csv
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from live_backprojection import _pathsetup  # noqa: F401

import cv2

from live_backprojection.detector.evaluate_against_oracle import auto_gain
from live_backprojection.detector.evaluate_downsampled_full import (
    AREA_HI, AREA_LO, BLOCK_SIZE, C_OFFSET, SCALE, detect_half_ring,
)

DATA_DIR = os.path.join(_pathsetup.PROJECT_ROOT, "data", "hik_DA0179014")

# The chosen production cutoff (see tune_ring_threshold_downsampled_full.py):
# detect_half_ring's own default (imported from evaluate_against_oracle.
# CENTER_RING_MIN) is 20, NOT the 25 this project settled on after the
# recall-vs-extras sweep -- must be passed explicitly on every call here.
CENTER_RING_MIN = 25

# Anchored to exactly "frame_" + 6 digits + ".png" -- a naive "frame_*.png"
# glob would also match derived artifacts like frame_000060_undistorted.png.
_FRAME_NAME_RE = re.compile(r"^frame_(\d{6})\.png$")

# Known physical cameras on the multi-camera rig
KNOWN_CAMERAS = [
    "hik_DA0179014",
    "hik_DA0883721",
    "hik_ED51079AAK00091",
    "hik_ED51079AAK00095",
]

DETECTION_CSV_FIELDS = (
    "camera_name",
    "frame_id",
    "frame_name",
    "timestamp",
    "capture_ts_ns",
    "u",
    "v",
    "size",
)


def canonical_camera_id(camera_name: Optional[str]) -> Optional[str]:
    """Resolves arbitrary camera name string/serial into exact canonical camera name."""
    if not camera_name:
        return None
    for cam in KNOWN_CAMERAS:
        if cam.lower() == camera_name.lower():
            return cam
        clean_cam = cam.replace("_", "").lower()
        clean_name = camera_name.replace("_", "").lower()
        if clean_cam in clean_name or clean_name in clean_cam:
            return cam
        serial = cam.split("_", 1)[1].lower()
        if serial in clean_name:
            return cam
    return camera_name


def detect_camera_name_from_path(path_str: Optional[str]) -> Optional[str]:
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
        return canonical_camera_id(m.group(1))
    return None


def find_capture_log_path(data_dir: Optional[str] = None) -> Optional[str]:
    """Finds capture_log.csv in data_dir, parent of data_dir, or PROJECT_ROOT/data."""
    candidates = []
    if data_dir:
        candidates.append(os.path.join(data_dir, "capture_log.csv"))
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(data_dir)), "capture_log.csv"))
    candidates.append(os.path.join(_pathsetup.PROJECT_ROOT, "data", "capture_log.csv"))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def load_capture_timestamps(
    capture_log_path: Optional[str] = None,
    camera_name: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Dict[Any, int]:
    """Loads capture_ts_ns from capture_log.csv keyed by frame_name, frame_id, and round_idx.
    
    Returns a dict mapping:
      - 'frame_000001.png' -> int(capture_ts_ns)
      - 1 (1-indexed frame_id int) -> int(capture_ts_ns)
      - '1' (1-indexed frame_id str) -> int(capture_ts_ns)
      - 0 (0-indexed round_idx int) -> int(capture_ts_ns)
    """
    path = capture_log_path or find_capture_log_path(data_dir)
    if not path or not os.path.isfile(path):
        return {}

    target_cam = canonical_camera_id(camera_name) or (detect_camera_name_from_path(data_dir) if data_dir else None)
    ts_map = {}
    try:
        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cam_id = row.get("camera_id", "")
                if target_cam:
                    if canonical_camera_id(cam_id) != target_cam:
                        continue
                try:
                    ridx = int(row["round_idx"])
                    ts = int(row["capture_ts_ns"])
                except (KeyError, ValueError):
                    continue

                fname = f"frame_{ridx + 1:06d}.png"
                ts_map[fname] = ts
                ts_map[ridx + 1] = ts
                ts_map[str(ridx + 1)] = ts
    except Exception:
        pass
    return ts_map


@dataclass
class Detection:
    u: float   # pixel x, full-resolution
    v: float   # pixel y, full-resolution
    size: float  # ring diameter estimate, px (from detect_half_ring)


@dataclass
class FrameDetections:
    frame_id: int          # 1-indexed number parsed from frame_name (e.g. 1 for frame_000001.png) --
                            # frame_name remains the authoritative VO-lookup key downstream regardless.
    frame_name: str
    timestamp: float        # time.monotonic() when this frame was processed; informational only
    detections: List[Detection] = field(default_factory=list)
    capture_ts_ns: Optional[int] = None  # Hardware/sensor synchronization capture timestamp in nanoseconds


def frame_source_from_directory(
    data_dir: str = DATA_DIR,
    name_re: "re.Pattern" = _FRAME_NAME_RE,
) -> Iterator[Tuple[int, str, "cv2.typing.MatLike"]]:
    """Yields (frame_id, frame_name, raw_gray) in sorted filename order,
    replaying saved frames as a stand-in for a live camera callback feed.
    raw_gray is read via cv2.imread(path, cv2.IMREAD_GRAYSCALE) -- these
    PNGs are natively single-channel, so this is a zero-accuracy-cost direct
    read (see evaluate_against_oracle.py's imread optimization)."""
    names = sorted(f for f in os.listdir(data_dir) if name_re.match(f))
    for name in names:
        frame_id = int(name_re.match(name).group(1))
        raw_gray = cv2.imread(os.path.join(data_dir, name), cv2.IMREAD_GRAYSCALE)
        if raw_gray is None:
            continue
        yield frame_id, name, raw_gray


def process_frame(raw_gray, frame_id: int, frame_name: str, block_size: int = BLOCK_SIZE,
                   c_offset: int = C_OFFSET, scale: float = SCALE,
                   center_ring_min: float = CENTER_RING_MIN,
                   area_lo: float = AREA_LO, area_hi: float = AREA_HI,
                   capture_ts_ns: Optional[int] = None) -> FrameDetections:
    """auto_gain -> convertScaleAbs -> detect_half_ring: the exact
    production detection order established in evaluate_downsampled_full.py.
    Detector parameters default to the project's chosen production config
    but are real parameters (not hardcoded), so a caller (e.g. the GUI) can
    override them. area_lo/area_hi default to the historical fixed bounds
    but should normally come from teacher.py's Stage C derivation instead
    (see live_backprojection/teacher.py's DEFAULT_AREA_MARGIN_FRAC comment
    for why a fixed pixel-area window doesn't adapt to a different camera/
    marker-size setup)."""
    gain = auto_gain(raw_gray)
    gray_full = cv2.convertScaleAbs(raw_gray, alpha=gain, beta=0)
    det = detect_half_ring(gray_full, block_size=block_size, c_offset=c_offset, scale=scale,
                            center_ring_min=center_ring_min, area_lo=area_lo, area_hi=area_hi)
    detections = [Detection(u=float(cx), v=float(cy), size=float(s)) for cx, cy, s in det]
    return FrameDetections(frame_id=frame_id, frame_name=frame_name,
                            timestamp=time.monotonic(), detections=detections,
                            capture_ts_ns=capture_ts_ns)


class LatestFrameSlot:
    """Thread-safe single-slot "keep only the newest" buffer for a
    push-based live source (e.g. a real camera SDK callback) feeding a
    consumer that might be slower. put() always overwrites whatever frame
    was waiting unconsumed, so a producer that outpaces the consumer can
    never build an unbounded backlog -- the older, unconsumed frame is
    DROPPED (counted in dropped_count), not queued. This bounds the
    pipeline's latency at the cost of skipping some frames, which is the
    standard trade for a live perception system: always working on
    close-to-current data beats falling further and further behind trying
    to process every single frame."""

    def __init__(self):
        self._cv = threading.Condition()
        self._pending = None  # (frame_id, frame_name, raw_gray) or None
        self.dropped_count = 0
        self.closed = False

    def put(self, frame_id: int, frame_name: str, raw_gray) -> None:
        """Called by the producer for every new frame. Overwrites any
        not-yet-consumed previous frame."""
        with self._cv:
            if self._pending is not None:
                self.dropped_count += 1
            self._pending = (frame_id, frame_name, raw_gray)
            self._cv.notify()

    def get(self, timeout: Optional[float] = None) -> Optional[Tuple[int, str, "cv2.typing.MatLike"]]:
        """Called by the consumer. Blocks until a frame is available, the
        slot is closed, or timeout elapses; returns and clears whatever was
        pending, or None if there was nothing (closed/timed out)."""
        with self._cv:
            if self._pending is None and not self.closed:
                self._cv.wait(timeout=timeout)
            item = self._pending
            self._pending = None
            return item

    def close(self) -> None:
        with self._cv:
            self.closed = True
            self._cv.notify_all()


class LiveDetectionRun:
    """Holds the in-memory, growing list of FrameDetections for one capture
    session. run() drains a frame source to completion ("the full capture
    finishes"); a real live system would instead call ingest() per camera-
    callback invocation and read .frames at any time without an explicit
    finish step.

    Detector parameters are fixed for the lifetime of one run (set at
    construction, defaulting to the project's chosen production config) and
    recorded in to_json_dict()'s meta -- so the in-memory/exported detection
    records always reflect what was actually used, even if a caller (e.g. the GUI)
    overrides them."""

    def __init__(self, block_size: int = BLOCK_SIZE, c_offset: int = C_OFFSET, scale: float = SCALE,
                 center_ring_min: float = CENTER_RING_MIN, area_lo: float = AREA_LO, area_hi: float = AREA_HI,
                 camera_name: Optional[str] = None, data_dir: Optional[str] = None,
                 capture_log_path: Optional[str] = None):
        self.frames: List[FrameDetections] = []
        self.detector_kwargs = {"block_size": block_size, "c_offset": c_offset, "scale": scale,
                                 "center_ring_min": center_ring_min, "area_lo": area_lo, "area_hi": area_hi}
        self.data_dir = data_dir or DATA_DIR
        self.camera_name = camera_name or (detect_camera_name_from_path(self.data_dir) if self.data_dir else None) or "hik_DA0179014"
        self.capture_log_path = capture_log_path or find_capture_log_path(self.data_dir)
        self._timestamps = load_capture_timestamps(self.capture_log_path, self.camera_name, self.data_dir)

    def ingest(self, raw_gray, frame_id: int, frame_name: str, capture_ts_ns: Optional[int] = None) -> FrameDetections:
        if capture_ts_ns is None and getattr(self, "_timestamps", None):
            capture_ts_ns = self._timestamps.get(frame_name, self._timestamps.get(frame_id, self._timestamps.get(frame_id - 1)))
        rec = process_frame(raw_gray, frame_id, frame_name, capture_ts_ns=capture_ts_ns, **self.detector_kwargs)
        self.frames.append(rec)
        return rec

    def run(self, frame_iter, target_hz: Optional[float] = None,
            progress_cb: Optional[Callable[[FrameDetections], None]] = None,
            camera_name: Optional[str] = None, data_dir: Optional[str] = None,
            capture_log_path: Optional[str] = None) -> List[FrameDetections]:
        """Drains frame_iter to completion. target_hz, if given, sleeps
        between frames so the loop doesn't run faster than that rate -- not
        needed for correctness on a batch replay, but keeps a future live-
        camera-paced throttle a one-line flip rather than a redesign."""
        if data_dir is not None:
            self.data_dir = data_dir
        if camera_name is not None:
            self.camera_name = camera_name
        if capture_log_path is not None:
            self.capture_log_path = capture_log_path
        if camera_name is not None or data_dir is not None or capture_log_path is not None:
            self._timestamps = load_capture_timestamps(self.capture_log_path, self.camera_name, self.data_dir)

        period = (1.0 / target_hz) if target_hz else None
        for frame_id, frame_name, raw_gray in frame_iter:
            t0 = time.monotonic()
            rec = self.ingest(raw_gray, frame_id, frame_name)
            if progress_cb is not None:
                progress_cb(rec)
            if period is not None:
                elapsed = time.monotonic() - t0
                if elapsed < period:
                    time.sleep(period - elapsed)
        return self.frames

    def run_live(self, frame_slot: "LatestFrameSlot", stop_event: Optional[threading.Event] = None,
                 progress_cb: Optional[Callable[[FrameDetections], None]] = None,
                 poll_timeout: float = 0.5) -> None:
        """Consumer loop for a push-based live source (see LatestFrameSlot):
        repeatedly pulls the LATEST available frame -- dropping any backlog
        the producer built up in the meantime -- and processes it, until
        stop_event is set or frame_slot is closed. Unlike run(), this never
        falls behind: if detection is slower than the arrival rate, stale
        frames are dropped rather than queued, so processing latency stays
        bounded instead of growing without limit."""
        while stop_event is None or not stop_event.is_set():
            item = frame_slot.get(timeout=poll_timeout)
            if item is None:
                if frame_slot.closed:
                    break
                continue  # poll timeout with nothing pending -- check stop_event and retry
            frame_id, frame_name, raw_gray = item
            rec = self.ingest(raw_gray, frame_id, frame_name)
            if progress_cb is not None:
                progress_cb(rec)

    def to_json_dict(self, data_dir: Optional[str] = None, camera_name: Optional[str] = None) -> dict:
        d_dir = data_dir or getattr(self, "data_dir", DATA_DIR)
        cam_name = (camera_name or getattr(self, "camera_name", None)
                    or (detect_camera_name_from_path(d_dir) if d_dir else None) or "hik_DA0179014")
        scale = self.detector_kwargs.get("scale", SCALE)
        if abs(scale - 1.0) < 1e-9:
            resolution_mode = "full_resolution_1x"
        elif abs(scale - 0.5) < 1e-9:
            resolution_mode = "downsampled_0.5x"
        else:
            resolution_mode = f"custom_{scale}x"
        return {
            "meta": {
                "source": "live_backprojection.live_detector",
                "camera_name": cam_name,
                "data_dir": d_dir,
                "detection_resolution_mode": resolution_mode,
                "detector": dict(self.detector_kwargs),
            },
            "frames": [
                {
                    "frame_id": rec.frame_id,
                    "frame_name": rec.frame_name,
                    "timestamp": rec.timestamp,
                    "capture_ts_ns": rec.capture_ts_ns,
                    "detections": [asdict(d) for d in rec.detections],
                }
                for rec in self.frames
            ],
        }

    def save(self, path: str, data_dir: Optional[str] = None, camera_name: Optional[str] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json_dict(data_dir=data_dir, camera_name=camera_name), f, indent=2)

    def detection_csv_rows(self, data_dir: Optional[str] = None,
                           camera_name: Optional[str] = None) -> List[dict]:
        """Flatten this run into one CSV row per pixel detection.

        The camera name is repeated on every row so one shared ``detect.csv``
        can hold results from multiple cameras. Empty frames intentionally
        produce no rows; the frame-level JSON representation remains available
        to the projection stage through :meth:`to_json_dict`.
        """
        d = self.to_json_dict(data_dir=data_dir, camera_name=camera_name)
        cam_name = str(d["meta"]["camera_name"])
        rows = []
        for frame in d["frames"]:
            for detection in frame["detections"]:
                rows.append({
                    "camera_name": cam_name,
                    "frame_id": int(frame["frame_id"]),
                    "frame_name": frame["frame_name"],
                    "timestamp": frame.get("timestamp"),
                    "capture_ts_ns": frame.get("capture_ts_ns"),
                    "u": float(detection["u"]),
                    "v": float(detection["v"]),
                    "size": float(detection["size"]),
                })
        return rows


def merge_detection_csv(path: str, rows: List[dict], camera_name: str) -> None:
    """Replace one camera's rows in *path* and preserve all other cameras.

    A run is authoritative for its camera: re-running an existing camera
    removes its prior rows, including when the new run found no detections.
    Rows for cameras not involved in this run are retained unchanged. The
    replacement is written atomically beside the destination file.
    """
    destination = os.path.abspath(path)
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    camera_name = str(camera_name)
    existing = []
    if os.path.isfile(destination):
        with open(destination, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "camera_name" not in reader.fieldnames:
                raise ValueError(f"existing detection CSV lacks camera_name: {destination}")
            existing = [dict(row) for row in reader]
    preserved = [row for row in existing if str(row.get("camera_name", "")) != camera_name]
    merged = preserved + [{field: row.get(field) for field in DETECTION_CSV_FIELDS} for row in rows]

    fd, temporary = tempfile.mkstemp(prefix=".detect-", suffix=".csv", dir=parent, text=True)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DETECTION_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(merged)
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def save_detection_csv(run: LiveDetectionRun, path: str,
                       data_dir: Optional[str] = None,
                       camera_name: Optional[str] = None) -> None:
    """Merge a completed run into the shared camera-keyed detection CSV."""
    cam_name = str(camera_name or run.camera_name)
    merge_detection_csv(path, run.detection_csv_rows(data_dir=data_dir, camera_name=cam_name), cam_name)


def load_detection_csv(path: str, camera_name: Optional[str] = None):
    """Load the shared detection CSV into the projection API's frame shape.

    With no camera filter, returns ``{camera_name: detector_output}``. With a
    filter, returns one detector-output dictionary (or an empty one when that
    camera has no rows). Frames that produced no detections cannot appear in a
    flat CSV and are therefore absent from the reconstructed sequence.
    """
    grouped: Dict[str, Dict[int, dict]] = {}
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"camera_name", "frame_id", "frame_name", "u", "v", "size"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"detection CSV missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            cam = str(row["camera_name"]).strip()
            if camera_name is not None and cam != str(camera_name):
                continue
            frame_id = int(row["frame_id"])
            frames = grouped.setdefault(cam, {})
            frame = frames.setdefault(frame_id, {
                "frame_id": frame_id,
                "frame_name": row["frame_name"],
                "timestamp": float(row["timestamp"]) if row.get("timestamp") else None,
                "capture_ts_ns": int(row["capture_ts_ns"]) if row.get("capture_ts_ns") else None,
                "detections": [],
            })
            frame["detections"].append({
                "u": float(row["u"]), "v": float(row["v"]), "size": float(row["size"]),
            })

    outputs = {}
    for cam, frames_by_id in grouped.items():
        outputs[cam] = {
            "meta": {"source": "live_backprojection.detect_csv", "camera_name": cam},
            "frames": [frames_by_id[key] for key in sorted(frames_by_id)],
        }
    if camera_name is not None:
        return outputs.get(str(camera_name), {
            "meta": {"source": "live_backprojection.detect_csv", "camera_name": str(camera_name)},
            "frames": [],
        })
    return outputs
