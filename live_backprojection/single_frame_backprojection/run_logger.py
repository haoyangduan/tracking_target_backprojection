"""
run_logger.py
=============
Shared logging utility, adapted from defect_visualization's
pipeline/run_logger.py (generic timestamped run folders; no Isaac-Sim
specific behaviour).

Creates a timestamped run directory under:
    <project_root>/logs/YY/MM/DD/HHMMSS_<suffix>/

Each run directory contains:
    - run_data.json   : all key numerical inputs and outputs for the run
    - <image_name>.*  : a copy of the selected input image (if provided)
"""

import json
import os
import shutil
from datetime import datetime


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
_LOGS_ROOT = os.path.join(_PROJECT_ROOT, "logs")


class RunLogger:
    def __init__(self, script_name: str = "unknown", logs_root: str = None, suffix: str = "backprojection"):
        self.script_name = script_name
        self._data: dict = {}
        self._image_src: str = None

        now = datetime.now()
        self._timestamp = now.strftime("%Y-%m-%d_%H%M%S")
        rel_path = os.path.join(
            now.strftime("%y"), now.strftime("%m"), now.strftime("%d"), now.strftime("%H%M%S"),
        )
        root = logs_root if logs_root else _LOGS_ROOT
        base_dir = os.path.join(root, rel_path)
        self.run_dir = f"{base_dir}_{suffix}" if suffix else base_dir
        os.makedirs(self.run_dir, exist_ok=True)

        self._data["_meta"] = {
            "script": script_name,
            "timestamp": now.isoformat(timespec="seconds"),
            "run_dir": self.run_dir,
        }
        print(f"[RunLogger] Run log folder: {self.run_dir}")

    def log(self, key: str, value):
        self._data[key] = _to_serialisable(value)

    def log_many(self, mapping: dict):
        for k, v in mapping.items():
            self.log(k, v)

    def set_image(self, image_path: str):
        if image_path and os.path.isfile(image_path):
            self._image_src = image_path
            self._data["input_image_original_path"] = image_path
        else:
            print(f"[RunLogger] Warning: image not found, skipping copy: {image_path}")

    def save(self) -> str:
        if self._image_src:
            img_name = os.path.basename(self._image_src)
            dst = os.path.join(self.run_dir, img_name)
            try:
                shutil.copy2(self._image_src, dst)
                self._data["input_image_log_copy"] = dst
            except Exception as exc:
                print(f"[RunLogger] Warning: could not copy image: {exc}")

        json_path = os.path.join(self.run_dir, "run_data.json")
        with open(json_path, "w") as f:
            json.dump(self._data, f, indent=2)
        print(f"[RunLogger] Run data saved to: {json_path}")
        return json_path

    @property
    def run_folder(self) -> str:
        return self.run_dir


def _to_serialisable(value):
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
    except ImportError:
        pass

    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {k: _to_serialisable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serialisable(v) for v in value]
    return value
