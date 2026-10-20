# Tracking target backprojection

Recorded detection, 3D mesh projection, clustering, and PyQt/web viewers.

## Setup

Use Python 3.11 or newer, install the dependencies, then launch:

```bash
python -m pip install -r requirements.txt
python gui_qt.py
```

The repository includes the `single_frame_backprojection` and `calib_target` packages required by the GUI and projection workflow. Runtime data, meshes, and generated viewers belong in local `data/` and `out/` directories and are intentionally excluded from Git.
