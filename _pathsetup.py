"""Shared sys.path bootstrap so everything in this package (including the
live_backprojection.detector subpackage) can import the pipeline/
calib_target packages -- pipeline/ has its own equivalent, pipeline/
_pathsetup.py, for the same reason.

Import this first, before any pipeline/calib_target/evaluate_* import:
    from live_backprojection import _pathsetup  # noqa: F401
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
