"""Real-time Detection & 3D Back-Projection SDK Demonstration.

Runs real-time 2D sticker detection and performance metrics using the abstract
Camera SDK framework (`BaseCameraSDK` / `FolderSimulatedSDK` / `HikvisionMVSSDKAdapter`).

Simulates a live high-speed industrial camera stream from a local folder or hardware device,
demonstrating zero-latency async acquisition and detection.
"""

import os
import sys
import time
import argparse

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from live_backprojection.camera_sdk import (
    FolderSimulatedSDK,
    OpenCVCameraSDK,
    HikvisionMVSSDKAdapter,
    RealTimeDetectionPipelineManager,
    FramePacket,
)
from live_backprojection.live_detector import detect_half_ring, auto_gain

_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
_SAMPLE_DIR = os.path.join(_DATA_DIR, "hik_DA0179014")


def main():
    parser = argparse.ArgumentParser(description="Real-time Camera SDK Detection Demo")
    parser.add_argument("--source", choices=["folder", "opencv", "hikvision"], default="folder",
                        help="Camera SDK stream source (default: folder)")
    parser.add_argument("--data-dir", default=_SAMPLE_DIR,
                        help="Image directory for simulated folder SDK")
    parser.add_argument("--fps", type=float, default=15.0,
                        help="Simulated camera capture FPS (default: 15.0)")
    parser.add_argument("--device", default="0",
                        help="Device ID or camera index for OpenCV SDK (default: 0)")
    parser.add_argument("--camera-name", default="hik_DA0179014",
                        help="Camera identifier (default: hik_DA0179014)")
    parser.add_argument("--no-view", action="store_true",
                        help="Disable live OpenCV preview window")
    args = parser.parse_args()

    # 1. Instantiate selected Camera SDK Implementation
    if args.source == "folder":
        print(f"[SDK Demo] Initializing FolderSimulatedSDK from {args.data_dir!r} at {args.fps} FPS...")
        sdk = FolderSimulatedSDK(data_dir=args.data_dir, camera_name=args.camera_name,
                                 target_fps=args.fps, loop=True)
    elif args.source == "opencv":
        dev_id = int(args.device) if args.device.isdigit() else args.device
        print(f"[SDK Demo] Initializing OpenCVCameraSDK for device {dev_id!r}...")
        sdk = OpenCVCameraSDK(device_id=dev_id, camera_name=args.camera_name)
    elif args.source == "hikvision":
        print(f"[SDK Demo] Initializing HikvisionMVSSDKAdapter template...")
        sdk = HikvisionMVSSDKAdapter(camera_name=args.camera_name)
    else:
        raise ValueError(f"Unknown camera source: {args.source}")

    if not sdk.connect():
        print(f"FAILED to connect Camera SDK: {sdk.get_info()}", file=sys.stderr)
        sys.exit(1)

    print(f"Connected to SDK: {sdk.get_info()}")

    # 2. Define real-time detector callback
    def detector_fn(image_bgr: np.ndarray):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        gained = cv2.convertScaleAbs(gray, alpha=auto_gain(gray), beta=0)
        # Run production 2D ring detector
        detections = detect_half_ring(gained, scale=0.5, block_size=61, c_offset=6.0, center_ring_min=25.0)
        return detections

    # 3. Define processor callback for visualization & metrics
    def processor_fn(packet: FramePacket, detections: list):
        h, w = packet.image.shape[:2]
        vis = packet.image.copy()

        # Draw detected 2D sticker blobs
        for det in detections:
            cx, cy = det[0], det[1]
            size = det[2] if len(det) > 2 else 32.0
            radius = max(4, int(round(size / 2.0)))
            cv2.circle(vis, (int(round(cx)), int(round(cy))), radius, (0, 255, 0), 2)
            cv2.drawMarker(vis, (int(round(cx)), int(round(cy))), (0, 255, 255),
                           cv2.MARKER_CROSS, 10, 1)

        info_str = f"Frame: {packet.frame_id} | Detections: {len(detections)} | Cam: {packet.camera_name}"
        fps_str = f"Live Processing FPS: {pipeline_manager.processing_fps:.1f}"
        
        cv2.putText(vis, info_str, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(vis, fps_str, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        print(f"[{packet.camera_name} RealTime] {packet.frame_id} (seq {packet.frame_index}): "
              f"{len(detections)} targets detected | Proc FPS: {pipeline_manager.processing_fps:.1f}")

        if not args.no_view:
            # Resize preview for desktop display
            disp_h = 720
            disp_w = int(w * (disp_h / h))
            preview = cv2.resize(vis, (disp_w, disp_h))
            cv2.imshow("Real-Time Camera SDK Target Detection Demo", preview)
            cv2.waitKey(1)

    # 4. Start Real-Time Pipeline Manager
    pipeline_manager = RealTimeDetectionPipelineManager(
        camera_sdk=sdk,
        detector_fn=detector_fn,
        processor_fn=processor_fn,
        max_queue_size=2,
    )

    print("\n" + "=" * 65)
    print("  REAL-TIME CAMERA SDK TARGET DETECTION RUNNING")
    print("  Press Ctrl+C in terminal (or Esc in window) to stop...")
    print("=" * 65 + "\n")

    if not pipeline_manager.start():
        print("FAILED to start RealTimeDetectionPipelineManager", file=sys.stderr)
        sys.exit(1)

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping Real-time Camera SDK Demo...")
    finally:
        pipeline_manager.stop()
        sdk.disconnect()
        if not args.no_view:
            cv2.destroyAllWindows()
        print("Real-time Camera SDK Demo stopped cleanly.")


if __name__ == "__main__":
    main()
