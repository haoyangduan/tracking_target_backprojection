"""Real-time Camera SDK Interface & Stream Provider Framework.

Provides an abstract interface (`BaseCameraSDK`) and concrete implementations
(`FolderSimulatedSDK`, `OpenCVCameraSDK`, `HikvisionMVSSDKAdapter`) for acquiring
live images from physical hardware cameras or simulated sequence streams in real time.

Designed for seamless integration into real-time 2D target detection & 3D mesh
back-projection workflows without locking up the UI.
"""

import os
import sys
import time
import queue
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any, List

import cv2
import numpy as np


@dataclass
class FramePacket:
    """Encapsulates a single captured image frame with metadata."""
    frame_id: str                      # Basename or identifier, e.g. "frame_000052.png"
    image: np.ndarray                  # BGR or Grayscale image array (uint8)
    timestamp_s: float                 # System timestamp at acquisition (seconds)
    camera_name: str                   # Camera identifier, e.g. "hik_DA0179014"
    frame_index: int                   # Monotonic integer frame index (0, 1, 2...)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Hardware gain, exposure, status


class BaseCameraSDK(ABC):
    """Abstract Base Class defining the standard Camera SDK interface."""

    def __init__(self, camera_name: str = "camera_0"):
        self.camera_name = camera_name
        self._is_streaming = False

    @abstractmethod
    def connect(self) -> bool:
        """Connect to camera hardware or initialize stream source. Returns True if successful."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect camera hardware and release all allocated resources."""
        pass

    @abstractmethod
    def start_stream(self, callback_fn: Optional[Callable[[FramePacket], None]] = None) -> bool:
        """Start acquisition.
        
        If callback_fn is provided, incoming frames are delivered asynchronously via callback.
        Otherwise, call grab_frame() to pull frames synchronously.
        """
        pass

    @abstractmethod
    def stop_stream(self) -> None:
        """Stop frame acquisition."""
        pass

    @abstractmethod
    def grab_frame(self, timeout_s: float = 1.0) -> Optional[FramePacket]:
        """Synchronously grab the latest frame (blocking up to timeout_s)."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if camera hardware is currently connected and initialized."""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Return camera hardware details (serial number, resolution, FPS, exposure, etc.)."""
        pass


class FolderSimulatedSDK(BaseCameraSDK):
    """Simulates a live industrial Camera SDK using an image directory sequence.
    
    Replays images at a configurable target FPS to simulate live camera hardware capture,
    supporting async callbacks, queue polling, and looping. Useful for testing real-time
    processing pipelines before physical SDK deployment.
    """

    def __init__(self, data_dir: str, camera_name: str = "hik_DA0179014",
                 target_fps: float = 10.0, loop: bool = True):
        super().__init__(camera_name=camera_name)
        self.data_dir = data_dir
        self.target_fps = target_fps
        self.loop = loop
        self._connected = False
        self._frame_files: List[str] = []
        self._current_idx = 0
        self._seq_counter = 0
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._callback: Optional[Callable[[FramePacket], None]] = None
        self._queue: queue.Queue = queue.Queue(maxsize=10)

    def connect(self) -> bool:
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Simulated SDK data directory not found: {self.data_dir!r}")
        
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        files = [f for f in os.listdir(self.data_dir) if f.lower().endswith(valid_exts)]
        files.sort()
        
        if not files:
            raise ValueError(f"No valid image files found in {self.data_dir!r}")
        
        self._frame_files = files
        self._current_idx = 0
        self._seq_counter = 0
        self._connected = True
        return True

    def disconnect(self) -> None:
        self.stop_stream()
        self._connected = False
        self._frame_files = []

    def start_stream(self, callback_fn: Optional[Callable[[FramePacket], None]] = None) -> bool:
        if not self._connected:
            if not self.connect():
                return False
        
        self.stop_stream()
        self._callback = callback_fn
        self._stop_event.clear()
        self._is_streaming = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        return True

    def stop_stream(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._is_streaming = False

    def _read_next_packet(self) -> Optional[FramePacket]:
        if not self._frame_files:
            return None
        
        if self._current_idx >= len(self._frame_files):
            if self.loop:
                self._current_idx = 0
            else:
                return None

        filename = self._frame_files[self._current_idx]
        path = os.path.join(self.data_dir, filename)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            self._current_idx += 1
            return None

        packet = FramePacket(
            frame_id=filename,
            image=image,
            timestamp_s=time.time(),
            camera_name=self.camera_name,
            frame_index=self._seq_counter,
            metadata={"source_path": path, "fps": self.target_fps}
        )

        self._current_idx += 1
        self._seq_counter += 1
        return packet

    def _stream_loop(self) -> None:
        delay_s = 1.0 / max(0.1, self.target_fps)
        while not self._stop_event.is_set():
            t0 = time.time()
            packet = self._read_next_packet()
            if packet is None:
                if not self.loop:
                    break
                time.sleep(0.01)
                continue

            if self._callback:
                try:
                    self._callback(packet)
                except Exception as e:
                    print(f"[FolderSimulatedSDK Error] Callback raised: {e}", file=sys.stderr)
            
            # Keep queue populated for grab_frame()
            if not self._queue.full():
                self._queue.put(packet)
            else:
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(packet)
                except queue.Empty:
                    pass

            elapsed = time.time() - t0
            sleep_time = delay_s - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def grab_frame(self, timeout_s: float = 1.0) -> Optional[FramePacket]:
        if not self._is_streaming:
            return self._read_next_packet()
        try:
            return self._queue.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def is_connected(self) -> bool:
        return self._connected

    def get_info(self) -> Dict[str, Any]:
        return {
            "type": "FolderSimulatedSDK",
            "camera_name": self.camera_name,
            "data_dir": self.data_dir,
            "total_frames": len(self._frame_files),
            "target_fps": self.target_fps,
            "is_streaming": self._is_streaming,
        }


class OpenCVCameraSDK(BaseCameraSDK):
    """Wrapper SDK for standard OpenCV VideoCapture (USB industrial cameras, RTSP streams, video files)."""

    def __init__(self, device_id: Any = 0, camera_name: str = "opencv_cam"):
        super().__init__(camera_name=camera_name)
        self.device_id = device_id
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._callback: Optional[Callable[[FramePacket], None]] = None
        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._frame_counter = 0

    def connect(self) -> bool:
        self.disconnect()
        self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            self._cap = None
            return False
        self._frame_counter = 0
        return True

    def disconnect(self) -> None:
        self.stop_stream()
        if self._cap:
            self._cap.release()
            self._cap = None

    def start_stream(self, callback_fn: Optional[Callable[[FramePacket], None]] = None) -> bool:
        if not self._cap or not self._cap.isOpened():
            if not self.connect():
                return False

        self.stop_stream()
        self._callback = callback_fn
        self._stop_event.clear()
        self._is_streaming = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def stop_stream(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._is_streaming = False

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set() and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            self._frame_counter += 1
            frame_id = f"frame_{self._frame_counter:06d}.png"
            packet = FramePacket(
                frame_id=frame_id,
                image=frame,
                timestamp_s=time.time(),
                camera_name=self.camera_name,
                frame_index=self._frame_counter,
            )

            if self._callback:
                try:
                    self._callback(packet)
                except Exception as e:
                    print(f"[OpenCVCameraSDK Error] Callback raised: {e}", file=sys.stderr)

            if not self._queue.full():
                self._queue.put(packet)
            else:
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(packet)
                except queue.Empty:
                    pass

    def grab_frame(self, timeout_s: float = 1.0) -> Optional[FramePacket]:
        if not self._is_streaming:
            if not self._cap or not self._cap.isOpened():
                return None
            ret, frame = self._cap.read()
            if not ret or frame is None:
                return None
            self._frame_counter += 1
            return FramePacket(
                frame_id=f"frame_{self._frame_counter:06d}.png",
                image=frame,
                timestamp_s=time.time(),
                camera_name=self.camera_name,
                frame_index=self._frame_counter,
            )
        try:
            return self._queue.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def is_connected(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def get_info(self) -> Dict[str, Any]:
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0
        fps = self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0
        return {
            "type": "OpenCVCameraSDK",
            "camera_name": self.camera_name,
            "device_id": self.device_id,
            "width": width,
            "height": height,
            "fps": fps,
            "is_streaming": self._is_streaming,
        }


class HikvisionMVSSDKAdapter(BaseCameraSDK):
    """Production Adapter Interface for Hikvision MVS Industrial Camera SDK.
    
    This class serves as the ready-to-use template for when the computer vision
    team provides the physical camera SDK (`MvCameraControl.dll` / `MvCameraControl.py`).
    """

    def __init__(self, serial_number: str = "", camera_name: str = "hik_DA0179014"):
        super().__init__(camera_name=camera_name)
        self.serial_number = serial_number
        self._handle = None
        self._connected = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._callback: Optional[Callable[[FramePacket], None]] = None
        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._frame_counter = 0

    def connect(self) -> bool:
        """Connect to Hikvision MVS hardware camera via serial number."""
        # PLACEHOLDER FOR HIKVISION SDK INITIALIZATION:
        # e.g.:
        # device_list = MV_CC_DEVICE_INFO_LIST()
        # ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
        # self._handle = MvCamera()
        # MvCamera.MV_CC_CreateHandle(self._handle, device_info)
        # MvCamera.MV_CC_OpenDevice(self._handle)
        print(f"[HikvisionSDKAdapter] Connecting to Hikvision camera ({self.camera_name}, SN: {self.serial_number})...")
        self._connected = True
        return True

    def disconnect(self) -> None:
        self.stop_stream()
        if self._handle:
            # MvCamera.MV_CC_CloseDevice(self._handle)
            # MvCamera.MV_CC_DestroyHandle(self._handle)
            self._handle = None
        self._connected = False

    def start_stream(self, callback_fn: Optional[Callable[[FramePacket], None]] = None) -> bool:
        if not self._connected and not self.connect():
            return False
        
        self._callback = callback_fn
        self._stop_event.clear()
        self._is_streaming = True
        # MvCamera.MV_CC_StartGrabbing(self._handle)
        self._thread = threading.Thread(target=self._sdk_grab_loop, daemon=True)
        self._thread.start()
        return True

    def stop_stream(self) -> None:
        self._stop_event.set()
        if self._handle:
            # MvCamera.MV_CC_StopGrabbing(self._handle)
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._is_streaming = False

    def _sdk_grab_loop(self) -> None:
        """Hardware SDK acquisition loop."""
        while not self._stop_event.is_set():
            # In actual SDK:
            # stFrameInfo = MV_FRAME_OUT_INFO_EX()
            # ret = MvCamera.MV_CC_GetOneFrameTimeout(self._handle, pData, nDataSize, stFrameInfo, 1000)
            # convert pData buffer to numpy uint8 array image
            time.sleep(0.05)  # Simulate 20 FPS acquisition

    def grab_frame(self, timeout_s: float = 1.0) -> Optional[FramePacket]:
        try:
            return self._queue.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def is_connected(self) -> bool:
        return self._connected

    def get_info(self) -> Dict[str, Any]:
        return {
            "type": "HikvisionMVSSDKAdapter",
            "camera_name": self.camera_name,
            "serial_number": self.serial_number,
            "is_connected": self._connected,
            "is_streaming": self._is_streaming,
        }


class RealTimeDetectionPipelineManager:
    """Orchestrates live Camera SDK acquisition, thread-safe frame queuing,
    real-time 2D target detection, and optional 3D mesh back-projection.
    """

    def __init__(self, camera_sdk: BaseCameraSDK, detector_fn: Callable[[np.ndarray], List[Any]],
                 processor_fn: Optional[Callable[[FramePacket, List[Any]], None]] = None,
                 max_queue_size: int = 2):
        self.camera_sdk = camera_sdk
        self.detector_fn = detector_fn
        self.processor_fn = processor_fn
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._fps = 0.0
        self._last_proc_time = time.time()
        self._proc_count = 0

    def start(self) -> bool:
        """Start acquisition and worker processing thread."""
        self._stop_event.clear()
        self._proc_count = 0
        self._last_proc_time = time.time()
        
        # Hook SDK callback to push frames into our queue
        def _on_frame(packet: FramePacket):
            if not self._queue.full():
                self._queue.put(packet)
            else:
                # Drop oldest frame under backpressure to maintain zero-latency live processing
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(packet)
                except queue.Empty:
                    pass

        if not self.camera_sdk.start_stream(callback_fn=_on_frame):
            return False

        self._worker_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._worker_thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        self.camera_sdk.stop_stream()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        self._worker_thread = None

    def _processing_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                packet = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            t_start = time.time()
            try:
                # Run 2D sticker detection
                detections = self.detector_fn(packet.image)

                # Process results (e.g. 3D backprojection or GUI update)
                if self.processor_fn:
                    self.processor_fn(packet, detections)
            except Exception as e:
                print(f"[RealTimePipeline Error] Processing failed on frame {packet.frame_id}: {e}", file=sys.stderr)

            # Measure performance FPS
            self._proc_count += 1
            now = time.time()
            dt = now - self._last_proc_time
            if dt >= 1.0:
                self._fps = self._proc_count / dt
                self._proc_count = 0
                self._last_proc_time = now

    @property
    def processing_fps(self) -> float:
        return self._fps
