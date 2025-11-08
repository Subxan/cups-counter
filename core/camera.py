"""Camera abstraction for multiple sources (Picamera2, OpenCV, GStreamer)."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Generator, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameSource(ABC):
    """Abstract frame source."""

    def __init__(self, width: int, height: int, fps: int, rotate: int = 0):
        self.width = width
        self.height = height
        self.fps = fps
        self.rotate = rotate
        self._frame_count = 0
        self._start_time = time.time()
        self._last_frame_time = None

    @abstractmethod
    def frames(self) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Yield (frame_bgr, timestamp_ns) tuples."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release resources."""
        pass

    def fps_estimate(self) -> float:
        """Estimate current FPS."""
        if self._frame_count == 0:
            return 0.0
        elapsed = time.time() - self._start_time
        return self._frame_count / elapsed if elapsed > 0 else 0.0

    def _rotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Rotate frame if needed."""
        if self.rotate == 0:
            return frame
        elif self.rotate == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotate == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame


class Picamera2Source(FrameSource):
    """Picamera2 frame source."""

    def __init__(self, width: int, height: int, fps: int, rotate: int = 0):
        super().__init__(width, height, fps, rotate)
        try:
            from picamera2 import Picamera2

            self.camera = Picamera2()
            config = self.camera.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"},
                controls={"FrameRate": fps},
            )
            self.camera.configure(config)
            self.camera.start()
            logger.info(f"Picamera2 initialized: {width}x{height} @ {fps}fps")
        except ImportError:
            raise ImportError("picamera2 not installed. Install with: pip install picamera2")
        except Exception as e:
            logger.error(f"Failed to initialize Picamera2: {e}")
            raise

    def frames(self) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Yield frames from Picamera2."""
        frame_time_ns = time.time_ns()
        while True:
            try:
                array = self.camera.capture_array()
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                frame_bgr = self._rotate_frame(frame_bgr)
                self._frame_count += 1
                self._last_frame_time = time.time_ns()
                yield (frame_bgr, self._last_frame_time)
                # Throttle to target FPS
                time.sleep(1.0 / self.fps)
            except Exception as e:
                logger.error(f"Picamera2 frame capture error: {e}")
                time.sleep(0.1)

    def close(self) -> None:
        """Stop camera."""
        try:
            self.camera.stop()
            self.camera.close()
        except Exception as e:
            logger.error(f"Error closing Picamera2: {e}")


class OpenCVSource(FrameSource):
    """OpenCV (USB) camera source."""

    def __init__(self, index: int, width: int, height: int, fps: int, rotate: int = 0):
        super().__init__(width, height, fps, rotate)
        self.index = index
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        logger.info(f"OpenCV camera {index} initialized: {width}x{height} @ {fps}fps")

    def frames(self) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Yield frames from OpenCV camera."""
        frame_interval = 1.0 / self.fps
        last_frame_time = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.1)
                continue

            frame = self._rotate_frame(frame)
            timestamp_ns = time.time_ns()
            self._frame_count += 1
            self._last_frame_time = timestamp_ns

            # Throttle to target FPS
            elapsed = time.time() - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_frame_time = time.time()

            yield (frame, timestamp_ns)

    def close(self) -> None:
        """Release camera."""
        self.cap.release()


class GStreamerSource(FrameSource):
    """GStreamer camera source (placeholder - can be extended)."""

    def __init__(self, pipeline: str, width: int, height: int, fps: int, rotate: int = 0):
        super().__init__(width, height, fps, rotate)
        self.pipeline_str = pipeline
        # For now, fall back to OpenCV with GStreamer backend
        logger.warning("GStreamer source using OpenCV backend")
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open GStreamer pipeline: {pipeline}")

    def frames(self) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Yield frames from GStreamer."""
        frame_interval = 1.0 / self.fps
        last_frame_time = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame from GStreamer")
                time.sleep(0.1)
                continue

            frame = self._rotate_frame(frame)
            timestamp_ns = time.time_ns()
            self._frame_count += 1
            self._last_frame_time = timestamp_ns

            elapsed = time.time() - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_frame_time = time.time()

            yield (frame, timestamp_ns)

    def close(self) -> None:
        """Release camera."""
        self.cap.release()


def create_frame_source(config) -> FrameSource:
    """Factory function to create appropriate frame source."""
    source_type = config.camera.source
    width = config.camera.width
    height = config.camera.height
    fps = config.camera.fps
    rotate = config.camera.rotate

    if source_type == "picamera2":
        return Picamera2Source(width, height, fps, rotate)
    elif source_type == "opencv":
        return OpenCVSource(config.camera.index, width, height, fps, rotate)
    elif source_type == "gstreamer":
        # Default pipeline - can be customized
        pipeline = (
            f"v4l2src device=/dev/video{config.camera.index} ! "
            f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
            "videoconvert ! appsink"
        )
        return GStreamerSource(pipeline, width, height, fps, rotate)
    else:
        raise ValueError(f"Unknown camera source: {source_type}")

