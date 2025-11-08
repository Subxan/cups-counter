"""Hailo inference wrapper with mock mode support."""

import logging
import random
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Detection format: (x1, y1, x2, y2, conf, cls_id)
Detection = Tuple[float, float, float, float, float, int]


class HailoInfer:
    """Hailo inference engine with mock mode."""

    def __init__(self, config):
        self.config = config
        self.mock_mode = config.model.mock_mode
        self.hef_path = Path(config.model.hef_path)
        self.device = None
        self.network_group = None
        self.input_vstreams = None
        self.output_vstreams = None
        self.model_height = 640  # Default YOLO input size
        self.model_width = 640

        # Mock mode state
        self._mock_tracks = []
        self._mock_frame_count = 0

        if not self.mock_mode:
            self._init_hailo()
        else:
            logger.info("Hailo inference in MOCK MODE - generating synthetic detections")

    def _init_hailo(self):
        """Initialize Hailo device and load model."""
        try:
            from hailo_platform import Device, HEF

            if not self.hef_path.exists():
                raise FileNotFoundError(f"HEF model not found: {self.hef_path}")

            # Initialize device
            self.device = Device()
            logger.info("Hailo device initialized")

            # Load HEF
            hef = HEF(str(self.hef_path))
            network_group = self.device.configure(hef)
            network_group_params = network_group.create_params()

            # Get input/output vstreams
            self.input_vstreams = network_group.get_input_vstreams()
            self.output_vstreams = network_group.get_output_vstreams()

            # Activate network group
            network_group.activate(network_group_params)
            self.network_group = network_group

            # Try to infer model input size from HEF metadata
            # This is a placeholder - actual implementation depends on Hailo API
            logger.info(f"Hailo model loaded: {self.hef_path}")

        except ImportError:
            logger.error(
                "hailo_platform not available. Install HailoRT or use --mock mode."
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Hailo: {e}")
            raise

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Preprocess frame: letterbox resize to model input size.

        Returns (preprocessed_frame, scale_x, scale_y) for coordinate mapping.
        """
        h, w = frame.shape[:2]
        target_h, target_w = self.model_height, self.model_width

        # Calculate scaling to fit while maintaining aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create letterbox (pad to target size)
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left

        letterboxed = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Convert BGR to RGB if needed (depends on model)
        # For now, assume model expects RGB
        letterboxed_rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] or model-specific range
        # Placeholder - adjust based on model requirements
        normalized = letterboxed_rgb.astype(np.float32) / 255.0

        # Calculate scale factors for coordinate mapping
        scale_x = w / new_w
        scale_y = h / new_h
        offset_x = -left
        offset_y = -top

        return normalized, scale_x, scale_y, offset_x, offset_y

    def _generate_mock_detections(self, frame: np.ndarray) -> List[Detection]:
        """Generate synthetic cup detections for mock mode."""
        self._mock_frame_count += 1
        h, w = frame.shape[:2]

        detections = []

        # Create 1-3 synthetic cups that move across frame
        if len(self._mock_tracks) == 0 or random.random() < 0.02:
            # Spawn new cup
            x = random.randint(50, w - 50)
            y = random.randint(100, h - 100)
            vx = random.uniform(2, 5)  # pixels per frame
            vy = random.uniform(-1, 1)
            self._mock_tracks.append({"x": x, "y": y, "vx": vx, "vy": vy, "age": 0})

        # Update existing tracks
        for track in self._mock_tracks[:]:
            track["x"] += track["vx"]
            track["y"] += track["vy"]
            track["age"] += 1

            # Remove if off-screen or too old
            if track["x"] < -50 or track["x"] > w + 50 or track["age"] > 300:
                self._mock_tracks.remove(track)
                continue

            # Generate detection
            box_size = random.randint(40, 80)
            x1 = max(0, int(track["x"] - box_size / 2))
            y1 = max(0, int(track["y"] - box_size / 2))
            x2 = min(w, int(track["x"] + box_size / 2))
            y2 = min(h, int(track["y"] + box_size / 2))

            conf = random.uniform(0.5, 0.95)
            cls_id = 41  # Placeholder class ID for "cup"

            detections.append((float(x1), float(y1), float(x2), float(y2), conf, cls_id))

        return detections

    def infer(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on frame and return detections.

        Returns list of (x1, y1, x2, y2, conf, cls_id) in image coordinates.
        """
        if self.mock_mode:
            return self._generate_mock_detections(frame)

        try:
            # Preprocess
            preprocessed, scale_x, scale_y, offset_x, offset_y = self._preprocess(frame)

            # Prepare input tensor
            # Shape depends on model - typically (1, 3, H, W) or (1, H, W, 3)
            input_data = np.expand_dims(preprocessed, axis=0)
            input_data = np.transpose(input_data, (0, 3, 1, 2))  # NHWC -> NCHW

            # Run inference
            input_dict = {vstream.name: input_data for vstream in self.input_vstreams}
            output_dict = self.network_group.run(input_dict)

            # Parse outputs (this is model-specific)
            # YOLO typically outputs boxes in format [x, y, w, h, conf, cls_scores...]
            # This is a placeholder - actual parsing depends on model architecture
            raw_output = list(output_dict.values())[0]

            # Post-process raw output to get detections
            # This would typically involve:
            # 1. Decode boxes from model format
            # 2. Apply confidence threshold
            # 3. Scale boxes back to image coordinates using scale_x, scale_y, offset_x, offset_y
            # 4. Filter by class

            # For now, return empty list (actual implementation needed)
            logger.warning("Hailo inference placeholder - implement model-specific parsing")
            return []

        except Exception as e:
            logger.error(f"Hailo inference error: {e}")
            return []

    def warmup(self, num_frames: int = 5):
        """Run warmup inferences."""
        if self.mock_mode:
            logger.info("Skipping warmup in mock mode")
            return

        logger.info(f"Warming up Hailo with {num_frames} frames...")
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for _ in range(num_frames):
            self.infer(dummy_frame)
        logger.info("Warmup complete")

    def close(self):
        """Release Hailo resources."""
        if self.mock_mode:
            return

        try:
            if self.network_group:
                self.network_group.deactivate()
            if self.device:
                self.device.release()
            logger.info("Hailo device released")
        except Exception as e:
            logger.error(f"Error closing Hailo: {e}")

