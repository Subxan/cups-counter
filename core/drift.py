"""Drift monitoring: SSIM, edge IoU, brightness variance."""

import logging
from collections import deque
from typing import Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


class DriftMonitor:
    """Monitor scene drift: camera shift, lighting changes."""

    def __init__(self, config):
        self.config = config
        self.enabled = config.drift.enabled

        if not self.enabled:
            return

        # Reference frame (set on first frame or after calibration)
        self.reference_frame = None
        self.reference_edges = None
        self.reference_brightness = None

        # Rolling metrics
        self.ssim_history = deque(maxlen=30)  # Last 30 frames
        self.edge_iou_history = deque(maxlen=30)
        self.brightness_history = deque(maxlen=30)

        # State
        self.last_recal_time = None
        self.drift_detected = False

    def set_reference(self, frame: np.ndarray):
        """Set reference frame for drift detection."""
        if not self.enabled:
            return

        self.reference_frame = frame.copy()
        self.reference_edges = self._compute_edges(frame)
        self.reference_brightness = self._compute_brightness(frame)
        logger.info("Drift monitor reference frame set")

    def _compute_edges(self, frame: np.ndarray) -> np.ndarray:
        """Compute Canny edges."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges

    def _compute_brightness(self, frame: np.ndarray) -> float:
        """Compute brightness variance."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.var(gray))

    def _compute_edge_iou(self, edges1: np.ndarray, edges2: np.ndarray) -> float:
        """Compute IoU between two edge maps."""
        intersection = np.logical_and(edges1 > 0, edges2 > 0).sum()
        union = np.logical_or(edges1 > 0, edges2 > 0).sum()
        return intersection / union if union > 0 else 0.0

    def update(self, frame: np.ndarray) -> dict:
        """Update drift metrics with new frame.

        Returns dict with metrics and flags.
        """
        if not self.enabled or self.reference_frame is None:
            return {
                "ssim": 1.0,
                "edge_iou": 1.0,
                "brightness_var": 0.0,
                "camera_shifted": False,
                "lighting_bad": False,
                "drift_score": 0.0,
            }

        # Compute SSIM
        gray_ref = cv2.cvtColor(self.reference_frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ssim_val = ssim(gray_ref, gray_curr, data_range=255)
        self.ssim_history.append(ssim_val)

        # Compute edge IoU
        curr_edges = self._compute_edges(frame)
        edge_iou = self._compute_edge_iou(self.reference_edges, curr_edges)
        self.edge_iou_history.append(edge_iou)

        # Compute brightness
        brightness_var = self._compute_brightness(frame)
        brightness_diff = abs(brightness_var - self.reference_brightness)
        self.brightness_history.append(brightness_var)

        # Check thresholds
        camera_shifted = (
            ssim_val < self.config.drift.ssim_threshold
            or edge_iou < self.config.drift.edge_iou_threshold
        )
        lighting_bad = brightness_var < self.config.drift.brightness_var_min

        # Drift score (0-1, higher = more drift)
        drift_score = (
            (1.0 - ssim_val) * 0.4
            + (1.0 - edge_iou) * 0.4
            + min(brightness_diff / 50.0, 1.0) * 0.2
        )

        self.drift_detected = camera_shifted or lighting_bad

        return {
            "ssim": float(ssim_val),
            "edge_iou": float(edge_iou),
            "brightness_var": float(brightness_var),
            "camera_shifted": camera_shifted,
            "lighting_bad": lighting_bad,
            "drift_score": float(drift_score),
        }

    def should_recalibrate(self) -> bool:
        """Check if recalibration should be triggered."""
        if not self.enabled or not self.config.drift.re_calibrate_on_drift:
            return False

        if not self.drift_detected:
            return False

        # Check cooldown
        if self.last_recal_time is not None:
            import time

            minutes_since = (time.time() - self.last_recal_time) / 60.0
            if minutes_since < self.config.drift.min_minutes_between_recal:
                return False

        return True

    def mark_recalibrated(self):
        """Mark that recalibration was performed."""
        import time

        self.last_recal_time = time.time()
        self.drift_detected = False
        logger.info("Drift monitor: recalibration marked")

    def get_metrics(self) -> dict:
        """Get current drift metrics."""
        if not self.enabled:
            return {}

        avg_ssim = np.mean(self.ssim_history) if self.ssim_history else 1.0
        avg_edge_iou = np.mean(self.edge_iou_history) if self.edge_iou_history else 1.0
        avg_brightness = np.mean(self.brightness_history) if self.brightness_history else 0.0

        return {
            "ssim_avg": float(avg_ssim),
            "edge_iou_avg": float(avg_edge_iou),
            "brightness_avg": float(avg_brightness),
            "camera_shifted": self.camera_shifted(),
            "lighting_bad": self.lighting_bad(),
            "last_recal_time": self.last_recal_time,
        }

    def camera_shifted(self) -> bool:
        """Check if camera appears shifted."""
        if not self.enabled or not self.ssim_history:
            return False

        recent_ssim = list(self.ssim_history)[-10:]  # Last 10 frames
        return np.mean(recent_ssim) < self.config.drift.ssim_threshold

    def lighting_bad(self) -> bool:
        """Check if lighting is problematic."""
        if not self.enabled or not self.brightness_history:
            return False

        recent_brightness = list(self.brightness_history)[-10:]
        return np.mean(recent_brightness) < self.config.drift.brightness_var_min

