"""ROI (Region of Interest) masking around counting line."""

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ROIMasker:
    """ROI masking around the counting line."""

    def __init__(self, config):
        self.config = config
        self.enabled = config.roi.enabled
        self.band_height = config.roi.band_height_px
        self.margin = config.roi.margin_px

        self.line_start = None
        self.line_end = None
        self.mask = None
        self.roi_bbox = None

        # Skip masking if too many detections would be clipped
        self.clipped_detection_count = 0
        self.total_detection_count = 0
        self.skip_mask_frames = 0

    def set_line(self, line_start: Tuple[int, int], line_end: Tuple[int, int]):
        """Set counting line and compute ROI mask."""
        if not self.enabled:
            return

        self.line_start = line_start
        self.line_end = line_end

        # Compute line midpoint and angle
        mid_x = (line_start[0] + line_end[0]) // 2
        mid_y = (line_start[1] + line_end[1]) // 2

        # Create horizontal band around line
        half_height = self.band_height // 2
        y_top = max(0, mid_y - half_height - self.margin)
        y_bottom = mid_y + half_height + self.margin

        self.roi_bbox = (0, y_top, None, y_bottom)  # x1, y1, x2, y2 (x2 = full width)

        logger.info(f"ROI mask set: y={y_top}-{y_bottom}")

    def get_mask(self, frame_shape: Tuple[int, int, int]) -> np.ndarray | None:
        """Get ROI mask for frame.

        Returns mask (white=keep, black=clip) or None if disabled/skipped.
        """
        if not self.enabled or self.roi_bbox is None:
            return None

        if self.skip_mask_frames > 0:
            return None

        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        _, y_top, _, y_bottom = self.roi_bbox
        mask[y_top:y_bottom, :] = 255
        return mask

    def apply_mask(self, frame: np.ndarray) -> np.ndarray:
        """Apply ROI mask to frame (black out non-ROI regions)."""
        if not self.enabled:
            return frame

        mask = self.get_mask(frame.shape)
        if mask is None:
            return frame

        masked = frame.copy()
        masked[mask == 0] = 0
        return masked

    def crop_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Crop frame to ROI region.

        Returns (cropped_frame, offset) where offset is (x, y) of top-left corner.
        """
        if not self.enabled or self.roi_bbox is None:
            return frame, (0, 0)

        h, w = frame.shape[:2]
        _, y_top, _, y_bottom = self.roi_bbox
        y_top = max(0, y_top)
        y_bottom = min(h, y_bottom)

        cropped = frame[y_top:y_bottom, :]
        return cropped, (0, y_top)

    def check_detection_clipping(
        self, detections: list, frame_shape: Tuple[int, int, int]
    ) -> bool:
        """Check if too many detections would be clipped by ROI.

        Returns True if masking should be skipped.
        """
        if not self.enabled or self.roi_bbox is None:
            return False

        if not detections:
            return False

        h, w = frame_shape[:2]
        _, y_top, _, y_bottom = self.roi_bbox

        clipped = 0
        for det in detections:
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            # Check if detection center is outside ROI
            center_y = (y1 + y2) / 2
            if center_y < y_top or center_y > y_bottom:
                clipped += 1

        self.total_detection_count += len(detections)
        self.clipped_detection_count += clipped

        # If >30% clipped over recent frames, skip masking
        if self.total_detection_count > 0:
            clip_ratio = self.clipped_detection_count / self.total_detection_count
            if clip_ratio > 0.3:
                self.skip_mask_frames = 30  # Skip for 30 frames
                logger.warning(
                    f"ROI masking skipped: {clip_ratio:.1%} detections would be clipped"
                )
                return True

        return False

    def reset_clipping_stats(self):
        """Reset clipping statistics."""
        self.clipped_detection_count = 0
        self.total_detection_count = 0

    def update_skip_counter(self):
        """Update skip counter (call each frame)."""
        if self.skip_mask_frames > 0:
            self.skip_mask_frames -= 1
            if self.skip_mask_frames == 0:
                self.reset_clipping_stats()

    def get_coverage(self) -> float:
        """Get ROI coverage ratio (0-1)."""
        if not self.enabled or self.roi_bbox is None:
            return 1.0

        # This would need frame dimensions - simplified for now
        return self.band_height / 720.0  # Assuming 720p default

