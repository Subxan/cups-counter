"""Object tracking with ByteTrack or OCSORT."""

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Detection format: (x1, y1, x2, y2, conf, cls_id)
Detection = Tuple[float, float, float, float, float, int]
# TrackedDetection format: (x1, y1, x2, y2, conf, cls_id, track_id)
TrackedDetection = Tuple[float, float, float, float, float, int, int]


class ByteTracker:
    """Simplified ByteTrack implementation."""

    def __init__(self, config):
        self.config = config
        self.track_thresh = config.tracking.track_thresh
        self.match_thresh = config.tracking.match_thresh
        self.min_box_area = config.tracking.min_box_area
        self.lost_ttl = config.tracking.lost_ttl

        self.tracks = {}  # track_id -> {bbox, conf, cls_id, age, lost_frames}
        self.next_track_id = 1

    def _iou(self, box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
        """Calculate IoU between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        inter_area = (xi2 - xi1) * (yi2 - yi1)

        # Union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _area(self, bbox: Tuple[float, float, float, float]) -> float:
        """Calculate box area."""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def update(self, detections: List[Detection]) -> List[TrackedDetection]:
        """Update tracks with new detections.

        Returns list of TrackedDetection (x1, y1, x2, y2, conf, cls_id, track_id).
        """
        # Filter by min area
        valid_dets = [
            det
            for det in detections
            if self._area((det[0], det[1], det[2], det[3])) >= self.min_box_area
        ]

        # Split into high and low confidence
        high_conf = [d for d in valid_dets if d[4] >= self.track_thresh]
        low_conf = [d for d in valid_dets if d[4] < self.track_thresh]

        # Match high confidence detections to existing tracks
        matched_tracks = set()
        matched_dets = set()
        tracked = []

        for track_id, track in self.tracks.items():
            if track["lost_frames"] > 0:
                continue  # Skip lost tracks for high-conf matching

            best_iou = 0.0
            best_det_idx = -1

            for idx, det in enumerate(high_conf):
                if idx in matched_dets:
                    continue

                iou = self._iou(
                    (track["bbox"][0], track["bbox"][1], track["bbox"][2], track["bbox"][3]),
                    (det[0], det[1], det[2], det[3]),
                )

                if iou > best_iou and iou >= self.match_thresh:
                    best_iou = iou
                    best_det_idx = idx

            if best_det_idx >= 0:
                # Update track
                det = high_conf[best_det_idx]
                self.tracks[track_id] = {
                    "bbox": (det[0], det[1], det[2], det[3]),
                    "conf": det[4],
                    "cls_id": det[5],
                    "age": track["age"] + 1,
                    "lost_frames": 0,
                }
                matched_tracks.add(track_id)
                matched_dets.add(best_det_idx)
                tracked.append((det[0], det[1], det[2], det[3], det[4], det[5], track_id))

        # Create new tracks for unmatched high-conf detections
        for idx, det in enumerate(high_conf):
            if idx not in matched_dets:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = {
                    "bbox": (det[0], det[1], det[2], det[3]),
                    "conf": det[4],
                    "cls_id": det[5],
                    "age": 1,
                    "lost_frames": 0,
                }
                tracked.append((det[0], det[1], det[2], det[3], det[4], det[5], track_id))

        # Match low confidence detections to lost tracks
        for track_id, track in list(self.tracks.items()):
            if track_id in matched_tracks:
                continue

            if track["lost_frames"] >= self.lost_ttl:
                # Remove expired track
                del self.tracks[track_id]
                continue

            best_iou = 0.0
            best_det_idx = -1

            for idx, det in enumerate(low_conf):
                iou = self._iou(
                    (track["bbox"][0], track["bbox"][1], track["bbox"][2], track["bbox"][3]),
                    (det[0], det[1], det[2], det[3]),
                )

                if iou > best_iou and iou >= self.match_thresh:
                    best_iou = iou
                    best_det_idx = idx

            if best_det_idx >= 0:
                # Reactivate track
                det = low_conf[best_det_idx]
                self.tracks[track_id] = {
                    "bbox": (det[0], det[1], det[2], det[3]),
                    "conf": det[4],
                    "cls_id": det[5],
                    "age": track["age"] + 1,
                    "lost_frames": 0,
                }
                tracked.append((det[0], det[1], det[2], det[3], det[4], det[5], track_id))
            else:
                # Increment lost frames
                self.tracks[track_id]["lost_frames"] += 1

        return tracked


class OCSORTTracker:
    """OCSORT tracker (placeholder - simplified implementation)."""

    def __init__(self, config):
        self.config = config
        # For now, use ByteTrack implementation
        logger.warning("OCSORT using ByteTrack implementation")
        self.tracker = ByteTracker(config)

    def update(self, detections: List[Detection]) -> List[TrackedDetection]:
        """Update tracks."""
        return self.tracker.update(detections)


def create_tracker(config) -> ByteTracker | OCSORTTracker:
    """Factory function to create tracker."""
    tracker_type = config.tracking.type
    if tracker_type == "bytetrack":
        return ByteTracker(config)
    elif tracker_type == "ocsort":
        return OCSORTTracker(config)
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")

