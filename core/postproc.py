"""Post-processing: NMS, class filtering, coordinate scaling."""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Detection format: (x1, y1, x2, y2, conf, cls_id)
Detection = Tuple[float, float, float, float, float, int]


class PostProcessor:
    """Post-process detections: NMS, class filtering, label mapping."""

    def __init__(self, config):
        self.config = config
        self.class_filter = set(config.model.class_filter)
        self.conf_thresh = config.model.conf_thresh
        self.iou_thresh = config.model.iou_thresh
        self.max_detections = config.model.max_detections

        # Load label map
        self.label_map = self._load_label_map()

    def _load_label_map(self) -> dict:
        """Load class ID to name mapping from labelmap.json."""
        labelmap_path = Path("models/labelmap.json")
        if labelmap_path.exists():
            try:
                with open(labelmap_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load labelmap: {e}")
        # Default mapping (COCO format - cup is class 41)
        return {"cup": 41}

    def _get_class_name(self, cls_id: int) -> str | None:
        """Get class name from class ID."""
        for name, id_val in self.label_map.items():
            if id_val == cls_id:
                return name
        return None

    def process(self, detections: List[Detection]) -> List[Detection]:
        """Apply NMS, class filtering, and confidence thresholding.

        Args:
            detections: List of (x1, y1, x2, y2, conf, cls_id)

        Returns:
            Filtered and NMS'd detections
        """
        if not detections:
            return []

        # Filter by confidence and class
        filtered = []
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            if conf < self.conf_thresh:
                continue

            class_name = self._get_class_name(cls_id)
            if class_name and class_name in self.class_filter:
                filtered.append(det)

        if not filtered:
            return []

        # Prepare for NMS
        boxes = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2, _, _ in filtered], dtype=np.float32)
        scores = np.array([conf for _, _, _, _, conf, _ in filtered], dtype=np.float32)
        class_ids = np.array([cls_id for _, _, _, _, _, cls_id in filtered], dtype=np.int32)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.conf_thresh,
            self.iou_thresh,
        )

        if len(indices) == 0:
            return []

        # Extract kept detections
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        else:
            indices = [indices]

        kept = []
        for idx in indices[: self.max_detections]:
            x1, y1, x2, y2 = boxes[idx]
            conf = scores[idx]
            cls_id = class_ids[idx]
            kept.append((float(x1), float(y1), float(x2), float(y2), float(conf), int(cls_id)))

        return kept

