"""Frame overlay drawing for visualization."""

import cv2
import numpy as np

from core.counting import LineCounter
from core.utils import centroid

# TrackedDetection format: (x1, y1, x2, y2, conf, cls_id, track_id)
TrackedDetection = tuple


def annotate(
    frame: np.ndarray,
    tracks: list[TrackedDetection],
    line_counter: LineCounter,
    stats: dict,
    roi_masker=None,
    drift_metrics: dict = None,
    auto_applied: bool = False,
) -> np.ndarray:
    """Draw overlays on frame: line, boxes, IDs, OSD, ROI band, drift indicators.

    Args:
        frame: Input frame (BGR)
        tracks: List of TrackedDetection
        line_counter: LineCounter instance
        stats: Dict with keys: in, out, net, fps
        roi_masker: ROIMasker instance (optional)
        drift_metrics: Dict with drift metrics (optional)
        auto_applied: Whether line was auto-applied (optional)

    Returns:
        Annotated frame
    """
    annotated = frame.copy()

    # Draw ROI band if enabled
    if roi_masker and roi_masker.enabled and roi_masker.roi_bbox:
        _, y_top, _, y_bottom = roi_masker.roi_bbox
        h, w = frame.shape[:2]
        # Draw semi-transparent band
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, y_top), (w, y_bottom), (255, 255, 0), -1)
        cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)
        # Draw band borders
        cv2.line(annotated, (0, y_top), (w, y_top), (255, 255, 0), 1)
        cv2.line(annotated, (0, y_bottom), (w, y_bottom), (255, 255, 0), 1)

    # Draw counting line
    line_start = line_counter.line_start
    line_end = line_counter.line_end
    cv2.line(annotated, line_start, line_end, (0, 255, 0), 2)

    # Draw direction arrow
    mid_x = (line_start[0] + line_end[0]) // 2
    mid_y = (line_start[1] + line_end[1]) // 2
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    length = (dx**2 + dy**2) ** 0.5
    if length > 0:
        # Perpendicular vector for arrow
        perp_x = -dy / length
        perp_y = dx / length
        arrow_len = 20
        arrow_x = int(mid_x + perp_x * arrow_len)
        arrow_y = int(mid_y + perp_y * arrow_len)
        cv2.arrowedLine(
            annotated,
            (mid_x, mid_y),
            (arrow_x, arrow_y),
            (0, 255, 0),
            2,
            tipLength=0.3,
        )

    # Draw bounding boxes and track IDs
    for track in tracks:
        x1, y1, x2, y2, conf, cls_id, track_id = track
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Box color based on confidence
        color = (0, int(255 * conf), int(255 * (1 - conf)))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Track ID label
        label = f"ID:{track_id}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            annotated,
            (x1, y1 - label_size[1] - 4),
            (x1 + label_size[0], y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    # Draw OSD (On-Screen Display)
    h, w = annotated.shape[:2]
    osd_y = 30
    osd_spacing = 25

    # Background for OSD
    osd_bg = np.zeros((100, 300, 3), dtype=np.uint8)
    osd_bg[:] = (0, 0, 0)
    annotated[10:110, 10:310] = cv2.addWeighted(
        annotated[10:110, 10:310], 0.7, osd_bg, 0.3, 0
    )

    # OSD text
    texts = [
        f"IN:  {stats.get('in', 0)}",
        f"OUT: {stats.get('out', 0)}",
        f"NET: {stats.get('net', 0)}",
        f"FPS: {stats.get('fps', 0):.1f}",
    ]

    # Add drift indicators
    if drift_metrics:
        if drift_metrics.get("camera_shifted"):
            texts.append("DRIFT: Camera")
        if drift_metrics.get("lighting_bad"):
            texts.append("DRIFT: Lighting")

    # Add auto-applied badge
    if auto_applied:
        cv2.putText(
            annotated,
            "AUTO",
            (w - 80, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    for i, text in enumerate(texts):
        cv2.putText(
            annotated,
            text,
            (20, osd_y + i * osd_spacing),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return annotated

