"""Line crossing detection for counting cups."""

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

from core.utils import centroid, segment_intersect

logger = logging.getLogger(__name__)

# TrackedDetection format: (x1, y1, x2, y2, conf, cls_id, track_id)
TrackedDetection = Tuple[float, float, float, float, float, int, int]

# Event format
Event = Dict[str, any]


class LineCounter:
    """Direction-aware line crossing counter with hysteresis."""

    def __init__(self, config):
        self.config = config
        self.line_start = tuple(config.counting.line.start)
        self.line_end = tuple(config.counting.line.end)
        self.direction = config.counting.direction
        self.hysteresis_px = config.counting.hysteresis_px
        self.min_visible_frames = config.counting.min_visible_frames

        # Track state per track_id
        self.track_states: Dict[int, Dict] = defaultdict(
            lambda: {
                "last_centroid": None,
                "side": None,  # "bar" or "counter"
                "visible_frames": 0,
                "crossed": False,  # Whether already counted for this crossing
            }
        )

        self.in_count = 0
        self.out_count = 0

    def _determine_side(self, point: Tuple[float, float]) -> str | None:
        """Determine which side of the line a point is on.

        Uses cross product to determine side relative to line direction.
        Returns "bar" or "counter" or None if on line.
        """
        px, py = point
        x1, y1 = self.line_start
        x2, y2 = self.line_end

        # Vector from line_start to point
        v1x = px - x1
        v1y = py - y1

        # Vector from line_start to line_end
        v2x = x2 - x1
        v2y = y2 - y1

        # Cross product
        cross = v1x * v2y - v1y * v2x

        if abs(cross) < self.hysteresis_px * ((v2x**2 + v2y**2) ** 0.5):
            return None  # Within hysteresis zone

        # Determine side based on direction
        if self.direction == "bar_to_counter":
            # Positive cross = counter side, negative = bar side
            return "counter" if cross > 0 else "bar"
        else:  # counter_to_bar
            # Inverted
            return "bar" if cross > 0 else "counter"

    def _check_crossing(
        self, track_id: int, prev_centroid: Tuple[float, float], curr_centroid: Tuple[float, float]
    ) -> str | None:
        """Check if track crossed the line.

        Returns "in" or "out" if crossing detected, None otherwise.
        """
        # Check if line segment intersects with movement segment
        intersection = segment_intersect(
            prev_centroid, curr_centroid, self.line_start, self.line_end
        )

        if intersection is None:
            return None

        # Determine direction of crossing
        prev_side = self._determine_side(prev_centroid)
        curr_side = self._determine_side(curr_centroid)

        # If both on same side or one is None (in hysteresis), no crossing
        if prev_side is None or curr_side is None or prev_side == curr_side:
            return None

        # Crossing detected
        if self.direction == "bar_to_counter":
            if prev_side == "bar" and curr_side == "counter":
                return "in"
            elif prev_side == "counter" and curr_side == "bar":
                return "out"
        else:  # counter_to_bar
            if prev_side == "counter" and curr_side == "bar":
                return "in"
            elif prev_side == "bar" and curr_side == "counter":
                return "out"

        return None

    def update(self, tracked_dets: List[TrackedDetection], timestamp_utc) -> List[Event]:
        """Update counter with new tracked detections.

        Args:
            tracked_dets: List of TrackedDetection
            timestamp_utc: UTC datetime for events

        Returns:
            List of Event dicts for new crossings
        """
        events = []

        # Get current centroids for all tracks
        current_tracks = {}
        for det in tracked_dets:
            x1, y1, x2, y2, conf, cls_id, track_id = det
            cent = centroid((x1, y1, x2, y2))
            current_tracks[track_id] = {
                "centroid": cent,
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
            }
            self.track_states[track_id]["visible_frames"] += 1

        # Check for crossings
        for track_id, state in list(self.track_states.items()):
            if track_id not in current_tracks:
                # Track lost - reset state but keep for a bit
                state["visible_frames"] = 0
                continue

            track_info = current_tracks[track_id]
            curr_cent = track_info["centroid"]

            if state["last_centroid"] is not None:
                # Check for crossing
                crossing_dir = self._check_crossing(
                    track_id, state["last_centroid"], curr_cent
                )

                if crossing_dir and not state["crossed"]:
                    # Only count if track has been visible enough
                    if state["visible_frames"] >= self.min_visible_frames:
                        # Create event
                        event = {
                            "ts_utc": timestamp_utc.isoformat(),
                            "direction": crossing_dir,
                            "track_id": track_id,
                            "bbox": list(track_info["bbox"]),
                            "conf": float(track_info["conf"]),
                        }
                        events.append(event)

                        # Update counts
                        if crossing_dir == "in":
                            self.in_count += 1
                        else:
                            self.out_count += 1

                        # Mark as crossed to prevent double-counting
                        state["crossed"] = True

            # Update state
            state["last_centroid"] = curr_cent
            state["side"] = self._determine_side(curr_cent)

            # Reset crossed flag if track moved away from line
            if state["side"] is not None:
                dist_to_line = abs(
                    (self.line_end[1] - self.line_start[1]) * curr_cent[0]
                    - (self.line_end[0] - self.line_start[0]) * curr_cent[1]
                    + self.line_end[0] * self.line_start[1]
                    - self.line_end[1] * self.line_start[0]
                ) / ((self.line_end[1] - self.line_start[1]) ** 2 + (self.line_end[0] - self.line_start[0]) ** 2) ** 0.5

                if dist_to_line > self.hysteresis_px * 2:
                    state["crossed"] = False

        # Clean up old tracks
        for track_id in list(self.track_states.keys()):
            if track_id not in current_tracks:
                if self.track_states[track_id]["visible_frames"] == 0:
                    # Remove after a delay
                    del self.track_states[track_id]

        return events

    def totals(self) -> Dict[str, int]:
        """Get current totals."""
        return {
            "in": self.in_count,
            "out": self.out_count,
            "net": self.in_count - self.out_count,
        }

    def reset(self):
        """Reset counts (for testing)."""
        self.in_count = 0
        self.out_count = 0
        self.track_states.clear()

