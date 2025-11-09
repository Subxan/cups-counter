"""Auto-calibration: motion heatmap + Hough line detection."""

import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.camera import create_frame_source
from core.config import AppConfig
from core.hailo_infer import HailoInfer
from core.postproc import PostProcessor
from core.tracking import create_tracker
from core.utils import safe_makedirs

logger = logging.getLogger(__name__)


class AutoCalibrator:
    """Auto-calibrate counting line from motion patterns."""

    def __init__(self, config):
        self.config = config
        self.enabled = config.autocal.enabled

        if not self.enabled:
            return

        self.warmup_seconds = config.autocal.warmup_seconds
        self.propose_top_k = config.autocal.propose_top_k
        self.min_flow_ratio = config.autocal.min_flow_ratio
        self.save_debug = config.autocal.save_debug

    def collect_warmup_data(
        self, camera, infer, postproc, tracker
    ) -> Tuple[List[np.ndarray], List[Dict], Dict[int, List[Tuple]]]:
        """Collect frames and track data during warmup.

        Returns:
            (frames, detections_list, track_trajectories)
        """
        logger.info(f"Collecting warmup data for {self.warmup_seconds} seconds...")

        frames = []
        detections_list = []
        track_trajectories = defaultdict(list)  # track_id -> [(x, y), ...]

        import time

        start_time = time.time()
        frame_count = 0

        for frame_bgr, _ in camera.frames():
            if time.time() - start_time >= self.warmup_seconds:
                break

            # Inference
            raw_detections = infer.infer(frame_bgr)
            detections = postproc.process(raw_detections)
            tracked_dets = tracker.update(detections)

            # Store frame (every Nth frame to save memory)
            if frame_count % 10 == 0:
                frames.append(frame_bgr.copy())

            # Store detections
            detections_list.append(tracked_dets)

            # Build trajectories
            for det in tracked_dets:
                x1, y1, x2, y2, _, _, track_id = det
                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                track_trajectories[track_id].append(centroid)

            frame_count += 1

        logger.info(f"Collected {frame_count} frames, {len(track_trajectories)} tracks")
        return frames, detections_list, dict(track_trajectories)

    def build_motion_heatmap(
        self, frames: List[np.ndarray], trajectories: Dict[int, List[Tuple]]
    ) -> np.ndarray:
        """Build motion heatmap from trajectories."""
        if not frames:
            return None

        h, w = frames[0].shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        # Draw trajectories as lines
        for track_id, points in trajectories.items():
            if len(points) < 2:
                continue

            # Draw line segments
            for i in range(len(points) - 1):
                pt1 = (int(points[i][0]), int(points[i][1]))
                pt2 = (int(points[i + 1][0]), int(points[i + 1][1]))
                cv2.line(heatmap, pt1, pt2, 1.0, 2)

        # Blur to smooth
        heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)

        if self.save_debug:
            debug_path = Path(self.config.ops.debug_dir) / "motion_heatmap.png"
            safe_makedirs(debug_path.parent)
            cv2.imwrite(str(debug_path), (heatmap * 255).astype(np.uint8))
            logger.info(f"Saved motion heatmap to {debug_path}")

        return heatmap

    def compute_flow_vectors(
        self, trajectories: Dict[int, List[Tuple]]
    ) -> Tuple[Dict[str, float], str]:
        """Compute dominant flow direction from trajectories.

        Returns:
            (flow_stats, dominant_direction)
        """
        flows = {"bar_to_counter": 0, "counter_to_bar": 0}

        for track_id, points in trajectories.items():
            if len(points) < 5:
                continue

            # Compute average direction
            start = np.array(points[0])
            end = np.array(points[-1])
            direction = end - start

            # Assume horizontal line - check y component
            if direction[1] < -10:  # Moving up
                flows["bar_to_counter"] += 1
            elif direction[1] > 10:  # Moving down
                flows["counter_to_bar"] += 1

        total = sum(flows.values())
        if total == 0:
            return flows, "bar_to_counter"

        # Determine dominant direction
        bar_to_counter_ratio = flows["bar_to_counter"] / total
        if bar_to_counter_ratio >= self.min_flow_ratio:
            dominant = "bar_to_counter"
        elif (1 - bar_to_counter_ratio) >= self.min_flow_ratio:
            dominant = "counter_to_bar"
        else:
            dominant = "bar_to_counter"  # Default

        return flows, dominant

    def find_candidate_lines(
        self, frames: List[np.ndarray], trajectories: Dict[int, List[Tuple]]
    ) -> List[Dict]:
        """Find candidate lines using Hough transform on edge map."""
        if not frames:
            return []

        # Average frames for stable edge detection
        avg_frame = np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames], axis=0).astype(
            np.uint8
        )

        # Focus on lower third (where counter line typically is)
        h, w = avg_frame.shape
        lower_third = avg_frame[int(2 * h / 3) :, :]

        # Edge detection
        edges = cv2.Canny(lower_third, 50, 150)

        if self.save_debug:
            debug_path = Path(self.config.ops.debug_dir) / "edge_map.png"
            cv2.imwrite(str(debug_path), edges)
            logger.info(f"Saved edge map to {debug_path}")

        # Hough lines (prefer horizontal)
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=w // 3, maxLineGap=20
        )

        if lines is None:
            logger.warning("No lines found in Hough transform")
            return []

        # Convert to full image coordinates and filter
        candidates = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Adjust y coordinates (add offset for lower third)
            y1 += int(2 * h / 3)
            y2 += int(2 * h / 3)

            # Filter: prefer horizontal lines
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle > 15 and angle < 165:  # Not horizontal enough
                continue

            # Score line
            score = self._score_line((x1, y1, x2, y2), trajectories, (w, h))
            candidates.append(
                {
                    "start": [int(x1), int(y1)],
                    "end": [int(x2), int(y2)],
                    "score": score,
                }
            )

        # Sort by score and return top-k
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[: self.propose_top_k]

    def _score_line(
        self,
        line: Tuple[int, int, int, int],
        trajectories: Dict[int, List[Tuple]],
        frame_size: Tuple[int, int],
    ) -> float:
        """Score a candidate line.

        Higher score = better line.
        """
        x1, y1, x2, y2 = line
        w, h = frame_size

        score = 0.0

        # 1. Alignment with flow separator (trajectories crossing)
        crossings = 0
        for track_id, points in trajectories.items():
            if len(points) < 2:
                continue

            # Check if trajectory crosses line
            for i in range(len(points) - 1):
                p1, p2 = points[i], points[i + 1]
                # Simple line intersection check
                if self._segments_intersect((p1, p2), (x1, y1), (x2, y2)):
                    crossings += 1
                    break

        score += crossings * 10.0

        # 2. Distance from borders (prefer middle region)
        mid_y = (y1 + y2) / 2
        border_dist = min(mid_y, h - mid_y)
        score += border_dist / 10.0

        # 3. Length (prefer longer lines)
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        score += length / 100.0

        return score

    def _segments_intersect(
        self, seg1: Tuple[Tuple, Tuple], seg2_start: Tuple, seg2_end: Tuple
    ) -> bool:
        """Check if two line segments intersect."""
        p1, p2 = seg1
        p3, p4 = seg2_start, seg2_end

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def run(self, camera, infer, postproc, tracker) -> List[Dict]:
        """Run auto-calibration.

        Returns list of candidate lines with scores.
        """
        if not self.enabled:
            return []

        logger.info("Starting auto-calibration...")

        # Collect warmup data
        frames, detections_list, trajectories = self.collect_warmup_data(
            camera, infer, postproc, tracker
        )

        if not trajectories:
            logger.warning("No trajectories collected, cannot auto-calibrate")
            return []

        # Build motion heatmap
        heatmap = self.build_motion_heatmap(frames, trajectories)

        # Compute flow direction
        flow_stats, dominant_direction = self.compute_flow_vectors(trajectories)
        logger.info(f"Dominant flow direction: {dominant_direction} ({flow_stats})")

        # Find candidate lines
        candidates = self.find_candidate_lines(frames, trajectories)

        # Add direction to candidates
        for candidate in candidates:
            candidate["direction"] = dominant_direction
            candidate["confidence"] = min(1.0, candidate["score"] / 100.0)

        logger.info(f"Found {len(candidates)} candidate lines")
        return candidates


def main():
    """Standalone auto-calibration tool."""
    import argparse

    parser = argparse.ArgumentParser(description="Auto-calibrate counting line")
    parser.add_argument("--config", type=str, default="configs/site-default.yaml")
    args = parser.parse_args()

    config = AppConfig.from_yaml(args.config)

    camera = None
    try:
        camera = create_frame_source(config)
        infer = HailoInfer(config)
        postproc = PostProcessor(config)
        tracker = create_tracker(config)

        calibrator = AutoCalibrator(config)
        candidates = calibrator.run(camera, infer, postproc, tracker)

        print(f"\nFound {len(candidates)} candidate lines:")
        for i, cand in enumerate(candidates):
            print(f"{i+1}. {cand['start']} -> {cand['end']} (score={cand['score']:.1f})")

    finally:
        if camera:
            camera.close()


if __name__ == "__main__":
    main()

