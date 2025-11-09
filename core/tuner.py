"""Parameter tuning via grid search on replay clips."""

import itertools
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class ParameterTuner:
    """Tune parameters via grid search on replay clips."""

    def __init__(self, config):
        self.config = config
        self.enabled = config.tuner.enabled

        if not self.enabled:
            return

        self.grid = config.tuner.grid
        self.optimize_for = config.tuner.optimize_for
        self.keep_best_profile = config.tuner.keep_best_profile

    def generate_grid(self) -> List[Dict]:
        """Generate parameter combinations from grid."""
        combinations = list(
            itertools.product(
                self.grid.conf_thresh,
                self.grid.match_thresh,
                self.grid.min_box_area,
            )
        )

        return [
            {
                "conf_thresh": c[0],
                "match_thresh": c[1],
                "min_box_area": c[2],
            }
            for c in combinations
        ]

    def score_run(
        self,
        events: List[Dict],
        tracks: Dict[int, int],  # track_id -> crossing_count
    ) -> float:
        """Score a parameter run.

        Returns score (higher is better).
        """
        if self.optimize_for == "stable_crossings":
            # Prefer: many unique tracks that cross exactly once
            unique_tracks = len(tracks)
            single_crossings = sum(1 for count in tracks.values() if count == 1)
            double_crossings = sum(1 for count in tracks.values() if count > 1)

            # Score: unique tracks * (single_crossings / total_crossings) - penalty for doubles
            if unique_tracks == 0:
                return 0.0

            single_ratio = single_crossings / unique_tracks
            penalty = double_crossings * 0.5
            score = unique_tracks * single_ratio - penalty

            return max(0.0, score)

        elif self.optimize_for == "min_double_counts":
            # Prefer: minimal double-counting
            total_crossings = len(events)
            unique_tracks = len(tracks)
            double_crossings = sum(1 for count in tracks.values() if count > 1)

            if total_crossings == 0:
                return 0.0

            # Score: unique tracks / total_crossings (higher = less double counting)
            score = unique_tracks / total_crossings
            penalty = double_crossings * 2.0
            return max(0.0, score - penalty)

        return 0.0

    def tune_on_clip(
        self,
        clip_path: str | Path,
        infer_fn,
        postproc_fn,
        tracker_fn,
        counter_fn,
    ) -> Dict:
        """Run grid search on a video clip.

        Args:
            clip_path: Path to video file
            infer_fn: Function(frame) -> detections
            postproc_fn: Function(detections, params) -> filtered_detections
            tracker_fn: Function(detections, params) -> tracked_detections
            counter_fn: Function(tracked_detections) -> events

        Returns:
            Best parameter set and scores
        """
        if not self.enabled:
            return {}

        logger.info(f"Tuning on clip: {clip_path}")

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            logger.error(f"Failed to open clip: {clip_path}")
            return {}

        # Generate parameter grid
        param_combos = self.generate_grid()
        logger.info(f"Testing {len(param_combos)} parameter combinations")

        best_score = -1.0
        best_params = None
        all_scores = []

        for params in param_combos:
            logger.debug(f"Testing params: {params}")

            # Reset components for this run
            tracker = tracker_fn(params)
            counter = counter_fn(params)

            # Process clip with these params
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind
            events = []
            track_crossings = {}  # track_id -> count

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Inference
                raw_detections = infer_fn(frame)

                # Post-process with params
                detections = postproc_fn(raw_detections, params)

                # Track with params
                tracked_dets = tracker.update(detections)

                # Count
                from datetime import datetime, timezone

                run_events = counter.update(tracked_dets, datetime.now(timezone.utc))

                # Aggregate
                events.extend(run_events)
                for event in run_events:
                    track_id = event.get("track_id", 0)
                    track_crossings[track_id] = track_crossings.get(track_id, 0) + 1

                frame_count += 1

            # Score this run
            score = self.score_run(events, track_crossings)
            all_scores.append((params, score))

            logger.debug(f"Params {params}: score={score:.2f}")

            if score > best_score:
                best_score = score
                best_params = params

        cap.release()

        logger.info(f"Best params: {best_params} (score={best_score:.2f})")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "all_scores": all_scores,
        }

    def save_best_profile(self, best_params: Dict, output_path: str | Path):
        """Save best parameters to override YAML."""
        if not self.keep_best_profile:
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        override = {
            "model": {"conf_thresh": best_params["conf_thresh"]},
            "tracking": {
                "match_thresh": best_params["match_thresh"],
                "min_box_area": best_params["min_box_area"],
            },
        }

        with open(output_path, "w") as f:
            yaml.dump(override, f, default_flow_style=False)

        logger.info(f"Best profile saved to {output_path}")

    def load_override(self, override_path: str | Path) -> Dict | None:
        """Load parameter override from YAML."""
        override_path = Path(override_path)
        if not override_path.exists():
            return None

        with open(override_path, "r") as f:
            data = yaml.safe_load(f)

        params = {}
        if "model" in data and "conf_thresh" in data["model"]:
            params["conf_thresh"] = data["model"]["conf_thresh"]
        if "tracking" in data:
            if "match_thresh" in data["tracking"]:
                params["match_thresh"] = data["tracking"]["match_thresh"]
            if "min_box_area" in data["tracking"]:
                params["min_box_area"] = data["tracking"]["min_box_area"]

        return params if params else None

