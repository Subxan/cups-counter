"""Main edge service application."""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from apps.auto_calibrate import AutoCalibrator
from core.camera import create_frame_source
from core.config import AppConfig
from core.counting import LineCounter
from core.drift import DriftMonitor
from core.hailo_infer import HailoInfer
from core.metrics import Metrics
from core.overlay import annotate
from core.postproc import PostProcessor
from core.roi import ROIMasker
from core.storage import Storage
from core.tracking import create_tracker
from core.utils import now_utc
from webui.api import start_web_server

# Global state for web UI
_shared_state = {"frame": None, "stats": {}, "last_update": 0}


def setup_logging(log_level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("cups-counter.log"),
        ],
    )


def main():
    """Main application loop."""
    parser = argparse.ArgumentParser(description="Cups Counter Edge Service")
    parser.add_argument("--config", type=str, default="configs/site-default.yaml", help="Config file path")
    parser.add_argument("--headless", action="store_true", help="Disable overlay drawing")
    parser.add_argument("--mock", action="store_true", help="Force mock mode")
    args = parser.parse_args()

    # Load config
    try:
        config = AppConfig.from_yaml(args.config)
        if args.mock:
            config.model.mock_mode = True
    except Exception as e:
        print(f"Failed to load config: {e}")
        return 1

    setup_logging(config.ops.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting Cups Counter Edge Service")
    logger.info(f"Config: {args.config}")
    logger.info(f"Mock mode: {config.model.mock_mode}")
    logger.info(f"Headless: {args.headless}")

    # Initialize components
    camera = None
    infer = None
    tracker = None
    counter = None
    storage = None
    metrics = None
    web_server = None
    drift_monitor = None
    roi_masker = None
    auto_calibrator = None
    auto_applied = False

    try:
        # Camera
        camera = create_frame_source(config)
        logger.info("Camera initialized")

        # Inference
        infer = HailoInfer(config)
        infer.warmup()
        logger.info("Inference engine ready")

        # Post-processing
        postproc = PostProcessor(config)

        # Tracker
        tracker = create_tracker(config)
        logger.info(f"Tracker initialized: {config.tracking.type}")

        # Counter
        counter = LineCounter(config)
        logger.info("Line counter initialized")

        # Drift monitor
        drift_monitor = DriftMonitor(config)
        logger.info("Drift monitor initialized")

        # ROI masker
        roi_masker = ROIMasker(config)
        if roi_masker.enabled:
            roi_masker.set_line(
                tuple(config.counting.line.start), tuple(config.counting.line.end)
            )
            logger.info("ROI masker initialized")

        # Auto-calibrator
        auto_calibrator = AutoCalibrator(config) if config.autocal.enabled else None

        # Storage
        storage = Storage(config)
        logger.info("Storage initialized")

        # Metrics
        metrics = Metrics(port=config.ops.metrics_port)
        logger.info("Metrics initialized")

        # Web UI (in background thread)
        web_server = start_web_server(config, _shared_state)
        logger.info(f"Web UI started on port {config.ui.http_port}")

        # Auto-calibration on startup
        if auto_calibrator and config.autocal.enabled:
            logger.info("Running auto-calibration...")
            candidates = auto_calibrator.run(camera, infer, postproc, tracker)

            if candidates:
                metrics.increment_autocal_runs()
                _shared_state["autocal_candidates"] = candidates

                # Auto-apply if confident and enabled
                if (
                    config.autocal.auto_apply_if_confident
                    and candidates[0]["confidence"] > 0.7
                ):
                    best = candidates[0]
                    config.counting.line.start = best["start"]
                    config.counting.line.end = best["end"]
                    config.counting.direction = best["direction"]
                    config.to_yaml(args.config)

                    # Update counter and ROI
                    counter.line_start = tuple(best["start"])
                    counter.line_end = tuple(best["end"])
                    if roi_masker.enabled:
                        roi_masker.set_line(tuple(best["start"]), tuple(best["end"]))

                    auto_applied = True
                    metrics.increment_autocal_applied()
                    logger.info(f"Auto-applied line: {best['start']} -> {best['end']}")
                else:
                    logger.info(f"Found {len(candidates)} candidates (manual selection needed)")

        # Set reference frame for drift monitoring (will be set from first frame in loop)

        # Signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Main loop
        logger.info("Entering main loop...")
        frame_count = 0
        start_time = time.time()
        reference_set = False

        for frame_bgr, timestamp_ns in camera.frames():
            try:
                frame_count += 1
                timestamp_utc = now_utc()

                # Set reference frame on first frame
                if drift_monitor and drift_monitor.enabled and not reference_set:
                    drift_monitor.set_reference(frame_bgr)
                    reference_set = True

                # Drift monitoring (every 30 frames)
                drift_metrics = {}
                if drift_monitor and drift_monitor.enabled and frame_count % 30 == 0:
                    drift_metrics = drift_monitor.update(frame_bgr)
                    metrics.update_drift_ssim(drift_metrics.get("ssim", 1.0))
                    metrics.update_edge_iou(drift_metrics.get("edge_iou", 1.0))
                    metrics.update_brightness_var(drift_metrics.get("brightness_var", 0.0))

                    # Check for recalibration
                    if drift_monitor.should_recalibrate() and auto_calibrator:
                        logger.warning("Drift detected, triggering recalibration...")
                        metrics.increment_recalibrations()
                        candidates = auto_calibrator.run(camera, infer, postproc, tracker)
                        if candidates:
                            best = candidates[0]
                            config.counting.line.start = best["start"]
                            config.counting.line.end = best["end"]
                            config.to_yaml(args.config)
                            counter.line_start = tuple(best["start"])
                            counter.line_end = tuple(best["end"])
                            if roi_masker.enabled:
                                roi_masker.set_line(tuple(best["start"]), tuple(best["end"]))
                            drift_monitor.set_reference(frame_bgr)
                            drift_monitor.mark_recalibrated()
                            logger.info("Recalibration complete")

                # ROI masking (optional)
                if roi_masker and roi_masker.enabled:
                    roi_masker.update_skip_counter()
                    # Check if detections would be clipped (before inference)
                    # For now, apply mask to frame
                    frame_for_infer = frame_bgr
                else:
                    frame_for_infer = frame_bgr

                # Inference
                raw_detections = infer.infer(frame_for_infer)

                # Post-process
                detections = postproc.process(raw_detections)

                # Check ROI clipping
                if roi_masker and roi_masker.enabled:
                    roi_masker.check_detection_clipping(detections, frame_bgr.shape)

                # Track
                tracked_dets = tracker.update(detections)

                # Count
                events = counter.update(tracked_dets, timestamp_utc)

                # Store events
                if events:
                    storage.write_events(events)
                    for event in events:
                        if event["direction"] == "in":
                            metrics.increment_in()
                        else:
                            metrics.increment_out()

                # Update metrics
                metrics.increment_frames()
                metrics.update_active_tracks(len(tracked_dets))

                # ROI coverage
                if roi_masker and roi_masker.enabled:
                    metrics.update_roi_coverage(roi_masker.get_coverage())

                # Calculate FPS
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0.0
                metrics.update_fps(fps)

                # Get totals
                totals = counter.totals()
                stats = {
                    "in": totals["in"],
                    "out": totals["out"],
                    "net": totals["net"],
                    "fps": fps,
                }

                # Draw overlay
                if config.ui.draw_overlays and not args.headless:
                    annotated_frame = annotate(
                        frame_bgr,
                        tracked_dets,
                        counter,
                        stats,
                        roi_masker=roi_masker,
                        drift_metrics=drift_metrics,
                        auto_applied=auto_applied,
                    )
                else:
                    annotated_frame = frame_bgr

                # Update shared state for web UI
                _shared_state["frame"] = annotated_frame
                _shared_state["stats"] = stats
                _shared_state["last_update"] = time.time()
                _shared_state["drift_metrics"] = drift_metrics
                if drift_monitor:
                    _shared_state["last_recal_time"] = drift_monitor.last_recal_time

                # Save debug video if enabled
                # (implementation would use VideoWriter)

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                metrics.increment_dropped()
                time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        if camera:
            camera.close()
        if infer:
            infer.close()
        if storage:
            storage.stop()
        if metrics:
            metrics.stop()
        if web_server:
            # Stop web server (would need proper shutdown)
            pass

        logger.info("Shutdown complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())

