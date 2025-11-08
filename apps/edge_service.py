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

from core.camera import create_frame_source
from core.config import AppConfig
from core.counting import LineCounter
from core.hailo_infer import HailoInfer
from core.metrics import Metrics
from core.overlay import annotate
from core.postproc import PostProcessor
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

        # Storage
        storage = Storage(config)
        logger.info("Storage initialized")

        # Metrics
        metrics = Metrics(port=config.ops.metrics_port)
        logger.info("Metrics initialized")

        # Web UI (in background thread)
        web_server = start_web_server(config, _shared_state)
        logger.info(f"Web UI started on port {config.ui.http_port}")

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

        for frame_bgr, timestamp_ns in camera.frames():
            try:
                frame_count += 1
                timestamp_utc = now_utc()

                # Inference
                raw_detections = infer.infer(frame_bgr)

                # Post-process
                detections = postproc.process(raw_detections)

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
                    annotated_frame = annotate(frame_bgr, tracked_dets, counter, stats)
                else:
                    annotated_frame = frame_bgr

                # Update shared state for web UI
                _shared_state["frame"] = annotated_frame
                _shared_state["stats"] = stats
                _shared_state["last_update"] = time.time()

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

