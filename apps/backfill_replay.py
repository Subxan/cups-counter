"""Backfill/replay tool for processing video files."""

import sys
from pathlib import Path

import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import AppConfig
from core.counting import LineCounter
from core.hailo_infer import HailoInfer
from core.postproc import PostProcessor
from core.storage import Storage
from core.tracking import create_tracker
from core.utils import now_utc


def main():
    """Backfill main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Process video file and generate counts")
    parser.add_argument("video", type=str, help="Input video file")
    parser.add_argument("--config", type=str, default="configs/site-default.yaml", help="Config file")
    parser.add_argument("--output", type=str, help="Output CSV file (default: auto-generated)")
    args = parser.parse_args()

    # Load config
    try:
        config = AppConfig.from_yaml(args.config)
    except Exception as e:
        print(f"Failed to load config: {e}")
        return 1

    print(f"Processing video: {args.video}")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Failed to open video: {args.video}")
        return 1

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames @ {fps} fps")

    # Initialize components
    infer = HailoInfer(config)
    postproc = PostProcessor(config)
    tracker = create_tracker(config)
    counter = LineCounter(config)
    storage = Storage(config)

    print("Processing frames...")
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")

            # Inference
            raw_detections = infer.infer(frame)

            # Post-process
            detections = postproc.process(raw_detections)

            # Track
            tracked_dets = tracker.update(detections)

            # Count
            timestamp_utc = now_utc()
            events = counter.update(tracked_dets, timestamp_utc)

            # Store events
            if events:
                storage.write_events(events)

        # Flush storage
        storage.stop()

        # Get totals
        totals = counter.totals()
        print(f"\nProcessing complete!")
        print(f"Total counts - IN: {totals['in']}, OUT: {totals['out']}, NET: {totals['net']}")

        # Export CSV
        if args.output:
            csv_path = Path(args.output)
        else:
            # Auto-generate from video name
            video_name = Path(args.video).stem
            csv_path = Path(config.storage.csv_dir) / f"{video_name}_counts.csv"

        # Get all events and export
        from datetime import datetime

        day_str = datetime.now().strftime("%Y-%m-%d")
        storage.start()  # Restart for export
        csv_path = storage.export_csv(day_str, csv_path.parent)
        print(f"CSV exported: {csv_path}")

    except KeyboardInterrupt:
        print("\nCancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        cap.release()
        infer.close()
        storage.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())

