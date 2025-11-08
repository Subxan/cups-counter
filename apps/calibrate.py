"""Calibration tool for setting counting line."""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.camera import create_frame_source
from core.config import AppConfig
from core.overlay import annotate
from core.counting import LineCounter
from core.utils import safe_makedirs

# Global state for mouse callback
_calib_state = {"points": [], "config": None, "counter": None}


def mouse_callback(event, x, y, flags, param):
    """Mouse callback for selecting line points."""
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(_calib_state["points"]) < 2:
            _calib_state["points"].append((x, y))
            print(f"Point {len(_calib_state['points'])}: ({x}, {y})")


def main():
    """Calibration main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate counting line")
    parser.add_argument("--config", type=str, default="configs/site-default.yaml", help="Config file")
    parser.add_argument("--duration", type=int, default=120, help="Calibration duration (seconds)")
    args = parser.parse_args()

    # Load config
    try:
        config = AppConfig.from_yaml(args.config)
    except Exception as e:
        print(f"Failed to load config: {e}")
        return 1

    print("Calibration Tool")
    print("=" * 50)
    print("1. Click two points to define the counting line")
    print("2. Line will be saved to config")
    print("3. Live counter will run for 2 minutes")
    print("=" * 50)

    # Initialize camera
    camera = None
    try:
        camera = create_frame_source(config)
        print("Camera initialized")

        # Get first frame to determine size
        frame_gen = camera.frames()
        frame, _ = next(frame_gen)
        h, w = frame.shape[:2]
        print(f"Frame size: {w}x{h}")

        # Create counter (will be updated with new line)
        counter = LineCounter(config)
        _calib_state["config"] = config
        _calib_state["counter"] = counter

        # Create window
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Calibration", mouse_callback)

        print("\nClick two points to define the line...")

        # Wait for two points
        points = []
        start_time = cv2.getTickCount()

        while len(points) < 2:
            frame, _ = next(frame_gen)

            # Draw current points
            display = frame.copy()
            for i, pt in enumerate(_calib_state["points"]):
                cv2.circle(display, pt, 5, (0, 255, 0), -1)
                cv2.putText(
                    display,
                    f"P{i+1}",
                    (pt[0] + 10, pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # Draw line if both points set
            if len(_calib_state["points"]) == 2:
                cv2.line(display, _calib_state["points"][0], _calib_state["points"][1], (0, 255, 0), 2)

            cv2.imshow("Calibration", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Cancelled")
                return 1

            points = _calib_state["points"]

        # Update config with new line
        config.counting.line.start = list(points[0])
        config.counting.line.end = list(points[1])
        config.to_yaml(args.config)
        print(f"\nLine saved to config: {points[0]} -> {points[1]}")

        # Update counter
        counter.line_start = points[0]
        counter.line_end = points[1]

        # Run live counter for specified duration
        print(f"\nRunning live counter for {args.duration} seconds...")
        print("Press 'q' to quit early")

        end_time = start_time + args.duration * cv2.getTickFrequency()
        frame_count = 0

        while cv2.getTickCount() < end_time:
            frame, _ = next(frame_gen)

            # Simple mock detections for calibration (moving box)
            # In real use, this would come from inference
            mock_x = int(100 + (frame_count % 200))
            mock_y = h // 2
            mock_box = (mock_x - 30, mock_y - 30, mock_x + 30, mock_y + 30)
            mock_track = (*mock_box, 0.8, 41, 1)  # track_id=1

            # Update counter
            from datetime import datetime, timezone

            counter.update([mock_track], datetime.now(timezone.utc))
            totals = counter.totals()

            # Draw overlay
            stats = {"in": totals["in"], "out": totals["out"], "net": totals["net"], "fps": 30.0}
            display = annotate(frame, [mock_track], counter, stats)

            cv2.imshow("Calibration", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            frame_count += 1

        # Save calibration screenshot
        safe_makedirs(config.ops.debug_dir)
        calib_path = Path(config.ops.debug_dir) / "calib.png"
        cv2.imwrite(str(calib_path), display)
        print(f"\nCalibration screenshot saved: {calib_path}")

        print("\nCalibration complete!")
        print(f"Final counts - IN: {totals['in']}, OUT: {totals['out']}, NET: {totals['net']}")

    except KeyboardInterrupt:
        print("\nCancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if camera:
            camera.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())

