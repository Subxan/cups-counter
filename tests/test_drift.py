"""Tests for drift monitoring."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from core.config import AppConfig
from core.drift import DriftMonitor


def test_drift_monitor_init():
    """Test drift monitor initialization."""
    config = AppConfig()
    monitor = DriftMonitor(config)
    assert monitor.enabled == config.drift.enabled


def test_reference_setting():
    """Test setting reference frame."""
    config = AppConfig()
    monitor = DriftMonitor(config)

    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    monitor.set_reference(frame)

    assert monitor.reference_frame is not None
    assert monitor.reference_edges is not None
    assert monitor.reference_brightness is not None


def test_drift_detection():
    """Test drift detection on altered frames."""
    config = AppConfig()
    config.drift.ssim_threshold = 0.90
    monitor = DriftMonitor(config)

    # Set reference
    reference = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    monitor.set_reference(reference)

    # Same frame (no drift)
    metrics1 = monitor.update(reference.copy())
    assert metrics1["ssim"] > 0.95  # Should be very similar
    assert not metrics1["camera_shifted"]

    # Shifted frame (drift)
    shifted = np.roll(reference, 50, axis=1)  # Shift horizontally
    metrics2 = monitor.update(shifted)
    assert metrics2["ssim"] < 0.90  # Should detect drift
    assert metrics2["camera_shifted"]


def test_brightness_detection():
    """Test brightness variance detection."""
    config = AppConfig()
    config.drift.brightness_var_min = 6.0
    monitor = DriftMonitor(config)

    # Normal frame
    normal_frame = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
    monitor.set_reference(normal_frame)

    # Very dark frame (low variance)
    dark_frame = np.full((480, 640, 3), 10, dtype=np.uint8)
    metrics = monitor.update(dark_frame)
    assert metrics["brightness_var"] < 6.0
    assert metrics["lighting_bad"]


def test_recalibration_cooldown():
    """Test recalibration cooldown."""
    config = AppConfig()
    config.drift.min_minutes_between_recal = 60
    monitor = DriftMonitor(config)

    # Set drift
    monitor.drift_detected = True

    # Should not recalibrate immediately
    assert not monitor.should_recalibrate()

    # Mark as recalibrated
    monitor.mark_recalibrated()
    assert monitor.last_recal_time is not None

    # Should not recalibrate again within cooldown
    assert not monitor.should_recalibrate()


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])

