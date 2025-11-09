"""Tests for auto-calibration."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apps.auto_calibrate import AutoCalibrator
from core.config import AppConfig


def test_line_proposal():
    """Test that line proposals are generated correctly."""
    config = AppConfig()
    config.autocal.enabled = True
    config.autocal.warmup_seconds = 5  # Short for testing

    calibrator = AutoCalibrator(config)
    assert calibrator.enabled

    # Test scoring function
    line = (100, 100, 200, 100)
    trajectories = {
        1: [(50, 120), (60, 110), (70, 100), (80, 90)],  # Crosses line
        2: [(150, 80), (160, 90), (170, 100)],  # Near line
    }
    frame_size = (1280, 720)

    score = calibrator._score_line(line, trajectories, frame_size)
    assert score > 0  # Should have positive score


def test_flow_direction():
    """Test flow direction detection."""
    config = AppConfig()
    calibrator = AutoCalibrator(config)

    # Simulate trajectories moving up (bar_to_counter)
    trajectories = {
        1: [(100, 200), (100, 180), (100, 160), (100, 140)],
        2: [(200, 220), (200, 200), (200, 180)],
    }

    flow_stats, direction = calibrator.compute_flow_vectors(trajectories)
    assert direction in ["bar_to_counter", "counter_to_bar"]
    assert "bar_to_counter" in flow_stats
    assert "counter_to_bar" in flow_stats


def test_segment_intersect():
    """Test line segment intersection."""
    config = AppConfig()
    calibrator = AutoCalibrator(config)

    # Two intersecting segments
    seg1 = ((50, 50), (150, 150))
    seg2_start = (50, 150)
    seg2_end = (150, 50)

    result = calibrator._segments_intersect(seg1, seg2_start, seg2_end)
    assert result is True

    # Non-intersecting segments
    seg3 = ((10, 10), (20, 20))
    seg4_start = (100, 100)
    seg4_end = (110, 110)

    result = calibrator._segments_intersect(seg3, seg4_start, seg4_end)
    assert result is False


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])

