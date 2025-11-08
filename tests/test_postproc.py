"""Tests for post-processing (NMS, filtering)."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import AppConfig
from core.postproc import PostProcessor


def test_confidence_filtering():
    """Test that low-confidence detections are filtered."""
    config = AppConfig()
    config.model.conf_thresh = 0.5
    postproc = PostProcessor(config)

    detections = [
        (100.0, 100.0, 150.0, 150.0, 0.8, 41),  # High conf
        (200.0, 200.0, 250.0, 250.0, 0.3, 41),  # Low conf
    ]

    result = postproc.process(detections)
    assert len(result) == 1
    assert result[0][4] >= 0.5  # Only high conf kept


def test_class_filtering():
    """Test that only filtered classes are kept."""
    config = AppConfig()
    config.model.class_filter = ["cup"]
    postproc = PostProcessor(config)

    # Assuming cup is class 41, person is class 0
    detections = [
        (100.0, 100.0, 150.0, 150.0, 0.8, 41),  # Cup
        (200.0, 200.0, 250.0, 250.0, 0.8, 0),   # Person
    ]

    result = postproc.process(detections)
    # Should only keep cup (class 41)
    assert len(result) == 1
    assert result[0][5] == 41


def test_nms():
    """Test that NMS removes overlapping boxes."""
    config = AppConfig()
    config.model.iou_thresh = 0.5
    postproc = PostProcessor(config)

    # Two overlapping boxes (high IoU)
    detections = [
        (100.0, 100.0, 150.0, 150.0, 0.9, 41),
        (105.0, 105.0, 155.0, 155.0, 0.8, 41),  # Overlaps with first
    ]

    result = postproc.process(detections)
    # NMS should keep only the higher confidence one
    assert len(result) == 1
    assert result[0][4] == 0.9  # Higher confidence kept


def test_nms_non_overlapping():
    """Test that non-overlapping boxes are both kept."""
    config = AppConfig()
    config.model.iou_thresh = 0.5
    postproc = PostProcessor(config)

    # Two non-overlapping boxes
    detections = [
        (100.0, 100.0, 150.0, 150.0, 0.9, 41),
        (300.0, 300.0, 350.0, 350.0, 0.8, 41),  # Far from first
    ]

    result = postproc.process(detections)
    # Both should be kept
    assert len(result) == 2


def test_max_detections():
    """Test that max_detections limit is enforced."""
    config = AppConfig()
    config.model.max_detections = 3
    postproc = PostProcessor(config)

    # Many detections
    detections = [
        (100.0 + i * 10, 100.0, 150.0 + i * 10, 150.0, 0.9, 41)
        for i in range(10)
    ]

    result = postproc.process(detections)
    assert len(result) <= 3


def test_empty_input():
    """Test that empty input returns empty result."""
    config = AppConfig()
    postproc = PostProcessor(config)

    result = postproc.process([])
    assert len(result) == 0


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])

