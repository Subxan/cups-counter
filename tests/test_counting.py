"""Tests for line crossing counting logic."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone

from core.counting import LineCounter
from core.config import AppConfig, CountingConfig, LineConfig


def test_basic_crossing():
    """Test basic line crossing detection."""
    config = AppConfig()
    config.counting.line.start = [100, 100]
    config.counting.line.end = [200, 100]
    config.counting.direction = "bar_to_counter"

    counter = LineCounter(config)

    # Track moving from bar side (below) to counter side (above)
    # Start below line
    track1 = (150.0, 120.0, 160.0, 130.0, 0.8, 41, 1)
    counter.update([track1], datetime.now(timezone.utc))

    # Move across line
    track2 = (150.0, 90.0, 160.0, 100.0, 0.8, 41, 1)
    events = counter.update([track2], datetime.now(timezone.utc))

    assert len(events) == 1
    assert events[0]["direction"] == "in"
    assert events[0]["track_id"] == 1

    totals = counter.totals()
    assert totals["in"] == 1
    assert totals["out"] == 0
    assert totals["net"] == 1


def test_hysteresis():
    """Test that hysteresis prevents jitter."""
    config = AppConfig()
    config.counting.line.start = [100, 100]
    config.counting.line.end = [200, 100]
    config.counting.hysteresis_px = 10

    counter = LineCounter(config)

    # Track near line (within hysteresis)
    track1 = (150.0, 95.0, 160.0, 105.0, 0.8, 41, 1)
    events1 = counter.update([track1], datetime.now(timezone.utc))
    assert len(events1) == 0  # Should not count

    # Move slightly but still near line
    track2 = (150.0, 98.0, 160.0, 108.0, 0.8, 41, 1)
    events2 = counter.update([track2], datetime.now(timezone.utc))
    assert len(events2) == 0  # Still within hysteresis


def test_min_visible_frames():
    """Test that min_visible_frames prevents premature counting."""
    config = AppConfig()
    config.counting.line.start = [100, 100]
    config.counting.line.end = [200, 100]
    config.counting.min_visible_frames = 5

    counter = LineCounter(config)

    # Track appears and crosses immediately
    track1 = (150.0, 120.0, 160.0, 130.0, 0.8, 41, 1)
    counter.update([track1], datetime.now(timezone.utc))

    # Cross line on frame 2 (should not count yet)
    track2 = (150.0, 90.0, 160.0, 100.0, 0.8, 41, 1)
    events = counter.update([track2], datetime.now(timezone.utc))
    assert len(events) == 0  # Not enough visible frames

    # Continue tracking for required frames
    for i in range(4):
        track = (150.0, 90.0 + i, 160.0, 100.0 + i, 0.8, 41, 1)
        counter.update([track], datetime.now(timezone.utc))

    # Now should have enough frames, but already crossed
    # (This test shows the limitation - would need to cross again)
    # For now, just verify the counter tracks frames


def test_double_crossing_prevention():
    """Test that same crossing is not counted twice."""
    config = AppConfig()
    config.counting.line.start = [100, 100]
    config.counting.line.end = [200, 100]

    counter = LineCounter(config)

    # Cross line
    track1 = (150.0, 120.0, 160.0, 130.0, 0.8, 41, 1)
    counter.update([track1], datetime.now(timezone.utc))

    track2 = (150.0, 90.0, 160.0, 100.0, 0.8, 41, 1)
    events1 = counter.update([track2], datetime.now(timezone.utc))
    assert len(events1) == 1

    # Move back and forth near line (should not count again)
    track3 = (150.0, 95.0, 160.0, 105.0, 0.8, 41, 1)
    events2 = counter.update([track3], datetime.now(timezone.utc))
    assert len(events2) == 0  # Should not count again


def test_bidirectional_counting():
    """Test counting in both directions."""
    config = AppConfig()
    config.counting.line.start = [100, 100]
    config.counting.line.end = [200, 100]

    counter = LineCounter(config)

    # Cross in
    track1 = (150.0, 120.0, 160.0, 130.0, 0.8, 41, 1)
    counter.update([track1], datetime.now(timezone.utc))

    track2 = (150.0, 90.0, 160.0, 100.0, 0.8, 41, 1)
    events1 = counter.update([track2], datetime.now(timezone.utc))
    assert len(events1) == 1
    assert events1[0]["direction"] == "in"

    # Cross out
    track3 = (150.0, 80.0, 160.0, 90.0, 0.8, 41, 1)
    counter.update([track3], datetime.now(timezone.utc))

    track4 = (150.0, 110.0, 160.0, 120.0, 0.8, 41, 1)
    events2 = counter.update([track4], datetime.now(timezone.utc))
    assert len(events2) == 1
    assert events2[0]["direction"] == "out"

    totals = counter.totals()
    assert totals["in"] == 1
    assert totals["out"] == 1
    assert totals["net"] == 0


def test_reset():
    """Test counter reset."""
    config = AppConfig()
    counter = LineCounter(config)

    # Add some counts
    track1 = (150.0, 120.0, 160.0, 130.0, 0.8, 41, 1)
    counter.update([track1], datetime.now(timezone.utc))
    track2 = (150.0, 90.0, 160.0, 100.0, 0.8, 41, 1)
    counter.update([track2], datetime.now(timezone.utc))

    totals_before = counter.totals()
    assert totals_before["in"] > 0

    counter.reset()
    totals_after = counter.totals()
    assert totals_after["in"] == 0
    assert totals_after["out"] == 0
    assert totals_after["net"] == 0


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])

