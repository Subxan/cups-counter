"""Tests for parameter tuning."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import AppConfig
from core.tuner import ParameterTuner


def test_tuner_init():
    """Test tuner initialization."""
    config = AppConfig()
    tuner = ParameterTuner(config)
    assert tuner.enabled == config.tuner.enabled


def test_grid_generation():
    """Test parameter grid generation."""
    config = AppConfig()
    tuner = ParameterTuner(config)

    grid = tuner.generate_grid()
    assert len(grid) > 0

    # Check all combinations are present
    conf_values = config.tuner.grid.conf_thresh
    match_values = config.tuner.grid.match_thresh
    area_values = config.tuner.grid.min_box_area

    expected_count = len(conf_values) * len(match_values) * len(area_values)
    assert len(grid) == expected_count

    # Check first combination
    first = grid[0]
    assert "conf_thresh" in first
    assert "match_thresh" in first
    assert "min_box_area" in first


def test_scoring_stable_crossings():
    """Test scoring for stable crossings."""
    config = AppConfig()
    config.tuner.optimize_for = "stable_crossings"
    tuner = ParameterTuner(config)

    # Ideal case: many unique tracks, each crossing once
    events = [
        {"track_id": 1, "direction": "in"},
        {"track_id": 2, "direction": "in"},
        {"track_id": 3, "direction": "out"},
    ]
    tracks = {1: 1, 2: 1, 3: 1}  # Each crossed once

    score = tuner.score_run(events, tracks)
    assert score > 0

    # Bad case: same track crossing multiple times
    tracks_bad = {1: 3, 2: 1}  # Track 1 crossed 3 times
    score_bad = tuner.score_run(events, tracks_bad)
    assert score_bad < score  # Should score lower


def test_scoring_min_double_counts():
    """Test scoring for minimizing double counts."""
    config = AppConfig()
    config.tuner.optimize_for = "min_double_counts"
    tuner = ParameterTuner(config)

    # Good case: unique tracks, no double counting
    events = [
        {"track_id": 1, "direction": "in"},
        {"track_id": 2, "direction": "in"},
    ]
    tracks = {1: 1, 2: 1}

    score = tuner.score_run(events, tracks)
    assert score > 0

    # Bad case: double counting
    tracks_bad = {1: 2}  # Track 1 crossed twice
    score_bad = tuner.score_run(events, tracks_bad)
    assert score_bad < score


def test_save_best_profile():
    """Test saving best profile."""
    config = AppConfig()
    tuner = ParameterTuner(config)

    best_params = {
        "conf_thresh": 0.35,
        "match_thresh": 0.80,
        "min_box_area": 150,
    }

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        output_path = f.name

    try:
        tuner.save_best_profile(best_params, output_path)
        # Check file exists
        assert Path(output_path).exists()
    finally:
        Path(output_path).unlink()


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])

