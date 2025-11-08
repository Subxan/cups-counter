"""Utility functions for time, geometry, and file operations."""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import pytz


def now_utc() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def to_iso8601(dt: datetime) -> str:
    """Convert datetime to ISO8601 string."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def utc_to_local(utc_dt: datetime, tz_name: str = "Asia/Baku") -> datetime:
    """Convert UTC datetime to local timezone."""
    tz = pytz.timezone(tz_name)
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    return utc_dt.astimezone(tz)


def centroid(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Calculate centroid of bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def segment_intersect(
    seg1_start: Tuple[float, float],
    seg1_end: Tuple[float, float],
    seg2_start: Tuple[float, float],
    seg2_end: Tuple[float, float],
) -> Tuple[float, float] | None:
    """Find intersection point of two line segments, or None if no intersection.

    Returns intersection point (x, y) or None.
    """
    x1, y1 = seg1_start
    x2, y2 = seg1_end
    x3, y3 = seg2_start
    x4, y4 = seg2_end

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # Parallel lines

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)

    return None


def point_to_line_distance(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> float:
    """Calculate perpendicular distance from point to line segment."""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    # Vector from line_start to line_end
    dx = x2 - x1
    dy = y2 - y1
    line_len_sq = dx * dx + dy * dy

    if line_len_sq < 1e-10:
        # Line is a point
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5

    # Project point onto line
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / line_len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5


def safe_makedirs(path: str | Path) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def rotate_file_path(base_path: str | Path, max_files: int = 10) -> Path:
    """Generate rotated file path (base.0, base.1, ..., base.N-1).

    Returns path for next available slot.
    """
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    # Find first available slot
    for i in range(max_files):
        candidate = base_path.parent / f"{base_path.stem}.{i}{base_path.suffix}"
        if not candidate.exists():
            return candidate

    # If all slots full, use oldest (0)
    return base_path.parent / f"{base_path.stem}.0{base_path.suffix}"


def parse_bbox(bbox: List[float] | Tuple[float, ...]) -> Tuple[float, float, float, float]:
    """Parse bbox to (x1, y1, x2, y2) format."""
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 elements, got {len(bbox)}")
    return tuple(bbox[:4])

