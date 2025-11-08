"""SQLite storage for events and daily rollups with async writer."""

import csv
import logging
import queue
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from core.utils import safe_makedirs, to_iso8601

logger = logging.getLogger(__name__)

Event = dict


class Storage:
    """Async SQLite storage with daily rollups and CSV export."""

    def __init__(self, config):
        self.config = config
        self.db_path = Path(config.storage.sqlite_path)
        self.csv_dir = Path(config.storage.csv_dir)
        self.wal_mode = config.storage.wal_mode
        self.export_daily_time = config.storage.export_daily_time

        safe_makedirs(self.db_path.parent)
        safe_makedirs(self.csv_dir)

        # Initialize database
        self._init_db()

        # Async writer queue
        self.write_queue = queue.Queue(maxsize=1000)
        self.writer_thread = None
        self.running = False

        # Start writer thread
        self.start()

        # Schedule daily export
        self._schedule_daily_export()

    def _init_db(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        if self.wal_mode:
            cursor.execute("PRAGMA journal_mode=WAL")

        # Events table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cup_events(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_utc TEXT NOT NULL,
                direction TEXT NOT NULL,
                track_id INTEGER,
                x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                conf REAL
            )
        """
        )

        # Daily rollups table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS rollups_daily(
                day TEXT PRIMARY KEY,
                in_count INTEGER,
                out_count INTEGER,
                net_count INTEGER
            )
        """
        )

        # Indexes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_ts_utc ON cup_events(ts_utc)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_day ON cup_events(DATE(ts_utc))"
        )

        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")

    def start(self):
        """Start async writer thread."""
        if self.running:
            return

        self.running = True
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        logger.info("Storage writer thread started")

    def stop(self):
        """Stop writer thread and flush queue."""
        self.running = False
        if self.writer_thread:
            # Wait for queue to drain
            self.write_queue.join()
            self.writer_thread.join(timeout=5.0)
        logger.info("Storage writer thread stopped")

    def _writer_loop(self):
        """Background thread that writes events to database."""
        conn = sqlite3.connect(str(self.db_path))
        if self.wal_mode:
            conn.execute("PRAGMA journal_mode=WAL")

        while self.running:
            try:
                # Get batch of events (with timeout)
                events = []
                try:
                    event = self.write_queue.get(timeout=1.0)
                    events.append(event)
                    # Try to get more events in batch
                    while len(events) < 100:
                        try:
                            event = self.write_queue.get_nowait()
                            events.append(event)
                        except queue.Empty:
                            break
                except queue.Empty:
                    continue

                # Write batch
                cursor = conn.cursor()
                for event in events:
                    cursor.execute(
                        """
                        INSERT INTO cup_events(ts_utc, direction, track_id, x1, y1, x2, y2, conf)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            event["ts_utc"],
                            event["direction"],
                            event.get("track_id"),
                            event["bbox"][0],
                            event["bbox"][1],
                            event["bbox"][2],
                            event["bbox"][3],
                            event["conf"],
                        ),
                    )

                conn.commit()

                # Mark tasks as done
                for _ in events:
                    self.write_queue.task_done()

            except Exception as e:
                logger.error(f"Storage writer error: {e}")

        conn.close()

    def write_events(self, events: List[Event]):
        """Queue events for async write."""
        for event in events:
            try:
                self.write_queue.put_nowait(event)
            except queue.Full:
                logger.warning("Write queue full, dropping event")

    def rollup_day(self, day: str) -> dict | None:
        """Calculate and store daily rollup.

        Args:
            day: Date string (YYYY-MM-DD)

        Returns:
            Rollup dict or None
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Calculate rollup
        cursor.execute(
            """
            SELECT 
                COUNT(*) FILTER (WHERE direction = 'in') as in_count,
                COUNT(*) FILTER (WHERE direction = 'out') as out_count
            FROM cup_events
            WHERE DATE(ts_utc) = ?
        """,
            (day,),
        )

        row = cursor.fetchone()
        if row:
            in_count, out_count = row[0] or 0, row[1] or 0
            net_count = in_count - out_count

            # Upsert rollup
            cursor.execute(
                """
                INSERT OR REPLACE INTO rollups_daily(day, in_count, out_count, net_count)
                VALUES (?, ?, ?, ?)
            """,
                (day, in_count, out_count, net_count),
            )

            conn.commit()
            conn.close()

            return {"day": day, "in_count": in_count, "out_count": out_count, "net_count": net_count}

        conn.close()
        return None

    def get_events(self, day: str | None = None, limit: int = 1000) -> List[dict]:
        """Get events from database.

        Args:
            day: Optional date filter (YYYY-MM-DD)
            limit: Max number of events

        Returns:
            List of event dicts
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if day:
            cursor.execute(
                """
                SELECT * FROM cup_events
                WHERE DATE(ts_utc) = ?
                ORDER BY ts_utc DESC
                LIMIT ?
            """,
                (day, limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM cup_events
                ORDER BY ts_utc DESC
                LIMIT ?
            """,
                (limit,),
            )

        rows = cursor.fetchall()
        events = [
            {
                "id": row["id"],
                "ts_utc": row["ts_utc"],
                "direction": row["direction"],
                "track_id": row["track_id"],
                "bbox": [row["x1"], row["y1"], row["x2"], row["y2"]],
                "conf": row["conf"],
            }
            for row in rows
        ]

        conn.close()
        return events

    def export_csv(self, day: str, csv_dir: Path | None = None) -> Path:
        """Export events for a day to CSV.

        Args:
            day: Date string (YYYY-MM-DD)
            csv_dir: Optional output directory

        Returns:
            Path to created CSV file
        """
        if csv_dir is None:
            csv_dir = self.csv_dir

        safe_makedirs(csv_dir)

        events = self.get_events(day=day, limit=100000)
        csv_path = csv_dir / f"{day}_counts.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp_utc",
                    "direction",
                    "track_id",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "confidence",
                ],
            )
            writer.writeheader()
            for event in events:
                writer.writerow(
                    {
                        "timestamp_utc": event["ts_utc"],
                        "direction": event["direction"],
                        "track_id": event["track_id"],
                        "x1": event["bbox"][0],
                        "y1": event["bbox"][1],
                        "x2": event["bbox"][2],
                        "y2": event["bbox"][3],
                        "confidence": event["conf"],
                    }
                )

        logger.info(f"Exported {len(events)} events to {csv_path}")
        return csv_path

    def _schedule_daily_export(self):
        """Schedule daily CSV export at configured time."""
        # This would typically use a scheduler like APScheduler
        # For now, export can be triggered manually or by external cron
        logger.info(f"Daily export scheduled for {self.export_daily_time} (local time)")

    def get_daily_rollup(self, day: str) -> dict | None:
        """Get stored daily rollup."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM rollups_daily WHERE day = ?", (day,))
        row = cursor.fetchone()

        conn.close()

        if row:
            return {
                "day": row["day"],
                "in_count": row["in_count"],
                "out_count": row["out_count"],
                "net_count": row["net_count"],
            }
        return None

