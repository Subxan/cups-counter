"""Nightly jobs: CSV export, parameter tuning, POS reconciliation."""

import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import AppConfig
from core.storage import Storage
from core.tuner import ParameterTuner

logger = logging.getLogger(__name__)


class NightlyJobs:
    """Scheduler for nightly maintenance jobs."""

    def __init__(self, config):
        self.config = config
        self.storage = Storage(config)
        self.tuner = ParameterTuner(config) if config.tuner.enabled else None

    def export_yesterday_csv(self):
        """Export CSV for yesterday."""
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info(f"Exporting CSV for {yesterday}")

        try:
            csv_path = self.storage.export_csv(yesterday)
            logger.info(f"CSV exported: {csv_path}")
            return csv_path
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            return None

    def run_tuner(self, clip_path: str | Path = None):
        """Run parameter tuning on a clip."""
        if not self.tuner:
            logger.info("Tuner disabled")
            return

        if clip_path is None:
            # Use yesterday's debug clip if available
            debug_dir = Path(self.config.ops.debug_dir)
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            clip_path = debug_dir / f"clip_{yesterday}.mp4"

            if not clip_path.exists():
                logger.warning(f"No clip found for tuning: {clip_path}")
                return

        logger.info(f"Running parameter tuner on {clip_path}")

        # This would need the actual inference/tracking functions
        # For now, just log
        logger.info("Tuner would run here (needs integration with edge_service)")

    def pos_reconciliation(self):
        """Reconcile counts with POS data."""
        if not self.config.pos.recon_enabled:
            return

        pos_dir = Path(self.config.pos.pos_csv_dir)
        if not pos_dir.exists():
            logger.warning(f"POS directory not found: {pos_dir}")
            return

        # Find POS CSV for yesterday
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        pos_csv = pos_dir / f"{yesterday}_pos.csv"

        if not pos_csv.exists():
            logger.info(f"No POS data for {yesterday}")
            return

        # Compare counts (simplified)
        logger.info(f"POS reconciliation for {yesterday} (placeholder)")

    def run_all(self):
        """Run all nightly jobs."""
        logger.info("Starting nightly jobs...")

        # Export CSV
        self.export_yesterday_csv()

        # Run tuner
        if self.config.tuner.enabled:
            self.run_tuner()

        # POS reconciliation
        self.pos_reconciliation()

        logger.info("Nightly jobs complete")


def main():
    """Standalone nightly jobs runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Run nightly maintenance jobs")
    parser.add_argument("--config", type=str, default="configs/site-default.yaml")
    args = parser.parse_args()

    config = AppConfig.from_yaml(args.config)
    jobs = NightlyJobs(config)
    jobs.run_all()


if __name__ == "__main__":
    main()

