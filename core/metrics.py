"""Prometheus metrics exporter."""

import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from prometheus_client import Counter, Gauge, generate_latest

logger = logging.getLogger(__name__)


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics."""

    def do_GET(self):
        """Handle GET request."""
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(generate_latest())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress access logs."""
        pass


class Metrics:
    """Prometheus metrics collector."""

    def __init__(self, port: int = 9109):
        self.port = port
        self.server = None
        self.server_thread = None

        # Define metrics
        self.cups_in_total = Counter("cups_in_total", "Total cups counted in")
        self.cups_out_total = Counter("cups_out_total", "Total cups counted out")
        self.frames_total = Counter("frames_total", "Total frames processed")
        self.fps = Gauge("fps", "Current frames per second")
        self.active_tracks = Gauge("active_tracks", "Number of active tracks")
        self.dropped_frames = Counter("dropped_frames", "Total dropped frames")

        # Drift metrics
        self.drift_ssim = Gauge("drift_ssim", "Structural similarity index (drift)")
        self.edge_iou = Gauge("edge_iou", "Edge map IoU (drift)")
        self.brightness_var = Gauge("brightness_var", "Brightness variance")

        # ROI metrics
        self.roi_coverage = Gauge("roi_coverage", "ROI coverage ratio")

        # Auto-calibration metrics
        self.autocal_runs_total = Counter("autocal_runs_total", "Total auto-calibration runs")
        self.autocal_applied_total = Counter("autocal_applied_total", "Total auto-applied calibrations")
        self.recalibrations_total = Counter("recalibrations_total", "Total recalibrations triggered")

        self._start_server()

    def _start_server(self):
        """Start metrics HTTP server in background thread."""
        try:
            self.server = HTTPServer(("", self.port), MetricsHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            logger.info(f"Metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")

    def update_fps(self, fps_value: float):
        """Update FPS gauge."""
        self.fps.set(fps_value)

    def update_active_tracks(self, count: int):
        """Update active tracks gauge."""
        self.active_tracks.set(count)

    def increment_in(self, count: int = 1):
        """Increment in counter."""
        self.cups_in_total.inc(count)

    def increment_out(self, count: int = 1):
        """Increment out counter."""
        self.cups_out_total.inc(count)

    def increment_frames(self, count: int = 1):
        """Increment frames counter."""
        self.frames_total.inc(count)

    def increment_dropped(self, count: int = 1):
        """Increment dropped frames counter."""
        self.dropped_frames.inc(count)

    def update_drift_ssim(self, value: float):
        """Update drift SSIM gauge."""
        self.drift_ssim.set(value)

    def update_edge_iou(self, value: float):
        """Update edge IoU gauge."""
        self.edge_iou.set(value)

    def update_brightness_var(self, value: float):
        """Update brightness variance gauge."""
        self.brightness_var.set(value)

    def update_roi_coverage(self, value: float):
        """Update ROI coverage gauge."""
        self.roi_coverage.set(value)

    def increment_autocal_runs(self, count: int = 1):
        """Increment auto-calibration runs counter."""
        self.autocal_runs_total.inc(count)

    def increment_autocal_applied(self, count: int = 1):
        """Increment auto-applied calibrations counter."""
        self.autocal_applied_total.inc(count)

    def increment_recalibrations(self, count: int = 1):
        """Increment recalibrations counter."""
        self.recalibrations_total.inc(count)

    def stop(self):
        """Stop metrics server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logger.info("Metrics server stopped")

