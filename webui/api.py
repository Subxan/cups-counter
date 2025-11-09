"""FastAPI web server for UI and API."""

import logging
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, Query, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from webui.ws import get_ws_manager

logger = logging.getLogger(__name__)

app = FastAPI(title="Cups Counter API")

# Shared state (updated by edge_service)
_shared_state = None


def start_web_server(config, shared_state):
    """Start web server in background thread."""
    global _shared_state
    _shared_state = shared_state
    shared_state["start_time"] = time.time()

    def run_server():
        import uvicorn
        import asyncio

        # Start WebSocket broadcast when server is ready
        async def on_startup():
            manager = get_ws_manager(shared_state)
            asyncio.create_task(manager.broadcast())

        # Use lifespan events (FastAPI 0.93+)
        # For compatibility, we'll start broadcast in a separate way
        # The broadcast will start when first WebSocket connects

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=config.ui.http_port,
            log_level="warning",
        )

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    return server_thread


@app.get("/")
async def root():
    """Serve main HTML page."""
    return FileResponse("webui/static/index.html")


@app.get("/healthz")
async def healthz():
    """Health check endpoint with detailed status."""
    if _shared_state is None:
        return {"ok": False, "reason": "Service not initialized"}

    reasons = []
    stats = _shared_state.get("stats", {})
    drift_metrics = _shared_state.get("drift_metrics", {})

    # Check FPS
    if stats.get("fps", 0) < 10:
        reasons.append("Low FPS")

    # Check drift
    if drift_metrics.get("camera_shifted"):
        reasons.append("Camera drift detected")
    if drift_metrics.get("lighting_bad"):
        reasons.append("Lighting issues")

    ok = len(reasons) == 0
    return {"ok": ok, "reasons": reasons, "fps": stats.get("fps", 0)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live stats."""
    if _shared_state is None:
        await websocket.close(code=1003, reason="Service not initialized")
        return

    manager = get_ws_manager(_shared_state)
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except Exception as e:
        logger.debug(f"WebSocket error: {e}")
    finally:
        await manager.disconnect(websocket)


@app.get("/stats")
async def get_stats():
    """Get current statistics."""
    if _shared_state is None:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)

    stats = _shared_state.get("stats", {})
    return {
        "in": stats.get("in", 0),
        "out": stats.get("out", 0),
        "net": stats.get("net", 0),
        "fps": stats.get("fps", 0.0),
        "uptime": time.time() - _shared_state.get("start_time", time.time()),
        "last_event_ts": _shared_state.get("last_event_ts"),
    }


@app.get("/events")
async def get_events(day: str | None = Query(None, description="Date filter (YYYY-MM-DD)")):
    """Get events (would need storage access - placeholder)."""
    # In real implementation, this would query the storage
    return {"events": [], "message": "Storage integration needed"}


@app.get("/live.jpg")
async def get_live_frame():
    """Get current annotated frame as JPEG."""
    if _shared_state is None:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)

    frame = _shared_state.get("frame")
    if frame is None:
        # Return placeholder
        placeholder = cv2.imread("webui/static/placeholder.jpg")
        if placeholder is None:
            placeholder = cv2.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "No frame available",
                (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        _, buffer = cv2.imencode(".jpg", placeholder)
        return StreamingResponse(
            iter([buffer.tobytes()]),
            media_type="image/jpeg",
        )

    # Encode frame as JPEG
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return StreamingResponse(
        iter([buffer.tobytes()]),
        media_type="image/jpeg",
    )


@app.get("/autocal/proposals")
async def get_autocal_proposals():
    """Get auto-calibration proposals."""
    if _shared_state is None:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)

    candidates = _shared_state.get("autocal_candidates", [])
    return {"candidates": candidates}


@app.post("/autocal/apply")
async def apply_autocal_proposal(index: int = Query(0, ge=0)):
    """Apply an auto-calibration proposal."""
    if _shared_state is None:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)

    candidates = _shared_state.get("autocal_candidates", [])
    if index >= len(candidates):
        return JSONResponse({"error": "Invalid proposal index"}, status_code=400)

    # This would trigger a config update in edge_service
    # For now, just acknowledge
    _shared_state["autocal_apply_request"] = index
    return {"status": "applied", "proposal": candidates[index]}


@app.get("/drift/status")
async def get_drift_status():
    """Get drift monitoring status."""
    if _shared_state is None:
        return JSONResponse({"error": "Service not initialized"}, status_code=503)

    drift_metrics = _shared_state.get("drift_metrics", {})
    return {
        "ssim": drift_metrics.get("ssim", 1.0),
        "edge_iou": drift_metrics.get("edge_iou", 1.0),
        "brightness_var": drift_metrics.get("brightness_var", 0.0),
        "camera_shifted": drift_metrics.get("camera_shifted", False),
        "lighting_bad": drift_metrics.get("lighting_bad", False),
        "last_recal_time": _shared_state.get("last_recal_time"),
    }


@app.get("/admin/audit/thumbnails")
async def get_audit_thumbnails(day: str | None = Query(None)):
    """Get audit thumbnails for low-confidence events."""
    # Placeholder - would query storage for low-confidence events
    return {"thumbnails": [], "message": "Audit feature placeholder"}


# Mount static files
try:
    app.mount("/static", StaticFiles(directory="webui/static"), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

