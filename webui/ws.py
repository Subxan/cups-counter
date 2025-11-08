"""WebSocket server for live updates."""

import asyncio
import json
import logging
import time
from typing import Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manage WebSocket connections for live stats."""

    def __init__(self, shared_state):
        self.shared_state = shared_state
        self.connections: Set[WebSocket] = set()
        self.broadcast_task = None

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.connections)}")
        
        # Start broadcast if not already running
        if self.broadcast_task is None:
            import asyncio
            self.broadcast_task = asyncio.create_task(self.broadcast())

    async def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        self.connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.connections)}")

    async def broadcast(self):
        """Broadcast stats to all connected clients."""
        while True:
            try:
                if not self.connections:
                    await asyncio.sleep(0.5)
                    continue

                stats = self.shared_state.get("stats", {})
                message = {
                    "in": stats.get("in", 0),
                    "out": stats.get("out", 0),
                    "net": stats.get("net", 0),
                    "fps": stats.get("fps", 0.0),
                    "timestamp": time.time(),
                }

                # Send to all connections
                disconnected = set()
                for connection in self.connections:
                    try:
                        await connection.send_json(message)
                    except Exception as e:
                        logger.warning(f"Error sending to WebSocket: {e}")
                        disconnected.add(connection)

                # Remove disconnected
                for conn in disconnected:
                    self.connections.discard(conn)

                # Broadcast at 2-5 Hz
                await asyncio.sleep(0.2)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                await asyncio.sleep(1.0)


# Global manager instance
_ws_manager = None


def get_ws_manager(shared_state):
    """Get or create WebSocket manager."""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager(shared_state)
    return _ws_manager

