import asyncio
import json
from typing import AsyncIterator


class WebSocketHandler:
    def __init__(self):
        self._connections: list = []

    def register(self, websocket):
        self._connections.append(websocket)

    def unregister(self, websocket):
        if websocket in self._connections:
            self._connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self._connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unregister(ws)

    def __len__(self) -> int:
        return len(self._connections)
