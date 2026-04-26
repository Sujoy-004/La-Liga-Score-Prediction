from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio
import random
from src.domain.models import PulseUpdate

class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

async def handle_pulse_stream(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_text(json.dumps({
            "type": "CONNECTION_ESTABLISHED", 
            "data": "Tactical DDD Stream Active"
        }))
        
        base_prob = 0.52
        while True:
            drift = random.uniform(-0.015, 0.015)
            base_prob = max(0.1, min(0.9, base_prob + drift))
            
            update = PulseUpdate(
                match="Real Madrid vs Barcelona",
                home_win_prob=round(base_prob, 4),
                away_win_prob=round(1 - base_prob, 4),
                event=random.choice([
                    "High press from Barcelona",
                    "Real Madrid transitioning fast",
                    "Midfield battle intensifying",
                    "Normal play"
                ]),
                timestamp=asyncio.get_event_loop().time()
            )
            
            await websocket.send_text(json.dumps({
                "type": "PROBABILITY_UPDATE",
                **update.__dict__
            }))
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
