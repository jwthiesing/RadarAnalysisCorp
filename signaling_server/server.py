"""RadarAnalysisCorp signaling server.

Stateless-ish WebSocket relay used **only** for WebRTC SDP offer/answer
exchange so that peers can find each other and establish DataChannels. Once a
DataChannel is up, all game traffic goes peer-to-peer (host-and-spoke); this
server is no longer in the data path.

Wire protocol (JSON over WebSocket):

  Client → server:
    {"op": "host", "name": "<host display name>"}            # create room
    {"op": "join", "room": "<code>", "name": "..."}          # join existing
    {"op": "offer",  "to": "<peer_id>", "sdp": "..."}        # forward to peer
    {"op": "answer", "to": "<peer_id>", "sdp": "..."}
    {"op": "ice",    "to": "<peer_id>", "candidate": {...}}

  Server → client:
    {"op": "hosted", "room": "<code>", "host_id": "<id>"}
    {"op": "joined", "room": "<code>", "host_id": "...", "peer_id": "..."}
    {"op": "peer_joined", "peer_id": "...", "name": "..."}   # to host
    {"op": "forward", "from": "<peer_id>", ...payload}       # relayed offer/answer/ice
    {"op": "peer_left", "peer_id": "..."}
    {"op": "error", "message": "..."}

Run with:
    python -m signaling_server.server   # listens on 0.0.0.0:8765
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import string
import uuid
from dataclasses import dataclass, field

from aiohttp import WSMsgType, web

log = logging.getLogger(__name__)

ROOM_CODE_WORDS = (
    "STORM", "SUPERCELL", "MESO", "WALL", "FUNNEL", "HOOK", "RFD",
    "SHEAR", "OUTFLOW", "BOWE", "WEDGE", "ROPE", "ANVIL", "OVERSHOOT",
    "COUPLET", "TVS", "DEBRIS", "RADAR", "DRYLINE", "DERECHO",
    "OUTBREAK", "RING", "VAULT", "MARGIN", "TILT", "BUST", "BUDGE", "STREAK",
)


def _gen_room_code(rng: random.Random | None = None) -> str:
    """Generate a memorable room code like 'STORM-FROG-72'."""
    rng = rng or random
    word1 = rng.choice(ROOM_CODE_WORDS)
    word2 = rng.choice(ROOM_CODE_WORDS)
    number = rng.randint(10, 99)
    return f"{word1}-{word2}-{number}"


@dataclass
class Peer:
    id: str
    name: str
    ws: web.WebSocketResponse


@dataclass
class Room:
    code: str
    host_id: str
    peers: dict[str, Peer] = field(default_factory=dict)

    @property
    def host(self) -> Peer | None:
        return self.peers.get(self.host_id)


class SignalingServer:
    """In-memory rooms; reset on process restart. Suitable for one host process."""

    def __init__(self) -> None:
        self.rooms: dict[str, Room] = {}
        self.peer_room: dict[str, str] = {}

    async def handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        peer_id: str | None = None
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    await self._send(ws, {"op": "error", "message": "bad json"})
                    continue
                op = data.get("op")
                if op == "host":
                    peer_id = await self._on_host(ws, data)
                elif op == "join":
                    peer_id = await self._on_join(ws, data)
                elif op in ("offer", "answer", "ice"):
                    if peer_id is None:
                        await self._send(ws, {"op": "error", "message": "not in a room"})
                        continue
                    await self._on_forward(peer_id, data)
                else:
                    await self._send(ws, {"op": "error", "message": f"unknown op {op!r}"})
        finally:
            if peer_id is not None:
                await self._on_disconnect(peer_id)
        return ws

    # ---- operations ---------------------------------------------------

    async def _on_host(self, ws: web.WebSocketResponse, data: dict) -> str:
        name = str(data.get("name") or "Host")
        peer_id = uuid.uuid4().hex[:8]
        code = _gen_room_code()
        while code in self.rooms:
            code = _gen_room_code()
        peer = Peer(id=peer_id, name=name, ws=ws)
        self.rooms[code] = Room(code=code, host_id=peer_id, peers={peer_id: peer})
        self.peer_room[peer_id] = code
        log.info("Hosted room %s by %s (%s)", code, peer_id, name)
        await self._send(ws, {"op": "hosted", "room": code, "host_id": peer_id})
        return peer_id

    async def _on_join(self, ws: web.WebSocketResponse, data: dict) -> str | None:
        code = str(data.get("room") or "").upper()
        name = str(data.get("name") or "Peer")
        room = self.rooms.get(code)
        if room is None:
            await self._send(ws, {"op": "error", "message": f"no such room {code}"})
            return None
        peer_id = uuid.uuid4().hex[:8]
        room.peers[peer_id] = Peer(id=peer_id, name=name, ws=ws)
        self.peer_room[peer_id] = code
        log.info("Peer %s (%s) joined %s", peer_id, name, code)
        # Tell the joiner who's hosting
        await self._send(ws, {"op": "joined", "room": code, "host_id": room.host_id,
                              "peer_id": peer_id})
        # Tell the host who joined (so the host can initiate the WebRTC offer)
        host = room.host
        if host is not None:
            await self._send(host.ws, {"op": "peer_joined", "peer_id": peer_id, "name": name})
        return peer_id

    async def _on_forward(self, sender_id: str, data: dict) -> None:
        target_id = str(data.get("to") or "")
        code = self.peer_room.get(sender_id)
        if not code:
            return
        room = self.rooms.get(code)
        if room is None:
            return
        target = room.peers.get(target_id)
        if target is None:
            return
        payload = dict(data)
        payload.pop("to", None)
        payload["from"] = sender_id
        payload["op"] = "forward"
        payload["kind"] = data.get("op")    # original op kept so peer can route
        await self._send(target.ws, payload)

    async def _on_disconnect(self, peer_id: str) -> None:
        code = self.peer_room.pop(peer_id, None)
        if not code:
            return
        room = self.rooms.get(code)
        if room is None:
            return
        room.peers.pop(peer_id, None)
        log.info("Peer %s left %s", peer_id, code)
        if peer_id == room.host_id:
            # Host gone → tear down the room and notify everyone left.
            for p in list(room.peers.values()):
                await self._send(p.ws, {"op": "peer_left", "peer_id": peer_id, "host_gone": True})
            del self.rooms[code]
            log.info("Room %s torn down (host gone)", code)
            return
        # Otherwise notify remaining peers (mostly the host)
        for p in room.peers.values():
            await self._send(p.ws, {"op": "peer_left", "peer_id": peer_id})

    async def _send(self, ws: web.WebSocketResponse, payload: dict) -> None:
        try:
            await ws.send_str(json.dumps(payload, separators=(",", ":")))
        except (ConnectionResetError, RuntimeError):
            pass


def build_app() -> web.Application:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    server = SignalingServer()
    app = web.Application()
    app.router.add_get("/ws", server.handle_ws)
    app.router.add_get("/", lambda r: web.Response(
        text="RadarAnalysisCorp signaling server — connect to /ws"))
    app["server"] = server
    return app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    app = build_app()
    web.run_app(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
