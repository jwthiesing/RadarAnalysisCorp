"""WebRTC DataChannel transport with hosted-signaling handshake.

Star topology (plan §10): each non-host peer maintains exactly one WebRTC
DataChannel to the room host. The host runs N parallel transports (one per
peer) and acts as a message router, broadcasting peer messages to all other
peers.

This module exposes two classes:
  - :class:`HostTransport` — listens for ``peer_joined`` on the signaling
    socket, creates an :class:`RTCPeerConnection` per joining peer, sends an
    SDP offer, and bridges DataChannel messages to caller-supplied callbacks.
  - :class:`ClientTransport` — connects to the signaling socket, waits for the
    host's SDP offer (relayed via the signaling server), answers, and exposes
    a single DataChannel to the host.

All callbacks fire on the asyncio event loop. The Qt UI integrates by running
a ``qasync`` loop or by hopping back to Qt via QMetaObject; for v1 we let the
caller manage the loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import aiohttp
from aiortc import (
    RTCConfiguration,
    RTCDataChannel,
    RTCIceCandidate,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.contrib.signaling import object_from_string, object_to_string

# Public STUN servers for NAT traversal (cone NATs). Symmetric NATs still need
# a TURN server, which we don't ship; document this in the README.
DEFAULT_ICE_SERVERS = [
    RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
    RTCIceServer(urls=["stun:global.stun.twilio.com:3478"]),
]


def _new_pc() -> RTCPeerConnection:
    return RTCPeerConnection(configuration=RTCConfiguration(iceServers=DEFAULT_ICE_SERVERS))

log = logging.getLogger(__name__)

DEFAULT_SIGNALING_URL = "ws://localhost:8765/ws"


def normalize_signaling_url(url: str) -> str:
    """Auto-bracket a raw IPv6 host in a ``ws://...`` URL.

    URL parsers require IPv6 literals to be wrapped in ``[...]`` so the
    colons in the address aren't confused with the host/port separator.
    Users who paste a raw address like
    ``ws://2600:1700:bb50:c070::1:8765/ws`` end up with the trailing
    ``:1:8765/ws`` getting mis-parsed (the URL library can't tell which
    colon is the port). Detect that case and insert the brackets.

    Inputs that already look correct (hostnames, IPv4, or properly
    bracketed IPv6) are passed through unchanged.
    """
    s = url.strip()
    if not s:
        return s
    scheme = ""
    rest = s
    if "://" in s:
        scheme, rest = s.split("://", 1)
        scheme += "://"
    # Already bracketed → trust it.
    if rest.startswith("["):
        return scheme + rest
    # Split off path so we only look at host:port.
    if "/" in rest:
        hostport, path = rest.split("/", 1)
        path = "/" + path
    else:
        hostport, path = rest, ""
    # Raw IPv6 literals contain at least two colons (an IPv4 has at most
    # one — the port separator). Bracket the host portion, treating the
    # *last* colon as the port separator if it's followed by digits only.
    if hostport.count(":") >= 2:
        last_colon = hostport.rfind(":")
        tail = hostport[last_colon + 1:]
        if tail.isdigit():
            host = hostport[:last_colon]
            port = ":" + tail
        else:
            host = hostport
            port = ""
        hostport = f"[{host}]{port}"
    return scheme + hostport + path

# Message handler: callable(peer_id, raw_str) -> None
MessageHandler = Callable[[str, str], None]
PeerEventHandler = Callable[[str], None]   # peer_id


# ---------------------------- helpers ----------------------------------------

def _sdp_to_dict(desc: RTCSessionDescription) -> dict:
    return {"type": desc.type, "sdp": desc.sdp}


def _sdp_from_dict(d: dict) -> RTCSessionDescription:
    return RTCSessionDescription(sdp=d["sdp"], type=d["type"])


def _candidate_to_dict(c: RTCIceCandidate) -> dict:
    return object_from_string(object_to_string(c)) if False else {
        "candidate": c.candidate, "sdpMid": c.sdpMid, "sdpMLineIndex": c.sdpMLineIndex,
    }


# ---------------------------- HostTransport ----------------------------------

class HostTransport:
    """One transport instance held by the room host, handling every joining peer.

    Usage::

        host = HostTransport(name="Alice", on_message=router, on_peer_joined=lambda pid: ...)
        room_code = await host.start()
        # ... later, send to one or all peers
        await host.broadcast(msg_json)
        await host.send_to(peer_id, msg_json)
        await host.stop()
    """

    def __init__(
        self,
        *,
        name: str,
        signaling_url: str = DEFAULT_SIGNALING_URL,
        on_message: MessageHandler | None = None,
        on_peer_joined: PeerEventHandler | None = None,
        on_peer_left: PeerEventHandler | None = None,
    ) -> None:
        self.name = name
        self.signaling_url = signaling_url
        # Public callback slots; safe to reassign after construction.
        self.on_message: MessageHandler | None = on_message
        self.on_peer_joined: PeerEventHandler | None = on_peer_joined
        self.on_peer_left: PeerEventHandler | None = on_peer_left
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._room: str | None = None
        self._host_id: str | None = None
        self._peers: dict[str, _PeerConn] = {}
        self._sig_task: asyncio.Task | None = None

    @property
    def room_code(self) -> str | None:
        return self._room

    @property
    def host_id(self) -> str | None:
        return self._host_id

    async def start(self) -> str:
        """Host a new room. Returns the room code."""
        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(self.signaling_url)
        await self._ws.send_str(json.dumps({"op": "host", "name": self.name}))
        msg = await self._ws.receive_json()
        if msg.get("op") != "hosted":
            raise RuntimeError(f"Unexpected signaling response: {msg}")
        self._room = msg["room"]
        self._host_id = msg["host_id"]
        self._sig_task = asyncio.create_task(self._signaling_loop())
        log.info("Host started: room=%s id=%s", self._room, self._host_id)
        return self._room

    async def stop(self) -> None:
        if self._sig_task:
            self._sig_task.cancel()
        for peer in list(self._peers.values()):
            await peer.close()
        self._peers.clear()
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()

    async def broadcast(self, payload: str) -> None:
        for peer in list(self._peers.values()):
            await peer.send(payload)

    async def send_to(self, peer_id: str, payload: str) -> None:
        peer = self._peers.get(peer_id)
        if peer:
            await peer.send(payload)

    @property
    def peer_ids(self) -> list[str]:
        return list(self._peers.keys())

    # ---- signaling loop ------------------------------------------------

    async def _signaling_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if raw.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(raw.data)
                op = data.get("op")
                if op == "peer_joined":
                    await self._on_new_peer(data["peer_id"], data.get("name", "?"))
                elif op == "peer_left":
                    pid = data["peer_id"]
                    peer = self._peers.pop(pid, None)
                    if peer:
                        await peer.close()
                    if self.on_peer_left:
                        self.on_peer_left(pid)
                elif op == "forward":
                    await self._handle_forward(data)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Host signaling loop error")

    async def _on_new_peer(self, peer_id: str, name: str) -> None:
        log.info("Host: new peer %s (%s) joined", peer_id, name)
        peer = _PeerConn(peer_id=peer_id, transport=self, host_initiator=True)
        await peer.setup_offerer()
        self._peers[peer_id] = peer
        if self.on_peer_joined:
            self.on_peer_joined(peer_id)

    async def _handle_forward(self, data: dict) -> None:
        from_id = data.get("from")
        if from_id is None:
            return
        peer = self._peers.get(from_id)
        if peer is None:
            return
        await peer.handle_signaling_payload(data)

    async def _send_signaling(self, target_peer_id: str, op: str, payload: dict) -> None:
        if self._ws is None:
            return
        await self._ws.send_str(json.dumps({"op": op, "to": target_peer_id, **payload}))

    def _deliver_message(self, peer_id: str, raw: str) -> None:
        if self.on_message:
            self.on_message(peer_id, raw)


# ---------------------------- ClientTransport --------------------------------

class ClientTransport:
    """Non-host peer side: connects to signaling, waits for host's offer, answers."""

    def __init__(
        self,
        *,
        name: str,
        signaling_url: str = DEFAULT_SIGNALING_URL,
        on_message: MessageHandler | None = None,
        on_host_left: Callable[[], None] | None = None,
    ) -> None:
        self.name = name
        self.signaling_url = signaling_url
        self.on_message: MessageHandler | None = on_message
        self.on_host_left = on_host_left
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._room: str | None = None
        self._peer_id: str | None = None
        self._host_id: str | None = None
        self._peer: _PeerConn | None = None
        self._sig_task: asyncio.Task | None = None

    async def join(self, room_code: str) -> str:
        """Join an existing room. Returns this client's assigned peer_id."""
        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(self.signaling_url)
        await self._ws.send_str(json.dumps({"op": "join", "room": room_code, "name": self.name}))
        msg = await self._ws.receive_json()
        if msg.get("op") == "error":
            raise RuntimeError(f"Join failed: {msg.get('message')}")
        if msg.get("op") != "joined":
            raise RuntimeError(f"Unexpected response: {msg}")
        self._room = msg["room"]
        self._host_id = msg["host_id"]
        self._peer_id = msg["peer_id"]
        self._sig_task = asyncio.create_task(self._signaling_loop())
        log.info("Client joined room=%s peer_id=%s host=%s",
                 self._room, self._peer_id, self._host_id)
        return self._peer_id

    async def stop(self) -> None:
        if self._sig_task:
            self._sig_task.cancel()
        if self._peer:
            await self._peer.close()
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()

    async def send(self, payload: str) -> None:
        if self._peer:
            await self._peer.send(payload)

    @property
    def peer_id(self) -> str | None:
        return self._peer_id

    @property
    def is_connected(self) -> bool:
        return self._peer is not None and self._peer.is_open

    async def _signaling_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if raw.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(raw.data)
                op = data.get("op")
                if op == "forward":
                    await self._handle_forward(data)
                elif op == "peer_left" and data.get("host_gone"):
                    if self.on_host_left:
                        self.on_host_left()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Client signaling loop error")

    async def _handle_forward(self, data: dict) -> None:
        if self._peer is None:
            self._peer = _PeerConn(peer_id=self._host_id or "?", transport=self,
                                   host_initiator=False)
        await self._peer.handle_signaling_payload(data)

    async def _send_signaling(self, target_peer_id: str, op: str, payload: dict) -> None:
        if self._ws is None:
            return
        await self._ws.send_str(json.dumps({"op": op, "to": target_peer_id, **payload}))

    def _deliver_message(self, peer_id: str, raw: str) -> None:
        if self.on_message:
            self.on_message(peer_id, raw)


# ---------------------------- one-peer connection ----------------------------

@dataclass
class _PeerConn:
    """Internal: one WebRTC connection to a single remote peer.

    Used by both HostTransport (one per peer) and ClientTransport (one to host).
    """

    peer_id: str
    transport: object        # HostTransport or ClientTransport
    host_initiator: bool

    pc: RTCPeerConnection = field(default_factory=_new_pc)
    channel: RTCDataChannel | None = None
    _open_event: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def is_open(self) -> bool:
        return self.channel is not None and self.channel.readyState == "open"

    async def setup_offerer(self) -> None:
        """Host-side: create DataChannel, build offer, send via signaling."""
        self.channel = self.pc.createDataChannel("game")
        self._wire_channel()
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        await self.transport._send_signaling(
            self.peer_id, "offer", {"sdp": _sdp_to_dict(self.pc.localDescription)}
        )

    async def handle_signaling_payload(self, data: dict) -> None:
        kind = data.get("kind")
        if kind == "offer":
            # Client side: receive offer, build answer
            desc = _sdp_from_dict(data["sdp"])
            await self.pc.setRemoteDescription(desc)

            @self.pc.on("datachannel")
            def _on_datachannel(channel):
                self.channel = channel
                self._wire_channel()

            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)
            await self.transport._send_signaling(
                self.peer_id, "answer", {"sdp": _sdp_to_dict(self.pc.localDescription)}
            )
        elif kind == "answer":
            # Host side: complete handshake
            desc = _sdp_from_dict(data["sdp"])
            await self.pc.setRemoteDescription(desc)
        elif kind == "ice":
            cand = data.get("candidate") or {}
            if not cand:
                return
            try:
                rtc_cand = RTCIceCandidate(
                    component=cand.get("component", 1),
                    foundation=cand.get("foundation", ""),
                    ip=cand.get("ip", ""),
                    port=cand.get("port", 0),
                    priority=cand.get("priority", 0),
                    protocol=cand.get("protocol", "udp"),
                    type=cand.get("type", "host"),
                    sdpMid=cand.get("sdpMid"),
                    sdpMLineIndex=cand.get("sdpMLineIndex"),
                )
                await self.pc.addIceCandidate(rtc_cand)
            except Exception as e:  # noqa: BLE001
                log.debug("Failed to apply ICE candidate from %s: %s", self.peer_id, e)

    def _wire_channel(self) -> None:
        if self.channel is None:
            return

        @self.channel.on("open")
        def _on_open():
            self._open_event.set()
            log.debug("DataChannel open with %s", self.peer_id)

        @self.channel.on("message")
        def _on_message(msg):
            if isinstance(msg, bytes):
                msg = msg.decode("utf-8")
            self.transport._deliver_message(self.peer_id, msg)

    def wire_ice_forwarding(self) -> None:
        """Forward locally-gathered ICE candidates over the signaling channel.

        aiortc emits ``icegatheringstatechange`` and tracks candidates as part
        of the local description; we use the ``onicecandidate``-equivalent
        listener available via aiortc's internal `_ice_gatherer` events.
        """
        @self.pc.on("icegatheringstatechange")
        def _on_state():
            # When gathering completes, the local description already contains
            # all the candidates; nothing more to forward.
            pass

        @self.pc.on("connectionstatechange")
        def _on_conn_state():
            log.debug("PC %s state: %s", self.peer_id, self.pc.connectionState)

    async def wait_open(self, timeout: float = 10.0) -> None:
        await asyncio.wait_for(self._open_event.wait(), timeout=timeout)

    async def send(self, payload: str) -> None:
        if self.channel is None or self.channel.readyState != "open":
            log.debug("send() dropped — channel not open")
            return
        self.channel.send(payload)

    async def close(self) -> None:
        try:
            if self.channel is not None:
                self.channel.close()
        except Exception:  # noqa: BLE001
            pass
        await self.pc.close()
