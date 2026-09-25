"""ReviewWSServer – Python ↔ Electron WebSocket bridge for review mode.

Python side runs a lightweight WebSocket server.
Electron player connects as a client.

Protocol (JSON messages):
  Python → Electron:
    {"type": "goto", "name": "Aya_list_a_01"}   # switch to specific video
  Electron → Python:
    {"type": "now_playing", "name": "Aya_list_a_01"}  # notify current video
    {"type": "identify", "client_type": "electron_player"}
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Callable, Optional

import websockets

log = logging.getLogger(__name__)


class ReviewWSServer:
    """Lightweight asyncio WebSocket server running in a background thread."""

    def __init__(self, port: int = 0):
        self._requested_port = port
        self._actual_port: int = 0
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server: Optional[Any] = None
        self._client: Optional[Any] = None
        self._ready = threading.Event()
        self._stop_future: Optional[asyncio.Future] = None

        # Callback invoked on the *server thread* when a message arrives.
        self.on_message: Optional[Callable[[dict], None]] = None

    # ------------------------------------------------------------------
    # Public API (called from the main / GUI thread)
    # ------------------------------------------------------------------

    def start(self, timeout: float = 5.0) -> int:
        """Start the server in a background daemon thread.

        Returns the actual port once the server is ready.
        """
        if self._thread and self._thread.is_alive():
            return self._actual_port

        self._ready.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ReviewWS")
        self._thread.start()

        if not self._ready.wait(timeout):
            raise RuntimeError("ReviewWSServer failed to start within timeout")
        return self._actual_port

    def send_command(self, cmd: dict) -> bool:
        """Thread-safe: send a JSON command to the connected Electron client.

        Returns True if the send was scheduled, False otherwise.
        """
        if self._client is None or self._loop is None:
            return False
        data = json.dumps(cmd, ensure_ascii=False)
        try:
            asyncio.run_coroutine_threadsafe(self._safe_send(data), self._loop)
        except RuntimeError:
            # stop() と競合してループが閉じた直後は未送信として扱う
            return False
        return True

    def stop(self) -> None:
        """Shutdown server and background thread."""
        loop = self._loop
        if loop is not None and self._stop_future is not None:
            # Signal the serve coroutine to exit cleanly via asyncio Future.
            # The loop may be closing/closed concurrently (server thread
            # shutting down on its own) — treat that as already stopped.
            try:
                loop.call_soon_threadsafe(self._resolve_stop)
            except RuntimeError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._client = None
        self._actual_port = 0

    def _resolve_stop(self):
        """Set the stop future result (must be called on the event loop thread)."""
        if self._stop_future and not self._stop_future.done():
            self._stop_future.set_result(None)

    # ------------------------------------------------------------------
    # Internal – runs on the background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception:
            log.exception("ReviewWSServer crashed")
        finally:
            # Cancel any remaining tasks
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True))
            self._loop.close()
            self._loop = None

    async def _serve(self) -> None:
        self._server = await websockets.serve(
            self._handler,
            "127.0.0.1",
            self._requested_port,
        )
        # Retrieve the actual bound port
        for sock in self._server.sockets:
            addr = sock.getsockname()
            self._actual_port = addr[1]
            break

        # Create the stop future BEFORE signalling readiness, so a stop()
        # issued immediately after start() returns can always resolve it.
        self._stop_future = self._loop.create_future()

        self._ready.set()
        log.info("ReviewWSServer listening on port %d", self._actual_port)

        # Wait for stop signal using asyncio Future (clean shutdown)
        try:
            await self._stop_future
        except asyncio.CancelledError:
            pass

        self._server.close()
        await self._server.wait_closed()

    async def _handler(self, ws) -> None:
        # 単一クライアント前提: 2本目が接続してきた場合は旧クライアントを置き換える
        self._client = ws
        log.info("Electron player connected")
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if self.on_message is not None:
                    try:
                        self.on_message(msg)
                    except Exception:
                        log.exception("on_message callback error")
        except websockets.ConnectionClosed:
            pass
        finally:
            if self._client is ws:
                self._client = None
            log.info("Electron player disconnected")

    async def _safe_send(self, data: str) -> None:
        if self._client is not None:
            try:
                await self._client.send(data)
            except Exception:
                self._client = None
