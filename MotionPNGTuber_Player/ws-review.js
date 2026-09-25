/**
 * ws-review.js - WebSocket client for review mode.
 *
 * Connects to the Python ReviewWSServer to receive control commands
 * and send playback status notifications.
 *
 * Protocol (JSON):
 *   Python -> Electron:
 *     {"type": "goto", "name": "Aya_list_a_01"}
 *     {"type": "remove", "name": "Aya_list_a_01"}
 *     {"type": "stop"}
 *   Electron -> Python:
 *     {"type": "now_playing", "name": "Aya_list_a_01"}
 *     {"type": "removed", "name": "Aya_list_a_01", "ok": true, "reason": ""}
 *     {"type": "identify", "client_type": "electron_player"}
 */

class ReviewWSClient {
    constructor(wsPort, callbacks) {
        this.wsPort = wsPort;
        this.ws = null;
        this.callbacks = callbacks || {};
        this.reconnectTimer = null;
        this.connected = false;
    }

    connect() {
        if (!this.wsPort) return;

        const url = `ws://127.0.0.1:${this.wsPort}`;
        console.log('[Review-WS] Connecting to', url);

        try {
            this.ws = new WebSocket(url);
        } catch (e) {
            console.error('[Review-WS] Connection error:', e);
            this._scheduleReconnect();
            return;
        }

        this.ws.onopen = () => {
            console.log('[Review-WS] Connected');
            this.connected = true;
            this._send({ type: 'identify', client_type: 'electron_player' });
            if (this.callbacks.onConnected) {
                this.callbacks.onConnected();
            }
        };

        this.ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                console.log('[Review-WS] Received:', msg.type);
                if (msg.type === 'goto' && msg.name) {
                    if (this.callbacks.onGoto) {
                        this.callbacks.onGoto(msg.name);
                    }
                } else if (msg.type === 'remove' && msg.name) {
                    if (this.callbacks.onRemove) {
                        this.callbacks.onRemove(msg.name);
                    }
                } else if (msg.type === 'stop') {
                    if (this.callbacks.onStop) {
                        this.callbacks.onStop();
                    }
                }
            } catch (e) {
                console.error('[Review-WS] Parse error:', e);
            }
        };

        this.ws.onclose = () => {
            console.log('[Review-WS] Disconnected');
            this.connected = false;
            this._scheduleReconnect();
        };

        this.ws.onerror = (error) => {
            console.error('[Review-WS] Error:', error);
        };
    }

    sendNowPlaying(name) {
        this._send({ type: 'now_playing', name: name });
    }

    sendRemoved(name, ok, reason) {
        this._send({ type: 'removed', name: name, ok: ok, reason: reason || '' });
    }

    disconnect() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        if (this.ws) {
            this.ws.onclose = null;
            this.ws.close();
            this.ws = null;
        }
        this.connected = false;
    }

    _send(msg) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(msg));
        }
    }

    _scheduleReconnect() {
        if (this.reconnectTimer) return;
        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null;
            this.connect();
        }, 2000);
    }
}
