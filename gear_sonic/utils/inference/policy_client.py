"""Small GR00T PolicyServer wire client without a model-runtime dependency."""

from __future__ import annotations

from typing import Any

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype.kind == "O":
        raise TypeError("object-dtype arrays are not allowed on the policy wire")
    return mnp.encode(value)


def _decode(value: Any) -> Any:
    if isinstance(value, dict):
        nd = value.get(b"nd", value.get("nd"))
        kind = value.get(b"kind", value.get("kind"))
        if nd and kind in (b"O", "O"):
            raise ValueError("object-dtype arrays are not allowed on the policy wire")
    return mnp.decode(value)


def _pack(value: Any) -> bytes:
    return msgpack.packb(value, default=_encode, use_bin_type=True)


def _unpack(value: bytes) -> Any:
    return msgpack.unpackb(value, object_hook=_decode, raw=False)


class PolicyClient:
    """Only the get_action endpoint needed by the WBC inference loop."""

    def __init__(self, host: str, port: int, *, timeout_ms: int = 15_000) -> None:
        if not 1 <= port <= 65535:
            raise ValueError("policy port must be between 1 and 65535")
        if timeout_ms <= 0:
            raise ValueError("policy timeout must be positive")
        self.endpoint = f"tcp://{host}:{port}"
        self.timeout_ms = timeout_ms
        self._context = zmq.Context()
        self._socket: Any | None = None
        self._reset_socket()

    def _reset_socket(self) -> None:
        if self._socket is not None:
            self._socket.close()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.connect(self.endpoint)

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        request = {
            "endpoint": "get_action",
            "data": {"observation": observation, "options": options},
        }
        try:
            self._socket.send(_pack(request))
            response = _unpack(self._socket.recv())
        except zmq.Again:
            self._reset_socket()
            raise
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T PolicyServer error: {response['error']}")
        if not isinstance(response, (list, tuple)) or len(response) != 2:
            raise RuntimeError("GR00T PolicyServer returned an invalid get_action response")
        if not isinstance(response[0], dict) or not isinstance(response[1], dict):
            raise RuntimeError("GR00T PolicyServer action and info must be dictionaries")
        return response[0], response[1]

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._context.term()
