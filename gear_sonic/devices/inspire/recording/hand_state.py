"""Read-only Inspire feedback subscriber; never opens a command socket."""

from dataclasses import dataclass
import time

import msgpack
import zmq

from ..contract import ContractError, InspireHandState


@dataclass(frozen=True)
class ReceivedHandState:
    state: InspireHandState
    received_monotonic_ns: int


class StateClient:
    def __init__(self, endpoint, *, receive_timeout_ms=2):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        try:
            self._socket.setsockopt(zmq.SUBSCRIBE, b"")
            self._socket.setsockopt(zmq.CONFLATE, 1)
            self._socket.setsockopt(zmq.RCVTIMEO, receive_timeout_ms)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.connect(endpoint)
        except BaseException:
            self.close()
            raise
        self._last_sequence = -1
        self._instance = None

    def receive_timed(self):
        payload = msgpack.unpackb(self._socket.recv(), raw=False)
        received_ns = time.monotonic_ns()
        instance = payload.get("operator_session", {}).get("instance_id")
        if self._instance is None:
            self._instance = instance
        elif self._instance != instance:
            raise ContractError("hand gateway restarted during recording")
        state = InspireHandState.from_wire_dict(payload)
        if state.sequence < self._last_sequence:
            raise ContractError("hand state sequence regressed")
        if state.sequence == self._last_sequence:
            raise TimeoutError("repeated hand state; local receipt was not refreshed")
        self._last_sequence = state.sequence
        return ReceivedHandState(state, received_ns)

    def close(self):
        self._socket.close(0)
        self._context.term()
