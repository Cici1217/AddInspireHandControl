"""Shared asynchronous scheduler for native6 VLA simulation and real execution.

The policy request/response path is deliberately isolated from the 50 Hz
control thread.  All policy results pass through the existing strict
``PolicyChunk`` decoder before they can reach a body or hand sink.
"""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Any, Callable, Protocol

import numpy as np

from .contract import ContractError
from .policy import PolicyChunk, extract_policy_chunk


class PolicyLike(Protocol):
    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]: ...

    def close(self) -> None: ...


class LiveVlaError(RuntimeError):
    """Base class for fail-closed live inference errors."""


class StalePolicyChunkError(LiveVlaError):
    """Raised when inference consumed the complete action horizon."""


class ActionBufferUnderrun(LiveVlaError):
    """Raised when no replacement action arrived within the bounded hold time."""


class UnsafePolicyAction(LiveVlaError):
    """Raised when a policy action violates a body-token safety gate."""


@dataclass(frozen=True)
class PolicyRequest:
    generation: int
    observation: dict[str, Any]
    submitted_monotonic_ns: int
    options: dict[str, Any] | None = None


@dataclass(frozen=True)
class PolicyResult:
    generation: int
    submitted_monotonic_ns: int
    completed_monotonic_ns: int
    chunk: PolicyChunk | None
    info: dict[str, Any]
    error: BaseException | None = None
    request_options: dict[str, Any] | None = None

    @property
    def latency_s(self) -> float:
        return (self.completed_monotonic_ns - self.submitted_monotonic_ns) / 1e9


class AsyncPolicyWorker:
    """One persistent PolicyClient owned and used by one worker thread.

    ZeroMQ sockets are not thread safe.  The client factory is therefore
    executed inside the worker instead of constructing a client in the control
    thread and later borrowing its socket.
    """

    def __init__(self, client_factory: Callable[[], PolicyLike]) -> None:
        self._client_factory = client_factory
        self._requests: queue.Queue[PolicyRequest] = queue.Queue(maxsize=1)
        self._results: queue.Queue[PolicyResult] = queue.Queue()
        self._stop = threading.Event()
        self._busy = threading.Event()
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._generation = 0
        self._thread = threading.Thread(
            target=self._run,
            name="inspire-native6-policy",
            daemon=True,
        )
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._busy.is_set() or not self._requests.empty()

    def wait_ready(self, timeout_s: float) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if not self._ready.wait(timeout_s):
            raise TimeoutError("PolicyClient worker did not initialize before timeout")
        if self._startup_error is not None:
            raise LiveVlaError(f"PolicyClient initialization failed: {self._startup_error}")

    def submit(
        self,
        observation: dict[str, Any],
        *,
        options: dict[str, Any] | None = None,
        submitted_monotonic_ns: int | None = None,
    ) -> int | None:
        """Queue the newest observation, or return ``None`` while one is in flight."""
        if self._stop.is_set():
            raise RuntimeError("policy worker is closed")
        if self.busy:
            return None
        self._generation += 1
        request = PolicyRequest(
            generation=self._generation,
            observation=observation,
            submitted_monotonic_ns=(
                time.monotonic_ns()
                if submitted_monotonic_ns is None
                else submitted_monotonic_ns
            ),
            options=options,
        )
        try:
            self._requests.put_nowait(request)
        except queue.Full:
            return None
        return request.generation

    def poll(self) -> PolicyResult | None:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def close(self, *, join_timeout_s: float = 1.0) -> None:
        if join_timeout_s < 0:
            raise ValueError("join_timeout_s must be non-negative")
        self._stop.set()
        self._thread.join(timeout=join_timeout_s)

    def _run(self) -> None:
        client: PolicyLike | None = None
        try:
            client = self._client_factory()
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()
        try:
            while not self._stop.is_set():
                try:
                    request = self._requests.get(timeout=0.05)
                except queue.Empty:
                    continue
                self._busy.set()
                try:
                    action, info = client.get_action(
                        request.observation,
                        options=request.options,
                    )
                    chunk = extract_policy_chunk(action)
                    result = PolicyResult(
                        generation=request.generation,
                        submitted_monotonic_ns=request.submitted_monotonic_ns,
                        completed_monotonic_ns=time.monotonic_ns(),
                        chunk=chunk,
                        info=info,
                        request_options=request.options,
                    )
                except BaseException as exc:
                    result = PolicyResult(
                        generation=request.generation,
                        submitted_monotonic_ns=request.submitted_monotonic_ns,
                        completed_monotonic_ns=time.monotonic_ns(),
                        chunk=None,
                        info={},
                        error=exc,
                        request_options=request.options,
                    )
                self._results.put(result)
                self._busy.clear()
        finally:
            client.close()


@dataclass(frozen=True)
class ChunkActivation:
    generation: int
    latency_s: float
    skipped_frames: int
    remaining_frames: int


@dataclass(frozen=True)
class ScheduledAction:
    generation: int
    chunk_index: int
    motion_token: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    holding_last_frame: bool
    hold_age_s: float


class TimedActionBuffer:
    """Map asynchronous action chunks onto an absolute 50 Hz timeline."""

    def __init__(self, *, control_rate_hz: float, maximum_hold_s: float) -> None:
        if control_rate_hz <= 0:
            raise ValueError("control_rate_hz must be positive")
        if maximum_hold_s < 0:
            raise ValueError("maximum_hold_s must be non-negative")
        self.control_rate_hz = float(control_rate_hz)
        self.maximum_hold_s = float(maximum_hold_s)
        self._result: PolicyResult | None = None
        self._start_index = 0
        self._activated_monotonic_ns = 0

    @property
    def ready(self) -> bool:
        return self._result is not None

    @property
    def chunk(self) -> PolicyChunk:
        if self._result is None or self._result.chunk is None:
            raise ActionBufferUnderrun("no policy action chunk is available")
        return self._result.chunk

    def clear(self) -> None:
        self._result = None
        self._start_index = 0
        self._activated_monotonic_ns = 0

    def activate(
        self,
        result: PolicyResult,
        *,
        activated_monotonic_ns: int | None = None,
    ) -> ChunkActivation:
        if result.error is not None:
            raise LiveVlaError(f"PolicyServer request failed: {result.error}") from result.error
        if result.chunk is None:
            raise LiveVlaError("PolicyServer result did not contain an action chunk")
        now_ns = (
            time.monotonic_ns()
            if activated_monotonic_ns is None
            else activated_monotonic_ns
        )
        total_latency_s = (now_ns - result.submitted_monotonic_ns) / 1e9
        if total_latency_s < 0:
            raise LiveVlaError("policy result completed before its request timestamp")
        skipped = int(np.round(total_latency_s * self.control_rate_hz))
        if skipped >= result.chunk.horizon:
            raise StalePolicyChunkError(
                "complete policy chunk is stale: "
                f"latency={total_latency_s:.3f}s horizon={result.chunk.horizon} "
                f"rate={self.control_rate_hz:g}Hz"
            )
        self._result = result
        self._start_index = skipped
        self._activated_monotonic_ns = now_ns
        return ChunkActivation(
            generation=result.generation,
            latency_s=total_latency_s,
            skipped_frames=skipped,
            remaining_frames=result.chunk.horizon - skipped,
        )

    def sample(self, *, now_monotonic_ns: int | None = None) -> ScheduledAction:
        if self._result is None or self._result.chunk is None:
            raise ActionBufferUnderrun("no policy action chunk is available")
        now_ns = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        elapsed_after_activation_s = (now_ns - self._activated_monotonic_ns) / 1e9
        if elapsed_after_activation_s < 0:
            raise LiveVlaError("action sampling timestamp predates chunk activation")
        elapsed_frames = int(
            np.floor(elapsed_after_activation_s * self.control_rate_hz + 1e-9)
        )
        requested_index = self._start_index + elapsed_frames
        last_index = self._result.chunk.horizon - 1
        holding = requested_index > last_index
        hold_age_s = 0.0
        if holding:
            hold_age_s = (requested_index - last_index) / self.control_rate_hz
            if hold_age_s > self.maximum_hold_s:
                raise ActionBufferUnderrun(
                    "policy action buffer exhausted: "
                    f"generation={self._result.generation} hold_age={hold_age_s:.3f}s "
                    f"limit={self.maximum_hold_s:.3f}s"
                )
        index = min(requested_index, last_index)
        chunk = self._result.chunk
        return ScheduledAction(
            generation=self._result.generation,
            chunk_index=index,
            motion_token=chunk.motion_token[index].copy(),
            left_hand=chunk.left_hand[index].copy(),
            right_hand=chunk.right_hand[index].copy(),
            holding_last_frame=holding,
            hold_age_s=hold_age_s,
        )


class BodyTokenContinuityGuard:
    """Reject body tokens outside the real Gateway gate or training continuity envelope."""

    def __init__(self, *, absolute_limit: float = 1.0, maximum_step: float = 0.25) -> None:
        if absolute_limit <= 0 or maximum_step <= 0:
            raise ValueError("body token limits must be positive")
        self.absolute_limit = float(absolute_limit)
        self.maximum_step = float(maximum_step)
        self._previous: np.ndarray | None = None
        self.maximum_observed_step = 0.0

    def seed(self, motion_token: np.ndarray) -> None:
        """Set the measured/accepted token that the first policy frame must follow."""
        token = np.asarray(motion_token, dtype=np.float32)
        if token.shape != (64,) or not np.all(np.isfinite(token)):
            raise ContractError(f"motion_token must be a finite (64,) vector, got {token.shape}")
        maximum = float(np.max(np.abs(token)))
        if maximum > self.absolute_limit:
            raise UnsafePolicyAction(
                f"motion token absolute value {maximum:.6g} exceeds {self.absolute_limit:.6g}"
            )
        self._previous = token.copy()

    def validate_chunk(self, chunk: PolicyChunk) -> None:
        maximum = float(np.max(np.abs(chunk.motion_token)))
        if maximum > self.absolute_limit:
            raise UnsafePolicyAction(
                f"motion token absolute value {maximum:.6g} exceeds {self.absolute_limit:.6g}"
            )
        if chunk.horizon > 1:
            step = float(np.max(np.abs(np.diff(chunk.motion_token, axis=0))))
            if step > self.maximum_step:
                raise UnsafePolicyAction(
                    f"motion token internal step {step:.6g} exceeds {self.maximum_step:.6g}"
                )

    def validate_transition(self, motion_token: np.ndarray) -> float:
        """Validate one token against the last dispatched token without committing it."""
        token = np.asarray(motion_token, dtype=np.float32)
        if token.shape != (64,) or not np.all(np.isfinite(token)):
            raise ContractError(f"motion_token must be a finite (64,) vector, got {token.shape}")
        maximum = float(np.max(np.abs(token)))
        if maximum > self.absolute_limit:
            raise UnsafePolicyAction(
                f"motion token absolute value {maximum:.6g} exceeds {self.absolute_limit:.6g}"
            )
        step = 0.0
        if self._previous is not None:
            step = float(np.max(np.abs(token - self._previous)))
            if step > self.maximum_step:
                raise UnsafePolicyAction(
                    f"motion token cross-frame step {step:.6g} exceeds {self.maximum_step:.6g}"
                )
        return step

    def validate_and_commit(self, motion_token: np.ndarray) -> float:
        token = np.asarray(motion_token, dtype=np.float32)
        step = self.validate_transition(token)
        self.maximum_observed_step = max(self.maximum_observed_step, step)
        self._previous = token.copy()
        return step
