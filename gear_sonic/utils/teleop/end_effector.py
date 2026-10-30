"""One device-independent optical control loop; SDK and drivers have separate owners."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time

from gear_sonic.utils.hand_control.interface import ActionReceipt, EffectorAction, EffectorError, EffectorState


@dataclass(frozen=True)
class SubmittedEffectors:
    receipt: ActionReceipt
    intent: EffectorAction
    measured: EffectorState
    tracking_valid: bool


class OpticalSession:
    def __init__(self, reader, deployment, *, retargeter=None, backend_factory=None, observer=None):
        self.reader, self.deployment = reader, deployment
        self.observer = observer
        self.retargeter = retargeter or deployment.retargeter()
        self.backend_factory = backend_factory or deployment.create
        self.rate_hz = deployment.registration.description.command_hz
        self.period_s = 1 / self.rate_hz
        self.timeout_s = deployment.registration.description.stale_after_s
        self.operation_timeout_s = float(deployment.options.get("operation_timeout_s", 3))
        self._stop, self._ready = threading.Event(), threading.Event()
        self._lock = threading.Lock()
        self._thread = self._error = self._latest = None
        self.submissions = self.overruns = self.started_ns = self.last_submit_ns = 0
        self.feedback_samples = self.first_feedback_ns = self.last_feedback_ns = 0

    def retarget(self, sample):
        if sample is None:
            raise EffectorError("tracking_unavailable", "no PICO frame")
        tracked = sample.get("tracked_hands")
        if tracked is None or tracked.timestamp_ns != int(sample["timestamp_ns"]):
            raise EffectorError("retargeting", "body/hand timestamp mismatch")
        return self.retargeter.map(tracked)

    def start(self):
        if self._thread is not None:
            raise EffectorError("lifecycle", "session already started")
        sample = self.reader.get_latest()
        self._map(sample)
        age = time.monotonic() - float(sample["timestamp_monotonic"])
        if not 0 <= age <= self.timeout_s:
            raise EffectorError("tracking_unavailable", "PICO tracking is stale before hand startup")
        self._thread = threading.Thread(target=self._run, name="pico-end-effector", daemon=True)
        self._thread.start()
        if not self._ready.wait(self.operation_timeout_s + 1):
            self.close()
            raise TimeoutError("end-effector session startup timed out")
        self.check()

    def _map(self, sample):
        # Use the generic mapping even for compatibility wrappers exposing old tuple APIs.
        return OpticalSession.retarget(self, sample)

    def check(self):
        if self._error is not None:
            raise RuntimeError(f"PICO hand session stopped: {self._error}") from self._error

    def is_running(self):
        self.check()
        return self._thread is not None and self._thread.is_alive()

    def request_stop(self):
        self._stop.set()

    def latest(self):
        self.check()
        with self._lock:
            return self._latest

    def _run(self):
        backend = None
        try:
            backend = self.backend_factory()
            backend.connect()
            state = backend.get_state()
            previous = backend.seed_action(state)
            desired = previous
            seed = True
            deadline = last_sent_s = last_valid_s = time.monotonic()
            stamp = 0
            tracking_valid, feedback_paused = True, False
            while not self._stop.is_set():
                sample = self.reader.get_latest()
                now = time.monotonic()
                fresh = sample is not None and 0 <= now-float(sample["timestamp_monotonic"]) <= self.timeout_s
                valid = False
                if fresh:
                    try:
                        candidate = self._map(sample)
                    except EffectorError as exc:
                        if exc.category not in ("tracking_unavailable", "retargeting"):
                            raise
                    else:
                        new_stamp = int(sample["timestamp_ns"])
                        if new_stamp < stamp:
                            raise EffectorError("retargeting", "PICO device timestamp regressed")
                        if new_stamp > stamp:
                            desired, stamp = candidate, new_stamp
                            last_valid_s = float(sample["timestamp_monotonic"])
                        valid = now - last_valid_s <= self.timeout_s
                if valid != tracking_valid:
                    print("PICO hand tracking restored." if valid else "PICO hand tracking unavailable; holding last target.", flush=True)
                    tracking_valid = valid
                target = previous
                if not seed and valid:
                    target = backend.limit_action(previous, desired, min(self.period_s, now-last_sent_s))
                try:
                    receipt = backend.send_action(target)
                except EffectorError as exc:
                    if exc.category != "feedback_stale":
                        raise
                    if not feedback_paused:
                        print("PICO hand feedback paused; waiting for fresh feedback.", flush=True)
                        feedback_paused = True
                    self._stop.wait(self.period_s)
                    deadline = time.monotonic()
                    continue
                if feedback_paused:
                    print("PICO hand feedback restored; control resumed.", flush=True)
                    feedback_paused = False
                previous = target
                state = backend.get_state()
                if state.received_monotonic_ns > self.last_feedback_ns:
                    self.first_feedback_ns = self.first_feedback_ns or state.received_monotonic_ns
                    self.last_feedback_ns = state.received_monotonic_ns
                    self.feedback_samples += 1
                sent_ns = receipt.submitted_monotonic_ns
                last_sent_s = sent_ns / 1e9
                self.started_ns = self.started_ns or sent_ns
                self.last_submit_ns = sent_ns
                self.submissions += 1
                with self._lock:
                    # Keep the actual source timestamp even while holding a previous target.
                    receipt = ActionReceipt(EffectorAction(target.values, stamp), sent_ns, receipt.sequence)
                    self._latest = SubmittedEffectors(receipt, desired, state, valid)
                    snapshot = self._latest
                if self.observer is not None:
                    self.observer(snapshot, backend)
                seed = False
                self._ready.set()
                deadline += self.period_s
                remaining = deadline - time.monotonic()
                if remaining < 0:
                    self.overruns += 1
                    deadline = time.monotonic() + self.period_s
                    remaining = self.period_s
                self._stop.wait(remaining)
        except BaseException as exc:
            self._error = exc
        finally:
            if backend is not None:
                try:
                    backend.close()
                except BaseException as exc:
                    self._error = RuntimeError(f"{self._error or 'shutdown'}; DISARM/close failed: {exc}")
            self._ready.set()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(self.operation_timeout_s + 1)
            if self._thread.is_alive():
                raise RuntimeError("end-effector session did not stop; shutdown is unconfirmed")
        self.check()

    def summary(self):
        elapsed = (self.last_submit_ns-self.started_ns)/1e9
        feedback_elapsed = (self.last_feedback_ns-self.first_feedback_ns)/1e9
        return {"device": self.deployment.registration.description.device_id, "backend": self.deployment.backend,
                "target_rate_hz": self.rate_hz, "submissions": self.submissions,
                "submission_rate_hz": (self.submissions-1)/elapsed if elapsed>0 else None,
                "observed_feedback_hz": (self.feedback_samples-1)/feedback_elapsed if feedback_elapsed>0 else None,
                "overruns": self.overruns, "serial_feedback_rate_verified": False}
