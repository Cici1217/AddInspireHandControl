"""Device-independent end-effector contracts, with legacy Inspire boundary types.

EndEffector uses separately described state/action channels keyed by installation.
HandBackend remains the existing paired-hand driver protocol behind InspireAdapter.
Importing this module opens no devices and requires no simulation or hardware SDK.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Sequence
from collections.abc import Mapping
import math
from types import MappingProxyType


@dataclass(frozen=True)
class HandDescription:
    """Ordered control channels; both positions and targets use these units.

    Limits and actuator conversion belong to the selected backend's configuration.
    Channels may be coupled finger coordinates rather than every physical joint.
    """

    model: str
    left_joint_names: tuple[str, ...]
    right_joint_names: tuple[str, ...]
    position_unit: str = "rad"


@dataclass(frozen=True)
class HandState:
    """Latest measured feedback, never a fabricated copy of the target.

    None means unavailable. received_monotonic_ns is the time of receipt in
    this process, not a timestamp from a remote machine. Backends must document
    whether left/right measurements are simultaneous and how stale data is handled.
    """

    left_position: tuple[float, ...] | None = None
    right_position: tuple[float, ...] | None = None
    received_monotonic_ns: int | None = None
    error: str | None = None


class HandBackend(ABC):
    """Interface for a pair of hands in simulation or on hardware.

    Calls are serialized by the owner; thread safety is not implied. Constructing
    a backend must not connect or move hardware. The backend owns limit/rate
    checks and communication failures; these are not implemented by this base.
    """

    @property
    @abstractmethod
    def description(self) -> HandDescription:
        """Return the model, command channel order and position unit."""

    @abstractmethod
    def connect(self) -> None:
        """Initialize resources without issuing an unsolicited motion command.

        On failure, release resources acquired during this attempt and raise.
        """

    @abstractmethod
    def set_target(self, left: Sequence[float], right: Sequence[float]) -> None:
        """Submit positions in description order after connecting.

        Validate both sides' length, finite values and configured limits before
        sending either side. A normal return means submission, not physical
        arrival or a simultaneous two-hand hardware transaction. Raise on invalid
        input or disconnected/stopped state. Backends must bound queued commands.
        """

    @abstractmethod
    def read_state(self) -> HandState:
        """Return feedback after connecting, with absent measurements as None.

        Implementations must document timeout and feedback freshness behavior.
        """

    @abstractmethod
    def stop(self) -> None:
        """Cancel pending targets and reject further targets until reconnected.

        Document device behavior (e.g. hold last position); this is not an
        emergency stop or a guarantee that motors are de-energized. No automatic
        open/close gesture. Repeated calls must be safe.
        """

    @abstractmethod
    def close(self) -> None:
        """Cancel pending work and release resources, including after failures.

        Repeated calls must be safe. No implicit open/close gesture. The backend
        must document what the physical hand does after communication closes.
        """


# Generic API. The HandBackend types above remain only at the legacy Inspire boundary.
class EffectorError(RuntimeError):
    """A stable diagnostic category, independent of the underlying driver."""

    def __init__(self, category: str, message: str, *, device: str = "", mount: str = ""):
        self.category, self.device, self.mount = category, device, mount
        super().__init__(f"{category} [{device or '?'}{':' + mount if mount else ''}]: {message}")


@dataclass(frozen=True)
class Channel:
    name: str
    unit: str
    lower: float
    upper: float
    max_rate: float | None = None
    numeric_tolerance: float = 1e-7

    def __post_init__(self):
        if not self.name or not self.unit or not all(math.isfinite(v) for v in (self.lower, self.upper)):
            raise ValueError("channel requires a name, unit and finite limits")
        if self.lower >= self.upper:
            raise ValueError(f"invalid limits for {self.name}")
        if not math.isfinite(self.numeric_tolerance) or not 0 <= self.numeric_tolerance <= 1e-4:
            raise ValueError("numeric_tolerance must be between 0 and 1e-4")
        if self.max_rate is not None and (not math.isfinite(self.max_rate) or self.max_rate <= 0):
            raise ValueError(f"invalid max_rate for {self.name}")


@dataclass(frozen=True)
class DeviceDescription:
    device_id: str
    version: str
    label: str
    physical_joints: tuple[str, ...]
    state_channels: tuple[Channel, ...]
    action_channels: tuple[Channel, ...]
    control_mode: str
    command_hz: float
    feedback_hz: float
    stale_after_s: float

    def __post_init__(self):
        for name in ("device_id", "version", "label", "control_mode"):
            if not getattr(self, name):
                raise ValueError(f"missing {name}")
        for channels in (self.state_channels, self.action_channels):
            if not channels or len({c.name for c in channels}) != len(channels):
                raise ValueError("channels must be nonempty with unique names")
        if not self.physical_joints or len(set(self.physical_joints)) != len(self.physical_joints):
            raise ValueError("physical joints must be nonempty and unique")
        for value in (self.command_hz, self.feedback_hz, self.stale_after_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("frequencies and timeout must be finite and positive")

    @property
    def action_dim(self):
        return len(self.action_channels)

    @property
    def state_dim(self):
        return len(self.state_channels)

    def to_dict(self):
        return asdict(self) | {"action_dim": self.action_dim, "state_dim": self.state_dim}

    def validate_vector(self, values, *, action=True, mount="") -> tuple[float, ...]:
        channels = self.action_channels if action else self.state_channels
        category = "action_rejected" if action else "feedback_invalid"
        try:
            vector = tuple(float(v) for v in values)
        except (TypeError, ValueError) as exc:
            raise EffectorError(category, "expected numeric vector", device=self.device_id, mount=mount) from exc
        if len(vector) != len(channels):
            raise EffectorError(category, f"expected {len(channels)} values, received {len(vector)}", device=self.device_id, mount=mount)
        for value, channel in zip(vector, channels):
            if not math.isfinite(value) or not channel.lower-channel.numeric_tolerance <= value <= channel.upper+channel.numeric_tolerance:
                raise EffectorError(category, f"{channel.name}={value:.8g} outside [{channel.lower}, {channel.upper}] {channel.unit}", device=self.device_id, mount=mount)
        return vector


@dataclass(frozen=True)
class EffectorAction:
    """Per-installation action vectors. No padding or assumed left/right pair."""
    values: Mapping[str, tuple[float, ...]]
    source_timestamp_ns: int = 0

    def __post_init__(self):
        if not isinstance(self.values, Mapping):
            raise ValueError("action values must be keyed by mount")
        object.__setattr__(self, "values", MappingProxyType({k: tuple(v) for k, v in self.values.items()}))
        if type(self.source_timestamp_ns) is not int or self.source_timestamp_ns < 0:
            raise ValueError("action timestamp must be a nonnegative integer")


@dataclass(frozen=True)
class EffectorState:
    values: Mapping[str, tuple[float, ...] | None]
    received_monotonic_ns: int
    valid: Mapping[str, bool]
    fault: str | None = None
    source_timestamp_ns: int | None = None
    sequence: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "values", MappingProxyType({k: tuple(v) if v is not None else None for k, v in self.values.items()}))
        object.__setattr__(self, "valid", MappingProxyType(dict(self.valid)))
        if set(self.values) != set(self.valid):
            raise ValueError("feedback validity must cover exactly the state mounts")
        if type(self.received_monotonic_ns) is not int or self.received_monotonic_ns < 0:
            raise ValueError("feedback receipt time must be a nonnegative integer")

    def require_fresh(self, now_ns: int, description: DeviceDescription):
        if self.fault:
            raise EffectorError("backend_fault", self.fault, device=description.device_id)
        age = (now_ns - self.received_monotonic_ns) / 1e9
        if not 0 <= age <= description.stale_after_s:
            raise EffectorError("feedback_stale", f"feedback age {age:.3f}s", device=description.device_id)
        if not self.values or any(not self.valid.get(k) or v is None for k, v in self.values.items()):
            raise EffectorError("feedback_invalid", "missing valid measured state", device=description.device_id)
        for mount, values in self.values.items():
            description.validate_vector(values, action=False, mount=mount)


@dataclass(frozen=True)
class ActionReceipt:
    """Only issued after submission; it does not claim arrival at the target."""
    action: EffectorAction
    submitted_monotonic_ns: int
    sequence: int


class EndEffector(ABC):
    description: DeviceDescription
    mounts: tuple[str, ...]

    @abstractmethod
    def connect(self):
        """Allocate resources without an unsolicited motion command."""

    @abstractmethod
    def get_state(self) -> EffectorState:
        """Return measured feedback; retain its real freshness timestamp."""

    @abstractmethod
    def send_action(self, action: EffectorAction) -> ActionReceipt:
        """Validate all mounts before submitting. Do not equate submission with arrival."""

    @abstractmethod
    def stop(self):
        """Stop targets; device-specific hold/disable semantics must be documented."""

    @abstractmethod
    def reset(self):
        """Clear local session state while disconnected. Never home or auto-enable."""

    @abstractmethod
    def close(self):
        """Release resources idempotently, without implicit open/close motion."""

    def validate_action(self, action: EffectorAction):
        if set(action.values) != set(self.mounts):
            raise EffectorError("action_rejected", f"expected mounts {self.mounts}", device=self.description.device_id)
        for mount, values in action.values.items():
            self.description.validate_vector(values, mount=mount)

    def seed_action(self, state: EffectorState) -> EffectorAction:
        """Seed named position channels from measured state, never from padding."""
        import time
        state.require_fresh(time.monotonic_ns(), self.description)
        state_channels = {c.name: (i, c.unit) for i, c in enumerate(self.description.state_channels)}
        if self.description.control_mode != "position" or any(
            c.name not in state_channels or c.unit != state_channels[c.name][1]
            for c in self.description.action_channels
        ):
            raise EffectorError("configuration", "adapter must define seed_action for this action space", device=self.description.device_id)
        action = EffectorAction({mount: tuple(values[state_channels[c.name][0]] for c in self.description.action_channels)
                                 for mount, values in state.values.items()})
        self.validate_action(action)
        return action

    def limit_action(self, previous: EffectorAction, desired: EffectorAction, elapsed_s: float) -> EffectorAction:
        """Device-independent slew limit; adapters may apply additional collision limits."""
        self.validate_action(previous)
        self.validate_action(desired)
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise EffectorError("action_rejected", "invalid elapsed time", device=self.description.device_id)
        values = {}
        for mount in self.mounts:
            result = []
            for old, new, channel in zip(previous.values[mount], desired.values[mount], self.description.action_channels):
                delta = (channel.max_rate * elapsed_s) if channel.max_rate else float("inf")
                result.append(old + max(-delta, min(delta, new - old)))
            values[mount] = tuple(result)
        return EffectorAction(values, desired.source_timestamp_ns)
