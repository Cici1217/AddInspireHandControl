"""Compatibility names for existing Inspire callers; control loop is device-neutral."""
from dataclasses import dataclass
import numpy as np

from gear_sonic.devices.inspire.adapter import InspireAdapter
from gear_sonic.devices.inspire.config import hardware_settings
from gear_sonic.devices.inspire.optical import OpticalRetargetingError, safe_seed, slew_target
from gear_sonic.utils.hand_control.registry import deployment
from gear_sonic.devices.inspire.retargeting import InspireRetargeter
from gear_sonic.utils.hand_control.interface import EffectorError
from gear_sonic.utils.teleop.end_effector import OpticalSession
from gear_sonic.utils.teleop.hand_tracking import OpticalPicoReader, xrt_hand_state_to_tracked

CONTROL_HZ = 50
PERIOD_S = 1 / CONTROL_HZ

@dataclass(frozen=True)
class SubmittedHands:
    left: np.ndarray
    right: np.ndarray
    device_timestamp_ns: int
    submitted_monotonic_ns: int
    tracking_valid: bool


from gear_sonic.devices.inspire.integration import InspireSonicIntegration

class OpticalHandSession(OpticalSession):
    def __init__(self, reader, calibration, config_path, *, simulation=False, backend_factory=None):
        config = deployment("inspire", "sim_shared" if simulation else "real", options={"hand_config": str(config_path)})
        retargeter = InspireRetargeter(config, calibration=calibration)
        settings = hardware_settings(config_path)
        if float(settings["command_rate_hz"]) != CONTROL_HZ:
            raise ValueError("PICO hand control requires hardware.command_rate_hz: 50")
        factory = (lambda: InspireAdapter(config, backend=backend_factory())) if backend_factory else config.create
        super().__init__(reader, config, retargeter=retargeter, backend_factory=factory)
        self._compat_source = self._compat_result = None
        self.timeout_s = float(settings["state_timeout_s"])
        self.operation_timeout_s = float(settings["operation_timeout_s"])

    def start(self):
        self.retarget(self.reader.get_latest())
        super().start()

    def retarget(self, sample):
        try:
            action = super().retarget(sample)
        except EffectorError as exc:
            raise OpticalRetargetingError(str(exc)) from exc
        return np.asarray(action.values["left"]), np.asarray(action.values["right"])

    def latest(self):
        result = super().latest()
        if result is None:
            return None
        if result is self._compat_source:
            return self._compat_result
        action = result.receipt.action
        self._compat_source = result
        self._compat_result = SubmittedHands(np.asarray(action.values["left"]), np.asarray(action.values["right"]),
                              action.source_timestamp_ns, result.receipt.submitted_monotonic_ns, result.tracking_valid)

        return self._compat_result
