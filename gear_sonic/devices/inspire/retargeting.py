"""Human hand tracking to device actions; no driver imports."""
import math
from gear_sonic.utils.hand_control.interface import EffectorAction, EffectorError
from gear_sonic.utils.hand_control.registry import asset_path

class InspireRetargeter:
    version = "inspire-optical-v1"

    def __init__(self, deployment, *, calibration=None):
        from .config import load_mapping
        from .optical import InspireOpticalCalibrationBundle
        self.deployment = deployment
        self.mapping = load_mapping(deployment.path(deployment.options["hand_config"]))
        self.calibration = calibration
        self.path = asset_path(deployment.options.get("calibration", "outputs/inspire_calibrations/pico_hands.json"))
        if self.path.resolve().is_relative_to(deployment.registration.source.parent):
            raise EffectorError('configuration', 'calibration must be stored outside the device package')
        if self.calibration is None and self.path.is_file():
            self.calibration = InspireOpticalCalibrationBundle.load(self.path)

    def prepare(self, reader):
        if self.calibration is None:
            from .calibration import calibrate
            self.calibration = calibrate(reader, str(self.path), str(self.deployment.path(self.deployment.options["hand_config"])))
        for side in self.deployment.mounts:
            hand = getattr(self.calibration, side)
            self.mapping.validate_limits(hand.open_q, field_name="calibration.open_q", side=side)
            self.mapping.validate_limits(hand.closed_q, field_name="calibration.closed_q", side=side)

    def map(self, tracked):
        if self.calibration is None:
            raise EffectorError("calibration", "prepare the retargeter before starting", device="inspire")
        from .optical import OpticalRetargetingError
        try:
            values = {}
            for side in self.deployment.mounts:
                hand = getattr(tracked, side, None)
                if hand is None:
                    raise EffectorError("tracking_unavailable", "both optical hands are required", device="inspire", mount=side)
                values[side] = tuple(getattr(self.calibration, side).retarget(hand, mapping=self.mapping))
            return EffectorAction(values, tracked.timestamp_ns)
        except OpticalRetargetingError as exc:
            raise EffectorError("retargeting", str(exc), device="inspire") from exc
