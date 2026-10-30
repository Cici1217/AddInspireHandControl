"""G1/SONIC integration for the native6 Inspire package."""
import numpy as np

class InspireSonicIntegration:
    """Existing G1 model and native6 recording wire format, outside generic PICO."""

    def __init__(self, config):
        self.config = config

    def body_model(self):
        return self.config.robot_model()

    def simulation(self, wbc_config, launch):
        from .config import deployment_config
        from .simulation.environment import configure_simulation
        from .simulation.gateway import InspireSimGateway
        hand_config = deployment_config(self.config)
        options = {'hand_controller_factory': lambda model, data: InspireSimGateway(model, data, hand_config)}
        if launch.enable_depth_publish:
            if not launch.enable_image_publish:
                raise ValueError('depth publishing requires image publishing')
            from .simulation.rgbd import InspireRGBDEnv
            options['env_factory'] = InspireRGBDEnv
        result = configure_simulation(wbc_config, hand_config)
        return result, options

    def body_preflight(self, args):
        from .config import deployment_config
        from .driver.supervisor import preflight, DEFAULT_DRIVER
        return preflight(args.mode, deployment_config(self.config),
                         args.driver or self.config.options.get('driver_path') or DEFAULT_DRIVER, remote=args.remote)

    def run_body(self, args, command, *, lease=None):
        from .config import deployment_config
        from .driver.supervisor import run, DEFAULT_DRIVER
        return run(args.mode, deployment_config(self.config),
                   args.driver or self.config.options.get('driver_path') or DEFAULT_DRIVER, command, remote=args.remote,
                   child_observer=lease.add_process if lease else None)

    def legacy_target_fields(self, submitted, intent, valid):
        action = submitted.receipt.action
        return {
            'left_hand_native6': np.asarray(action.values['left']),
            'right_hand_native6': np.asarray(action.values['right']),
            'left_hand_intent': np.asarray(intent.values['left']),
            'right_hand_intent': np.asarray(intent.values['right']),
            'native6_target_valid': np.array([valid and submitted.tracking_valid]),
            'native6_command_sent': np.array([True]),
        }
