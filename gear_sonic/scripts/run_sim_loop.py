"""Entry point for running a MuJoCo simulation loop with the G1 robot model.

Parses a YAML-based WBC config via tyro CLI, instantiates the G1 robot model,
and launches the simulator (optionally with offscreen image publishing).
"""

from pathlib import Path
from typing import Dict

import tyro

from gear_sonic.utils.hand_control.registry import DEFAULT_REGISTRY, deployment_for_mode
from gear_sonic.data.robot_model.robot_model import RobotModel
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory

ArgsConfig = SimLoopConfig


class SimWrapper:
    def __init__(self, robot_model: RobotModel, env_name: str, config: Dict[str, any], **kwargs):
        self.robot_model = robot_model
        self.config = config

        # BaseSimulator initializes DDS before constructing its SDK bridge.
        self.sim = SimulatorFactory.create_simulator(
            config=self.config,
            env_name=env_name,
            **kwargs,
        )


def main(config: ArgsConfig):
    wbc_config = config.load_wbc_yaml()
    # NOTE: we will override the interface to local if it is not specified
    wbc_config["ENV_NAME"] = config.env_name

    if config.enable_image_publish:
        assert config.enable_offscreen, (
            "enable_offscreen must be True when enable_image_publish is True"
        )

    device = deployment_for_mode(config.hand, 'sim', config_path=config.effector_deployment,
                                 registry_dir=config.device_registry or DEFAULT_REGISTRY,
                                 options={'hand_config': str(config.hand_config)} if config.hand_config else None)
    if 'body' not in device.registration.backends[device.backend]['capabilities']:
        raise ValueError('device has no full-body simulation integration; use run_end_effector for standalone tools')
    integration = device.body_integration()
    wbc_config, hand_options = integration.simulation(wbc_config, config)
    if config.scene:
        wbc_config['ROBOT_SCENE'] = str(Path(config.scene).expanduser().resolve())
    if config.initial_keyframe:
        wbc_config["INITIAL_KEYFRAME"] = config.initial_keyframe
    robot_model = device.robot_model()

    from gear_sonic.utils.hand_control.management import DeviceLease
    with DeviceLease(device, 'full_body_simulation') as lease:
        sim_wrapper = SimWrapper(
            robot_model=robot_model,
            env_name=config.env_name,
            config=wbc_config,
            onscreen=wbc_config.get("ENABLE_ONSCREEN", True),
            offscreen=wbc_config.get("ENABLE_OFFSCREEN", False),
            enable_image_publish=config.enable_image_publish,
            **hand_options,
        )
        lease.status('connected')
        SimulatorFactory.start_simulator(
            sim_wrapper.sim,
            as_thread=False,
            enable_image_publish=config.enable_image_publish,
            mp_start_method=config.mp_start_method,
            camera_port=config.camera_port,
        )


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
