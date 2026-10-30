"""Inspire native6 adapter. Driver and simulation behavior are unchanged."""
from dataclasses import replace
import time
from gear_sonic.utils.hand_control.interface import ActionReceipt, EffectorAction, EffectorError, EffectorState, EndEffector
from .simulation.viewer import _MuJoCoViewer

class InspireAdapter(EndEffector):
    """Preserve the native gateway/session and MuJoCo controller unchanged.

    The existing gateway owns a pair of hands atomically; this particular adapter
    therefore requires left/right installations.
    """

    def __init__(self, deployment, *, backend=None):
        from .config import hardware_settings, load_mapping, deployment_config
        self.deployment = deployment
        self.description = deployment.registration.description
        self.mounts = deployment.mounts
        if set(self.mounts) != {"left", "right"}:
            raise EffectorError("configuration", "Inspire gateway requires left and right mounts", device=self.description.device_id)
        self.config_path = deployment_config(deployment)
        self.mapping = load_mapping(self.config_path)
        self.settings = hardware_settings(self.config_path)
        if self.description.command_hz != 50 or self.settings["command_rate_hz"] != 50:
            raise EffectorError("configuration", "existing Inspire control requires 50 Hz", device=self.description.device_id)
        for side in self.mounts:
            joints = getattr(self.mapping, side)
            for joint, channel in zip(joints, self.description.action_channels):
                if (joint.name, joint.lower, joint.upper, joint.velocity_limit) != (channel.name, channel.lower, channel.upper, channel.max_rate):
                    raise EffectorError("configuration", "registration and native mapping differ", device=self.description.device_id, mount=side)
            if len(joints) != self.description.action_dim or self.description.state_dim != len(joints):
                raise EffectorError("configuration", "registration dimensions differ from native mapping", device=self.description.device_id)
        self.description = replace(self.description, stale_after_s=self.settings["state_timeout_s"])
        if deployment.registration.backends[deployment.backend]['mode'] == 'sim':
            # Full-body motion produces small soft-limit residuals in MuJoCo
            # (measured -1.08e-5 rad during preparation). Allow 0.0001 rad only
            # in simulated feedback; raw readings and command limits stay intact.
            self.description = replace(self.description, state_channels=tuple(
                replace(channel, numeric_tolerance=1e-4) for channel in self.description.state_channels))
        self._backend = backend
        self._connected = False
        self._sequence = 0
        self._renderer = None
        self._viewer = None

    def connect(self):
        if self._connected:
            return
        if self._backend is None:
            if self.deployment.backend == "sim":
                from .simulation.backend import InspireSimBackend
                self._backend = InspireSimBackend(config_path=self.config_path)
            else:
                from .driver.client import InspireGatewayBackend
                self._backend = InspireGatewayBackend(self.config_path, simulation=self.deployment.backend == "sim_shared", pause_on_stale=True)
        try:
            self._backend.connect()
            self._connected = True
        except Exception as exc:
            self._backend.close()
            raise EffectorError("connection", str(exc), device=self.description.device_id) from exc

    def get_state(self):
        if not self._connected:
            raise EffectorError("connection", "not connected", device=self.description.device_id)
        native = self._backend.read_state()
        values = {"left": native.left_position, "right": native.right_position}
        return EffectorState(values, native.received_monotonic_ns or 0,
                             {k: v is not None and not native.error for k, v in values.items()},
                             native.error)

    def send_action(self, action):
        self.validate_action(action)
        if not self._connected:
            raise EffectorError("connection", "not connected", device=self.description.device_id)
        from .contract import FeedbackUnavailable
        try:
            import numpy as np
            self._backend.set_target(np.asarray(action.values["left"]), np.asarray(action.values["right"]))
        except FeedbackUnavailable as exc:
            raise EffectorError("feedback_stale", str(exc), device=self.description.device_id) from exc
        except ValueError as exc:
            raise EffectorError("action_rejected", str(exc), device=self.description.device_id) from exc
        except Exception as exc:
            raise EffectorError("backend_fault", str(exc), device=self.description.device_id) from exc
        self._sequence += 1
        return ActionReceipt(action, time.monotonic_ns(), self._sequence)

    def seed_action(self, state):
        from .optical import safe_seed
        state.require_fresh(time.monotonic_ns(), self.description)
        values = state.values
        if self.deployment.registration.backends[self.deployment.backend]['mode'] == 'sim':
            # Project only validated simulation round-off into the legal command
            # range; never replace the measured state or soften hardware checks.
            values = {side: tuple(min(c.upper, max(c.lower, q))
                                  for c, q in zip(self.description.action_channels, values[side]))
                      for side in self.mounts}
        return EffectorAction({side: tuple(safe_seed(values[side], self.mapping, side)) for side in self.mounts})

    def limit_action(self, previous, desired, elapsed_s):
        from .optical import slew_target
        self.validate_action(desired)
        return EffectorAction({side: tuple(slew_target(previous.values[side], desired.values[side], mapping=self.mapping, side=side, elapsed_s=elapsed_s)) for side in self.mounts}, desired.source_timestamp_ns)

    def stop(self):
        if self._backend is not None:
            self._backend.stop()

    def reset(self):
        if self._connected:
            raise EffectorError("lifecycle", "close before reset; hardware fault reset remains explicit", device=self.description.device_id)
        self._sequence = 0
        self._backend = None

    def close(self):
        try:
            if self._viewer is not None:
                self._viewer.close()
                self._viewer = None
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None
            if self._backend is not None:
                self._backend.close()
        finally:
            self._connected = False

    def render(self, width=320, height=240):
        """Bench-only camera; shared-world cameras remain separate subscribers."""
        if self.deployment.backend != "sim" or not self._connected:
            raise EffectorError("capability", "local rendering requires sim backend", device=self.description.device_id)
        import mujoco
        with self._backend._lock:
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self._backend.model, height=height, width=width)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0, 0, .8]
            camera.distance, camera.azimuth, camera.elevation = 2.0, 135, -20
            self._renderer.update_scene(self._backend.data, camera=camera)
            return self._renderer.render().copy()

    def body_observation(self):
        if self.deployment.backend != "sim" or not self._connected:
            raise EffectorError("capability", "use the SONIC state stream for the shared world", device=self.description.device_id)
        with self._backend._lock:
            return self._backend.data.qpos[self._backend._body_q].copy()

    def body_state_names(self):
        if self.deployment.backend != 'sim' or not self._connected:
            raise EffectorError('capability', 'local body channels require sim backend')
        import mujoco
        names = []
        model = self._backend.model
        for jid in range(model.njnt):
            if int(model.jnt_qposadr[jid]) not in self._backend._body_q:
                continue
            name = model.joint(jid).name or f'joint_{jid}'
            if model.jnt_type[jid] == int(mujoco.mjtJoint.mjJNT_FREE):
                names.extend(f'{name}.{suffix}' for suffix in ('x','y','z','qw','qx','qy','qz'))
            else:
                names.append(name)
        return tuple(names)

    def open_viewer(self):
        if self.deployment.backend != 'sim' or not self._connected:
            raise EffectorError('capability', 'local viewer requires sim backend')
        with self._backend._lock:
            self._viewer = _MuJoCoViewer(self._backend.model, self._backend.data)

    def sync_viewer(self):
        if self._viewer is None or not self._viewer.is_running():
            return False
        with self._backend._lock:
            self._viewer.sync()
        return True
