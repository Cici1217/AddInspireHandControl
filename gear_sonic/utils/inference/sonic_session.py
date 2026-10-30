"""Shared SONIC transport extracted unchanged from the verified dataset replay."""
import time
import numpy as np

class SonicSession:
    """Existing SONIC planner/pose protocol with feedback and owned socket cleanup."""
    def __init__(self, command_endpoint, state_endpoint, *, state_timeout=.5):
        import zmq
        self.zmq = zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.states = self.context.socket(zmq.SUB)
        self.started = False
        self.last_feedback = None
        self.state = None
        self.state_timeout = state_timeout
        self.pose_selected = False
        try:
            self.socket.setsockopt(zmq.LINGER, 0); self.socket.setsockopt(zmq.SNDHWM, 3)
            self.socket.bind(command_endpoint)
            self.states.setsockopt(zmq.LINGER, 0); self.states.setsockopt(zmq.CONFLATE, 1)
            self.states.setsockopt(zmq.SUBSCRIBE, b'g1_debug'); self.states.connect(state_endpoint)
        except BaseException:
            self.close(); raise

    def command(self, *, planner, stop=False):
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message
        if not stop: self.started = True
        self.socket.send(build_command_message(start=not stop, stop=stop, planner=planner))

    def idle(self):
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_planner_message
        self.socket.send(build_planner_message(0, [0., 0., 0.], [1., 0., 0.], -1., -1.))

    def feedback(self, *, required=True):
        import msgpack
        import msgpack_numpy as mnp
        try:
            raw = self.states.recv(self.zmq.NOBLOCK)
        except self.zmq.Again:
            raw = None
        if raw is not None:
            row = msgpack.unpackb(raw[len(b'g1_debug'):], raw=False, object_hook=mnp.decode)
            for key, shape in (('body_q', (29,)), ('token_state', (64,))):
                value = np.asarray(row.get(key))
                if value.shape != shape or not np.all(np.isfinite(value)):
                    raise RuntimeError(f'incomplete SONIC feedback: {key}')
            self.state = row
            self.last_feedback = time.monotonic()
        if required and (self.last_feedback is None or time.monotonic() - self.last_feedback > self.state_timeout):
            raise RuntimeError('SONIC body feedback stopped during replay')
        return raw is not None

    def pose(self, data, *, version=3):
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message
        if version not in (1, 3):
            raise ValueError('body reference requires joint protocol v1 or SMPL protocol v3')
        self.socket.send(pack_pose_message(data, version=version))

    def token(self, value, frame_index):
        """Official SONIC v4; end effectors keep their independent driver."""
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (64,) or not np.isfinite(value).all():
            raise ValueError('SONIC requires a finite motion token64')
        self.socket.send(pack_pose_message({'token_state': value[None],
                         'frame_index': np.array([frame_index], dtype=np.int64)}, version=4))

    def stop(self):
        if self.started:
            for _ in range(3):
                self.command(planner=not self.pose_selected, stop=True); time.sleep(.02)
            self.started = False

    def close(self):
        self.states.close(0); self.socket.close(0); self.context.term()
