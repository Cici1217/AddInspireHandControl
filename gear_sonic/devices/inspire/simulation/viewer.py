"""Owned MuJoCo viewer thread for the standalone Inspire simulation."""
import threading
from gear_sonic.utils.hand_control.interface import EffectorError


class _MuJoCoViewer:
    """Own the render thread so GLFW shutdown cannot race Python process exit.

    MuJoCo's public launch_passive starts a daemon thread and close only requests
    exit. Use its installed launcher with an owned thread, retaining its renderer,
    synchronization and model-loading implementation rather than copying them.
    """

    def __init__(self, model, data):
        import queue
        import mujoco.viewer
        self._ready = queue.Queue(1)
        self._error = None
        self._handle = None

        def render():
            try:
                mujoco.viewer._launch_internal(model, data, run_physics_thread=False, handle_return=self._ready)
            except BaseException as exc:
                self._error = exc
                if self._handle is None:
                    try:
                        self._ready.put_nowait(None)
                    except queue.Full:
                        pass

        self._thread = threading.Thread(target=render, name='end-effector-viewer', daemon=False)
        self._thread.start()
        self._handle = self._ready.get()
        if self._handle is None:
            self._thread.join()
            raise EffectorError('backend_fault', f'MuJoCo viewer initialization failed: {self._error}')

    def is_running(self):
        if self._error is not None:
            raise EffectorError('backend_fault', f'MuJoCo viewer failed: {self._error}')
        return self._handle.is_running()

    def sync(self):
        self._handle.sync()

    def close(self):
        self._handle.close()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise EffectorError('lifecycle', 'MuJoCo viewer did not finish closing')
