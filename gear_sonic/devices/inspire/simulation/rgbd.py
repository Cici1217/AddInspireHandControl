"""Optional RGB-D environment using the official simulator's env_factory seam."""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

from gear_sonic.camera.sensor_server import ImageMessageSchema, SensorServer
from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv


def metric_depth_to_uint16(depth, far):
    """Store millimetres; clipped background and out-of-range pixels are invalid."""
    value = np.asarray(depth)
    valid = np.isfinite(value) & (value > 0) & (value < far * 0.999) & (value <= 65.535)
    result = np.zeros(value.shape, dtype=np.uint16)
    result[valid] = np.maximum(1, np.rint(value[valid] * 1000)).astype(np.uint16)
    return result


class RGBDPublisher:
    """A bounded latest-frame queue keeps PNG encoding off the physics loop."""

    def __init__(self, port):
        self.queue = queue.Queue(maxsize=1)
        self.stopped = threading.Event()
        self.started = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, args=(port,), daemon=True)
        self.thread.start()
        if not self.started.wait(5):
            raise TimeoutError("RGB-D publisher startup timeout")
        if self.error:
            raise self.error

    def _run(self, port):
        server = SensorServer()
        try:
            server.start_server(port)
            print("INSPIRE_RGBD_READY encoding=uint16_png depth_scale_m=0.001", flush=True)
            self.started.set()
            while not self.stopped.is_set():
                try:
                    frame = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                server.send_message(ImageMessageSchema(**frame).serialize())
        except BaseException as exc:
            self.error = exc
            self.started.set()
        finally:
            if hasattr(server, "socket"):
                server.stop_server()

    def publish(self, frame):
        if self.error:
            raise RuntimeError("RGB-D publisher failed") from self.error
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait(frame)

    def stop(self):
        self.stopped.set()
        self.thread.join(timeout=5)


class InspireRGBDEnv(DefaultEnv):
    def start_image_publish_subprocess(self, start_method="spawn", camera_port=5555):
        self.image_publish_process = RGBDPublisher(camera_port)
        self._rgbd_frame = 0

    def update_render_caches(self):
        if self.image_publish_process is None:
            return super().update_render_caches()
        images, timestamps, metadata, cache = {}, {}, {}, {}
        stamp = time.time()
        for name, cfg in self.camera_configs.items():
            renderer = self.renderers[name]
            camera_name = cfg.get("mjcf_name", name)
            camera_id = self.mj_model.camera(camera_name).id
            renderer.disable_depth_rendering()
            renderer.update_scene(self.mj_data, camera=camera_name)
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            try:
                depth_m = renderer.render().copy()
            finally:
                renderer.disable_depth_rendering()
            far = float(self.mj_model.vis.map.zfar * self.mj_model.stat.extent)
            depth = metric_depth_to_uint16(depth_m, far)
            h, w = depth.shape
            sensor_size = self.mj_model.cam_sensorsize[camera_id]
            if np.all(sensor_size > 0):
                focal = self.mj_model.cam_intrinsic[camera_id, :2] / sensor_size * [w, h]
                fx, fy = map(float, focal)
            else:
                fx = fy = float(h / (2 * np.tan(np.deg2rad(self.mj_model.cam_fovy[camera_id]) / 2)))
            intrinsics = dict(
                width=w,
                height=h,
                fx=fx,
                fy=fy,
                ppx=(w - 1) / 2,
                ppy=(h - 1) / 2,
                distortion_model="none",
                distortion_coeffs=[0.0] * 5,
            )
            calibration = dict(
                device_name="MuJoCo virtual camera",
                serial_number=camera_name,
                depth_scale_m=0.001,
                depth_intrinsics=intrinsics,
                color_intrinsics=intrinsics,
                depth_to_color={"rotation": np.eye(3).reshape(-1).tolist(), "translation": [0.0, 0.0, 0.0]},
                depth_aligned_to_color=True,
                invalid_depth_value=0,
                world_extrinsics=None,
                simulated=True,
            )
            optical_to_world = self.mj_data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
            metadata[name] = {
                "calibration": calibration,
                "frame": {
                    "frame_number": self._rgbd_frame,
                    "timestamp_ms": float(self.mj_data.time * 1000),
                    "timestamp_domain": "simulation_time",
                    "camera_to_world": {
                        "rotation_row_major": optical_to_world.reshape(-1).tolist(),
                        "translation_m": self.mj_data.cam_xpos[camera_id].tolist(),
                    },
                },
            }
            images[name], images[name + "_depth"] = rgb, depth
            timestamps[name] = timestamps[name + "_depth"] = stamp
            cache[name + "_image"] = rgb
        self._rgbd_frame += 1
        self.image_publish_process.publish(dict(images=images, timestamps=timestamps, metadata=metadata))
        return cache
