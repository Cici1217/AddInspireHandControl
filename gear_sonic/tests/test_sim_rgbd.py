"""Render actual MuJoCo depth and round-trip it through the RGB-D publisher."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import time

import mujoco
import numpy as np

from gear_sonic.camera.sensor_server import ImageMessageSchema, SensorClient
from gear_sonic.devices.inspire.simulation.rgbd import InspireRGBDEnv, metric_depth_to_uint16


def test_metric_depth_clipping():
    value = metric_depth_to_uint16(np.array([[0.0, np.nan, np.inf, -1, 2.0, 80.0, 5.0]]), 5.0)
    assert value.tolist() == [[0, 0, 0, 0, 2000, 0, 0]]


def test_mujoco_rgb_depth_pair_through_real_zmq():
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    model = mujoco.MjModel.from_xml_string("""<mujoco><visual><global offwidth="64" offheight="48"/></visual>
      <worldbody><camera name="head_camera" pos="0 0 2"/>
      <geom type="plane" size="5 5 .1" rgba=".5 .5 .5 1"/></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = object.__new__(InspireRGBDEnv)
    env.mj_model, env.mj_data = model, data
    env.camera_configs = {"ego_view": {"mjcf_name": "head_camera", "width": 64, "height": 48}}
    env.renderers = {"ego_view": mujoco.Renderer(model, height=48, width=64)}
    client = SensorClient()
    client.start_client("127.0.0.1", port)
    env.start_image_publish_subprocess(camera_port=port)
    message = None
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and message is None:
            env.update_render_caches()
            message = client.receive_message_nonblocking(50)
        assert message is not None
        decoded = ImageMessageSchema.deserialize(message).asdict()
        assert decoded["images"]["ego_view"].shape == (48, 64, 3)
        depth = decoded["images"]["ego_view_depth"]
        assert depth.dtype == np.uint16
        assert abs(int(depth[24, 32]) - 2000) <= 1
        assert decoded["timestamps"]["ego_view"] == decoded["timestamps"]["ego_view_depth"]
        assert decoded["metadata"]["ego_view"]["calibration"]["simulated"] is True
    finally:
        env.image_publish_process.stop()
        client.stop_client()
        env.renderers["ego_view"].close()
