"""RGB-D transport and frame integration, without a connected camera/robot."""

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor, ComposedCameraSensor
from gear_sonic.camera.sensor_server import ImageMessageSchema, ImageUtils
from gear_sonic.tests.test_inspire_recording import build, make_sources
from gear_sonic.devices.inspire.recording.depth import DepthFrame
from gear_sonic.devices.inspire.contract import ContractError


def rgbd_message():
    camera = make_sources()[1]
    depth = depth_frame()
    camera["images"]["ego_view_depth"] = depth.image
    camera["timestamps"]["ego_view_depth"] = camera["timestamps"]["ego_view"]
    camera["metadata"] = {"ego_view": depth.metadata}
    return camera


@pytest.mark.parametrize("full_resolution", [False, True])
def test_depth_round_trip_through_composed_server_msgpack_client_and_recording_frame(full_resolution):
    source = rgbd_message()
    if full_resolution:
        source["images"]["ego_view_depth"] = np.arange(480 * 640, dtype=np.uint16).reshape(480, 640)
        source["metadata"]["ego_view"]["calibration"]["depth_intrinsics"].update(width=640, height=480)
    server = object.__new__(ComposedCameraSensor)
    wire = server.serialize_message({"ego_view": source})
    assert wire["encodings"] == {"ego_view_depth": "png16"}
    decoded = ImageMessageSchema.deserialize(msgpack.unpackb(msgpack.packb(wire), raw=False)).asdict()
    frame = build(camera=decoded)
    np.testing.assert_array_equal(frame.depth.image, source["images"]["ego_view_depth"])
    assert frame.depth.source_timestamp_ns == frame.camera_source_timestamp_ns
    assert frame.depth.metadata == source["metadata"]["ego_view"]
    assert frame.ego_view.shape == (480, 640, 3)
    assert "ego_view_depth" not in frame.to_exporter_dict()


def test_legacy_rgb_message_still_decodes_without_depth_metadata():
    source = make_sources()[1]
    wire = ImageMessageSchema(**source).serialize()
    decoded = ImageMessageSchema.deserialize(wire).asdict()
    assert set(decoded) == {"timestamps", "images"}
    assert build(camera=decoded).depth is None


@pytest.mark.parametrize("policy", ["off", "required", "auto"])
@pytest.mark.parametrize("encoding", ["legacy_jpeg", "png16"])
def test_camera_depth_selection_precedes_wire_decoding(monkeypatch, policy, encoding):
    from gear_sonic.devices.inspire.recording.depth import DepthPolicyCamera

    source = rgbd_message()
    wire = ImageMessageSchema(**source).serialize()
    if encoding == "legacy_jpeg":
        # The old RealSense server included JPEG depth even in RGB-only capture.
        wire.pop("encodings")
        wire.pop("metadata")
        wire["images"]["ego_view_depth"] = ImageUtils.encode_image(
            source["images"]["ego_view_depth"].astype(np.uint8)
        )
    original = copy.deepcopy(wire)
    monkeypatch.setattr(ComposedCameraClientSensor, "start_client", lambda *args: None)
    client = ComposedCameraClientSensor(include_depth=policy != "off")
    monkeypatch.setattr(client, "receive_message_nonblocking", lambda **kwargs: wire)
    camera = DepthPolicyCamera(client, policy)
    if policy != "off" and encoding == "legacy_jpeg":
        with pytest.raises(ValueError, match="lossless depth encoding"):
            camera.read()
    else:
        if policy == "off":
            monkeypatch.setattr(ImageUtils, "decode_depth_image", lambda _: pytest.fail("depth decoded"))
        decoded = camera.read()
        assert decoded["images"]["ego_view"].shape == (480, 640, 3)
        frame = build(camera=decoded)
        if policy == "off":
            assert set(decoded["images"]) == set(decoded["timestamps"]) == {"ego_view"}
            assert frame.depth is None
        else:
            np.testing.assert_array_equal(frame.depth.image, source["images"]["ego_view_depth"])
    assert wire == original


def test_depth_never_silently_uses_jpeg_or_uint8():
    message = rgbd_message()
    message["images"]["ego_view_depth"] = message["images"]["ego_view_depth"].astype(np.uint8)
    with pytest.raises(ValueError, match="uint16"):
        ImageMessageSchema(**message).serialize()
    with pytest.raises(ValueError, match="lossless"):
        ImageMessageSchema.deserialize({"images": {"ego_view_depth": "legacy-jpeg"}})


@pytest.mark.parametrize("bad", ["scale", "intrinsics", "metadata", "timestamp"])
def test_depth_requires_usable_calibration_and_time(bad):
    message = copy.deepcopy(rgbd_message())
    if bad == "scale":
        message["metadata"]["ego_view"]["calibration"]["depth_scale_m"] = 0
    elif bad == "intrinsics":
        message["metadata"]["ego_view"]["calibration"]["depth_intrinsics"]["width"] = 9
    elif bad == "metadata":
        del message["metadata"]
    else:
        message["timestamps"]["ego_view_depth"] = float("nan")
    with pytest.raises(ContractError):
        DepthFrame.from_camera_message(message)


def test_realsense_driver_preserves_native_values_calibration_and_sdk_timestamps(monkeypatch):
    """Exercise the existing driver against a small SDK double, including serial selection."""
    pixels = np.array([[0, 1000], [12345, 65535]], dtype=np.uint16)
    selected = []
    intrinsics = SimpleNamespace(
        width=2, height=2, fx=300.0, fy=301.0, ppx=0.5, ppy=0.5, model="brown_conrady", coeffs=[0.0] * 5
    )
    stream_profile = SimpleNamespace(
        get_intrinsics=lambda: intrinsics,
        get_extrinsics_to=lambda other: SimpleNamespace(
            rotation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], translation=[0.015, 0.0, 0.0]
        ),
    )
    stream_profile.as_video_stream_profile = lambda: stream_profile
    device = SimpleNamespace(
        get_info=lambda key: {"serial": "G1-test", "name": "RealSense", "firmware": "test"}[key],
        first_depth_sensor=lambda: SimpleNamespace(get_depth_scale=lambda: 0.001),
    )
    profile = SimpleNamespace(get_stream=lambda stream: stream_profile, get_device=lambda: device)

    def sdk_frame(data, number):
        return SimpleNamespace(
            get_data=lambda: data,
            get_frame_number=lambda: number,
            get_timestamp=lambda: 500.25,
            get_frame_timestamp_domain=lambda: "hardware_clock",
        )

    frames = SimpleNamespace(
        get_color_frame=lambda: sdk_frame(np.zeros((2, 2, 3), dtype=np.uint8), 11),
        get_depth_frame=lambda: sdk_frame(pixels, 12),
    )
    pipeline = SimpleNamespace(start=lambda config: profile, wait_for_frames=lambda: frames, stop=lambda: None)
    sdk = SimpleNamespace(
        context=lambda: SimpleNamespace(query_devices=lambda: [device]),
        pipeline=lambda: pipeline,
        config=lambda: SimpleNamespace(enable_device=selected.append, enable_stream=lambda *args: None),
        camera_info=SimpleNamespace(serial_number="serial", name="name", firmware_version="firmware"),
        stream=SimpleNamespace(color="color", depth="depth"),
        format=SimpleNamespace(rgb8="rgb8", z16="z16"),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", sdk)
    path = Path(__file__).resolve().parents[1] / "camera/drivers/realsense.py"
    spec = importlib.util.spec_from_file_location("depth_test_realsense", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    camera = driver.RealSenseSensor(device_id="G1-test")
    message = ImageMessageSchema.deserialize(camera.serialize(camera.read())).asdict()
    result = DepthFrame.from_camera_message(message)
    assert selected == ["G1-test"]
    np.testing.assert_array_equal(result.image, pixels)
    assert result.metadata["calibration"]["depth_scale_m"] == 0.001
    assert result.metadata["calibration"]["depth_to_color"]["translation_m"] == [0.015, 0.0, 0.0]
    assert result.metadata["frames"]["depth"] == {
        "frame_number": 12,
        "timestamp_ms": 500.25,
        "timestamp_domain": "hardware_clock",
    }
    pixels[:] = 0
    assert result.image[1, 1] == 65535
    camera.close()


def depth_frame():
    return DepthFrame(
        np.array([[0, 1000], [12345, 65535]], dtype=np.uint16),
        123000000000,
        {
            "calibration": {
                "depth_scale_m": 0.001,
                "depth_intrinsics": {"height": 2, "width": 2, "fx": 300, "fy": 300, "ppx": 0.5, "ppy": 0.5},
            }
        },
    )


def test_depth_policy_is_explicit_and_does_not_mutate_message():
    from gear_sonic.devices.inspire.recording.depth import DepthPolicyCamera

    message = rgbd_message()
    source = SimpleNamespace(read=lambda: message)
    assert "ego_view_depth" not in DepthPolicyCamera(source, "off").read()["images"]
    assert "ego_view_depth" in message["images"]
    assert DepthPolicyCamera(source, "required").read() is message
    with pytest.raises(RuntimeError, match="RGB"):
        DepthPolicyCamera(SimpleNamespace(read=lambda: make_sources()[1]), "required").read()
