"""Physics acceptance and portable training data contracts; no hardware or downloads."""

import json

import numpy as np
import pytest

from gear_sonic.devices.inspire.training.basket import assess
from gear_sonic.devices.inspire.training.job import data_paths, digest, verify_files
from gear_sonic.devices.inspire.training.prepare import base_model_metadata


def test_patch_projection_preserves_outputs_gradients_and_checkpoint_keys():
    from copy import deepcopy
    from types import SimpleNamespace
    import torch
    from gear_sonic.devices.inspire.training.job import linear_patch_projection

    torch.manual_seed(17)
    original = torch.nn.Conv3d(3, 8, (2, 4, 4), stride=(2, 4, 4)).double()
    projection = deepcopy(original)
    module = SimpleNamespace(proj=projection, in_channels=3, temporal_patch_size=2, patch_size=4)
    x = torch.randn(5, 3 * 2 * 4 * 4, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    expected = original(x.reshape(-1, 3, 2, 4, 4)).reshape(-1, 8)
    actual = linear_patch_projection(module, y)
    torch.testing.assert_close(actual, expected)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(y.grad, x.grad)
    for a, b in zip(projection.parameters(), original.parameters()):
        torch.testing.assert_close(a.grad, b.grad)
    assert projection.state_dict().keys() == original.state_dict().keys()
    projection.padding = (1, 1, 1)
    with pytest.raises(ValueError, match="geometry"):
        linear_patch_projection(module, y)


def successful_physics():
    n = 301
    t = np.arange(n) * 20_000_000
    height = np.r_[np.full(100, 0.43), np.linspace(0.43, 0.65, 100), np.full(101, 0.65)]
    return {
        "monotonic_ns": t,
        "qpos": np.tile([0.0, 0.0, 0.76], (n, 1)),
        "basket_xyz": np.column_stack([np.zeros(n), np.zeros(n), height]),
        "hand_contact_force": np.full((n, 2), 3.0),
        "other_contact_force": np.zeros(n),
        "external_force": np.zeros(n),
    }


def test_acceptance_requires_sustained_unsupported_bilateral_lift():
    physics = successful_physics()
    report = assess(physics, 0, 6_000_000_000)
    assert report["passed"] and report["physics_grasp_verified"]
    assert report["minimum_hold_lift_m"] > 0.2


@pytest.mark.parametrize("bad", ["no_lift", "one_hand", "chair_support", "body_support", "fall", "nan", "short"])
def test_unsuccessful_simulations_never_enter_training(bad):
    physics = successful_physics()
    if bad == "no_lift":
        physics["basket_xyz"][:, 2] = 0.43
    if bad == "one_hand":
        physics["hand_contact_force"][-100:, 0] = 0
    if bad == "chair_support":
        physics["other_contact_force"][-100:] = 1
    if bad == "body_support":
        physics["external_force"][100] = 100
    if bad == "fall":
        physics["qpos"][200, 2] = 0.3
    if bad == "nan":
        physics["qpos"][200, 0] = np.nan
    end = 10_000_000 if bad == "short" else 6_000_000_000
    assert not assess(physics, 0, end)["passed"]


def test_bundle_can_move_without_importing_wbc_or_hardware(tmp_path):
    (tmp_path / "datasets").mkdir()
    m = {"dataset_root": "/not/available", "datasets": {"sim_train": "sim/task/train"}}
    root, paths = data_paths(tmp_path, m)
    assert root == tmp_path / "datasets" and paths["sim_train"] == root / "sim/task/train"
    m["datasets"]["sim_train"] = "../../outside"
    with pytest.raises(ValueError):
        data_paths(tmp_path, m)


def test_wrong_checkpoint_statistics_and_changed_labels_are_rejected(tmp_path):
    bundle = tmp_path / "bundle"
    checkpoint = tmp_path / "checkpoint"
    data = tmp_path / "data"
    for path in (bundle, checkpoint, data):
        path.mkdir()
    configs = ("config.json", "processor_config.json", "statistics.json", "embodiment_id.json")
    for name in configs:
        (checkpoint / name).write_text("{}")
        (bundle / name).write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"weight fixture")
    (data / "episode.parquet").write_bytes(b"label fixture")
    manifest = {
        "checkpoint_shards_sha256": {"model.safetensors": digest(checkpoint / "model.safetensors")},
        "checkpoint_config_sha256": {name: digest(checkpoint / name) for name in configs},
        "dataset_files_sha256": {"episode.parquet": digest(data / "episode.parquet")},
    }
    verify_files(bundle, manifest, checkpoint, data)
    (checkpoint / "statistics.json").write_text('{"wrong": true}')
    with pytest.raises(ValueError, match="checkpoint hash differs: statistics.json"):
        verify_files(bundle, manifest, checkpoint, data)
    (checkpoint / "statistics.json").write_text("{}")
    (data / "episode.parquet").write_bytes(b"changed labels")
    with pytest.raises(ValueError, match="dataset hash differs"):
        verify_files(bundle, manifest, checkpoint, data)


def test_task_finetuned_weights_cannot_be_used_as_n17_base(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "Gr00tN1d7", "action_horizon": 40, "max_action_dim": 132, "max_state_dim": 132})
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model.safetensors"}})
    )
    (tmp_path / "model.safetensors").write_bytes(b"task-finetuned weight fixture")
    with pytest.raises(ValueError, match="official nvidia/GR00T-N1.7-3B base weights"):
        base_model_metadata(tmp_path)
