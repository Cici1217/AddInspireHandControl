"""Resolve Inspire assets and instantiate the 41-joint G1 robot model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

from .config import DEFAULT_MAPPING_PATH

from gear_sonic.utils.hand_control.registry import ROOT as PACKAGE_PARENT


@dataclass(frozen=True)
class InspireAssetPaths:
    model: Path
    scene: Path
    urdf: Path


def load_asset_paths(config_path: str | Path = DEFAULT_MAPPING_PATH) -> InspireAssetPaths:
    """Resolve configured assets and check URDF meshes before loading Pinocchio.

    Configured relative paths are relative to the package parent, even when a
    custom configuration is used. Absolute paths are accepted for local assets.
    """
    with Path(config_path).expanduser().open(encoding="utf-8") as stream:
        simulation = yaml.safe_load(stream)["simulation"]

    def resolve(key: str) -> Path:
        value = simulation[key]
        path = ((Path(__file__).parent / value[8:]) if value.startswith('package:')
                else PACKAGE_PARENT / Path(value).expanduser()).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing Inspire asset: {path}")
        return path

    assets = InspireAssetPaths(resolve("model_path"), resolve("scene_path"), resolve("urdf_path"))
    for mesh in ET.parse(assets.urdf).findall(".//mesh"):
        path = (assets.urdf.parent / mesh.attrib["filename"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing Inspire mesh: {path}")
        with path.open("rb") as stream:
            if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(
                    f"Mesh is a Git LFS pointer: {path}. Run git lfs pull "
                    '--include="gear_sonic/data/robots/g1/meshes/**" --exclude=""'
                )
    return assets


def instantiate_g1_rh56dfx_robot_model(
    *,
    waist_location: str = "lower_and_upper_body",
    high_elbow_pose: bool = False,
    config_path: str | Path | None = None,
):
    """Create the 41-joint model from the selected hand configuration.

    Pinocchio is imported only when this factory is called.
    """
    from gear_sonic.data.robot_model.robot_model import RobotModel
    from .supplemental import (
        G1Rh56dfxSupplementalInfo,
    )
    from gear_sonic.data.robot_model.supplemental_info.g1.g1_supplemental_info import (
        ElbowPose,
        WaistLocation,
    )

    waist = WaistLocation(waist_location)
    elbow = ElbowPose.HIGH if high_elbow_pose else ElbowPose.LOW
    from .config import load_mapping
    supplemental = G1Rh56dfxSupplementalInfo(waist_location=waist, elbow_pose=elbow,
                                          mapping=load_mapping(config_path or DEFAULT_MAPPING_PATH))
    assets = load_asset_paths() if config_path is None else load_asset_paths(config_path)
    model = RobotModel(
        str(assets.urdf),
        str(assets.urdf.parent),
        supplemental_info=supplemental,
    )
    if model.num_joints != 41:
        raise ValueError(f"RH56DFX RobotModel must contain 41 joints, got {model.num_joints}")
    return model


def create_model(deployment):
    return instantiate_g1_rh56dfx_robot_model(
        config_path=deployment.path(deployment.options['hand_config']),
        waist_location=deployment.options.get('waist_location', 'lower_and_upper_body'),
        high_elbow_pose=deployment.options.get('high_elbow_pose', False),
    )


def asset_resources(deployment):
    assets = load_asset_paths(deployment.path(deployment.options['hand_config']))
    return {'urdf': assets.urdf, 'mjcf': assets.model, 'scene': assets.scene}
