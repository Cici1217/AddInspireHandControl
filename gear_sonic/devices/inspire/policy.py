"""G1/Inspire policy contract, ported from the original native6 VLA adapters.

Policy input is measured body29 + left6 + right6. Policy output is a SONIC
token64 and two native6 targets, never a 29-joint body command.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .contract import ContractError

HORIZON = 40
ACTION_DIMS = {'motion_token': 64, 'left_hand_joints': 6, 'right_hand_joints': 6}
STATE_DIMS = {'left_leg': 6, 'right_leg': 6, 'waist': 3, 'left_arm': 7, 'right_arm': 7,
              'left_hand': 6, 'right_hand': 6, 'projected_gravity': 3}


@dataclass(frozen=True)
class PolicyChunk:
    motion_token: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray

    @property
    def horizon(self):
        return len(self.motion_token)


def extract_policy_chunk(action):
    if not isinstance(action, dict):
        raise ContractError('policy response must contain an action dictionary')
    values = []
    for name, width in ACTION_DIMS.items():
        if name in action and f'action.{name}' in action:
            raise ContractError(f'ambiguous policy field: {name}')
        value = np.asarray(action.get(name, action.get(f'action.{name}')), dtype=np.float32)
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 2 or value.shape[1] != width or not len(value) or not np.isfinite(value).all():
            raise ContractError(f'policy {name} must be finite [T, {width}], got {value.shape}')
        value = value.copy(); value.setflags(write=False); values.append(value)
    if len({len(value) for value in values}) != 1:
        raise ContractError('policy body/left/right horizons do not match')
    return PolicyChunk(*values)


def rtc_options(chunk, overlap=20):
    if chunk.horizon != HORIZON or not 5 < overlap <= HORIZON:
        raise ContractError('RTC requires a 40-frame history and overlap greater than 5')
    return {'rtc_previous_actions': {'motion_token': chunk.motion_token[None].copy(),
                                    'left_hand_joints': chunk.left_hand[None].copy(),
                                    'right_hand_joints': chunk.right_hand[None].copy()},
            'action_horizon': HORIZON, 'rtc_overlap_steps': overlap,
            'rtc_frozen_steps': 5, 'rtc_ramp_rate': 1.0}


def constant_chunk(token, action):
    return PolicyChunk(*(np.repeat(np.asarray(values, dtype=np.float32)[None], HORIZON, axis=0)
                         for values in (token, action.values['left'], action.values['right'])))


class ObservationBuilder:
    def __init__(self, camera, model, description, prompt):
        self.camera, self.model, self.description, self.prompt = camera, model, description, prompt
        self.last_timestamp = 0.

    def build(self, body, hands, now_ns):
        from gear_sonic.utils.inference.vla_utils import prepare_observation_for_eval
        from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
        hands.require_fresh(now_ns, self.description)
        image = self.camera.read()
        if image is None:
            return None
        timestamp = image.get('timestamps', {}).get('ego_view')
        if isinstance(timestamp, bool) or not isinstance(timestamp, (float, int)) or not np.isfinite(timestamp) or timestamp <= 0:
            raise ContractError('camera requires a positive ego_view timestamp')
        if timestamp <= self.last_timestamp:
            return None
        rgb = np.asarray(image.get('images', {}).get('ego_view'))
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ContractError('ego_view must be an RGB uint8 image')
        q, quat = np.asarray(body.get('body_q')), np.asarray(body.get('base_quat'))
        if q.shape != (29,) or not np.isfinite(q).all():
            raise ContractError('body observation requires finite body29 feedback')
        if quat.shape != (4,) or not np.isfinite(quat).all() or abs(np.linalg.norm(quat)-1) > .05:
            raise ContractError('body observation requires a unit base quaternion')
        # RobotModel places the left hand before the right arm. DDS body29
        # followed by native6 hands is a transport order, not its configuration.
        configuration = self.model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=q,
            left_hand_actuated_joint_values=np.asarray(hands.values['left']),
            right_hand_actuated_joint_values=np.asarray(hands.values['right']),
        )
        observation = {'video': {'ego_view': rgb[None, None].copy()}, 'state': {},
                       'language': {'annotation.human.task_description': [[self.prompt]]},
                       'q': configuration[None, None].astype(np.float32)}
        prepare_observation_for_eval(self.model, observation)
        observation['state']['projected_gravity'] = np.asarray(compute_projected_gravity(quat), dtype=np.float32)[None, None]
        # Native6 coordinates are already the policy channels, without Dex3 remapping.
        for side in ('left', 'right'):
            observation['state'][side+'_hand'] = np.asarray(hands.values[side], dtype=np.float32)[None, None]
        self.last_timestamp = timestamp
        return observation


def check_checkpoint(path):
    """Check the actual native6 modality and all weight shards before loading a GPU."""
    root = Path(path).expanduser().resolve()
    def read(name):
        return json.loads((root/name).read_text())
    processor = read('processor_config.json')['processor_kwargs']['modality_configs']['unitree_g1_sonic']
    stats = read('statistics.json')['unitree_g1_sonic']
    for kind, dims in (('state', STATE_DIMS), ('action', ACTION_DIMS)):
        if set(processor[kind]['modality_keys']) != set(dims):
            raise ValueError(f'checkpoint {kind} does not match G1/Inspire native6')
        for name, width in dims.items():
            if len(stats[kind][name]['mean']) != width:
                raise ValueError(f'checkpoint {kind}.{name} must have {width} channels')
    if processor['action']['delta_indices'] != list(range(HORIZON)):
        raise ValueError('checkpoint must predict 40 actions at 50 Hz')
    if processor['video']['modality_keys'] != ['ego_view'] or processor['video']['delta_indices'] != [0]:
        raise ValueError('checkpoint must use a single current ego_view frame')
    index = root/'model.safetensors.index.json'
    shards = set(read(index.name)['weight_map'].values()) if index.is_file() else {'model.safetensors'}
    if not shards:
        raise ValueError('checkpoint has no weight shards')
    for name in shards:
        file = (root/name).resolve()
        if not file.is_relative_to(root) or not file.is_file() or file.stat().st_size < 1024:
            raise ValueError(f'missing or incomplete model weights: {name}')
        with file.open('rb') as stream:
            if stream.read(64).startswith(b'version https://git-lfs'):
                raise ValueError(f'model weights are only a Git LFS pointer: {name}')
    read('config.json')
    return {'model': str(root), 'embodiment': 'UNITREE_G1_SONIC', 'action_horizon': HORIZON,
            'action_dims': ACTION_DIMS, 'weight_shards': sorted(shards), 'control_connections_opened': False}
