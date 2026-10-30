"""State replay validates its own saved scene and never becomes a robot command."""
from argparse import Namespace
import json

import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')
from gear_sonic.devices.inspire.recording.simulation import SCHEMA, digest, load, run


@pytest.fixture
def recording(tmp_path):
    (tmp_path/'meta').mkdir()
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
    mujoco.mj_saveModel(model, str(tmp_path/'scene.mjb'))
    qpos=np.tile(model.qpos0,(5,1));qpos[:,2]=np.arange(5)*.01
    np.savez(tmp_path/'state.npz',qpos=qpos,qvel=np.zeros((5,model.nv)),timestamp=np.arange(5)/50)
    meta={'schema':SCHEMA,'mujoco_version':mujoco.__version__,'fps':50,
          'scene':'scene.mjb','scene_sha256':digest(tmp_path/'scene.mjb'),
          'episodes':[{'file':'state.npz','sha256':digest(tmp_path/'state.npz'),'frames':5,'light_scale':1.0}]}
    (tmp_path/'meta/simulation_replay.json').write_text(json.dumps(meta))
    return tmp_path,meta,qpos


def test_saved_model_and_states_roundtrip(recording):
    root,_,expected=recording
    model,arrays=load(root,0)
    data=mujoco.MjData(model)
    for qpos in arrays['qpos']:
        data.qpos[:]=qpos;mujoco.mj_forward(model,data)
    np.testing.assert_array_equal(data.qpos,expected[-1])
    assert model.nq==7


@pytest.mark.parametrize('failure',['hash','path','version','frames','incomplete'])
def test_changed_or_incomplete_state_recordings_are_rejected(recording, failure):
    root,meta,_=recording
    if failure=='hash':meta['episodes'][0]['sha256']='0'*64
    if failure=='path':meta['episodes'][0]['file']='../state.npz'
    if failure=='version':meta['mujoco_version']='0.0'
    if failure=='frames':meta['episodes'][0]['frames']=6
    if failure=='incomplete':(root/'INCOMPLETE').touch()
    (root/'meta/simulation_replay.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError):load(root,0)


def test_real_target_is_rejected_before_loading_or_connecting():
    with pytest.raises(ValueError,match='只能在仿真'):
        run(Namespace(target='real',dataset='/does/not/exist'),None)
