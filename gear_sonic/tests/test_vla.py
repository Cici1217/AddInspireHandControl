"""VLA contract and timing regression; all endpoints remain local or unopened."""
from dataclasses import replace
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.devices.inspire.policy import (PolicyChunk, extract_policy_chunk, rtc_options,
                                              constant_chunk, ObservationBuilder, check_checkpoint)
from gear_sonic.devices.inspire.scheduling import (PolicyResult, TimedActionBuffer, BodyTokenContinuityGuard,
    StalePolicyChunkError, ActionBufferUnderrun, UnsafePolicyAction, AsyncPolicyWorker)
from gear_sonic.devices.inspire.inference import validate_result, load_pose
from gear_sonic.utils.hand_control.interface import EffectorAction, EffectorState, EffectorError
from gear_sonic.utils.hand_control.registry import deployment, ROOT
from gear_sonic.scripts.run_device_vla import build_parser, run


@pytest.mark.parametrize('cleanup_fails', [False, True])
def test_control_failure_survives_cleanup_and_still_propagates(cleanup_fails, capsys):
    from contextlib import ExitStack
    from gear_sonic.devices.inspire.inference import record_failure
    from gear_sonic.devices.inspire.driver.session import OperatorSessionError

    report = {}
    cleaned = []
    original = RuntimeError('gateway FAULT: serial feedback lost')
    stop_error = OperatorSessionError('DISARM could not be verified')

    def close_hand():
        cleaned.append('hand')
        if cleanup_fails:
            raise stop_error

    def execute():
        try:
            with ExitStack() as owned:
                owned.callback(lambda: cleaned.append('socket'))
                owned.callback(close_hand)
                owned.callback(lambda: cleaned.append('body'))
                raise original
        except BaseException as exc:
            record_failure(report, exc)
            raise

    with pytest.raises(RuntimeError) as caught:
        execute()
    assert caught.value is (stop_error if cleanup_fails else original)
    assert cleaned == ['body', 'hand', 'socket']
    assert report['error_chain'][0] == 'RuntimeError: gateway FAULT: serial feedback lost'
    assert report['error'] == report['error_chain'][-1]
    assert 'serial feedback lost' in report['traceback']
    output = capsys.readouterr().out
    if cleanup_fails:
        assert 'DISARM could not be verified' in report['error']
        assert 'VLA_CAUSE RuntimeError: gateway FAULT: serial feedback lost' in output
    else:
        assert len(report['error_chain']) == 1 and not output


def chunk():
    return PolicyChunk(np.zeros((40,64),np.float32),np.zeros((40,6),np.float32),np.zeros((40,6),np.float32))


def result(submitted=0, completed=0):
    return PolicyResult(3,submitted,completed,chunk(),{'rtc_applied':True})


def test_async_timeline_accounts_for_inference_and_poll_delay_once():
    timeline=TimedActionBuffer(control_rate_hz=50,maximum_hold_s=.10)
    activation=timeline.activate(result(1_000_000_000,1_080_000_000),activated_monotonic_ns=1_100_000_000)
    assert activation.skipped_frames==5 and activation.remaining_frames==35
    assert timeline.sample(now_monotonic_ns=1_140_000_000).chunk_index==7
    with pytest.raises(StalePolicyChunkError):
        timeline.activate(result(0,800_000_000),activated_monotonic_ns=800_000_000)
    timeline.activate(result(),activated_monotonic_ns=0)
    assert timeline.sample(now_monotonic_ns=820_000_000).holding_last_frame
    with pytest.raises(ActionBufferUnderrun):timeline.sample(now_monotonic_ns=920_000_000)


def test_token_limits_cover_within_chunk_cross_chunk_and_first_transition():
    guard=BodyTokenContinuityGuard();guard.seed(np.zeros(64))
    guard.validate_chunk(chunk());guard.validate_transition(np.full(64,.2))
    guard.validate_and_commit(np.full(64,-.1))  # Preview did not commit .2.
    with pytest.raises(UnsafePolicyAction):guard.validate_transition(np.full(64,.2))
    bad=chunk();bad.motion_token[8,2]=.3
    with pytest.raises(UnsafePolicyAction,match='internal step'):guard.validate_chunk(bad)
    bad.motion_token[8,2]=1.01
    with pytest.raises(UnsafePolicyAction,match='absolute'):guard.validate_chunk(bad)


@pytest.mark.parametrize('failure',['dimension','nan','horizon','ambiguous','missing'])
def test_policy_decoder_rejects_malformed_actions(failure):
    row={k:np.zeros((1,40,d)) for k,d in [('motion_token',64),('left_hand_joints',6),('right_hand_joints',6)]}
    if failure=='dimension':row['left_hand_joints']=np.zeros((40,7))
    if failure=='nan':row['motion_token'][0,1,4]=np.nan
    if failure=='horizon':row['right_hand_joints']=np.zeros((1,39,6))
    if failure=='ambiguous':row['action.motion_token']=row['motion_token']
    if failure=='missing':del row['motion_token']
    with pytest.raises(ValueError):extract_policy_chunk(row)


def test_startup_rtc_preserves_physical_units_and_copies_history():
    action=EffectorAction({'left':[.1]*6,'right':[.2]*6})
    c=constant_chunk(np.arange(64)/100,action)
    options=rtc_options(c)
    assert options['rtc_overlap_steps']==20 and options['rtc_frozen_steps']==5
    assert options['rtc_previous_actions']['left_hand_joints'].shape==(1,40,6)
    assert options['rtc_previous_actions']['left_hand_joints'][0,39,0]==pytest.approx(.1)
    options['rtc_previous_actions']['motion_token'][0,0,0]=99
    assert c.motion_token[0,0]==0
    with pytest.raises(ValueError):rtc_options(c,5)


def test_rtc_ack_and_native6_limits_are_checked_before_send():
    adapter=deployment('inspire','sim_shared').create()  # Never connected.
    guard=BodyTokenContinuityGuard()
    validate_result(result(),adapter,guard)
    with pytest.raises(RuntimeError,match='RTC'):validate_result(replace(result(),info={}),adapter,guard)
    bad=result();bad.chunk.left_hand[3,0]=5
    with pytest.raises(EffectorError):validate_result(bad,adapter,guard)
    with pytest.raises(ValueError,match='40'):validate_result(replace(result(),chunk=replace(chunk(),motion_token=np.zeros((1,64)))),adapter,guard)


def test_observation_uses_real_robot_model_native6_and_new_camera_frames():
    config=deployment('inspire','sim_shared');model=config.robot_model();now=time.monotonic_ns()
    state=EffectorState({'left':[.1]*6,'right':[.2]*6},now,{'left':True,'right':True})
    frame={'timestamps':{'ego_view':10.},'images':{'ego_view':np.zeros((32,48,3),np.uint8)}}
    camera=SimpleNamespace(read=lambda:frame)
    builder=ObservationBuilder(camera,model,config.registration.description,'lift basket')
    body={'body_q':np.arange(29,dtype=np.float32)/100,'base_quat':np.array([1.,0,0,0])}
    observation=builder.build(body,state,now)
    assert observation['state']['left_hand'].shape==(1,1,6)
    assert observation['state']['left_leg'].shape==(1,1,6)
    assert observation['state']['left_hand'][0,0,0]==pytest.approx(.1)
    np.testing.assert_array_equal(observation['state']['left_arm'][0,0],body['body_q'][15:22])
    np.testing.assert_array_equal(observation['state']['right_arm'][0,0],body['body_q'][22:29])
    np.testing.assert_array_equal(model.get_body_actuated_joints(observation['q'][0,0]),body['body_q'])
    assert observation['video']['ego_view'].shape==(1,1,32,48,3)
    assert builder.build(body,state,now) is None
    frame['timestamps']['ego_view']=11.
    with pytest.raises(ValueError,match='quaternion'):builder.build(body|{'base_quat':np.zeros(4)},state,now)
    with pytest.raises(EffectorError):builder.build(body,state,now+10_000_000_000)


def test_model_check_rejects_wrong_hands_missing_shard_and_lfs(tmp_path):
    from gear_sonic.devices.inspire.policy import STATE_DIMS,ACTION_DIMS
    modality={kind:{'modality_keys':list(dims)} for kind,dims in [('state',STATE_DIMS),('action',ACTION_DIMS)]}
    modality['action']['delta_indices']=list(range(40));modality['video']={'modality_keys':['ego_view'],'delta_indices':[0]}
    stats={kind:{k:{'mean':[0]*d} for k,d in dims.items()} for kind,dims in [('state',STATE_DIMS),('action',ACTION_DIMS)]}
    def save(name,data):(tmp_path/name).write_text(json.dumps(data))
    save('processor_config.json',{'processor_kwargs':{'modality_configs':{'unitree_g1_sonic':modality}}})
    save('statistics.json',{'unitree_g1_sonic':stats});save('config.json',{})
    with pytest.raises(ValueError,match='weights'):check_checkpoint(tmp_path)
    (tmp_path/'model.safetensors').write_bytes(b'version https://git-lfs'+b' '*2048)
    with pytest.raises(ValueError,match='LFS'):check_checkpoint(tmp_path)
    (tmp_path/'model.safetensors').write_bytes(b' '*2048)
    assert check_checkpoint(tmp_path)['action_dims']['left_hand_joints']==6
    stats['state']['left_hand']['mean']=[0]*7;save('statistics.json',{'unitree_g1_sonic':stats})
    with pytest.raises(ValueError,match='6 channels'):check_checkpoint(tmp_path)


def test_async_worker_owns_client_and_returns_errors_without_blocking_control():
    calls=[];owner=threading.get_ident()
    class Client:
        def __init__(self):calls.append(threading.get_ident())
        def get_action(self,obs,options=None):
            calls.append(threading.get_ident());time.sleep(.06)
            if obs.get('fail'):raise RuntimeError('policy offline')
            return {k:np.zeros((40,d)) for k,d in [('motion_token',64),('left_hand_joints',6),('right_hand_joints',6)]},{'rtc_applied':True}
        def close(self):calls.append(threading.get_ident())
    worker=AsyncPolicyWorker(Client)
    try:
        worker.wait_ready(2);assert worker.submit({})==1
        assert worker.submit({}) is None
        deadline=time.monotonic()+2;reply=None
        while reply is None and time.monotonic()<deadline:time.sleep(.01);reply=worker.poll()
        assert reply.chunk.horizon==40 and reply.info['rtc_applied']
        assert worker.submit({'fail':True})==2
        reply=None
        while reply is None and time.monotonic()<deadline:time.sleep(.01);reply=worker.poll()
        assert str(reply.error)=='policy offline'
    finally:worker.close(join_timeout_s=2)
    assert len(set(calls))==1 and calls[0]!=owner


@pytest.mark.parametrize('target',['sim','real'])
def test_cli_default_check_opens_no_control_socket(target,monkeypatch):
    from gear_sonic.devices.inspire import inference
    def forbidden(*args,**kwargs):raise AssertionError('created a control connection')
    monkeypatch.setattr(inference,'SonicSession',forbidden)
    summary=run(build_parser().parse_args(['--target',target,'--prompt','test']))
    assert summary['control_connections_opened'] is False and summary['control_hz']==50


def test_sim_pose_validates_limits_and_real_auto_start_is_rejected(tmp_path):
    config=deployment('inspire','sim_shared')
    pose=load_pose(ROOT/'gear_sonic/devices/inspire/assets/vla_initial_pose.txt',config)
    assert pose.shape==(29,)
    p=tmp_path/'bad';np.savetxt(p,np.full(29,100))
    with pytest.raises(ValueError,match='limit'):load_pose(p,config)
    args=build_parser().parse_args(['--target','real','--prompt','test','--execute','--auto-start'])
    with pytest.raises(ValueError,match='simulation'):run(args)


def test_short_catchup_tick_ramps_for_actual_interval_without_relaxing_driver():
    from gear_sonic.devices.inspire.inference import limit_for_tick
    from gear_sonic.devices.inspire.safety import CommandGuard
    from gear_sonic.devices.inspire.contract import InspireHandCommand
    adapter=deployment('inspire','sim_shared').create()
    previous=EffectorAction({'left':[0]*6,'right':[0]*6})
    desired=EffectorAction({'left':[1,1,1,1,0,0],'right':[1,1,1,1,0,0]})
    t=1_000_000_000;guard=CommandGuard(adapter.mapping);guard.arm()
    guard.accept(InspireHandCommand(1,t,np.zeros(6),np.zeros(6)),now_ns=t)
    for seq,dt in enumerate([.02,.027,.013,.025,.015],2):
        now=t+int(dt*1e9);action=limit_for_tick(adapter,previous,desired,t,now)
        guard.accept(InspireHandCommand(seq,now,np.asarray(action.values['left']),np.asarray(action.values['right'])),now_ns=now)
        assert 0<action.values['left'][0]-previous.values['left'][0]<=2*dt
        previous=action;t=now


def test_native_initial_pose_rejects_hardware_and_invalid_data_before_dds(tmp_path):
    import subprocess
    binary=ROOT/'gear_sonic_deploy/target/release/g1_deploy_onnx_ref'
    if not binary.is_file():pytest.skip('local SONIC build unavailable')
    pose=ROOT/'gear_sonic/devices/inspire/assets/vla_initial_pose.txt'
    for interface,file,expected in [('hardware-test',pose,'requires loopback'),('lo',tmp_path/'missing','cannot open')]:
        result=subprocess.run([str(binary),interface,'unused.onnx','unused-motion','--sim-initial-pose',str(file)],
                              capture_output=True,text=True,timeout=10)
        assert result.returncode==1 and expected in result.stdout+result.stderr
        assert 'Creating G1Deploy' not in result.stdout


def test_native_pose_limits_match_registered_model():
    import re
    import xml.etree.ElementTree as ET
    from gear_sonic.utils.hand_control.models import resource_path
    config=deployment('inspire','sim_shared')
    root=ET.parse(resource_path(config,'mjcf')).getroot()
    joints={j.get('name'):j for j in root.iter('joint') if j.get('name')}
    ranges=[list(map(float,joints[a.get('joint')].get('range').split())) for a in root.find('actuator') if '_inspire_' not in a.get('joint','')]
    header=(ROOT/'gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/body_pose.hpp').read_text()
    from gear_sonic.utils.inference.ready_pose import UPPER_ORDER
    native_order = re.search(r'ready_pose_upper_order = \{([^}]+)\}', header)[1]
    assert tuple(int(x) for x in native_order.split(',')) == UPPER_ORDER
    for column,side in enumerate(['lower','upper']):
        raw=re.search(r'body_pose_'+side+r' = \{([^}]+)\}',header)[1]
        np.testing.assert_allclose([float(x) for x in raw.split(',')],np.array(ranges)[:,column],rtol=0,atol=1e-8)


def test_named_scene_keyframe_initializes_actual_mujoco_data_and_rejects_missing():
    mujoco=pytest.importorskip('mujoco')
    from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
    env=DefaultEnv.__new__(DefaultEnv)
    env.config={'INITIAL_KEYFRAME':'photo_arms_down'};env.hand_controller=None
    env.mj_model=mujoco.MjModel.from_xml_path(str(ROOT/'scenes/inspire/basket_chair/scene.xml'))
    env.mj_data=mujoco.MjData(env.mj_model)
    DefaultEnv.reset(env)
    index=mujoco.mj_name2id(env.mj_model,mujoco.mjtObj.mjOBJ_KEY,'photo_arms_down')
    np.testing.assert_array_equal(env.mj_data.qpos,env.mj_model.key_qpos[index])
    assert env.mj_data.qpos[2]==pytest.approx(.793)
    env.config={'INITIAL_KEYFRAME':'missing'}
    with pytest.raises(ValueError,match='keyframe'):DefaultEnv.reset(env)
    env.config={};DefaultEnv.reset(env)
    np.testing.assert_array_equal(env.mj_data.qpos,env.mj_model.qpos0)


def test_failed_simulator_construction_preserves_original_scene_error(monkeypatch):
    pytest.importorskip('mujoco')
    from gear_sonic.utils.mujoco_sim import base_sim
    monkeypatch.setattr(base_sim, 'Robot', lambda config: None)
    def invalid_scene(*args, **kwargs):
        raise ValueError('initial keyframe not found: missing')
    simulator = base_sim.BaseSimulator.__new__(base_sim.BaseSimulator)
    with pytest.raises(ValueError, match='initial keyframe not found'):
        simulator.__init__({'SIMULATE_DT': .002}, env_factory=invalid_scene)
    simulator.close()
    simulator.__del__()
