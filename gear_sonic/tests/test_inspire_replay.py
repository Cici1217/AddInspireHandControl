"""Recorded action validation, protocol replay and actual MuJoCo hand feedback."""
from dataclasses import replace
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pa = pytest.importorskip('pyarrow')
import pyarrow.parquet as pq
import yaml

from gear_sonic.devices.inspire import replay
from gear_sonic.devices.inspire.config import DEFAULT_MAPPING_PATH
from gear_sonic.devices.inspire.recording.schema import capture_identity
from gear_sonic.utils.data_collection.end_effector import recording_metadata
from gear_sonic.utils.hand_control.registry import ROOT, deployment


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / 'recorded'; (root / 'meta').mkdir(parents=True)
    (root / 'data/chunk-000').mkdir(parents=True)
    count = 60
    hands = np.zeros((count, 6)); hands[:, 0] = np.linspace(0, .25, count)
    values = {'action.left_hand_joints': hands, 'action.right_hand_joints': hands,
              'teleop.smpl_pose': np.zeros((count, 63)), 'teleop.smpl_joints': np.zeros((count,72)),
              'teleop.body_quat_w': np.tile([1,0,0,0], (count,1)),
              'teleop.vr_3pt_position': np.zeros((count,9)),
              'teleop.vr_3pt_orientation': np.tile([1,0,0,0,1,0], (count,3))}
    features = {key:{'shape':[value.shape[1]], 'dtype':'float32'} for key,value in values.items()}
    for side in ('left','right'): features[f'action.{side}_hand_joints']['names'] = list(replay.JOINT_ORDER)
    info = {'fps':50,'total_episodes':1,'total_frames':count,'features':features}
    (root/'meta/info.json').write_text(json.dumps(info))
    capture = capture_identity(fps=50) | {'actual_frames':count,'actual_episodes':1}
    (root/'meta/inspire_capture.json').write_text(json.dumps(capture))
    meta = recording_metadata(deployment('inspire','sim_shared'),task='replay fixture',source='test')
    (root/'meta/end_effector.json').write_text(json.dumps(meta))
    values |= {'frame_index':np.arange(count), 'episode_index':np.zeros(count,dtype=int),
               'timestamp':np.arange(count)/50}
    pq.write_table(pa.table({k:v.tolist() for k,v in values.items()}), root/'data/chunk-000/episode_000000.parquet')
    return root


def mutate(root, key, value):
    path = root/'data/chunk-000/episode_000000.parquet'
    rows = pq.read_table(path).to_pydict(); rows[key] = value(rows[key])
    pq.write_table(pa.table(rows),path)


@pytest.fixture
def joint_dataset(dataset):
    from gear_sonic.devices.inspire.recording.sonic_reference import SonicReferenceWriter
    writer = SonicReferenceWriter()
    times = 1_000_000_000 + np.arange(60, dtype=np.int64) * 20_000_000
    for index, timestamp in enumerate(times):
        writer.append(SimpleNamespace(body_joint_positions=np.full(29, index / 1000),
                                     body_joint_velocities=np.full(29, .05),
                                     body_quaternion=np.array([.8, 0, 0, .6]),
                                     sample_monotonic_ns=int(timestamp)))
    path = dataset / 'meta/inspire_capture.json'
    capture = json.loads(path.read_text())
    capture['sonic_reference'] = writer.finish(dataset)
    capture['timing_trace'] = {'path': 'meta/inspire_timing.npz'}
    np.savez(dataset / 'meta/inspire_timing.npz', sample_monotonic_ns=times)
    path.write_text(json.dumps(capture))
    return dataset


@pytest.mark.parametrize('target', ['sim_shared', 'real'])
def test_measured_reference_uses_imu_and_joints_without_smpl(joint_dataset, target):
    mutate(joint_dataset, 'teleop.smpl_pose', lambda x: [[float('nan')] * 63] + x[1:])
    episode = replay.load_episode(joint_dataset, config=deployment('inspire', target), body_reference='joints')
    assert episode.smpl_pose is None and episode.vr_position is None
    assert episode.summary()['body_reference'] == 'joints'
    data = replay.pose_window(episode, 59)
    assert set(data) == {'joint_pos', 'joint_vel', 'body_quat_w', 'frame_index'}
    assert data['joint_pos'].shape == (50, 29)
    np.testing.assert_allclose(data['joint_pos'], .059)
    np.testing.assert_allclose(data['joint_vel'], .05)
    np.testing.assert_allclose(data['body_quat_w'], [[.8, 0, 0, .6]] * 50)
    assert data['frame_index'].tolist() == list(range(59, 109))
    with pytest.raises(ValueError, match='finite'):
        replay.load_episode(joint_dataset, config=deployment('inspire', target))


def test_old_capture_does_not_silently_substitute_smpl_for_joints(dataset):
    with pytest.raises(ValueError, match='没有实测关节角回放文件'):
        replay.load_episode(dataset, config=deployment('inspire', 'sim_shared'), body_reference='joints')
    assert replay.load_episode(dataset, config=deployment('inspire', 'sim_shared')).body_reference == 'smpl'


def test_joint_replay_hands_follow_capture_time_across_a_gap(tmp_path):
    from gear_sonic.devices.inspire.recording.sonic_reference import SonicReferenceWriter
    writer = SonicReferenceWriter()
    times = np.array([1_000_000_000, 1_020_000_000, 1_060_000_000])
    for timestamp in times:
        writer.append(SimpleNamespace(body_joint_positions=np.zeros(29), body_joint_velocities=np.zeros(29),
                                     body_quaternion=np.array([1., 0, 0, 0]), sample_monotonic_ns=timestamp))
    reference = writer.finish(tmp_path)
    np.savez(tmp_path / 'timing.npz', sample_monotonic_ns=times)
    capture = {'sonic_reference': reference, 'timing_trace': {'path': 'timing.npz'}}
    hand = np.tile([0, .02, .06], (6, 1)).T
    q, dq, quat, clock, hands, sources = replay.load_joint_reference(tmp_path, capture, 3, [hand, hand])
    np.testing.assert_allclose(clock, [0, .02, .04, .06])
    np.testing.assert_allclose(hands[0][:, 0], clock)
    np.testing.assert_allclose(hands[1][:, 0], clock)
    capture['sonic_reference']['frames'] = 3
    with pytest.raises(ValueError, match='counts disagree'):
        replay.load_joint_reference(tmp_path, capture, 3, [hand, hand])


@pytest.mark.parametrize('source', ['SIM_ONLY','REAL_CAPTURE_READ_ONLY'])
@pytest.mark.parametrize('target', ['sim_shared','real'])
def test_both_recording_sources_and_targets_keep_actions_without_connections(dataset, source, target, tmp_path):
    p = dataset/'meta/inspire_capture.json'; row=json.loads(p.read_text()); row['validation_level']=source;p.write_text(json.dumps(row))
    # Different network endpoints and model path spelling are deployment settings.
    raw=yaml.safe_load(DEFAULT_MAPPING_PATH.read_text());raw['hardware']['command_endpoint']='tcp://127.0.0.1:59991'
    raw['hardware']['left_device']='/dev/example_only'
    custom=tmp_path/'other-deployment.yaml';custom.write_text(yaml.safe_dump(raw,sort_keys=False))
    config=deployment('inspire',target,options={'hand_config':str(custom)})
    loaded=replay.load_episode(dataset,config=config)
    assert loaded.frames==60 and loaded.source==source
    assert loaded.left[-1,0]==pytest.approx(.25)
    assert loaded.vr_orientation.shape==(60,12)
    assert not loaded.left.flags.writeable


@pytest.mark.parametrize('failure', ['missing','nan','time','hand_limit','order','mapping','incomplete','quat','vr','frame_count','episode'])
def test_invalid_recordings_are_rejected_before_control(dataset,failure):
    if failure=='missing':
        p=dataset/'data/chunk-000/episode_000000.parquet';table=pq.read_table(p);pq.write_table(table.drop(['teleop.smpl_pose']),p)
    elif failure=='nan': mutate(dataset,'teleop.smpl_pose',lambda x:[[float('nan')]*63]+x[1:])
    elif failure=='time': mutate(dataset,'timestamp',lambda x:[0]*len(x))
    elif failure=='hand_limit': mutate(dataset,'action.left_hand_joints',lambda x:[[4]*6]+x[1:])
    elif failure=='quat': mutate(dataset,'teleop.body_quat_w',lambda x:[[0]*4]+x[1:])
    elif failure=='vr': mutate(dataset,'teleop.vr_3pt_orientation',lambda x:[[0]*18]+x[1:])
    elif failure=='episode': mutate(dataset,'episode_index',lambda x:[1]*len(x))
    elif failure=='incomplete': (dataset/'INCOMPLETE').touch()
    elif failure=='order':
        p=dataset/'meta/info.json';row=json.loads(p.read_text());row['features']['action.left_hand_joints']['names'].reverse();p.write_text(json.dumps(row))
    elif failure=='mapping':
        p=dataset/'meta/end_effector.json';row=json.loads(p.read_text());row['deployment']['registration']['action_channels'][0]['unit']='m';p.write_text(json.dumps(row))
    elif failure=='frame_count':
        p=dataset/'meta/inspire_capture.json';row=json.loads(p.read_text());row['actual_frames']=100;p.write_text(json.dumps(row))
    with pytest.raises((ValueError,KeyError)):
        replay.load_episode(dataset,config=deployment('inspire','sim_shared'))


def test_old_v3_schema_and_pose_protocol_dont_replay_feedback_token(dataset):
    p=dataset/'meta/inspire_capture.json';row=json.loads(p.read_text());row['capture_schema']='inspire.lerobot.capture.v3';p.write_text(json.dumps(row))
    (dataset/'meta/end_effector.json').unlink()
    episode=replay.load_episode(dataset,config=deployment('inspire','sim_shared'))
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message, HEADER_SIZE
    data=replay.pose_window(episode,59,replay.reconstruct_wrist_joint_positions(episode.smpl_pose))
    assert data['frame_index'].tolist()==[59,60,61,62,63]
    assert data['smpl_pose'].shape==(5,21,3) and 'token_state' not in data
    raw=pack_pose_message(data,version=3)
    header=json.loads(raw[4:4+HEADER_SIZE].rstrip(b'\0'))
    assert header['v']==3
    assert 'vr_position' in {f['name'] for f in header['fields']}


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));return sock.getsockname()[1]


@pytest.fixture
def hand_sim(tmp_path):
    commands,states=port(),port()
    raw=yaml.safe_load(DEFAULT_MAPPING_PATH.read_text())
    raw['hardware'].update(command_endpoint=f'tcp://127.0.0.1:{commands}',state_endpoint=f'tcp://127.0.0.1:{states}')
    path=tmp_path/'hand.yaml';path.write_text(yaml.safe_dump(raw,sort_keys=False))
    code='''
import sys,time,mujoco,numpy as np
from gear_sonic.devices.inspire.model import load_asset_paths
from gear_sonic.devices.inspire.simulation.gateway import InspireSimGateway
model=mujoco.MjModel.from_xml_path(str(load_asset_paths(sys.argv[1]).scene))
# Isolate hand physics from the body peer: no body controller or gravity in this harness.
model.opt.gravity[:]=0
data=mujoco.MjData(model);mujoco.mj_forward(model,data)
body_q=[];body_v=[]
for jid in range(model.njnt):
 if '_inspire_' in (model.joint(jid).name or ''): continue
 free=model.jnt_type[jid]==int(mujoco.mjtJoint.mjJNT_FREE)
 body_q.extend(range(model.jnt_qposadr[jid],model.jnt_qposadr[jid]+(7 if free else 1)))
 body_v.extend(range(model.jnt_dofadr[jid],model.jnt_dofadr[jid]+(6 if free else 1)))
pose=data.qpos[body_q].copy()
gateway=InspireSimGateway(model,data,sys.argv[1])
print('SIM_READY',flush=True)
try:
 while True:
  gateway.step();mujoco.mj_step(model,data)
  data.qpos[body_q]=pose;data.qvel[body_v]=0;mujoco.mj_forward(model,data)
  time.sleep(model.opt.timestep)
except KeyboardInterrupt: pass
finally: gateway.close()
'''
    import signal
    with (tmp_path/'sim.log').open('w+') as log:
        process=subprocess.Popen([str(ROOT/'.venv_sim/bin/python'),'-B','-c',code,str(path)],cwd=ROOT,stdout=log,stderr=log)
        try:
            deadline=time.monotonic()+15
            while 'SIM_READY' not in (tmp_path/'sim.log').read_text():
                if process.poll() is not None or time.monotonic()>deadline: pytest.fail((tmp_path/'sim.log').read_text())
                time.sleep(.05)
            yield deployment('inspire','sim_shared',options={'hand_config':str(path)})
        finally:
            process.send_signal(signal.SIGINT)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()


class BodyPeer:
    """Only the SONIC peer is emulated; hands run actual shared MuJoCo + ZMQ."""
    def __init__(self):
        self.command=f'tcp://127.0.0.1:{port()}';self.state=f'tcp://127.0.0.1:{port()}'
        self.messages=[];self.quit=threading.Event();self.pause=threading.Event()
        self.thread=threading.Thread(target=self.run);self.thread.start()

    def run(self):
        import zmq,msgpack,msgpack_numpy as mnp
        context=zmq.Context();sub=context.socket(zmq.SUB);pub=context.socket(zmq.PUB)
        sub.setsockopt(zmq.SUBSCRIBE,b'');sub.connect(self.command);pub.bind(self.state)
        row={'body_q':np.zeros(29,dtype=np.float32),'token_state':np.zeros(64,dtype=np.float32)}
        try:
            while not self.quit.wait(.005):
                while sub.poll(0):self.messages.append(sub.recv())
                if not self.pause.is_set():pub.send(b'g1_debug'+msgpack.packb(row,default=mnp.encode,use_bin_type=True))
        finally: sub.close(0);pub.close(0);context.term()

    def close(self):self.quit.set();self.thread.join(timeout=3)


@pytest.mark.parametrize('reference', ['smpl', 'joints'])
def test_actual_hand_simulation_replay_completes_and_releases_session(joint_dataset,hand_sim,reference):
    episode=replay.load_episode(joint_dataset,config=hand_sim,body_reference=reference);peer=BodyPeer()
    try:
        report=replay.execute(episode,hand_sim,frames=60,auto_start=True,body_endpoint=peer.command,state_endpoint=peer.state)
        assert report['completed'] and report['played_frames']==60
        assert report['hand_limited_frames']==0
        assert 45<report['command_hz']<55
        assert len([p for p in peer.messages if p.startswith(b'pose')])>=60
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import HEADER_SIZE
        for message in (p for p in peer.messages if p.startswith(b'pose')):
            header = json.loads(message[4:4+HEADER_SIZE].rstrip(b'\0'))
            assert header['v'] == (1 if reference == 'joints' else 3)
            assert ('smpl_pose' in {f['name'] for f in header['fields']}) == (reference == 'smpl')
        hand=hand_sim.create();hand.connect()
        try: assert hand.get_state().values['left'][0]>.1
        finally: hand.close()
    finally:peer.close()


@pytest.mark.parametrize('stop_reason', ['feedback', 'interrupt'])
def test_stopping_replay_stops_body_and_releases_hands(dataset,hand_sim,monkeypatch,stop_reason):
    episode=replay.load_episode(dataset,config=hand_sim);peer=BodyPeer()
    original=replay.BodyReplay.pose
    calls=[]
    def lose_feedback(self,data):
        original(self,data);calls.append(True)
        if stop_reason=='feedback': peer.pause.set()
        elif len(calls)>1: raise KeyboardInterrupt()
    monkeypatch.setattr(replay.BodyReplay,'pose',lose_feedback)
    try:
        with pytest.raises(RuntimeError if stop_reason=='feedback' else KeyboardInterrupt,
                           match='body feedback stopped' if stop_reason=='feedback' else None):
            replay.execute(episode,hand_sim,auto_start=True,body_endpoint=peer.command,state_endpoint=peer.state)
        hand=hand_sim.create();hand.connect()
        try:hand.send_action(hand.seed_action(hand.get_state()))
        finally:hand.close()
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message
        assert build_command_message(start=False,stop=True,planner=False) in peer.messages
    finally:peer.close()


def test_real_execution_cannot_skip_operator_gate(dataset):
    config=deployment('inspire','real');episode=replay.load_episode(dataset,config=config)
    with pytest.raises(ValueError,match='automatic start'):
        replay.execute(episode,config,auto_start=True)


def test_blocked_thumb_stops_before_body_replay_and_reports_channel(joint_dataset, tmp_path, monkeypatch, capsys):
    """A successful command submission must not bypass failed physical alignment."""
    from contextlib import nullcontext
    from gear_sonic.utils.hand_control.interface import EffectorAction
    config = deployment('inspire', 'real')
    episode = replay.load_episode(joint_dataset, config=config, body_reference='joints')
    # Match the reported failure: this legal thumb target can be submitted,
    # but the emulated measured thumb stays at 0.858 rad.
    right = episode.right.copy(); right[:, 5] = .129
    episode = replace(episode, right=right)
    clock = SimpleNamespace(now=1.)
    def sleep(seconds): clock.now += seconds
    monkeypatch.setattr(replay, 'time', SimpleNamespace(
        monotonic=lambda: clock.now, monotonic_ns=lambda: int(clock.now * 1e9),
        sleep=sleep, strftime=lambda _: 'blocked-thumb-test'))
    monkeypatch.setattr(replay, 'ROOT', tmp_path)
    monkeypatch.setattr(replay, 'DeviceLease', lambda *args: nullcontext())
    monkeypatch.setattr(replay, 'enter_ready', lambda: True)

    class Body:
        started = stopped = closed = False
        pose_selected = False
        last_feedback = None
        poses = []
        def idle(self): pass
        def feedback(self, **kwargs): return self.started
        def command(self, *, planner, stop=False): self.started = not stop
        def pose(self, data, **kwargs): self.poses.append(data)
        def stop(self): self.stopped = True
        def close(self): self.closed = True
    body = Body()
    monkeypatch.setattr(replay, 'BodyReplay', lambda *args: body)

    adapter = config.create()  # Pure limit calculations only; never connect it.
    class Hand:
        description = SimpleNamespace(stale_after_s=.3)
        closed = False
        values = {'left': episode.left[0].copy(), 'right': episode.right[0].copy()}
        values['right'][5] = .858
        def connect(self): pass
        def get_state(self):
            return SimpleNamespace(values={s: v.copy() for s, v in self.values.items()},
                                   fault=None, valid={'left': True, 'right': True},
                                   received_monotonic_ns=int(clock.now * 1e9))
        def seed_action(self, state): return EffectorAction(state.values)
        def limit_action(self, *args): return adapter.limit_action(*args)
        def send_action(self, action):
            self.values = {s: np.array(v) for s, v in action.values.items()}
            self.values['right'][5] = .858
        def close(self): self.closed = True
    hand = Hand()
    fake_config = SimpleNamespace(backend='real', registration=config.registration,
                                  create=lambda: hand, snapshot=lambda: {})
    with pytest.raises(RuntimeError, match='右手拇指旋转.*目标 0.129、发送 0.129、实测 0.858'):
        replay.execute(episode, fake_config)
    assert not body.poses and body.stopped and body.closed and hand.closed
    report_path, = (tmp_path / 'outputs/replay_runs').glob('*.json')
    report = json.loads(report_path.read_text())
    assert report['played_frames'] == 0 and report['completed'] is False
    assert report['hand_alignment']['max_command_error_rad'] < .005
    assert report['hand_alignment']['worst_joint']['error_rad'] == pytest.approx(.729)
    assert 'REPLAY_PROGRESS frame=0/60 time_s=0.00' in capsys.readouterr().out


def test_real_replay_protocol_with_original_driver_and_local_serials(dataset,tmp_path,monkeypatch):
    """Actual native driver, local PTYs and a body peer; no physical robot."""
    import os
    from gear_sonic.tests.test_inspire_serial_e2e import SerialEmulator, endpoint
    from gear_sonic.devices.inspire.config import driver_command
    binary=os.environ.get('INSPIRE_TEST_GATEWAY')
    if not binary: pytest.skip('set INSPIRE_TEST_GATEWAY for the local serial test')
    left,right=SerialEmulator(2),SerialEmulator(3)
    raw=yaml.safe_load(DEFAULT_MAPPING_PATH.read_text())
    raw['hardware'].update(left_device=left.path,right_device=right.path,left_id=2,right_id=3,
        command_endpoint=endpoint(),state_endpoint=endpoint(),lock_path=str(tmp_path/'writer.lock'))
    path=tmp_path/'real.yaml';path.write_text(yaml.safe_dump(raw,sort_keys=False))
    config=deployment('inspire','real',options={'hand_config':str(path)})
    process=peer=None;confirmations=[]
    def enter(): confirmations.append(True);return True
    monkeypatch.setattr(replay,'enter_ready',enter)
    try:
        with (tmp_path/'gateway.log').open('w') as log:
            process=subprocess.Popen(driver_command(binary,path),stdout=log,stderr=log,
                                     env=os.environ|{'G1_RL_SESSION_CONTROL':'1'})
            deadline=time.monotonic()+5
            while 'DIRECT_SERIAL_STATE_READY' not in (tmp_path/'gateway.log').read_text():
                assert process.poll() is None
                assert time.monotonic()<deadline
                time.sleep(.02)
            peer=BodyPeer()
            report=replay.execute(replay.load_episode(dataset,config=config),config,
                                  body_endpoint=peer.command,state_endpoint=peer.state)
            assert report['completed'] and report['played_frames']==60
            assert len(confirmations)==2
            assert 45<report['command_hz']<55
            for emulator in (left,right):
                assert len(emulator.writes)>150
                assert 40<1/np.median(np.diff(emulator.read_times[-50:]))<60
            process.terminate();assert process.wait(5)==0
            assert left.speed==right.speed==[400]*6
    finally:
        if process is not None and process.poll() is None:process.terminate();process.wait(5)
        if peer:peer.close()
        left.close();right.close()
