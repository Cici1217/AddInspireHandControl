"""Exercise actual HTTP, PTY processes and frozen command plans without a robot."""

from __future__ import annotations

import ast
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from gear_sonic.utils.console.bootstrap import ensure_venv
from gear_sonic.utils.console.catalog import FEATURES, Builder, build_plan
from gear_sonic.utils.console.config import ROOT, Profiles, defaults, validate, portable_config, resolve_paths
from gear_sonic.utils.console.runner import Runner
from gear_sonic.utils.console.server import Console, handler


def wait_until(fn, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if fn():
            return
        time.sleep(0.02)
    raise AssertionError("condition did not become true")


def config(tmp_path):
    # Test output roots are not full checkouts; keep source assets in this checkout.
    return resolve_paths(defaults()) | {"local.root": str(tmp_path)}


def test_logs_follow_project_and_legacy_profiles_load(tmp_path):
    legacy = config(tmp_path) | {"local.logs": "/old/external/logs"}
    directory = tmp_path / "settings"
    directory.mkdir()
    (directory / "profiles.json").write_text(json.dumps({"active": "默认", "profiles": {"默认": legacy}}))
    loaded = Profiles(directory).snapshot()["profiles"]["默认"]
    assert "local.logs" not in loaded
    for root in (tmp_path, tmp_path / "another-project"):
        plan = build_plan(legacy | {"local.root": str(root)}, "scene", "sim")
        assert plan["log_root"] == str(root / "outputs/console_logs")


def test_profiles_persist_and_validate(tmp_path):
    profiles = Profiles(tmp_path)
    c = defaults() | {"task.name": "抓篮子", "task.seconds": 12.5}
    profiles.save("新电脑", c)
    assert Profiles(tmp_path).snapshot()["profiles"]["新电脑"] == c
    for invalid in (
        {"task.seconds": 0.8},
        {"task.fps": 30},
        {"task.seconds": float("nan")},
        {"ports.camera": 5557},
        {"g1.ssh_host": "-oProxyCommand=bad"},
        {"local.root": "~/project"},
        {"task.depth": "false"},
    ):
        with pytest.raises(ValueError):
            validate(c | invalid)


@pytest.mark.parametrize('dataset_root', ['outputs/datasets', 'shared recordings', '/tmp/shared recordings'])
def test_capture_locations_follow_project_and_source_not_replay_target(tmp_path, dataset_root):
    c = config(tmp_path) | {'local.output': dataset_root, 'local.input': '/old/real-recording'}
    for mode in ('sim', 'real'):
        plan = build_plan(c, 'capture', mode)
        root = (tmp_path / 'outputs/datasets' / mode).resolve()
        step = next(s for s in plan['steps'] if s['id'] == 'capture')
        assert step['argv'][step['argv'].index('--output-root') + 1] == str(root)
        assert step['outputs'] == [str(root / c['task.name'])]
        assert 'output:' + str(root / c['task.name']) in step['resources']
        assert plan['output_directories'][mode] == str(root)
        replay = build_plan(c, 'replay', mode)
        assert all('/old/real-recording' in s['argv'] for s in replay['steps'] if s['id'] in ('replay_check', 'dataset_replay'))


def test_legacy_default_output_migrates_without_moving_old_data(tmp_path):
    old = str(tmp_path / 'outputs/inspire_capture_000')
    c = config(tmp_path) | {'local.output': str(tmp_path / 'outputs'), 'local.input': old}
    migrated = validate(c)
    assert 'local.output' not in migrated
    assert migrated['local.input'] == str(tmp_path / 'outputs/datasets/sim/inspire_capture_000')
    Path(old).mkdir(parents=True)
    marker = Path(old) / 'keep.txt';marker.write_text('old recording')
    assert validate(c)['local.input'] == old
    assert marker.read_text() == 'old recording'
    custom = c | {'local.output': '/external/captures', 'local.input': '/external/session'}
    assert 'local.output' not in validate(custom)
    assert validate(custom)['local.input'] == '/external/session'


def test_common_output_initialization_is_idempotent_and_local(tmp_path):
    from gear_sonic.utils.console.bootstrap import initialize_outputs
    c = config(tmp_path)
    paths = initialize_outputs(c)
    assert set(paths) == {'sim','real','logs','calibration','replay','vla','pretrained','finetuned'}
    assert all(Path(p).is_dir() and Path(p).is_relative_to(tmp_path) for p in paths.values())
    marker = Path(paths['sim']) / 'recording.txt';marker.write_text('keep')
    assert initialize_outputs(c) == paths
    assert marker.read_text() == 'keep'
    assert not (tmp_path / 'g1').exists()
    plan = build_plan(c, 'setup', 'local')
    step = next(s for s in plan['steps'] if s['id'] == 'directories')
    assert step['machine'] == 'local' and step['depends'] == []


def test_dataset_library_uses_recorded_source_and_excludes_incomplete(tmp_path):
    from gear_sonic.utils.data_collection.paths import list_datasets
    root = tmp_path / 'outputs/datasets'
    for name, source in [('sim/actually_real', 'REAL_CAPTURE_READ_ONLY'), ('real/actually_sim', 'SIM_ONLY')]:
        meta = root / name / 'meta'; meta.mkdir(parents=True)
        (meta/'info.json').write_text(json.dumps({'total_episodes':2, 'total_frames':100}))
        (meta/'inspire_capture.json').write_text(json.dumps({'capture_schema':'inspire.lerobot.capture.v4', 'validation_level':source}))
    rows = list_datasets(tmp_path)
    assert {row['label']:row['source'] for row in rows} == {'actually_real':'real', 'actually_sim':'sim'}
    (root/'real/actually_sim/INCOMPLETE').touch()
    assert [r['source'] for r in list_datasets(tmp_path)] == ['real']


def test_state_replay_has_no_body_or_hardware_steps(tmp_path):
    meta=tmp_path/'recorded/meta';meta.mkdir(parents=True)
    (meta/'info.json').write_text(json.dumps({'total_episodes':1,'total_frames':100}))
    (meta/'simulation_replay.json').write_text(json.dumps({'schema':'inspire.simulation_replay.v1'}))
    c=config(tmp_path)|{'local.input':str(meta.parent),'task.headless':True}
    steps=build_plan(c,'replay','sim')['steps']
    assert [s['id'] for s in steps]==['replay_check','dataset_replay']
    assert all('--playback' in s['argv'] and s['simulation_replay'] for s in steps)
    assert steps[-1]['inputs']==[] and '--headless' in steps[-1]['argv']
    with pytest.raises(ValueError, match='仿真物理状态'):
        build_plan(c,'replay','real')


def test_removed_capture_override_cannot_redirect_new_recordings(tmp_path):
    c=config(tmp_path)|{'task.capture_directory':'custom captures'}
    for mode in ('sim','real'):
        step=next(s for s in build_plan(c,'capture',mode)['steps'] if s['id']=='capture')
        assert step['argv'][step['argv'].index('--output-root')+1]==str(tmp_path/'outputs/datasets'/mode)
        assert ('--sim' in step['argv'])==(mode=='sim')
    assert 'task.capture_directory' not in validate(c)
    assert not (tmp_path/'custom captures').exists()

def test_initialization_preserves_existing_environments(tmp_path):
    env = tmp_path / "already-installed"
    (env / "bin").mkdir(parents=True)
    (env / "pyvenv.cfg").write_text("existing configuration")
    (env / "bin/python").symlink_to(sys.executable)
    ensure_venv(str(env / "bin/python"), "/nonexistent/base/python")
    assert (env / "pyvenv.cfg").read_text() == "existing configuration"
    unrecognized = tmp_path / "important-files"
    unrecognized.mkdir()
    (unrecognized / "keep.txt").write_text("keep me")
    with pytest.raises(ValueError, match="不覆盖"):
        ensure_venv(str(unrecognized / "bin/python"), sys.executable)
    assert (unrecognized / "keep.txt").read_text() == "keep me"


def test_every_plan_uses_existing_cli_flags():
    for feature in FEATURES:
        for mode in feature["modes"]:
            plan = build_plan(defaults(), feature["id"], mode)
            assert plan["pending"] == feature.get("pending", False)
            for step in plan["steps"]:
                assert shlex.split(step["launch_command"]) == step["launch_argv"]
                argv = step["argv"]
                if "-m" not in argv:
                    continue
                module = argv[argv.index("-m") + 1]
                path = ROOT / (module.replace(".", "/") + ".py")
                tree = ast.parse(path.read_text())
                options = {
                    arg.value
                    for n in ast.walk(tree)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "add_argument"
                    for arg in n.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--")
                }
                if module in ("gear_sonic.scripts.run_sim_loop", "gear_sonic.camera.composed_camera"):
                    # Tyro arguments come from inherited dataclass fields.
                    config_tree = (
                        ast.parse((ROOT / "gear_sonic/utils/mujoco_sim/configs.py").read_text())
                        if module.endswith("run_sim_loop")
                        else tree
                    )
                    for n in ast.walk(config_tree):
                        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                            options.add("--" + n.target.id.replace("_", "-"))
                            options.add("--no-" + n.target.id.replace("_", "-"))
                actual = {a for a in argv[argv.index("-m") + 2 :] if a.startswith("--")}
                assert actual <= options, (module, actual - options)


def test_preview_executes_exact_quoted_arguments(tmp_path):
    b = Builder(config(tmp_path), "scene", "sim")
    tricky = "目录 with 'quotes'; $(touch SHOULD_NOT_EXIST) `id`"
    b.add(
        "echo",
        "测试",
        [sys.executable, "-c", "import sys; print(sys.argv[1])", tricky],
        outputs=[str(tmp_path / "result")],
    )
    plan = b.result({"title": "测试"})
    runner = Runner()
    job = runner.start(plan, "echo")
    wait_until(lambda: runner.jobs[job["id"]].state == "completed")
    assert tricky in runner.jobs[job["id"]].output
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()
    saved = json.loads((Path(job["log_path"]).parent / "command.json").read_text())
    assert saved["step"]["launch_argv"] == plan["steps"][0]["launch_argv"]


def test_manual_dependencies_stdin_and_owned_stop(tmp_path):
    b = Builder(config(tmp_path), "scene", "sim")
    b.add(
        "service",
        "服务",
        [sys.executable, "-u", "-c", "import time; print('READY'); time.sleep(90)"],
        service=True,
        ready="READY",
        stop_order=10,
    )
    b.add(
        "input",
        "人工输入",
        [sys.executable, "-u", "-c", "print('WAIT'); print('GOT='+input())"],
        inputs=["START\n"],
    )
    plan = b.result({"title": "测试"})
    runner = Runner()
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(90)"])
    try:
        with pytest.raises(ValueError, match="前置"):
            runner.start(plan, "input")
        runner.start(plan, "service")
        wait_until(lambda: runner.latest(plan["id"], "service").ready)
        runner.start(plan, "input")
        child = runner.latest(plan["id"], "input")
        wait_until(lambda: "WAIT" in child.output)
        with pytest.raises(ValueError):
            runner.input(child.id, "SOMETHING ELSE\n")
        runner.input(child.id, "START\n")
        wait_until(lambda: child.state == "completed")
        assert "GOT=START" in child.output
        runner.stop_all()
        wait_until(lambda: runner.latest(plan["id"], "service").state == "stopped")
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait()
        runner.stop_all()


def test_ready_services_reused_across_plans_without_bypassing_checks(tmp_path):
    b = Builder(config(tmp_path), "scene", "sim")
    b.add("service", "共享服务", [sys.executable, "-u", "-c", "import time; print('READY'); time.sleep(90)"], service=True)
    b.add("client", "客户端", [sys.executable, "-c", "print('ok')"])
    original = b.result({"title": "original"})
    changed = b.result({"title": "changed task"})
    runner = Runner()
    try:
        data = runner.start(original, "service")
        job = runner.jobs[data["id"]]
        with pytest.raises(ValueError, match="前置"):
            runner.start(changed, "client")
        runner.mark_ready(job.id)
        other_command = json.loads(json.dumps(changed))
        other_command["steps"][0]["launch_argv"].append("different")
        with pytest.raises(ValueError, match="前置"):
            runner.start(other_command, "client")
        data = runner.start(changed, "client")
        wait_until(lambda: runner.jobs[data["id"]].state == "completed")
        runner.stop(job.id)
        wait_until(lambda: job.state == "stopped")
        with pytest.raises(ValueError, match="前置"):
            runner.start(changed, "client")
    finally:
        runner.stop_all()


def test_failed_step_can_be_retried_before_next_step(tmp_path):
    b = Builder(config(tmp_path), "scene", "sim")
    b.add(
        "check",
        "检查",
        [sys.executable, "-c", "from pathlib import Path; raise SystemExit(not Path('fixed').exists())"],
    )
    b.add("after", "后续", [sys.executable, "-c", "print('AFTER')"])
    plan = b.result({"title": "测试"})
    runner = Runner()
    runner.start(plan, "check")
    wait_until(lambda: runner.latest(plan["id"], "check").state == "failed")
    with pytest.raises(ValueError, match="前置"):
        runner.start(plan, "after")
    (tmp_path / "fixed").touch()
    runner.start(plan, "check")
    wait_until(lambda: runner.latest(plan["id"], "check").state == "completed")
    runner.start(plan, "after")
    wait_until(lambda: runner.latest(plan["id"], "after").state == "completed")


def test_close_terminal_waits_for_exit_and_does_not_interrupt_cleanup_twice(tmp_path):
    release = tmp_path / "finish-cleanup"
    code = (
        "import signal, time\nfrom pathlib import Path\n"
        "def stop(*args): print('CLEANUP', flush=True)\n"
        "signal.signal(signal.SIGINT, stop)\nprint('READY', flush=True)\n"
        f"while not Path({str(release)!r}).exists(): time.sleep(.02)\n"
        "print('SAVED', flush=True)\n"
    )
    b = Builder(config(tmp_path), "scene", "sim")
    b.add("record", "正在收尾", [sys.executable, "-u", "-c", code], ready="READY")
    runner = Runner()
    data = runner.start(b.result({"title": "关闭验证"}), "record")
    job = runner.jobs[data["id"]]
    try:
        wait_until(lambda: job.ready)
        closing = runner.close_terminal(job.id)
        assert closing["state"] == "stopping" and closing["close_requested"]
        assert not closing["terminal_closed"] and job.process.poll() is None
        wait_until(lambda: "CLEANUP" in job.output)
        # A repeated request (including after a page refresh) must not send a
        # second Ctrl+C while the recorder is still saving data.
        assert not runner.close_terminal(job.id)["terminal_closed"]
        assert not runner.snapshot()["jobs"][0]["terminal_closed"]
        release.touch()
        wait_until(lambda: job.state == "stopped")
        closed = runner.snapshot()["jobs"][0]
        assert closed["terminal_closed"] and job.process.poll() == 0
        assert job.output.count("CLEANUP") == 1 and "SAVED" in job.output
        assert "SAVED" in job.log_path.read_text()
        assert runner.close_terminal(job.id)["terminal_closed"]
    finally:
        release.touch()
        runner.stop_all()
        job.process.wait(timeout=8)


@pytest.mark.parametrize("returncode", [0, 7])
def test_close_ended_terminal_preserves_history_and_dependencies(tmp_path, returncode):
    b = Builder(config(tmp_path), "scene", "sim")
    b.add("check", "已结束", [sys.executable, "-c", f"raise SystemExit({returncode})"])
    b.add("after", "后续", [sys.executable, "-c", "print('AFTER')"])
    plan = b.result({"title": "关闭验证"})
    runner = Runner()
    data = runner.start(plan, "check")
    job = runner.jobs[data["id"]]
    wait_until(lambda: job.state in ("completed", "failed"))
    closed = runner.close_terminal(job.id)
    assert closed["terminal_closed"] and closed["returncode"] == returncode
    assert not job.stopped and runner.latest(plan["id"], "check") is job
    if returncode == 0:
        next_job = runner.start(plan, "after")
        wait_until(lambda: runner.jobs[next_job["id"]].state == "completed")
    else:
        with pytest.raises(ValueError, match="前置"):
            runner.start(plan, "after")


def test_exited_service_is_not_treated_as_ready(tmp_path):
    b = Builder(config(tmp_path), "scene", "sim")
    b.add("service", "退出的服务", [sys.executable, "-c", "print('exiting')"], service=True)
    b.add("client", "不得启动", [sys.executable, "-c", "print('bad')"])
    plan = b.result({"title": "test"})
    runner = Runner()
    runner.start(plan, "service")
    wait_until(lambda: runner.latest(plan["id"], "service").state == "completed")
    with pytest.raises(ValueError, match="前置"):
        runner.start(plan, "client")


def test_http_local_origin_token_and_frozen_preview(tmp_path):
    app = Console(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(app))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"

    def request(path, data=None, token=True, **headers):
        headers = {"Content-Type": "application/json", **headers}
        if token:
            headers["X-Console-Token"] = app.token
        return urlopen(
            Request(base + path, data=None if data is None else json.dumps(data).encode(), headers=headers)
        )

    try:
        first_session = request("/").headers['X-Console-Session']
        assert first_session == request('/api/state').headers['X-Console-Session']
        assert first_session == request('/api/jobs').headers['X-Console-Session']
        assert first_session != Console(tmp_path).session_id
        with pytest.raises(HTTPError) as error:
            request("/api/profiles", {"name": "x", "config": defaults()}, token=False)
        assert error.value.code == 403
        assert error.value.headers['X-Console-Session'] == first_session
        with pytest.raises(HTTPError):
            request("/api/state", Origin="https://untrusted.invalid")
        plan = json.load(request("/api/plan", {"config": defaults(), "feature": "scene", "mode": "sim"}))
        changed = defaults() | {"local.hand_config": "/tmp/another.yaml"}
        other = json.load(request("/api/plan", {"config": changed, "feature": "scene", "mode": "sim"}))
        assert plan["id"] != other["id"]
        assert app.plans[plan["id"]]["config"]["local.hand_config"] == resolve_paths(defaults())["local.hand_config"]
        assert json.load(request("/api/jobs")) == {"jobs": []}
        with pytest.raises(HTTPError) as denied:
            request('/api/config-check', {'config': config(tmp_path)}, token=False)
        assert denied.value.code == 403
        checked = json.load(request('/api/config-check', {'config': config(tmp_path)}))
        assert checked['checks'] and checked['remote_checked'] is False
        assert json.load(request('/api/jobs')) == {'jobs': []}
        with pytest.raises(HTTPError):
            request('/api/output-directories', {'config': config(tmp_path)}, token=False)
        assert not (tmp_path / 'outputs/datasets').exists()
        paths = json.load(request('/api/output-directories', {'config': config(tmp_path)}))
        assert Path(paths['sim']).is_dir() and Path(paths['real']).is_dir()
        with pytest.raises(HTTPError) as error:
            request("/api/sequence", {"plan_id": plan["id"]})
        assert error.value.code == 404
        b = Builder(config(tmp_path), "scene", "sim")
        b.add("done", "关闭验证", [sys.executable, "-c", "print('DONE')"])
        job = app.runner.start(b.result({"title": "测试"}), "done")
        wait_until(lambda: app.runner.jobs[job["id"]].state == "completed")
        with pytest.raises(HTTPError) as error:
            request("/api/close", {"job_id": job["id"]}, token=False)
        assert error.value.code == 403
        assert json.load(request("/api/close", {"job_id": job["id"]}))["terminal_closed"]
        assert json.load(request("/api/state"))["jobs"][0]["terminal_closed"]
    finally:
        server.shutdown()
        server.server_close()


def test_migrated_plans_resolve_new_checkout_and_new_entrypoints():
    assert ROOT == Path(__file__).resolve().parents[2]
    for feature in FEATURES:
        for mode in feature["modes"]:
            plan = build_plan(defaults(), feature["id"], mode)
            text = json.dumps(plan)
            assert "gear_sonic.extensions" not in text
            assert plan['config']['local.root'] == str(ROOT)
    plan = build_plan(defaults() | {"task.seconds": 8, "task.wait": 3, "task.depth": True}, "capture", "sim")
    steps = {s["id"]: s for s in plan["steps"]}
    assert "--enable-depth-publish" in steps["scene"]["argv"]
    argv = steps["capture"]["argv"]
    assert argv[argv.index("--seconds") + 1] == "8"
    assert argv[argv.index("--wait-seconds") + 1] == "3"
    assert argv[argv.index("--depth") + 1] == "required"
    assert not any("RESET" in v or "ARM HAND" in v for s in plan["steps"] for v in s["inputs"])


def test_removed_functions_and_legacy_configuration_cannot_execute():
    assert [f['id'] for f in FEATURES if not f.get('hidden')] == ['body','hands','teleop','capture','replay','vla','setup']
    legacy = defaults() | {'future.rgbd_input': '/old/data', 'future.train_steps': 10000}
    assert not any(k.startswith('future.') for k in validate(legacy))
    for feature in ("training", "reconstruction"):
        with pytest.raises(ValueError, match='功能或模式无效'):
            build_plan(legacy, feature, "real")


@pytest.mark.parametrize("mode", ["sim", "real"])
def test_body_control_is_independent_and_optional_for_hands(mode):
    standalone = build_plan(defaults(), "body", mode)
    steps = {s["id"]: s for s in standalone["steps"]}
    assert set(steps) == ({"scene", "body"} if mode == "sim" else {"body"})
    body = steps["body"]
    assert "keyboard" in body["argv"] or "--keyboard" in body["argv"]
    assert set("123wasdrO") <= set(body["inputs"])
    assert not body["ready"]  # Standing is still confirmed by the operator.
    with_body = build_plan(defaults() | {"task.with_body": True}, "hands", mode)
    optional = next(s for s in with_body["steps"] if s["id"] == "body")
    assert "可选" in optional["title"] and "独立身体控制入口" in optional["note"]
    without_body = build_plan(defaults() | {"task.with_body": False}, "hands", mode)
    assert not any(s["id"] == "body" for s in without_body["steps"])
    teleop = build_plan(defaults(), "teleop", mode)
    assert "1" not in next(s for s in teleop["steps"] if s["id"] == "body")["inputs"]


def test_body_keys_reach_owned_terminal_and_unknown_keys_are_rejected(tmp_path):
    inputs = next(s for s in build_plan(defaults(), "body", "real")["steps"] if s["id"] == "body")["inputs"]
    b = Builder(config(tmp_path), "body", "real")
    code = (
        "import os, tty\ntty.setraw(0)\nprint('READY', flush=True)\n"
        "while True:\n c=os.read(0,1)\n print('KEY='+c.hex(), flush=True)\n if c==b'O': break\n"
    )
    b.add("body", "模拟键盘程序", [sys.executable, "-u", "-c", code], inputs=inputs, ready="READY")
    runner = Runner()
    job = runner.start(b.result({"title": "键盘验证"}), "body")
    try:
        wait_until(lambda: runner.jobs[job["id"]].ready)
        for key in ("]", "\n", "1", "2", "3", "w", "r"):
            runner.input(job["id"], key)
        with pytest.raises(ValueError, match="不接受"):
            runner.input(job["id"], "run arbitrary command\n")
        runner.input(job["id"], "O")
        wait_until(lambda: runner.jobs[job["id"]].state == "completed")
        assert [
            line.strip() for line in runner.jobs[job["id"]].output.splitlines() if line.startswith("KEY=")
        ] == ["KEY=" + key.encode().hex() for key in ("]", "\n", "1", "2", "3", "w", "r", "O")]
    finally:
        runner.stop_all()


def test_deployment_confirmation_and_startup_events_survive_log_truncation(tmp_path):
    inputs = next(s for s in build_plan(defaults(), "body", "sim")["steps"] if s["id"] == "body")["inputs"]
    b = Builder(config(tmp_path), "body", "sim")
    # Exercise line input before the C++ keyboard switches the PTY to raw mode.
    code = (
        "print('Proceed with deployment? [Y/n]:', flush=True)\n"
        "answer=input()\nassert answer=='y', repr(answer)\n"
        "print('Init Done\\nCONTROL state\\n[Keyboard] motion name is planner_motion', flush=True)\n"
        "print('x'*170000, flush=True)\n"
    )
    b.add("body", "部署确认", [sys.executable, "-u", "-c", code], inputs=inputs)
    runner = Runner()
    data = runner.start(b.result({"title": "部署确认"}), "body")
    job = runner.jobs[data["id"]]
    try:
        wait_until(lambda: "deployment_confirm" in job.startup_events)
        runner.input(job.id, "y\n")
        wait_until(lambda: job.state == "completed")
        public = runner.snapshot()["jobs"][0]
        assert "Init Done" not in public["output"]
        assert set(public["startup_events"]) == {
            "deployment_confirm",
            "init_done",
            "body_control",
            "planner_ready",
        }
    finally:
        runner.stop_all()


@pytest.mark.parametrize("mode", ["sim", "real"])
def test_body_control_retains_native_motion_directory(mode):
    c = defaults() | {"task.replay_source": "sonic", "local.sonic_motion": "/tmp/motions 'test'", "g1.sonic_motion": "/home/unitree/my motions"}
    plan = build_plan(c, "body", mode)
    body = plan["steps"][-1]
    assert not plan["pending"] and body["control_mode"] == "planner"
    assert set("pntrhO") <= set(body["inputs"])
    if mode == 'sim':
        assert body['argv'][body['argv'].index('--motion-data') + 1] == c['local.sonic_motion']
    else:
        assert body['argv'] == ['bash', c['g1.body_script'], '--keyboard', '--motion-data', c['g1.sonic_motion']]
    assert not any(s["id"] in ("hand", "teleop", "capture") for s in plan["steps"])
    assert "--motion-data" not in build_plan(defaults(), "body", "real")["steps"][-1]["argv"]
    with pytest.raises(ValueError, match="绝对路径"):
        validate(c | {"g1.sonic_motion": "~/motions"})


@pytest.mark.parametrize('mode', ['sim','real'])
@pytest.mark.parametrize('reference', ['smpl','joints'])
def test_recorded_replay_plan_checks_local_dataset_and_reuses_current_control(mode, reference):
    c=defaults() | {'local.input':'/tmp/recordings/episode 1','task.episode':2,'task.replay_frames':150,
                    'task.replay_mode': reference}
    steps=build_plan(c,'replay',mode)['steps']
    assert [s['id'] for s in steps]==(['replay_check','scene','body','dataset_replay'] if mode=='sim' else
                                     ['replay_check','hand','tunnel','body','dataset_replay'])
    assert '--execute' not in steps[0]['argv'] and '--execute' in steps[-1]['argv']
    assert steps[0]['machine']==steps[-1]['machine']=='local'
    assert steps[0]['argv'][steps[0]['argv'].index('--target')+1]==mode
    assert c['local.input'] in steps[-1]['argv']
    assert 'local:hand_producer' in steps[-1]['resources'] and 'local:body_target' in steps[-1]['resources']
    assert '1' not in steps[-2]['inputs']
    assert all(s['dataset_replay'] for s in steps)
    for step in (steps[0], steps[-1]):
        assert step['argv'][step['argv'].index('--body-reference')+1] == reference
    if mode=='sim':
        with pytest.raises(ValueError,match='不支持'):
            build_plan(c | {'task.body_hand':'dex3'},'replay',mode)


def test_old_replay_config_migrates_to_dataset_smpl_choice():
    c = validate(defaults() | {'task.replay_source': 'sonic', 'task.replay_mode': 'source'})
    assert 'task.replay_source' not in c and c['task.replay_mode'] == 'smpl'
    steps = build_plan(c, 'replay', 'sim')['steps']
    assert steps[0]['id'] == 'replay_check' and steps[-1]['id'] == 'dataset_replay'


def test_replay_progress_survives_terminal_chunks_and_log_truncation(tmp_path):
    b=Builder(config(tmp_path),'replay','sim')
    code="import time\nprint('REPLAY_READY ready',flush=True)\ntime.sleep(.1)\nprint('REPLAY_STANDING stand',flush=True)\nprint('REPLAY_PRO',end='',flush=True)\ntime.sleep(.1)\nprint('GRESS frame=75/150 time_s=1.50',flush=True)\nprint('x'*85000,flush=True)\nprint('REPLAY_COMPLETE done',flush=True)\n"
    b.add('dataset_replay','回放进度测试',[sys.executable,'-u','-c',code],deps=[],inputs=['\n'])
    plan=b.result(next(f for f in FEATURES if f['id']=='replay'))
    runner=Runner()
    try:
        result=runner.start(plan,'dataset_replay');job=runner.jobs[result['id']]
        wait_until(lambda:job.state=='completed')
        assert job.public()['replay']==dict(stage='completed',frame=75,total=150,time_s=1.5)
    finally:runner.stop_all()


def test_replay_confirmations_cannot_queue_past_next_operator_gate(tmp_path):
    b=Builder(config(tmp_path),'replay','sim')
    code="import time\nprint('REPLAY_READY ready',flush=True)\ninput()\ntime.sleep(.3)\nprint('REPLAY_STANDING stand',flush=True)\ninput()\nprint('REPLAY_COMPLETE done',flush=True)\n"
    b.add('dataset_replay','回放确认测试',[sys.executable,'-u','-c',code],deps=[],inputs=['\n'])
    runner=Runner()
    try:
        result=runner.start(b.result(next(f for f in FEATURES if f['id']=='replay')),'dataset_replay')
        job=runner.jobs[result['id']]
        wait_until(lambda:job.replay['stage']=='ready')
        runner.input(job.id,'\n')
        with pytest.raises(ValueError,match='不能重复确认'):runner.input(job.id,'\n')
        wait_until(lambda:job.replay['stage']=='standing')
        runner.input(job.id,'\n')
        wait_until(lambda:job.state=='completed')
        assert job.replay['stage']=='completed'
    finally:runner.stop_all()


def test_capture_results_require_current_completed_episode(tmp_path):
    b = Builder(config(tmp_path), 'capture', 'sim')
    code = """import time
for episode in (1, 2):
    print('CAPTURE_READY {"episode": %d}' % episode, flush=True)
    input()
    time.sleep(.15)
    print('CAPTURE_LABEL {"episode": %d}' % episode, flush=True)
    outcome = input()
    print('RESULT %d %s' % (episode, outcome), flush=True)
print('CAPTURE_COMPLETE {"episodes": 2}', flush=True)
"""
    b.add('capture', '逐条结果标注', [sys.executable, '-u', '-c', code], deps=[],
          inputs=['\n', 'success\n', 'failure\n', 'unlabeled\n'])
    runner = Runner()
    try:
        result = runner.start(b.result({'title': '逐条结果标注'}), 'capture')
        job = runner.jobs[result['id']]
        wait_until(lambda: job.capture['stage'] == 'ready')
        with pytest.raises(ValueError, match='提前或重复'):
            runner.input(job.id, 'success\n', capture_episode=1)
        runner.input(job.id, '\n', capture_episode=1)
        with pytest.raises(ValueError, match='提前或重复'):
            runner.input(job.id, '\n', capture_episode=1)
        wait_until(lambda: job.capture['stage'] == 'label')
        runner.input(job.id, 'success\n', capture_episode=1)
        with pytest.raises(ValueError, match='提前或重复'):
            runner.input(job.id, 'failure\n', capture_episode=1)
        wait_until(lambda: job.capture['stage'] == 'ready' and job.capture['episode'] == 2)
        with pytest.raises(ValueError, match='提前或重复'):
            runner.input(job.id, '\n', capture_episode=1)
        assert 'RESULT 2' not in job.output
        runner.input(job.id, '\n', capture_episode=2)
        wait_until(lambda: job.capture['stage'] == 'label' and job.capture['episode'] == 2)
        with pytest.raises(ValueError, match='提前或重复'):
            runner.input(job.id, 'success\n', capture_episode=1)
        runner.input(job.id, 'failure\n', capture_episode=2)
        wait_until(lambda: job.state == 'completed')
        assert 'RESULT 1 success' in job.output and 'RESULT 2 failure' in job.output
        assert job.public()['capture']['stage'] == 'complete'
    finally:
        runner.stop_all()
    for mode in ('sim', 'real'):
        plan = build_plan(config(tmp_path) | {'task.outcome': 'success'}, 'capture', mode)
        step = next(s for s in plan['steps'] if s['id'] == 'capture')
        assert '--label-after-each' in step['argv'] and '--outcome' not in step['argv']


def test_native_motion_state_follows_output_and_survives_truncation(tmp_path):
    # Drive a real PTY: chunked lines, unordered names, Enter both ways, N and T/R.
    code = (
        "import os, tty\ntty.setraw(0)\n"
        "print('✓ Loaded z_first (100 timesteps)\\n✓ Loaded a_second (250 timesteps)', flush=True)\n"
        "print('Started with motion: z_first (paused at frame 0)\\nInit Done\\nCONTROL state', flush=True)\n"
        "planner=False\nwhile True:\n"
        " k=os.read(0,1)\n"
        " if k==b'O': break\n"
        " if k==b'\\n':\n  planner=not planner\n"
        "  print('Planner enabled' if planner else 'Planner disabled', flush=True)\n"
        " if k==b'n':\n  print('Reference motion ', end='', flush=True)\n  print('name: a_second', flush=True)\n"
        " if k==b't': print('Playing motion 1 from frame 0 to end (250 total frames)', flush=True)\n"
        " if k==b'r': print('Reset motion 1 to frame 0 (paused)\\nMotion set: 2', flush=True)\n"
        " if k==b'z': print('x'*170000, flush=True)\n"
    )
    b = Builder(config(tmp_path), "replay", "sim")
    b.add(
        "body", "native fixture", [sys.executable, "-u", "-c", code], inputs=["1", "n", "t", "r", "z", "O", "\n"]
    )
    runner = Runner()
    data = runner.start(b.result({"title": "native fixture"}), "body")
    job = runner.jobs[data["id"]]
    try:
        wait_until(lambda: job.sonic["current_motion"] == "z_first")
        assert [m["name"] for m in job.sonic["motions"]] == ["z_first", "a_second"]
        runner.input(job.id, "\n")
        wait_until(lambda: job.sonic["mode"] == "planner")
        runner.input(job.id, "\n")
        wait_until(lambda: job.sonic["mode"] == "reference")
        runner.input(job.id, "n")
        wait_until(lambda: job.sonic["current_motion"] == "a_second")
        runner.input(job.id, "t")
        wait_until(lambda: job.sonic["playing"])
        runner.input(job.id, "r")
        wait_until(lambda: not job.sonic["playing"] and job.sonic["motion_set"] == 2)
        runner.input(job.id, "z")
        wait_until(lambda: "Loaded" not in job.output)
        assert job.public()["sonic"]["current_motion"] == "a_second"
        runner.input(job.id, "O")
        wait_until(lambda: job.state == "completed")
    finally:
        runner.stop_all()


@pytest.mark.parametrize("feature", ["body"])
@pytest.mark.parametrize("mode", ["sim", "real"])
def test_body_end_effector_routes_scene_and_controller_together(feature, mode):
    for hand in ("dex3", "inspire"):
        c = defaults() | {"task.body_hand": hand, "task.replay_source":"sonic"}
        steps = {s["id"]: s for s in build_plan(c, feature, mode)["steps"]}
        body = steps["body"]["argv"]
        if mode == "sim":
            scene = steps["scene"]["argv"]
            assert scene[scene.index("--hand") + 1] == hand
            assert body[body.index("--hand") + 1] == hand
            assert ("--hand-config" in scene) == (hand == "inspire")
            assert ("--hand-config" in body) == (hand == "inspire")
            assert ("--hand-python" in body) == (hand == "inspire")
        else:
            # The G1 wrapper retains its own Inspire transport; do not pass
            # simulation-only deploy.sh flags or the simulation hand selection.
            assert body == ['bash', c['g1.body_script'], '--keyboard']
            assert "dex3" not in steps["body"]["launch_command"]
    for other in ("hands", "teleop", "capture"):
        original = build_plan(defaults(), other, mode)["steps"]
        if mode == 'sim':
            with pytest.raises(ValueError, match='不支持'):
                build_plan(defaults() | {"task.body_hand": "dex3"}, other, mode)
        else:
            selected = build_plan(defaults() | {"task.body_hand": "dex3"}, other, mode)["steps"]
            assert selected == original  # Separate machine selections preserve simulation preference.
    with pytest.raises(ValueError, match="选项无效"):
        validate(defaults() | {"task.body_hand": "unimplemented_gripper"})


def test_registered_target_limits_are_checked_without_importing_driver(monkeypatch):
    import importlib
    monkeypatch.setattr(importlib, 'import_module', lambda *a: pytest.fail('configuration imported driver'))
    for bad in ({'left':[2]*6}, {'left':[.03,.04]}, {'left':[float('nan')]*6}):
        with pytest.raises(ValueError, match='action_rejected'):
            validate(defaults() | {'task.effector_targets':{'inspire':bad}})


def test_missing_saved_registry_does_not_prevent_opening_configuration(tmp_path):
    saved = defaults() | {'local.effector_registry': str(tmp_path/'removed-registry')}
    (tmp_path/'profiles.json').write_text(json.dumps({'active':'saved','profiles':{'saved':saved}}))
    app = Console(tmp_path)
    server = ThreadingHTTPServer(('127.0.0.1',0), handler(app))
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        with urlopen(f'http://127.0.0.1:{server.server_port}/api/state') as response:
            state = json.load(response)
        assert state['devices'] == []
        assert 'custom device registries' in state['registry_error']
        assert state['profiles']['profiles']['saved']['local.effector_registry'] == str(tmp_path/'removed-registry')
    finally:
        server.shutdown();server.server_close();thread.join()


@pytest.mark.parametrize('mode', ['sim', 'real'])
@pytest.mark.parametrize('with_body', [True, False])
def test_hand_action_depends_on_transport_not_optional_body(tmp_path, mode, with_body):
    plan = build_plan(config(tmp_path) | {'task.with_body': with_body, 'task.pose': 'sequence', 'task.hand_action': 'close'}, 'hands', mode)
    steps = {s['id']: s for s in plan['steps']}
    assert steps['send_hand']['depends'] == (['scene'] if mode == 'sim' else ['hand', 'tunnel'])
    # Test the actual runner's readiness gate with harmless processes, never SSH or hardware.
    for step in plan['steps']:
        step['device_context'] = None
        step['launch_argv'] = [sys.executable, '-u', '-c', "print('READY'); import time; time.sleep(90)" if step['service'] else "print('HAND_STARTED')"]
    runner = Runner()
    try:
        with pytest.raises(ValueError, match='前置步骤尚未就绪：01'):
            runner.start(plan, 'send_hand')
        for dependency in steps['send_hand']['depends']:
            row = runner.start(plan, dependency)
            runner.mark_ready(row['id'])
        row = runner.start(plan, 'send_hand')
        wait_until(lambda: runner.jobs[row['id']].state == 'completed')
        assert 'HAND_STARTED' in runner.jobs[row['id']].output
        assert runner.latest(plan['id'], 'body') is None
    finally:
        runner.stop_all()


@pytest.mark.parametrize('mode',['sim','real'])
def test_vla_plan_uses_registered_contract_and_existing_hardware_services(mode):
    steps=build_plan(defaults(),'vla',mode)['steps'];by_id={s['id']:s for s in steps}
    assert [s['id'] for s in steps]==(['vla_check','policy','scene','body','vla'] if mode=='sim' else
                                    ['vla_check','policy','hand','camera','tunnel','body','vla'])
    assert '--check' in by_id['vla_check']['argv']
    assert 'gear_sonic.scripts.run_device_vla' in by_id['vla']['argv']
    assert '--execute' in by_id['vla']['argv'] and '--auto-start' not in by_id['vla']['argv']
    assert by_id['vla']['depends']==['body','policy']
    assert by_id['vla']['inputs']==['\n'] and by_id['vla']['stop_order']>by_id['body']['stop_order']
    assert all(s['vla_control'] for s in steps)
    assert '--input-type' in by_id['body']['argv'] or mode=='real'
    if mode=='sim':
        assert resolve_paths(defaults())['local.vla_sonic_checkpoint'] in by_id['body']['argv']
        assert '--enable-image-publish' in by_id['scene']['argv']
        assert 'LD_LIBRARY_PATH' in by_id['body']['launch_command']
    else:
        assert '-R' in by_id['tunnel']['argv'] and defaults()['g1.body_script'] in by_id['body']['argv']
    with pytest.raises(ValueError):build_plan(defaults()|{'task.body_hand':'dex3'},'vla','sim')


def test_vla_operator_gates_do_not_queue_enter_and_progress_survives_chunks(tmp_path):
    b=Builder(config(tmp_path),'vla','sim')
    code="import time;print('VLA_READY wait',flush=True);input();time.sleep(.2);print('VLA_STANDING wait',flush=True);input();print('VLA_PROG',end='',flush=True);time.sleep(.05);print('RESS frames=50 seconds=1.0',flush=True);print('VLA_COMPLETE done',flush=True)"
    b.add('vla','测试确认',[sys.executable,'-u','-c',code],deps=[],inputs=['\n'])
    runner=Runner()
    try:
        row=runner.start(b.result(next(f for f in FEATURES if f['id']=='vla')),'vla');job=runner.jobs[row['id']]
        wait_until(lambda:job.vla['stage']=='ready');runner.input(job.id,'\n')
        with pytest.raises(ValueError,match='不能重复'):runner.input(job.id,'\n')
        wait_until(lambda:job.vla['stage']=='standing');runner.input(job.id,'\n')
        with pytest.raises(ValueError,match='不能重复'):runner.input(job.id,'\n')
        wait_until(lambda:job.state=='completed')
        assert job.public()['vla']==dict(stage='completed',frames=50,seconds=1.)
    finally:runner.stop_all()


def test_bootstrap_repairs_project_binding_even_when_imports_succeed(tmp_path,monkeypatch):
    from gear_sonic.utils.console import bootstrap
    from types import SimpleNamespace
    env=tmp_path/'env';env.mkdir();(env/'pyvenv.cfg').write_text('')
    python=str(env/'bin/python');calls=[]
    monkeypatch.setattr(bootstrap.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0))
    monkeypatch.setattr(bootstrap,'run',lambda argv:calls.append(argv))
    bootstrap.install('capture',python,str(tmp_path),'/unused')
    assert any('install' in a and str(tmp_path)+'/gear_sonic[data_collection]' in a for a in calls)
    assert 'gear_sonic.data.exporter' in calls[-1][-1]


def test_optional_hand_calibration_does_not_block_teleop_environment(tmp_path,monkeypatch,capsys):
    from gear_sonic.utils.console import bootstrap
    from types import SimpleNamespace
    c=config(tmp_path)|{'setup.role':'teleop','local.teleop_python':sys.executable,'local.xrt_program':sys.executable,
                        'local.calibration':str(tmp_path/'not-calibrated.json')}
    monkeypatch.setattr(bootstrap.shutil,'which',lambda name:'/usr/bin/'+name)
    monkeypatch.setattr(bootstrap.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0,stderr=''))
    assert bootstrap.check(c)==0
    assert json.loads(capsys.readouterr().out)['passed']


def test_ready_pose_is_shared_by_body_teleop_vla_and_independent_of_scene(tmp_path):
    pose=ROOT/'gear_sonic/data/poses/g1/box_ready.txt'
    for mode in ('sim', 'real'):
        for feature in ('body','teleop','capture','vla'):
            c=config(tmp_path)|{'local.ready_pose':str(pose)}
            plan=build_plan(c,feature,mode)
            body=next(s for s in plan['steps'] if s['id']=='body')
            values=body['argv'][body['argv'].index('--ready-pose')+1]
            assert len(values.split())==29
            for step in plan['steps']:
                if step['id'] in ('teleop','vla'):
                    assert step['argv'][step['argv'].index('--ready-pose-values')+1]==values
                if step['id']=='scene':assert '--initial-keyframe' not in step['argv']
    with pytest.raises(ValueError,match='两套设置'):
        build_plan(c|{'task.vla_keyframe':'photo_arms_down'},'vla','sim')


@pytest.mark.parametrize('feature', ['body', 'hands', 'teleop', 'capture', 'vla'])
def test_real_start_validates_and_transmits_pose_values_not_local_path(tmp_path, feature):
    pose = str(ROOT/'gear_sonic/data/poses/g1/arms_raised.txt')
    c = config(tmp_path) | {'local.ready_pose': pose, 'task.with_body': True,
                            'task.vla_keyframe': 'sim-only-keyframe'}
    plan = build_plan(c, feature, 'real')
    body = next(s for s in plan['steps'] if s['id'] == 'body')
    values = body['argv'][body['argv'].index('--ready-pose') + 1]
    assert len(values.split()) == 29 and pose not in body['launch_command']
    for step in plan['steps']:
        if step['id'] in ('teleop', 'vla'):
            assert step['argv'][step['argv'].index('--ready-pose-values') + 1] == values
        assert '--initial-keyframe' not in step['argv']
    assert plan['config']['local.ready_pose'] == pose
    with pytest.raises(FileNotFoundError):
        build_plan(c | {'local.ready_pose': '/missing/pose.txt'}, feature, 'real')


def test_teleop_preparation_enter_cannot_queue_through_gates(tmp_path):
    b=Builder(config(tmp_path),'teleop','sim')
    code="import time\nprint('PICO_PREPARE_READY test',flush=True)\ninput()\ntime.sleep(.2)\nprint('PICO_PREPARED test',flush=True)\ninput()\nprint('PICO_STARTING test',flush=True)\n"
    b.add('teleop','准备姿态确认',[sys.executable,'-u','-c',code],deps=[],inputs=['\n'])
    runner=Runner()
    try:
        job=runner.jobs[runner.start(b.result({'title':'准备姿态确认'}),'teleop')['id']]
        wait_until(lambda:job.preparation=='ready')
        runner.input(job.id,'\n')
        with pytest.raises(ValueError,match='不能重复确认'):runner.input(job.id,'\n')
        wait_until(lambda:job.preparation=='prepared')
        runner.input(job.id,'\n')
        wait_until(lambda:job.state=='completed')
    finally:runner.stop_all()
