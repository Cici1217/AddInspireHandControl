"""Owned PTY processes, durable logs and explicit dependency checks."""

from __future__ import annotations

import codecs
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import pty
import re
import signal
import subprocess
import termios
import threading
import time
import uuid

STARTUP_MESSAGES = {
    "deployment_confirm": "Proceed with deployment?",
    "init_done": "Init Done",
    "body_control": "CONTROL state",
    "planner_ready": "motion name is planner_motion",
    "pico_ready": "PICO ready.",
    "pico_started": "[ThreePointPose] Calibration completed",
    "effector_started": "End-effector control active",
}


@dataclass
class Job:
    id: str
    plan_id: str
    step: dict
    log_path: Path
    process: subprocess.Popen
    master: int
    state: str = "running"
    ready: bool = False
    output: str = ""
    returncode: int | None = None
    started: float = field(default_factory=time.time)
    stopped: bool = False
    close_requested: bool = False
    startup_events: set[str] = field(default_factory=set)
    device_lease: object = None
    replay: dict = field(default_factory=lambda: dict(stage='waiting', frame=0, total=0, time_s=0.))
    vla: dict = field(default_factory=lambda: dict(stage='waiting', frames=0, seconds=0.))
    capture: dict = field(default_factory=lambda: dict(stage='waiting', episode=0))
    preparation: str = 'inactive'

    sonic: dict = field(
        default_factory=lambda: dict(
            mode="reference",
            motions=[],
            current_motion="",
            motion_set=0,
            playing=False,
        )
    )
    pending_line: str = ""

    def read_sonic(self, value):
        """Keep native state across PTY chunks and display-log truncation."""
        if self.step['id'] == 'teleop':
            lines = (self.pending_line + value).split('\n')
            self.pending_line = lines.pop()[-8192:]
            for line in lines:
                for marker, stage in (('PICO_PREPARE_READY ', 'ready'), ('PICO_PREPARING ', 'preparing'),
                                      ('PICO_PREPARED ', 'prepared'), ('PICO_STARTING ', 'starting')):
                    if line.startswith(marker): self.preparation = stage
            return
        if self.step['id'] == 'capture':
            lines = (self.pending_line + value).split('\n')
            self.pending_line = lines.pop()[-8192:]
            for line in lines:
                match = re.fullmatch(r'CAPTURE_(READY|RECORDING|LABEL|LABELED|COMPLETE) (\{.*\})\r?', line)
                if match:
                    try:
                        event = json.loads(match[2])
                    except ValueError:
                        continue
                    if isinstance(event, dict):
                        self.capture.update(event, stage=match[1].lower())
            return
        if self.step['id'] == 'vla':
            lines = (self.pending_line + value).split('\n')
            self.pending_line = lines.pop()[-8192:]
            for line in lines:
                for marker, stage in (('READY','ready'), ('STARTING','starting'), ('STANDING','standing'),
                                      ('STARTED','running'), ('COMPLETE','completed'), ('STOPPED','stopped'), ('FAILED','failed')):
                    if line.startswith('VLA_' + marker + ' '): self.vla['stage'] = stage
                progress = re.search(r'VLA_PROGRESS frames=(\d+) seconds=([\d.]+)', line)
                if progress: self.vla.update(frames=int(progress[1]), seconds=float(progress[2]))
            return
        if self.step['id'] == 'dataset_replay':
            lines = (self.pending_line + value).split('\n')
            self.pending_line = lines.pop()[-8192:]
            for line in lines:
                for marker, stage in (('READY','ready'), ('STARTING','starting'), ('STANDING','standing'), ('ALIGNING','aligning'),
                                      ('STARTED','playing'), ('COMPLETE','completed'), ('STOPPED','stopped'), ('FAILED','failed')):
                    if line.startswith('REPLAY_' + marker + ' '): self.replay['stage'] = stage
                progress = re.search(r'REPLAY_PROGRESS frame=(\d+)/(\d+) time_s=([\d.]+)', line)
                if progress:
                    self.replay.update(frame=int(progress[1]), total=int(progress[2]), time_s=float(progress[3]))
            return
        if self.step["id"] != "body" or "1" not in self.step["inputs"]:
            return
        lines = (self.pending_line + value).split("\n")
        self.pending_line = lines.pop()[-8192:]
        for line in lines:
            line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line).strip()
            loaded = re.search(r"✓ Loaded (.+) \((\d+) timesteps\)", line)
            if loaded and loaded[1] not in [m["name"] for m in self.sonic["motions"]]:
                self.sonic["motions"].append(dict(name=loaded[1], frames=int(loaded[2])))
            selected = re.search(r"Started with motion: (.+) \(paused at frame 0\)", line)
            selected = selected or re.search(r"Reference motion name: (.+)$", line)
            if selected and selected[1] in [m["name"] for m in self.sonic["motions"]]:
                if self.sonic["current_motion"] != selected[1]:
                    self.sonic["playing"] = False
                self.sonic["current_motion"] = selected[1]
            if line == "Planner enabled":
                self.sonic.update(mode="planner", playing=True)
            elif line == "Planner disabled":
                self.sonic.update(mode="reference", playing=False)
            group = re.search(r"Motion set: (\d+)", line)
            if group:
                self.sonic["motion_set"] = int(group[1])
            if re.search(r"Playing motion \d+ from frame", line):
                self.sonic["playing"] = True
            if re.search(r"Reset motion \d+ to frame 0|Motion index: .* completed\.", line):
                self.sonic["playing"] = False

    def public(self):
        return dict(
            id=self.id,
            plan_id=self.plan_id,
            step_id=self.step["id"],
            title=self.step["title"],
            machine=self.step["machine"],
            state=self.state,
            close_requested=self.close_requested,
            terminal_closed=self.close_requested and self.state in ("stopped", "completed", "failed"),
            startup_events=sorted(self.startup_events),
            control_mode=self.step.get("control_mode", ""),
            sonic=json.loads(json.dumps(self.sonic)),
            replay=dict(self.replay),
            vla=dict(self.vla),
            capture=dict(self.capture),
            preparation=self.preparation,
            vla_control=self.step.get('vla_control', False),
            dataset_replay=self.step.get('dataset_replay', False),
            ready=self.ready,
            service=self.step["service"],
            ready_pattern=self.step["ready"],
            returncode=self.returncode,
            started=self.started,
            pid=self.process.pid,
            log_path=str(self.log_path),
            output=self.output[-80000:],
            inputs=self.step["inputs"],
            effector_only=self.step.get('effector_only', False),
            outputs=self.step["outputs"],
            command=self.step["launch_command"],
        )


class Runner:
    def __init__(self):
        self.lock = threading.RLock()
        self.jobs: dict[str, Job] = {}
        self.closed = False

    def latest(self, plan_id, step_id):
        with self.lock:
            matches = [j for j in self.jobs.values() if j.plan_id == plan_id and j.step["id"] == step_id]
            return matches[-1] if matches else None

    def satisfied(self, job):
        return job and (
            (job.state == "completed" and not job.step["service"]) or (job.state == "running" and job.ready)
        )

    def dependency_ready(self, plan, step_id):
        if self.satisfied(self.latest(plan["id"], step_id)):
            return True
        step = next((s for s in plan["steps"] if s["id"] == step_id), None)
        if not step or not step["service"]:
            return False
        # Reuse only an owned, ready service with the same command and control contract.
        keys = ("id", "launch_argv", "resources", "ready", "inputs", "control_mode")
        return any(
            job.step["service"] and self.satisfied(job)
            and all(job.step.get(key) == step.get(key) for key in keys)
            for job in self.jobs.values()
        )

    def start(self, plan, step_id, external=False):
        with self.lock:
            if self.closed:
                raise ValueError("控制台正在关闭")
            step = next((s for s in plan["steps"] if s["id"] == step_id), None)
            if not step:
                raise ValueError("执行步骤不存在")
            for other in self.jobs.values():
                if other.state in ("running", "stopping") and set(other.step["resources"]) & set(
                    step["resources"]
                ):
                    raise ValueError(f"资源被本控制台进程占用：{other.step['title']} ({other.id})")
            if not external:
                missing = [d for d in step["depends"] if not self.dependency_ready(plan, d)]
                if missing:
                    labels = [f"{index + 1:02d} {item['title']}" for index, item in enumerate(plan['steps'])
                              if item['id'] in missing]
                    raise ValueError("前置步骤尚未就绪：" + "；".join(labels) + "。请先执行并在终端确认就绪。")
            lease = None
            if step.get('device_context'):
                from gear_sonic.utils.hand_control.registry import deployment
                from gear_sonic.utils.hand_control.management import DeviceLease
                context = step['device_context']; frozen = context['snapshot']
                selected = deployment(frozen['device'], frozen['backend'], mounts=frozen['mounts'], options=frozen['options'],
                                      registry_dir=context['registry_dir'])
                if selected.registration.sha256 != frozen['registration']['registration_sha256']:
                    raise ValueError('设备注册信息已改变，请重新生成任务命令')
                lease = DeviceLease(selected, 'console:' + step_id)
            job_id = uuid.uuid4().hex[:12]
            log_dir = Path(plan["log_root"]) / job_id
            log_dir.mkdir(parents=True, exist_ok=False)
            (log_dir / "command.json").write_text(
                json.dumps(
                    {
                        "plan_id": plan["id"],
                        "config": plan["config"],
                        "step": step,
                        "external_dependencies_confirmed": external,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            master, slave = pty.openpty()
            attributes = termios.tcgetattr(slave)
            attributes[3] &= ~termios.ECHO
            termios.tcsetattr(slave, termios.TCSANOW, attributes)
            try:
                if lease:
                    lease.start()
                process = subprocess.Popen(
                    step["launch_argv"],
                    stdin=slave,
                    stdout=slave,
                    stderr=slave,
                    start_new_session=True,
                    close_fds=True,
                )
            except BaseException:
                if lease:
                    lease.close()
                os.close(master)
                raise
            finally:
                os.close(slave)
            job = Job(job_id, plan["id"], step, log_dir / "stdout.log", process, master)
            if lease:
                from gear_sonic.utils.hand_control.management import atomic_json, process_identity
                lease.row.update(pid=process.pid, process_start=process_identity(process.pid))
                atomic_json(lease.path, lease.row)
                job.device_lease = lease
            self.jobs[job_id] = job
            threading.Thread(target=self._read, args=(job,), daemon=True).start()
            return job.public()

    def _read(self, job):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        with job.log_path.open("w", encoding="utf-8") as log:
            while True:
                try:
                    data = os.read(job.master, 8192)
                except OSError:
                    break
                if not data:
                    break
                value = decoder.decode(data)
                log.write(value)
                log.flush()
                with self.lock:
                    job.read_sonic(value)
                    recent = job.output[-128:] + value
                    job.startup_events.update(
                        key for key, message in STARTUP_MESSAGES.items() if message in recent
                    )
                    job.output = (job.output + value)[-160000:]
                    if job.step["ready"] and job.step["ready"] in job.output:
                        job.ready = True
        result = job.process.wait()
        if job.device_lease:
            job.device_lease.close()
        with self.lock:
            job.returncode = result
            job.state = "stopped" if job.stopped else "completed" if result == 0 else "failed"
            job.ready = False
            os.close(job.master)
            (job.log_path.parent / "result.json").write_text(
                json.dumps({"state": job.state, "returncode": result, "ended": time.time()}, indent=2)
            )

    def input(self, job_id, value, *, capture_episode=None):
        with self.lock:
            job = self.jobs[job_id]
            if job.state != "running":
                raise ValueError("进程已不在运行")
            if value not in job.step["inputs"]:
                raise ValueError("该步骤不接受此终端输入")
            if job.step['id'] == 'teleop' and job.preparation != 'inactive':
                next_stage = {'ready': 'preparing', 'prepared': 'starting'}.get(job.preparation)
                if value != '\n' or next_stage is None:
                    raise ValueError('请等待准备姿态提示，不能重复确认')
                job.preparation = next_stage
            replay_stage = None
            vla_stage = None
            capture_stage = None
            if job.step['id'] == 'capture':
                if (job.capture['stage'] == 'ready' and value == '\n'
                    and type(capture_episode) is int and capture_episode == job.capture['episode']):
                    capture_stage = 'recording'
                elif (job.capture['stage'] == 'label'
                      and value in ('success\n', 'failure\n', 'unlabeled\n')
                      and type(capture_episode) is int and capture_episode == job.capture['episode']):
                    capture_stage = 'labeling'
                else:
                    raise ValueError('请等待当前条目的开始或标注提示，不能提前或重复提交')
            if job.step['id'] == 'dataset_replay':
                replay_stage = {'ready':'starting', 'standing':'aligning'}.get(job.replay['stage'])
                if replay_stage is None:
                    raise ValueError('请等待回放程序显示下一步，不能重复确认')
            if job.step['id'] == 'vla':
                vla_stage = {'ready':'starting', 'standing':'inference_starting'}.get(job.vla['stage'])
                if vla_stage is None:
                    raise ValueError('请等待 VLA 程序显示下一步，不能重复确认')
            os.write(job.master, value.encode())
            if replay_stage: job.replay['stage'] = replay_stage
            if vla_stage: job.vla['stage'] = vla_stage
            if capture_stage: job.capture['stage'] = capture_stage

    def mark_ready(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            if job.state != "running" or not job.step["service"]:
                raise ValueError("只有运行中的服务可确认就绪")
            if job.step["ready"] and not job.ready:
                raise ValueError("尚未出现运行书要求的就绪日志，不能跳过")
            job.ready = True

    def stop(self, job_id, force=False):
        with self.lock:
            job = self.jobs[job_id]
            if job.state not in ("running", "stopping"):
                return
            job.stopped = True
            job.state = "stopping"
            try:
                if job.step["machine"] == "g1" and not force:
                    # The remote command owns SSH's PTY: terminal Ctrl+C targets
                    # its foreground process group rather than arbitrary G1 PIDs.
                    os.write(job.master, b"\x03")
                else:
                    os.killpg(job.process.pid, signal.SIGKILL if force else signal.SIGINT)
            except (ProcessLookupError, OSError):
                pass

    def close_terminal(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            job.close_requested = True
            if job.state == "running":
                self.stop(job_id)
            # Keep the terminal until _read has waited for the process to exit.
            # Retain job history for dependencies and saved logs.
            return job.public()

    def stop_all(self):
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda j: j.step["stop_order"], reverse=True)

        def ordered():
            for job in jobs:
                if job.state not in ("running", "stopping"):
                    continue
                self.stop(job.id)
                try:
                    # Dataset commit + stats can take minutes. Never kill it
                    # automatically or tear down its sources while finalizing.
                    job.process.wait(timeout=600 if job.step["id"] == "capture" else 20)
                except subprocess.TimeoutExpired:
                    return

        threading.Thread(target=ordered, daemon=True).start()

    def snapshot(self):
        with self.lock:
            return {
                "jobs": [job.public() for job in self.jobs.values()],
            }
