"""Run with python3 -m gear_sonic.utils.console.server."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
from urllib.parse import urlparse

from .catalog import FEATURES, build_plan
from .config import ROOT, Profiles, fields, effector_catalog
from gear_sonic.utils.hand_control.interface import EffectorError
from .runner import Runner


class Console:
    def __init__(self, directory=None):
        self.profiles = Profiles(directory)
        self.runner = Runner()
        self.plans = {}
        self.lock = threading.RLock()
        self.token = secrets.token_urlsafe(32)
        self.session_id = secrets.token_hex(16)


def handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, data, status=200, content_type="application/json; charset=utf-8"):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Console-Session", app.session_id)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(data)

        def allowed(self):
            host = self.headers.get("Host", "")
            valid = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            origin = self.headers.get("Origin")
            return host in valid and (not origin or origin == "http://" + host)

        def do_GET(self):
            if not self.allowed():
                return self.send({"error": "只接受本机同源访问"}, 403)
            path = urlparse(self.path).path
            if path == "/api/state":
                profiles = app.profiles.snapshot()
                try:
                    devices = effector_catalog(profiles['profiles'][profiles['active']])
                    registry_error = ''
                except (ValueError, KeyError, OSError, EffectorError) as exc:
                    devices, registry_error = [], str(exc)
                return self.send(
                    {
                        "profiles": profiles,
                        "devices": devices,
                        "registry_error": registry_error,
                        "fields": fields(),
                        "checkout": str(ROOT),
                        "features": FEATURES,
                        "token": app.token,
                        **app.runner.snapshot(),
                    }
                )
            if path == "/api/jobs":
                return self.send(app.runner.snapshot())
            if path == "/api/outputs":
                result = []
                for job in app.runner.snapshot()["jobs"]:
                    for value in job["outputs"]:
                        local = not value.startswith("G1:")
                        p = Path(value)
                        entries = (
                            sorted(str(e) for e in p.parent.glob(p.name + "*"))[:100]
                            if local and p.parent.is_dir()
                            else []
                        )
                        result.append({"path": value, "machine": "local" if local else "g1", "matches": entries})
                return self.send(result)
            static = {
                "/": ("index.html", "text/html"),
                "/app.js": ("app.js", "text/javascript"),
                "/settings.js": ("settings.js", "text/javascript"),
                "/i18n.js": ("i18n.js", "text/javascript"),
                "/ja.js": ("ja.js", "text/javascript"),
                "/style.css": ("style.css", "text/css"),
            }
            if path in static:
                name, mime = static[path]
                return self.send(
                    (Path(__file__).parent / "static" / name).read_bytes(), content_type=mime + "; charset=utf-8"
                )
            self.send({"error": "不存在"}, 404)

        def do_POST(self):
            if not self.allowed() or not secrets.compare_digest(
                self.headers.get("X-Console-Token", ""), app.token
            ):
                return self.send({"error": "本机会话校验失败，请刷新网页"}, 403)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 200000:
                    raise ValueError("请求大小无效")
                data = json.loads(self.rfile.read(size))
                path = urlparse(self.path).path
                if path == '/api/directories':
                    from .directories import list_directories
                    return self.send(list_directories(data.get('path'), project_root=data.get('root', str(ROOT))))
                if path == '/api/config-check':
                    from .bootstrap import inspect_configuration
                    return self.send(inspect_configuration(data['config']))
                if path == '/api/devices':
                    return self.send(effector_catalog(data['config']))
                if path == '/api/datasets':
                    from .config import validate
                    from gear_sonic.utils.data_collection.paths import list_datasets
                    config = validate(data['config'], check_devices=False)
                    return self.send(list_datasets(config['local.root']))
                if path == '/api/output-directories':
                    from .config import validate
                    from .bootstrap import initialize_outputs
                    return self.send(initialize_outputs(validate(data['config'])))
                if path == "/api/profiles":
                    return self.send(app.profiles.save(data["name"], data["config"]))
                if path == "/api/plan":
                    plan = build_plan(data["config"], data["feature"], data["mode"])
                    with app.lock:
                        app.plans[plan["id"]] = plan
                    return self.send(plan)
                if path == "/api/start":
                    plan = app.plans[data["plan_id"]]
                    return self.send(app.runner.start(plan, data["step_id"], external=data.get("external", False)))
                if path == "/api/input":
                    app.runner.input(data["job_id"], data["value"], capture_episode=data.get('capture_episode'))
                elif path == "/api/ready":
                    app.runner.mark_ready(data["job_id"])
                elif path == "/api/stop":
                    app.runner.stop(data["job_id"], force=data.get("force", False))
                elif path == "/api/close":
                    return self.send(app.runner.close_terminal(data["job_id"]))
                elif path == "/api/stop-all":
                    app.runner.stop_all()
                else:
                    return self.send({"error": "接口不存在"}, 404)
                self.send({"ok": True})
            except (ValueError, KeyError, OSError, EffectorError) as exc:
                self.send({"error": str(exc)}, 400)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state-dir", type=Path, help="配置目录；相对项目根目录，默认 configs/console")
    args = parser.parse_args()
    app = Console(ROOT / args.state_dir if args.state_dir is not None else None)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(app))
    print(f"SONIC 实验控制台 http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.runner.closed = True
        app.runner.stop_all()
        # Do not abandon recorder finalization on server shutdown.
        for job in list(app.runner.jobs.values()):
            try:
                job.process.wait(timeout=610)
            except Exception:
                print(f"进程尚未退出，检查日志 {job.log_path} (PID {job.process.pid})", flush=True)
        server.server_close()


if __name__ == "__main__":
    main()
