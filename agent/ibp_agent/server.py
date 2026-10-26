"""Loopback-only UI with authenticated requests and a single native GPU worker."""

import concurrent.futures, json, mimetypes, secrets, subprocess, threading, urllib.parse, uuid, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from .workflow import WorkflowAgent
from .dataset import Dataset
from .settings import Settings, atomic_json
from .parameter_model import ParameterModel, BOUNDS, DEFAULT_PARAMS
from .training_runtime import TrainingRuntime
from .personal import PersonalPolicy
from .knowledge import Knowledge
from .model import LocalModel
from .runtime import ModelProcess, locations

EXT = {
    ".cr2",
    ".cr3",
    ".nef",
    ".arw",
    ".dng",
    ".raf",
    ".rw2",
    ".orf",
    ".pef",
    ".srw",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}


class Application:
    def __init__(self):
        self.paths = locations()
        self.token = secrets.token_urlsafe(32)
        self.jobs = {}
        self.uploads = {}
        self.lock = threading.Lock()
        self.worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.model = ModelProcess(self.paths)
        self.settings = Settings(self.paths["data"])
        self.dataset = Dataset()
        self.training = TrainingRuntime(self.dataset)
        self.personal = PersonalPolicy(self.training)
        self.personal_state = {"status": "idle"}
        self.index_result = None
        self.ready = False
        self.error = None
        self.busy = False
        self.knowledge = Knowledge(
            self.paths["knowledge"], self.paths["data"] / "knowledge.sqlite"
        )

    def restore(self):
        for file in (self.paths["data"] / "uploads").glob("*/upload.json"):
            try:
                info = json.loads(file.read_text(encoding="utf-8"))
                source = file.parent / Path(info["file"]).name
                if source.exists():
                    self.uploads[file.parent.name] = {"name": info["name"], "path": source}
            except (ValueError, KeyError, OSError):
                continue
        for file in sorted((self.paths["data"] / "jobs").glob("*/job.json"), key=lambda p: p.stat().st_mtime):
            try:
                job = json.loads(file.read_text(encoding="utf-8"))
                if job["upload"] not in self.uploads:
                    continue
                job["report"] = json.loads((file.parent / "report.json").read_text(encoding="utf-8"))
                job.update(status="done", events=[])
                label = self.dataset.root / "samples" / job["id"] / "sample.json"
                if label.exists():
                    job["feedback"] = json.loads(label.read_text(encoding="utf-8"))["feedback"]
                self.jobs[job["id"]] = job
            except (ValueError, KeyError, OSError):
                continue

    def index_pdf(self, file, name):
        try:
            from pypdf import PdfReader
            from .dataset import sha256
            reader = PdfReader(file)
            if len(reader.pages) > 1500:
                raise ValueError("单篇论文上限 1500 页")
            title = str((reader.metadata or {}).get("/Title") or name)[:300]
            source = {"id": "local-" + sha256(file)[:20], "title": title, "access": "user supplied PDF; bibliographic identity unverified"}
            pages = [p.extract_text() or "" for p in reader.pages]
            chunks = self.knowledge.papers.add(source, file, pages, sha256(file))
            self.index_result = {"status": "done", "chunks": chunks, "title": title}
        except Exception as e:
            self.index_result = {"status": "error", "error": str(e)}
        finally:
            with self.lock:
                self.busy = False

    def initialize(self):
        try:
            endpoint = self.model.start()
            self.agent = WorkflowAgent(
                self.paths["bin"], self.knowledge, LocalModel(endpoint, self.model.key)
            )
            self.error = None
            self.ready = True
        except Exception as e:
            self.error = str(e)
            self.model.close()

    def predictor(self, role):
        if role == "student" and self.personal.predictor:
            return self.personal.predictor
        profile = self.settings.profile(role)
        local = role == "student" and not profile["base_url"]
        if local:
            profile = {"base_url": self.model.endpoint, "api_key": self.model.key, "model": "local"}
        elif role == "teacher" and not profile["api_key"] and not profile["base_url"].startswith("http://127.0.0.1"):
            raise ValueError("请在 API 设置中填写教师密钥")
        return ParameterModel(profile, local=local)

    def run_job(self, ident, prompt, upload, role="student", parent=None, manual=None):
        job = self.jobs[ident]
        source = self.uploads[upload]["path"]
        folder = self.paths["data"] / "jobs" / ident
        job["status"] = "running"
        try:

            def progress(event):
                job["events"].append(event)

            job["report"] = self.agent.run(source, prompt, folder, progress,
                predictor=None if parent else self.predictor(role), role=role, parent=parent, manual=manual)
            job["report"]["source_name"] = self.uploads[upload]["name"]
            if parent:
                job["report"]["parent_job"] = job.get("parent")
            atomic_json(folder / "report.json", job["report"])
            atomic_json(folder / "job.json", {"id": ident, "upload": upload, "parent": job.get("parent")})
            job["status"] = "done"
        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
            atomic_json(folder / "failure.json", {"error": str(e), "role": role, "prompt": prompt})
        finally:
            with self.lock:
                self.busy = False

    def export_job(self, ident):
        job = self.jobs[ident]
        job["export_status"] = "running"
        try:
            job["export"] = self.agent.export(
                self.uploads[job["upload"]]["path"],
                job["report"],
                self.paths["data"] / "jobs" / ident / "exports",
            )
            job["export_status"] = "done"
        except Exception as e:
            job["export_status"] = "error"
            job["export_error"] = str(e)
        finally:
            with self.lock:
                self.busy = False

    def train(self, kind):
        try:
            if kind == "qwen":
                self.personal.close()
                self.personal_state = {"status": "idle"}
                self.ready = False
                self.model.close()
            self.training.run(kind, self.settings.public()["training_model"])
        finally:
            if kind == "qwen":
                self.initialize()
            with self.lock:
                self.busy = False

    def activate_personal(self, name):
        try:
            if name:
                self.personal.activate(name)
            else:
                self.personal.close()
            self.personal_state = {"status": "done", "selected": self.personal.selected}
        except Exception as e:
            self.personal_state = {"status": "error", "error": str(e)}
        finally:
            with self.lock:
                self.busy = False

    def close(self):
        self.personal.close()
        self.worker.shutdown(wait=True, cancel_futures=True)
        self.model.close()


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, data, ctype="application/json; charset=utf-8"):
            payload = (
                json.dumps(data, ensure_ascii=False).encode()
                if isinstance(data, (dict, list))
                else data
            )
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(payload)

        def authorized(self):
            u = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(u.query)
            token = self.headers.get("X-IBP-Token", query.get("token", [""])[0])
            origin = self.headers.get("Origin")
            allowed = f"http://127.0.0.1:{self.server.server_port}"
            return (
                secrets.compare_digest(token, app.token)
                and (not origin or origin == allowed)
                and self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"
            )

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            if path in ("/", "/app.js", "/rag.js", "/style.css"):
                f = app.paths["web"] / ({"/": "index.html"}.get(path, path[1:]))
                return self.send(
                    200, f.read_bytes(), mimetypes.guess_type(str(f))[0] or "text/plain"
                )
            if not self.authorized():
                return self.send(403, {"error": "Unauthorized"})
            if path == "/api/status":
                return self.send(
                    200,
                    {
                        "ready": app.ready,
                        "error": app.error,
                        "busy": app.busy,
                        "model": app.paths.get("model", Path("Qwen3")).stem,
                        "data_dir": str(app.paths["data"]),
                    },
                )
            if path == "/api/settings":
                return self.send(200, app.settings.public())
            if path == "/api/parameters":
                return self.send(200, {"bounds": BOUNDS, "defaults": DEFAULT_PARAMS})
            if path == "/api/dataset":
                return self.send(200, app.dataset.status())
            if path == "/api/training":
                return self.send(200, {**app.training.status(), "personal": app.personal_state})
            if path == "/api/jobs":
                return self.send(200, [{"id": j["id"], "status": j["status"], "name": app.uploads[j["upload"]]["name"]} for j in reversed(list(app.jobs.values()))])
            if path == "/api/rag/status":
                return self.send(200, {**app.knowledge.papers.status(), "import": app.index_result})
            if path == "/api/rag/search":
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                try:
                    return self.send(
                        200,
                        app.knowledge.search_papers(
                            query.get("q", [""])[0], int(query.get("limit", ["5"])[0])
                        ),
                    )
                except (ValueError, TypeError) as e:
                    return self.send(400, {"error": str(e)})
            if path.startswith("/api/rag/papers/"):
                pdf = app.knowledge.papers.pdf(path.rsplit("/", 1)[-1])
                if pdf is None:
                    return self.send(404, {"error": "Indexed paper not found"})
                return self.send(200, pdf.read_bytes(), "application/pdf")
            if path == "/api/knowledge":
                return self.send(
                    200,
                    {"terms": app.knowledge.terms, "sources": app.knowledge.sources},
                )
            if path.startswith("/api/jobs/"):
                ident = path.split("/")[-1]
                job = app.jobs.get(ident)
                return (
                    self.send(200, job)
                    if job
                    else self.send(404, {"error": "Unknown job"})
                )
            if path.startswith("/files/"):
                parts = path.split("/")[2:]
                if len(parts) < 2 or parts[0] not in app.jobs:
                    return self.send(404, {"error": "Not found"})
                root = (app.paths["data"] / "jobs" / parts[0]).resolve()
                f = root.joinpath(*parts[1:]).resolve()
                if not f.is_relative_to(root) or not f.is_file():
                    return self.send(404, {"error": "Not found"})
                return self.send(
                    200,
                    f.read_bytes(),
                    mimetypes.guess_type(str(f))[0] or "application/octet-stream",
                )
            self.send(404, {"error": "Not found"})

        def do_POST(self):
            if not self.authorized():
                return self.send(403, {"error": "Unauthorized"})
            try:
                self.post()
            except Exception as e:
                self.send(400, {"error": str(e)})

        def post(self):
            path = urllib.parse.urlparse(self.path).path
            size = int(self.headers.get("Content-Length", "0"))
            if path == "/api/rag/import":
                if not 0 < size <= 50 * 1024 * 1024:
                    raise ValueError("论文文件上限 50 MB")
                with app.lock:
                    if app.busy:
                        return self.send(409, {"error": "已有任务运行"})
                    app.busy = True
                try:
                    name = Path(urllib.parse.unquote(self.headers.get("X-Filename", "paper.pdf"))).name
                    data = self.rfile.read(size)
                    if len(data) != size or not data.startswith(b"%PDF-"):
                        raise ValueError("PDF 文件无效")
                    import hashlib
                    file = app.dataset.root / "papers" / (hashlib.sha256(data).hexdigest() + ".pdf")
                    file.parent.mkdir(exist_ok=True)
                    file.write_bytes(data)
                    app.index_result = {"status": "running", "title": name}
                    app.worker.submit(app.index_pdf, file, name)
                    return self.send(202, {"status": "running"})
                except Exception:
                    app.busy = False
                    raise
            if path == "/api/upload":
                if not 0 < size <= 100 * 1024 * 1024:
                    raise ValueError("单张文件上限 100 MB")
                name = Path(
                    urllib.parse.unquote(self.headers.get("X-Filename", "image"))
                ).name
                ext = Path(name).suffix.lower()
                if ext not in EXT:
                    raise ValueError("不支持此图像格式")
                ident = uuid.uuid4().hex
                folder = app.paths["data"] / "uploads" / ident
                folder.mkdir(parents=True)
                f = folder / ("original" + ext)
                with f.open("xb") as stream:
                    left = size
                    while left:
                        chunk = self.rfile.read(min(left, 1024 * 1024))
                        if not chunk:
                            raise ValueError("Incomplete upload")
                        stream.write(chunk)
                        left -= len(chunk)
                app.uploads[ident] = {"path": f, "name": name}
                atomic_json(folder / "upload.json", {"name": name, "file": f.name})
                return self.send(200, {"id": ident, "name": name})
            if not 0 <= size <= 65536:
                raise ValueError("Request too large")
            body = json.loads(self.rfile.read(size) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("Request must be an object")
            if path == "/api/settings":
                if app.busy:
                    return self.send(409, {"error": "任务运行中"})
                return self.send(200, app.settings.update(body))
            if path == "/api/feedback":
                ident = body.get("id")
                job = app.jobs.get(ident)
                if not job or job["status"] != "done":
                    raise ValueError("图片尚未生成")
                result = app.dataset.label(ident, job["report"], app.uploads[job["upload"]]["path"],
                    app.paths["data"] / "jobs" / ident, body.get("accepted"), body.get("group_id", ""))
                job["feedback"] = {"accepted": body["accepted"], "revision": result["revision"]}
                return self.send(200, result)
            if path == "/api/manual":
                parent = app.jobs.get(body.get("id"))
                if not parent or parent["status"] != "done":
                    raise ValueError("请先生成图片")
                from .native import validate_params
                params = validate_params(body.get("params", {}))
                with app.lock:
                    if app.busy:
                        return self.send(409, {"error": "已有任务运行"})
                    app.busy = True
                ident = uuid.uuid4().hex
                app.jobs[ident] = {"id": ident, "upload": parent["upload"], "parent": parent["id"], "status": "queued", "events": []}
                app.worker.submit(app.run_job, ident, parent["report"]["prompt"], parent["upload"], parent["report"]["role"], parent["report"], params)
                return self.send(202, {"id": ident})
            if path == "/api/dataset/export":
                return self.send(200, app.dataset.export())
            if path == "/api/open-folder":
                import os
                folder = app.dataset.root / ("weights" if body.get("kind") == "weights" else "")
                os.startfile(str(folder))
                return self.send(200, {"path": str(folder)})
            if path == "/api/weights/activate":
                name = body.get("id")
                if name is not None and name not in {w["id"] for w in app.training.weights()}:
                    raise ValueError("未知个人权重")
                with app.lock:
                    if app.busy:
                        return self.send(409, {"error": "已有任务运行"})
                    app.busy = True
                app.personal_state = {"status": "loading"}
                app.worker.submit(app.activate_personal, name)
                return self.send(202, app.personal_state)
            if path == "/api/training/cancel":
                return self.send(200, app.training.cancel())
            if path == "/api/training":
                kind = body.get("kind")
                if kind not in ("mlp", "prepare", "qwen"):
                    raise ValueError("未知训练任务")
                if kind != "prepare":
                    from .training import split_rows
                    rows = [x for x in app.dataset.records() if x["feedback"]["accepted"]]
                    split_rows(rows)
                with app.lock:
                    if app.busy:
                        return self.send(409, {"error": "已有任务运行"})
                    app.busy = True
                app.training.state = {"status": "queued", "kind": kind}
                app.worker.submit(app.train, kind)
                return self.send(202, {"status": "queued"})
            if path == "/api/jobs":
                role = body.get("role", "student")
                if role not in ("teacher", "student"):
                    raise ValueError("未知模型角色")
                if not app.ready:
                    raise ValueError(app.error or "模型仍在加载")
                if body.get("upload") not in app.uploads:
                    raise ValueError("请先选择图片")
                prompt = body.get("prompt", "")
                if not isinstance(prompt, str) or not 0 < len(prompt) <= 1500:
                    raise ValueError("请输入 1–1500 字需求")
                app.predictor(role)
                with app.lock:
                    if app.busy:
                        return self.send(409, {"error": "已有任务运行，请等待完成"})
                    app.busy = True
                ident = uuid.uuid4().hex
                app.jobs[ident] = {
                    "id": ident,
                    "upload": body["upload"],
                    "status": "queued",
                    "events": [],
                }
                app.worker.submit(app.run_job, ident, prompt, body["upload"], role)
                return self.send(202, {"id": ident})
            if path == "/api/export":
                ident = body.get("id")
                job = app.jobs.get(ident)
                if not job or job["status"] != "done":
                    raise ValueError("预览尚未完成")
                with app.lock:
                    if app.busy:
                        return self.send(409, {"error": "已有任务运行"})
                    app.busy = True
                job["export_status"] = "queued"
                job.pop("export", None)
                job.pop("export_error", None)
                app.worker.submit(app.export_job, ident)
                return self.send(202, {"id": ident})
            if path == "/api/studio":
                source = []
                if body.get("id") in app.jobs:
                    source = [str(app.uploads[app.jobs[body["id"]]["upload"]]["path"])]
                subprocess.Popen(
                    [str(app.paths["bin"] / "ImageBatchProcessorStudio.exe"), *source],
                    cwd=app.paths["bin"],
                    env=dict(__import__("os").environ, IBP_AGENT_DATA=str(app.paths["data"])),
                )
                return self.send(
                    200,
                    {
                        "message": "Studio 已打开原图；通过配方菜单导入导出结果旁的 .imgarg 即可继续编辑。"
                    },
                )
            if path == "/api/quit":
                if app.busy:
                    return self.send(409, {"error": "请等待任务完成后退出"})
                self.send(200, {"message": "已退出"})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            self.send(404, {"error": "Not found"})

    return Handler


def serve(no_browser=False, port=0, studio=True):
    from .instance import acquire, reuse
    if not acquire():
        return reuse(no_browser, studio)
    app = Application()
    app.restore()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    server.daemon_threads = True
    url = f"http://127.0.0.1:{server.server_port}/#" + app.token
    (app.paths["data"] / "session.json").write_text(
        json.dumps({"url": url, "port": server.server_port, "pid": __import__("os").getpid()}), encoding="utf-8"
    )
    threading.Thread(target=app.initialize, daemon=True).start()
    if not no_browser:
        if studio:
            subprocess.Popen([str(app.paths["bin"] / "ImageBatchProcessorStudio.exe")], cwd=app.paths["bin"],
                env=dict(__import__("os").environ, IBP_AGENT_DATA=str(app.paths["data"])))
        else:
            webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.3)
    finally:
        server.server_close()
        app.close()
