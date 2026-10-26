"""Configuration and owned model-process lifetime. No global PATH or network changes."""

import os, socket, subprocess, sys, time, urllib.request
from pathlib import Path


def resources():
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))


def install_root():
    return (
        Path(sys.executable).parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parents[2] / ".local"
    )


def user_root():
    p = Path(
        os.environ.get(
            "IBP_AGENT_DATA",
            str(
                Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "IBPPhotoAgent"
            ),
        )
    )
    p.mkdir(parents=True, exist_ok=True)
    return p


def locations():
    base = install_root()
    return {
        "bin": base / "app" / "bin",
        "model": next((base / "models" / n for n in ("Qwen3-1.7B-Q4_K_M.gguf", "Qwen3-1.7B-Q8_0.gguf", "Qwen3-0.6B-Q8_0.gguf") if (base / "models" / n).exists()), base / "models/Qwen3-1.7B-Q4_K_M.gguf"),
        "llama": base / "llama" / "llama-server.exe",
        "knowledge": resources() / "knowledge",
        "web": resources() / "web",
        "data": user_root(),
    }


class ModelProcess:
    def __init__(self, paths):
        self.paths = paths
        self.process = None
        self.log = None
        self.endpoint = None
        self.key = None

    def start(self):
        import secrets

        for k in ("model", "llama"):
            if not self.paths[k].exists():
                raise FileNotFoundError(f"Missing {k}: {self.paths[k]}")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.endpoint = f"http://127.0.0.1:{port}"
        self.key = secrets.token_urlsafe(32)
        self.log = (self.paths["data"] / "model.log").open("w", encoding="utf-8")
        args = [
            str(self.paths["llama"]),
            "-m",
            str(self.paths["model"]),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-c",
            "6144",
            "-t",
            str(min(4, os.cpu_count() or 4)),
            "--parallel",
            "1",
            "--api-key",
            self.key,
            "--jinja",
            "--no-webui",
        ]
        self.process = subprocess.Popen(
            args,
            cwd=self.paths["llama"].parent,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        end = time.monotonic() + 120
        while time.monotonic() < end:
            if self.process.poll() is not None:
                raise RuntimeError("Model process exited; see model.log")
            try:
                req = urllib.request.Request(
                    self.endpoint + "/health",
                    headers={"Authorization": "Bearer " + self.key},
                )
                with opener.open(req, timeout=2) as r:
                    if r.status == 200:
                        return self.endpoint
            except Exception:
                pass
            time.sleep(0.5)
        raise TimeoutError("Model did not become ready within 120 seconds")

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log:
            self.log.close()
