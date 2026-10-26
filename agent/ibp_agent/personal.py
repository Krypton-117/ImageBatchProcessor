"""Activation of retained personal weights; normal rendering guards still apply."""
import json, os, secrets, socket, subprocess, time, urllib.request
from .parameter_model import ParameterModel, SYSTEM, PARAM_VERSION
from .training import predict_mlp

class MLPPredictor:
    def __init__(self, folder): self.folder = folder
    def propose(self, payload):
        params = predict_mlp(self.folder / "weights.npz", payload)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        return {"params": params, "messages": messages, "model": self.folder.name, "base_url": "personal-local-mlp",
                "raw_response": json.dumps({"params": params}), "prompt_version": PARAM_VERSION}

class PersonalPolicy:
    def __init__(self, training):
        self.training = training
        self.process = None
        self.log = None
        self.selected = None
        self.predictor = None
    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(10)
            except subprocess.TimeoutExpired: self.process.kill(); self.process.wait()
        if self.log: self.log.close()
        self.process = self.log = self.predictor = self.selected = None
    def activate(self, name):
        choices = {x["id"]: x["kind"] for x in self.training.weights()}
        if name not in choices: raise ValueError("未找到已完成的个人权重")
        folder = self.training.dataset.root / "weights" / name
        if not (folder / "evaluation.json").exists(): raise ValueError("权重缺少验证记录")
        self.close()
        if choices[name] == "mlp":
            self.predictor = MLPPredictor(folder)
            self.selected = name
            return
        base, source, python = self.training.paths()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
        key = secrets.token_urlsafe(32)
        url = "http://127.0.0.1:" + str(port)
        self.log = (self.training.dataset.root / "personal-model.log").open("w", encoding="utf-8")
        command = [str(python), "-s", str(source / "tools/agent/serve_adapter.py"), "--adapter", str(folder), "--port", str(port)]
        report = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
        bundled_base = base / "training/base" / report.get("base_model", "").split("/")[-1]
        if report.get("base_manifest_sha256") and bundled_base.is_dir():
            command += ["--base-dir", str(bundled_base)]
        self.process = subprocess.Popen(command,
            stdout=self.log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env=dict(os.environ, PYTHONNOUSERSITE="1", PYTHONUTF8="1", HF_HOME=str(self.training.dataset.root / "models"),
                     HF_HUB_DISABLE_TELEMETRY="1", IBP_ADAPTER_TOKEN=key, TOKENIZERS_PARALLELISM="false"))
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            for _ in range(240):
                if self.process.poll() is not None: raise ValueError("个人模型启动失败，请查看 personal-model.log")
                try:
                    request = urllib.request.Request(url + "/health", headers={"Authorization": "Bearer " + key})
                    with opener.open(request, timeout=2) as response: response.read()
                    self.predictor = ParameterModel({"base_url": url, "api_key": key, "model": name}, local=True)
                    self.selected = name
                    return
                except OSError: time.sleep(.5)
            raise ValueError("个人模型加载超时")
        except Exception:
            self.close()
            raise
