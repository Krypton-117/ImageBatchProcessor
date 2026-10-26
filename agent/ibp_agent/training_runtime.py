"""Owned training process and portable Python setup, outside the app runtime."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .runtime import install_root
from .training import train_mlp, split_rows


class TrainingRuntime:
    def __init__(self, dataset):
        self.dataset = dataset
        self.state = {"status": "idle"}
        self.process = None

    def paths(self):
        base = install_root()
        source = base / "sources" if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2]
        python = base / "training/python/python.exe"
        return base, source, python

    def status(self):
        data = dict(self.state)
        data["python_ready"] = self.paths()[2].exists()
        data["weights"] = self.weights()
        log = self.dataset.root / "training.log"
        if log.exists():
            with log.open("rb") as f:
                f.seek(max(0, f.seek(0, 2) - 5000))
                data["log_tail"] = f.read().decode("utf-8", errors="replace")
        return data

    def command(self, args, log):
        self.process = subprocess.Popen([str(x) for x in args], stdout=log, stderr=subprocess.STDOUT,
            cwd=self.dataset.root, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env=dict(os.environ, HF_HOME=str(self.dataset.root / "models"), HF_HUB_DISABLE_TELEMETRY="1", PYTHONUTF8="1", PYTHONNOUSERSITE="1", TOKENIZERS_PARALLELISM="false"))
        code = self.process.wait()
        self.process = None
        if code:
            raise RuntimeError(f"训练进程退出码 {code}；查看 training.log")

    def run(self, kind, model="Qwen/Qwen3-0.6B"):
        self.state = {"status": "running", "kind": kind, "started": time.time()}
        try:
            self.dataset.export()
            if kind == "mlp":
                result = train_mlp(self.dataset, lambda p: self.state.update(progress=p))
            elif kind in ("prepare", "qwen"):
                base, source, python = self.paths()
                with (self.dataset.root / "training.log").open("w", encoding="utf-8") as log:
                    if kind == "prepare":
                        bundled = base / "training/python"
                        if not bundled.exists():
                            raise ValueError("未找到安装包内的训练 Python")
                        self.command([python, "-s", "-c", "import torch,transformers,peft;print(torch.__version__,transformers.__version__,peft.__version__)"], log)
                        result = {"path": str(python.parent)}
                    else:
                        if not python.exists():
                            raise ValueError("请先准备 Qwen 训练环境")
                        rows = [json.loads(x) for x in (self.dataset.root / "exports/qwen-sft.jsonl").read_text(encoding="utf-8").splitlines() if x]
                        split_rows(rows)
                        output = self.dataset.root / "weights" / ("qwen-" + time.strftime("%Y%m%d-%H%M%S"))
                        command = [python, "-s", source / "tools/agent/train_qwen.py", "--data", self.dataset.root / "exports/qwen-sft.jsonl", "--output", output, "--model", model]
                        bundled_base = base / "training/base" / model.split("/")[-1]
                        if (bundled_base / "model.safetensors").exists():
                            command += ["--local-model", bundled_base]
                        self.command(command, log)
                        if not (output / "adapter_model.safetensors").exists():
                            raise ValueError("训练未生成适配器")
                        result = {"path": str(output)}
            else:
                raise ValueError("未知训练任务")
            self.state.update(status="done", result=result, seconds=round(time.time() - self.state["started"], 2))
        except Exception as e:
            self.state.update(status="error", error=str(e))
        return self.state

    def weights(self):
        return [{"id": p.parent.name, "kind": "qwen" if p.name == "adapter_model.safetensors" else "mlp"}
                for pattern in ("*/adapter_model.safetensors", "*/weights.npz")
                for p in sorted((self.dataset.root / "weights").glob(pattern))]

    def cancel(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            return {"cancelled": True}
        raise ValueError("当前没有可停止的训练进程")
