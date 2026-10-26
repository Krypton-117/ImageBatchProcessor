"""Portable, human-labelled records. Rejected attempts never enter SFT targets."""
import ctypes
import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path
from .parameter_model import DEFAULT_PARAMS, PARAM_VERSION
from .settings import atomic_json


def training_root():
    override = os.environ.get("IBP_TRAINING_DATA")
    if override:
        return Path(override)
    documents = Path.home() / "Documents"
    if os.name == "nt":
        buf = ctypes.create_unicode_buffer(32768)
        if ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf) == 0:
            documents = Path(buf.value)
    return documents / "IBP Training"


def sha256(file):
    with Path(file).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Dataset:
    def __init__(self, root=None):
        self.root = Path(root) if root else training_root()
        self.lock = threading.RLock()
        for folder in ("samples", "originals", "exports", "weights"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)

    def records(self):
        records = []
        for file in sorted((self.root / "samples").glob("*/sample.json")):
            records.append(json.loads(file.read_text(encoding="utf-8")))
        return records

    def status(self):
        with self.lock:
            rows = self.records()
            approved = [x for x in rows if x["feedback"]["accepted"]]
            return {"path": str(self.root), "total": len(rows), "accepted": len(approved),
                    "rejected": len(rows) - len(approved), "groups": len({x["source"]["group_id"] for x in approved}),
                    "weights": sorted(str(p.relative_to(self.root)) for p in (self.root / "weights").glob("*/*") if p.name in ("weights.npz", "adapter_model.safetensors", "evaluation.json"))}

    def label(self, ident, report, source, folder, accepted, group_id=""):
        if type(accepted) is not bool:
            raise ValueError("请选择满意或不满意")
        if not isinstance(group_id, str) or len(group_id) > 150:
            raise ValueError("拍摄组标识无效")
        if len(ident) != 32 or any(x not in "0123456789abcdef" for x in ident):
            raise ValueError("样本标识无效")
        digest = sha256(source)
        if digest != report["source_sha256"]:
            raise ValueError("原图已发生变化")
        with self.lock:
            original = self.root / "originals" / (digest + Path(source).suffix.lower())
            if not original.exists():
                tmp = original.with_suffix(original.suffix + ".tmp")
                shutil.copy2(source, tmp)
                if sha256(tmp) != digest:
                    tmp.unlink()
                    raise ValueError("原图复制校验失败")
                os.replace(tmp, original)
            sample_dir = self.root / "samples" / ident
            sample_dir.mkdir(exist_ok=True)
            revision = 1
            existing = sample_dir / "sample.json"
            if existing.exists():
                previous = json.loads(existing.read_text(encoding="utf-8"))
                revision = previous["feedback"]["revision"] + 1
                history = sample_dir / "history"
                history.mkdir(exist_ok=True)
                shutil.copy2(existing, history / f"sample-{revision-1}.json")
            rendered = {}
            for name in ("before.jpg", "after.jpg", "preview.imgarg"):
                file = Path(folder) / name
                if file.exists():
                    shutil.copy2(file, sample_dir / name)
                    rendered[name] = {"path": str((sample_dir / name).relative_to(self.root)), "sha256": sha256(file)}
            sample = {
                "schema": "ibp-personal-sample-v1", "id": ident,
                "source": {"path": str(original.relative_to(self.root)), "sha256": digest,
                           "name": report.get("source_name"), "group_id": group_id.strip() or digest},
                "feedback": {"accepted": accepted, "label": "satisfied" if accepted else "unsatisfied",
                             "rated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "revision": revision,
                             "scope": "preview", "reviewed_output_sha256": rendered["after.jpg"]["sha256"]},
                "artifacts": rendered, "report": report,
                "applied_parameters": {**DEFAULT_PARAMS, **report["params"]},
                "contract": {"parameter_policy": PARAM_VERSION, "semantics": "absolute from original; native operation order",
                             "engine": "IBP Core 1.1.0 / 3edbc74", "image_domain": report["metric_contract"],
                             "training_input": "before only; exclude after metrics, feedback and output image"},
            }
            atomic_json(existing, sample)
            self.export()
            return {"saved": True, "accepted": accepted, "sample": str(existing), "revision": revision, **self.status()}

    def export(self):
        with self.lock:
            records = self.records()
            accepted = [x for x in records if x["feedback"]["accepted"]]
            sft = []
            for row in accepted:
                prediction = row["report"].get("prediction")
                if not prediction or not prediction.get("messages"):
                    continue
                # Supervise actual accepted parameters, including manual corrections or safety projection.
                sft.append({"id": row["id"], "group_id": row["source"]["group_id"], "source_sha256": row["source"]["sha256"],
                    "messages": prediction["messages"] + [{"role": "assistant", "content": json.dumps({"params": row["applied_parameters"]}, separators=(",", ":"))}]})
            for name, rows in (("all-records.jsonl", records), ("accepted.jsonl", accepted), ("qwen-sft.jsonl", sft)):
                f = self.root / "exports" / name
                temp = f.with_suffix(".tmp")
                with temp.open("w", encoding="utf-8") as stream:
                    for row in rows:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                os.replace(temp, f)
            return {"records": len(records), "accepted": len(accepted), "sft": len(sft), "directory": str(self.root / "exports")}
