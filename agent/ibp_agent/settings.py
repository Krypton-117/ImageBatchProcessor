"""Versioned model profiles; Windows DPAPI protects API keys at rest."""
import base64
import ctypes as C
import json
import os
import threading
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlparse

DEFAULTS = {
    "version": 1,
    "teacher": {"base_url": "https://api.deepseek.com", "model": "deepseek-chat", "api_key": ""},
    "student": {"base_url": "", "model": "Qwen3-1.7B", "api_key": ""},
    "training_model": "Qwen/Qwen3-0.6B",
}


def endpoint(value):
    if not isinstance(value, str) or len(value) > 500:
        raise ValueError("API URL 无效")
    if not value:
        return ""
    u = urlparse(value)
    if (u.scheme not in ("http", "https") or not u.hostname or u.username or u.password
            or u.query or u.fragment or (u.scheme == "http" and u.hostname not in ("127.0.0.1", "localhost", "::1"))):
        raise ValueError("远程 API 需 HTTPS；本机 API 可使用 http://127.0.0.1")
    return value.rstrip("/")


def protect(value, decrypt=False):
    if not value:
        return ""
    if os.name != "nt":
        raise RuntimeError("API 密钥存储需要 Windows DPAPI")
    class Blob(C.Structure):
        _fields_ = [("size", C.c_ulong), ("data", C.POINTER(C.c_ubyte))]
    raw = base64.b64decode(value) if decrypt else value.encode("utf-8")
    buf = (C.c_ubyte * len(raw)).from_buffer_copy(raw)
    src, dst = Blob(len(raw), buf), Blob()
    f = C.windll.crypt32.CryptUnprotectData if decrypt else C.windll.crypt32.CryptProtectData
    if not f(C.byref(src), None, None, None, None, 1, C.byref(dst)):
        raise RuntimeError("Windows 密钥保护失败")
    try:
        result = C.string_at(dst.data, dst.size)
        return result.decode("utf-8") if decrypt else base64.b64encode(result).decode("ascii")
    finally:
        C.windll.kernel32.LocalFree(C.cast(dst.data, C.c_void_p))


def atomic_json(file, data):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temp = file.with_suffix(file.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, file)


class Settings:
    def __init__(self, root):
        self.file = Path(root) / "settings.json"
        self.lock = threading.RLock()
        self.data = deepcopy(DEFAULTS)
        if self.file.exists():
            stored = json.loads(self.file.read_text(encoding="utf-8"))
            for role in ("teacher", "student"):
                self.data[role].update(stored.get(role, {}))
                self.data[role]["api_key"] = protect(self.data[role].pop("key_dpapi", ""), True)
            self.data["training_model"] = stored.get("training_model", DEFAULTS["training_model"])

    def public(self):
        with self.lock:
            d = deepcopy(self.data)
        for role in ("teacher", "student"):
            d[role]["has_key"] = bool(d[role].pop("api_key"))
        return d

    def profile(self, role):
        if role not in ("teacher", "student"):
            raise ValueError("未知模型角色")
        with self.lock:
            return deepcopy(self.data[role])

    def update(self, data):
        with self.lock:
            updated = deepcopy(self.data)
            for role in ("teacher", "student"):
                if role not in data:
                    continue
                p = data[role]
                updated[role]["base_url"] = endpoint(p.get("base_url", updated[role]["base_url"]))
                model = p.get("model", updated[role]["model"])
                if not isinstance(model, str) or not model.strip() or len(model) > 150:
                    raise ValueError("模型名称无效")
                updated[role]["model"] = model.strip()
                key = p.get("api_key", "")
                if not isinstance(key, str) or len(key) > 4096 or any(c in key for c in "\r\n"):
                    raise ValueError("API key 无效")
                if p.get("clear_key"):
                    updated[role]["api_key"] = ""
                elif key:
                    updated[role]["api_key"] = key.strip()
            name = data.get("training_model", updated["training_model"])
            if name not in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"):
                raise ValueError("不支持的训练基座")
            updated["training_model"] = name
            stored = deepcopy(updated)
            for role in ("teacher", "student"):
                stored[role]["key_dpapi"] = protect(stored[role].pop("api_key"))
            atomic_json(self.file, stored)
            self.data = updated
        return self.public()
