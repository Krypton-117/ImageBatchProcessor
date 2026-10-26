"""Per-user launcher lock and authenticated reuse of a running UI service."""
import ctypes
import hashlib
import json
import os
import time
import urllib.request
import webbrowser
from .runtime import user_root

_handle = None

def acquire():
    global _handle
    if os.name != "nt":
        return True
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    name = "Local\\IBPPhotoAgent-" + hashlib.sha256(str(user_root().resolve()).lower().encode()).hexdigest()[:24]
    _handle = kernel.CreateMutexW(None, False, name)
    if not _handle:
        raise ctypes.WinError(ctypes.get_last_error())
    return ctypes.get_last_error() != 183

def reuse(no_browser=False, studio=True):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last = None
    for _ in range(40):
        try:
            data = json.loads((user_root() / "session.json").read_text(encoding="utf-8"))
            base, token = data["url"].split("/#")
            if not base.startswith("http://127.0.0.1:"):
                raise ValueError("Invalid local session")
            headers = {"X-IBP-Token": token, "Content-Type": "application/json"}
            with opener.open(urllib.request.Request(base + "/api/status", headers=headers), timeout=2) as response:
                json.load(response)
            if not no_browser:
                if studio:
                    with opener.open(urllib.request.Request(base + "/api/studio", b"{}", headers), timeout=5) as response:
                        response.read()
                else:
                    webbrowser.open(data["url"])
            return
        except (OSError, ValueError, KeyError) as e:
            last = e
            time.sleep(.25)
    raise RuntimeError("已有 IBP 服务正在启动或未响应，请稍后重试") from last
