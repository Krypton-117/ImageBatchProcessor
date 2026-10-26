"""Download the official Qwen training base at a recorded ModelScope revision."""
from pathlib import Path
import concurrent.futures, hashlib, json
import requests
from download_assets import download
ROOT = Path(__file__).resolve().parents[2]
REPO = "Qwen/Qwen3-0.6B"
REVISION = "09b42cad3d112e832108974449ccb5e8e0f5b5d1"
FILES = {"LICENSE", "README.md", "config.json", "generation_config.json", "merges.txt", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json"}
def range_download(url, target, expected, total):
    if target.exists():
        with target.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() == expected: return
    prefix = target.with_suffix(target.suffix + ".part")
    start = prefix.stat().st_size if prefix.exists() else 0
    partsdir = target.parent / (target.name + ".ranges")
    partsdir.mkdir(exist_ok=True)
    parts = [(a, min(total - 1, a + 8 * 1024 * 1024 - 1)) for a in range(start, total, 8 * 1024 * 1024)]
    def fetch_part(pair):
        a, b = pair
        part = partsdir / str(a)
        for attempt in range(6):
            offset = part.stat().st_size if part.exists() else 0
            if offset == b - a + 1: return
            try:
                with requests.get(url, headers={"Range": f"bytes={a+offset}-{b}"}, stream=True, timeout=(20, 50)) as response:
                    if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {a+offset}-"):
                        raise ValueError("Range not honored")
                    with part.open("ab") as stream:
                        for block in response.iter_content(262144): stream.write(block)
                if part.stat().st_size != b - a + 1: raise ValueError("Incomplete range")
                return
            except (OSError, ValueError, requests.RequestException) as e:
                print("Range retry", a, type(e).__name__, flush=True)
        raise RuntimeError(f"Failed range {a}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for i, _ in enumerate(pool.map(fetch_part, parts)):
            if i % 20 == 0: print("Ranges", i, "/", len(parts), flush=True)
    with target.open("wb") as stream:
        for part in ([prefix] if prefix.exists() else []) + [partsdir / str(a) for a, b in parts]:
            with part.open("rb") as source:
                while block := source.read(4194304): stream.write(block)
    with target.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError("Model hash mismatch")

def main():
    dest = ROOT / ".local/training/base/Qwen3-0.6B"
    response = requests.get(f"https://modelscope.cn/api/v1/models/{REPO}/repo/files", params={"Revision": REVISION, "Recursive": "true"}, timeout=30)
    response.raise_for_status()
    files = [f for f in response.json()["Data"]["Files"] if f["Path"] in FILES]
    if {f["Path"] for f in files} != FILES: raise ValueError("Incomplete official manifest")
    def fetch(f):
        url = f"https://modelscope.cn/models/{REPO}/resolve/{REVISION}/{f['Path']}"
        if f["Size"] > 100_000_000:
            range_download(url, dest / f["Path"], f["Sha256"], f["Size"])
        else:
            download(url, dest / f["Path"], f["Sha256"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, files))
    manifest = {"repo": REPO, "source": "official Qwen ModelScope repository", "revision": REVISION,
                "files": {f["Path"]: {"sha256": f["Sha256"], "size": f["Size"]} for f in files}}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Verified trainable Qwen3-0.6B base", flush=True)
if __name__ == "__main__": main()
