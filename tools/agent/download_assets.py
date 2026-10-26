"""Download explicitly recorded upstream assets, verifying hashes when available."""

import concurrent.futures, hashlib, json, os
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / ".local"


def download(url, path, expected=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if (
        path.exists()
        and expected
        and hashlib.sha256(path.read_bytes()).hexdigest() == expected
    ):
        return
    temp = path.with_suffix(path.suffix + ".part")
    for attempt in range(5):
        try:
            offset = temp.stat().st_size if temp.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            # Request-local proxy only; no machine network settings are changed.
            proxy_url = os.environ.get("IBP_DOWNLOAD_PROXY")
            proxy = {"https": proxy_url} if proxy_url and attempt % 2 else None
            with requests.get(
                url, headers=headers, proxies=proxy, stream=True, timeout=(20, 90)
            ) as r:
                r.raise_for_status()
                append = offset and r.status_code == 206
                if append and not r.headers.get("Content-Range", "").startswith(
                    f"bytes {offset}-"
                ):
                    raise ValueError("Unexpected Content-Range")
                with temp.open("ab" if append else "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
            with temp.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if expected and digest != expected:
                raise ValueError(f"hash mismatch: {path.name}")
            temp.replace(path)
            print(f"OK {path.name}: {digest}", flush=True)
            return
        except Exception as e:
            print(f"Retry {path.name}: {type(e).__name__}: {str(e)[:140]}", flush=True)
    raise RuntimeError(f"Download failed: {url}")


def main():
    spec = json.loads(
        (Path(__file__).with_name("assets.json")).read_text(encoding="utf-8")
    )
    model = spec["model"]
    jobs = [
        (
            f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{model['file']}",
            LOCAL / "models" / model["file"],
            model["sha256"],
        ),
        (
            "https://github.com/ggml-org/llama.cpp/releases/download/b11889/llama-b11889-bin-win-cpu-x64.zip",
            LOCAL / "llama.zip",
            spec["runtime"]["sha256"],
        ),
    ]
    for name in ("LICENSE", "README.md"):
        jobs.append(
            (
                f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{name}",
                LOCAL / "models" / name,
                None,
            )
        )
    for raw in spec["raw"]:
        jobs.append((raw["url"], LOCAL / "raw" / raw["file"], raw["sha256"]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, *j) for j in jobs]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    (LOCAL / "raw/sources.json").write_text(
        json.dumps(spec["raw"], indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
