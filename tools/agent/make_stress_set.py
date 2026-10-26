"""CC0 RAW fixtures plus reproducible post-decode stress cases (NOT synthetic RAW)."""

import hashlib, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent"))
from ibp_agent.native import Core


def main():
    out = ROOT / ".local/stress"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261026)
    manifest = []
    with Core(ROOT / ".local/app/bin") as core:
        for source in sorted((ROOT / ".local/raw").glob("*")):
            if source.suffix.lower() not in (".cr2", ".nef", ".arw"):
                continue
            base = core.load(source, 1024)
            original = out / (source.stem + "-reference.tif")
            if not original.exists():
                core.render(base, {}, original)
            manifest.append(
                {
                    "id": source.stem,
                    "path": str(source),
                    "type": "downloaded_camera_raw",
                    "prompt": "自然一点，尽量保留原来的感觉。",
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            )
            if source.stem != "canon_40d":
                continue
            cases = {}
            cases["overexposed_5ev"] = np.clip(base[:, :, :3] * 32, 0, 1)
            cases["underexposed_5ev_quantized"] = (
                np.round(np.clip(base[:, :, :3] / 32, 0, 1) * 255) / 255
            )
            cases["severe_warm_cast"] = np.clip(base[:, :, :3] * [3.5, 0.8, 0.15], 0, 1)
            cases["severe_cool_cast"] = np.clip(
                base[:, :, :3] * [0.12, 0.65, 3.5], 0, 1
            )
            cases["high_noise"] = np.clip(
                base[:, :, :3] + rng.normal(0, 0.09, base[:, :, :3].shape), 0, 1
            )
            cases["black_crush"] = np.clip((base[:, :, :3] - 0.12) * 1.8, 0, 1)
            rgb = np.clip(base[:, :, :3], 0, 1)
            encoded = np.where(
                rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055
            )
            blurred = (
                np.array(
                    Image.fromarray((encoded * 255).astype("uint8")).filter(
                        ImageFilter.GaussianBlur(6)
                    )
                ).astype("float32")
                / 255
            )
            cases["defocus_blur"] = np.where(
                blurred <= 0.04045, blurred / 12.92, ((blurred + 0.055) / 1.055) ** 2.4
            )
            cases["clipped_channel"] = np.clip(base[:, :, :3] * [8, 1, 1], 0, 1)
            prompts = {
                "overexposed_5ev": "压低刺眼的高光，尽量保留细节。",
                "underexposed_5ev_quantized": "亮一点，但噪点不要太明显。",
                "severe_warm_cast": "画面发黄，冷一点，颜色自然些。",
                "severe_cool_cast": "太冷了，暖一点。",
                "high_noise": "噪点太多，干净一点。",
                "black_crush": "暗部提亮，尽量保留细节。",
                "defocus_blur": "有点糊，清晰一点。",
                "clipped_channel": "颜色太艳，淡雅一点。",
            }
            for name, rgb in cases.items():
                image = np.concatenate(
                    [rgb, np.ones((*rgb.shape[:2], 1))], axis=-1
                ).astype("float32")
                path = out / (name + ".tif")
                if not path.exists():
                    core.render(image, {}, path)
                manifest.append(
                    {
                        "id": name,
                        "path": str(path),
                        "type": "simulated_postdecode_degradation",
                        "source": str(source),
                        "seed": 20261026,
                        "prompt": prompts[name],
                        "limitation": "不是拍摄参数真值或相机 RAW；截断/量化损失有意不可逆，不能用提亮或锐化声称恢复。",
                    }
                )
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    thumbs = []
    for row in manifest:
        if row["type"] != "simulated_postdecode_degradation":
            continue
        image = Image.open(row["path"]).convert("RGB")
        image.thumbnail((300, 200))
        thumbs.append((row["id"], image))
    from PIL import ImageDraw

    sheet = Image.new("RGB", (1200, 470), "#f4f3ed")
    draw = ImageDraw.Draw(sheet)
    for i, (name, image) in enumerate(thumbs):
        x = (i % 4) * 300
        y = (i // 4) * 235
        sheet.paste(image, (x, y))
        draw.text((x + 8, y + 205), name, fill="#20372b")
    sheet.save(out / "contact-sheet.jpg")
    print(
        json.dumps(
            {
                "fixtures": len(manifest),
                "raw": 3,
                "simulated": 8,
                "manifest": str(out / "manifest.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
