"""Held-out paraphrases: report failures, never silently widen acceptance."""

import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent"))
from ibp_agent.model import LocalModel
from ibp_agent.runtime import ModelProcess, locations

CASES = [
    ("整张照片曝光再加一些", {"brighten"}),
    ("希望整体看上去明亮点", {"brighten"}),
    ("整体显得刺眼，把亮部收一收", {"highlights"}),
    ("把暗处提起来一些", {"shadows"}),
    ("看起来不够通透，增强对比", {"contrast"}),
    ("反差别那么强，降低对比", {"soft"}),
    ("想要稍微暖和的氛围", {"warm"}),
    ("画面偏黄，请调冷一些", {"cool"}),
    ("让颜色更鲜艳一点", {"vivid"}),
    ("色彩浓度降一点，淡雅一些", {"muted"}),
    ("颗粒感太重，能否降噪", {"denoise"}),
    ("轮廓不够清楚，适当锐化", {"sharpen"}),
    ("自然些，别太夸张", {"natural"}),
    ("提亮一些并去噪", {"brighten", "denoise"}),
    ("更柔和，偏暖一点", {"soft", "warm"}),
    ("把曝光降低半档", {"darken"}),
    ("增加一点暗部层次", {"shadows"}),
    ("请把红色衣服变成蓝色", None),
    ("只调整天空的亮度", None),
    ("帮我变得高级一些", None),
]


def main():
    out = ROOT / ".local" / ("language-regression-" + time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir(exist_ok=True)
    paths = locations()
    paths["data"] = out
    runtime = ModelProcess(paths)
    rows = []
    try:
        model = LocalModel(runtime.start(), runtime.key)
        for text, expected in CASES:
            result = model.parse(text)

            def score(r):
                return (
                    (r["unsupported"] or r["clarification"] or not r["intents"])
                    if expected is None
                    else set(r["intents"]) == expected
                    and not r["unsupported"]
                    and not r["clarification"]
                )

            rows.append(
                {
                    "prompt": text,
                    "expected": sorted(expected)
                    if expected
                    else "clarify_or_unsupported",
                    "raw_pass": score(result["raw_model_result"]),
                    "guarded_pass": score(result),
                    "actual": result,
                }
            )
            print(rows[-1]["guarded_pass"], text, flush=True)
    finally:
        runtime.close()
        (out / "results.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
