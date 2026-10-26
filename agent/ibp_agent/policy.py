"""Observed limitations, relative targets, and non-destructive candidate selection."""

from pathlib import Path
import numpy as np
from .native import BOUNDS

RAW_EXT = {
    ".cr2",
    ".cr3",
    ".nef",
    ".arw",
    ".dng",
    ".raf",
    ".rw2",
    ".orf",
    ".pef",
    ".srw",
}


def diagnostics(path, metrics):
    notes = []
    sensor = None
    if metrics["whiteClip"] > 0.01:
        notes.append(
            "解码后存在较多近白/通道剪切像素。压低高光可改善亮度，但不能证明纹理恢复。"
        )
    if metrics["blackClip"] > 0.05:
        notes.append(
            "暗像素比例较高。可能是夜景或欠曝，不能仅凭直方图判定拍摄错误；提亮可能放大噪声。"
        )
    if metrics["noiseEstimate"] > 0.012:
        notes.append("空间噪声代理偏高。优先限制提亮幅度，并将降噪与细节保留一起评估。")
    if metrics["edgeEnergy"] < 0.008:
        notes.append(
            "边缘能量较低：可能是失焦、运动模糊或本来平滑的场景，单图无法可靠区分；锐化不能恢复丢失细节。"
        )
    if Path(path).suffix.lower() in RAW_EXT:
        try:
            import rawpy

            with rawpy.imread(str(path)) as raw:
                mosaic = raw.raw_image_visible
                colors = raw.raw_colors_visible
                levels = raw.camera_white_level_per_channel or [raw.white_level] * 4
                sat = np.zeros(mosaic.shape, dtype=bool)
                for c in range(4):
                    sat |= (colors == c) & (mosaic >= levels[c])
                sensor = {
                    "saturated_fraction": float(np.mean(sat)),
                    "white_level": int(raw.white_level),
                    "black_levels": list(raw.black_level_per_channel),
                    "basis": "原始马赛克在 LibRaw 报告白电平处的饱和比例；不是场景语义或恢复保证",
                }
                if sensor["saturated_fraction"] > 0.001:
                    notes.append(
                        "RAW 马赛克含传感器白电平饱和采样；这部分真实纹理不能靠全局曝光调节重建。"
                    )
        except Exception as e:
            notes.append("RAW 马赛克诊断不可用：" + str(e)[:180])
    else:
        notes.append(
            "当前为已渲染图像：无法反推真实 ISO、快门、传感器动态范围或 RAW 高光余量。"
        )
    return {
        "notes": notes,
        "sensor": sensor,
        "certainty": "局部统计证据；不能保证无失真或证明语义效果。所有警戒线为工程启发式。",
    }


def targets_for(ids, b):
    targets = []

    def add(metric, target, scale, term):
        targets.append(
            {
                "metric": metric,
                "baseline": b[metric],
                "target": target,
                "scale": scale,
                "intent": term,
                "basis": "相对原图的工程启发式目标，非论文审美阈值",
            }
        )

    for i in ids:
        if i == "brighten":
            add("median", min(1.0, b["median"] + 0.10), 0.10, i)
        elif i == "darken":
            add("median", max(0.0, b["median"] - 0.10), 0.10, i)
        elif i == "highlights":
            add("p95", max(0.0, b["p95"] - 0.08), 0.08, i)
        elif i == "shadows":
            add("p05", min(1.0, b["p05"] + 0.06), 0.06, i)
        elif i == "contrast":
            add(
                "rmsContrast",
                b["rmsContrast"] * 1.15,
                max(0.025, b["rmsContrast"] * 0.15),
                i,
            )
        elif i == "soft":
            add(
                "rmsContrast",
                b["rmsContrast"] * 0.85,
                max(0.025, b["rmsContrast"] * 0.15),
                i,
            )
        elif i in ("warm", "cool"):
            add("labB", b["labB"] + (4 if i == "warm" else -4), 4, i)
        elif i in ("vivid", "muted"):
            add(
                "colorfulness",
                b["colorfulness"] * (1.15 if i == "vivid" else 0.80),
                max(0.025, b["colorfulness"] * 0.2),
                i,
            )
        elif i == "denoise":
            add(
                "noiseEstimate",
                b["noiseEstimate"] * 0.8,
                max(0.0005, b["noiseEstimate"] * 0.2),
                i,
            )
        elif i == "sharpen":
            add(
                "edgeEnergy",
                b["edgeEnergy"] * 1.1,
                max(0.002, b["edgeEnergy"] * 0.1),
                i,
            )
        elif i == "natural":
            if b["median"] < 0.30:
                add("median", min(0.38, b["median"] + 0.08), 0.08, i)
            elif b["median"] > 0.65:
                add("median", max(0.6, b["median"] - 0.05), 0.05, i)
    return targets


def assess(metrics, baseline, targets, params, preserve=()):
    violations = []
    if "color" in preserve and metrics["colorfulness"] > baseline["colorfulness"] + max(
        0.015, baseline["colorfulness"] * 0.08
    ):
        violations.append("超过用户保留色彩鲜明度约束")
    if "brightness" in preserve and abs(metrics["median"] - baseline["median"]) > 0.03:
        violations.append("超过用户保留亮度约束")
    if "warmth" in preserve and abs(metrics["labB"] - baseline["labB"]) > 2:
        violations.append("超过用户保留暖色代理约束")
    if "highlights" in preserve and metrics["p95"] > baseline["p95"] + 0.015:
        violations.append("超过用户保留高光约束")
    if "details" in preserve and metrics["edgeEnergy"] < baseline["edgeEnergy"] * 0.85:
        violations.append("超过用户保留细节代理约束")
    # Hard gates compare to the immutable original. They do not certify visual fidelity.
    if metrics["whiteClip"] > baseline["whiteClip"] + 0.005:
        violations.append("新增高光剪切超过 0.5 个百分点")
    if metrics["blackClip"] > baseline["blackClip"] + 0.01:
        violations.append("新增暗部近黑超过 1 个百分点")
    if metrics["noiseEstimate"] > max(
        baseline["noiseEstimate"] * 1.8, baseline["noiseEstimate"] + 0.004
    ):
        violations.append("空间噪声代理增长超限")
    if metrics["edgeEnergy"] < baseline["edgeEnergy"] * 0.65:
        violations.append("边缘能量保留不足 65%（启发式）")
    loss = sum(
        ((metrics[t["metric"]] - t["target"]) / t["scale"]) ** 2 for t in targets
    ) / max(1, len(targets))
    # Favor a smaller change only when measured target distances are close.
    regularizer = (
        sum(
            (v - (1 if k in ("contrast", "saturation", "gamma") else 0)) ** 2
            for k, v in params.items()
        )
        * 0.015
    )
    return {
        "loss": float(loss + regularizer),
        "target_loss": float(loss),
        "accepted": not violations,
        "violations": violations,
    }


def candidate_params(terms, baseline):
    seed = {}
    for t in terms:
        seed.update(t["seed"])
    if any(t["id"] == "natural" for t in terms):
        if baseline["median"] < 0.30:
            seed.setdefault("exposure", 0.25)
        if baseline["whiteClip"] > 0.01:
            seed.setdefault("highlights", -0.25)
    if baseline["noiseEstimate"] > 0.012 and "exposure" in seed:
        seed["exposure"] = min(seed["exposure"], 0.3)
    for factor in (0.25, 0.5, 0.75, 1, 1.35, 1.7):
        p = {}
        for key, value in seed.items():
            default = 1 if key in ("contrast", "saturation", "gamma") else 0
            p[key] = round(
                min(
                    BOUNDS[key][1],
                    max(BOUNDS[key][0], default + (value - default) * factor),
                ),
                5,
            )
        if p:
            yield p
