"""Tool orchestration with an auditable, finite evaluation budget."""

import hashlib, json, time
from pathlib import Path
from .native import Core, BOUNDS
from .rag import model_context
from .policy import diagnostics, targets_for, assess, candidate_params


class ClarificationNeeded(ValueError):
    pass


class PhotoAgent:
    def __init__(self, bin_dir, knowledge, model):
        self.bin_dir = bin_dir
        self.knowledge = knowledge
        self.model = model

    def run(self, source, prompt, out_dir, progress=lambda x: None):
        start = time.monotonic()
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        trace = []

        def event(tool, **data):
            item = {"tool": tool, **data}
            trace.append(item)
            progress(item)

        literature = self.knowledge.search_papers(prompt)
        context = model_context(literature, self.knowledge.retrieve(prompt, limit=4))
        event(
            "retrieve_paper_passages",
            status="检索论文正文",
            citations=[h["citation"] for h in literature["hits"]],
            elapsed_ms=literature["elapsed_ms"],
            cache_hit=literature["cache_hit"],
        )
        event("translate_intent", status="本地模型结合检索资料理解需求")
        intent = self.model.parse(prompt, evidence=context)
        event("model_result", result=intent)
        if intent["unsupported"]:
            raise ClarificationNeeded(
                "需求含删除/生成物体、换背景或语义局部选择，当前参数式工具不支持。请改为全局光线、色彩或细节需求。"
            )
        ids = intent["intents"]
        conflicts = [
            ("brighten", "darken"),
            ("warm", "cool"),
            ("contrast", "soft"),
            ("vivid", "muted"),
        ]
        if (
            intent["clarification"]
            or not ids
            or any(a in ids and b in ids for a, b in conflicts)
        ):
            raise ClarificationNeeded(
                "需求含未可靠识别的表达、精确数值或冲突效果。本版本请使用明确方向，例如“亮一点”“压低高光”“暖一点”“降噪”；不自动猜测含义。"
            )
        terms = self.knowledge.get(ids)
        refs = self.knowledge.references(terms)
        event(
            "retrieve_knowledge",
            terms=[t["label"] for t in terms],
            sources=[r["id"] for r in refs],
        )
        with Core(self.bin_dir) as core:
            original = core.load(source, 1280)
            before = core.render(original, {}, out / "before.jpg")
            diagnosis = diagnostics(source, before)
            targets = targets_for(ids, before)
            event(
                "inspect_image",
                metrics=before,
                diagnostics=diagnosis,
                device=core.device,
            )
            best = {}
            best_metrics = before
            best_assessment = assess(
                before, before, targets, {}, intent.get("preserve", [])
            )
            candidates = [
                {"index": 0, "params": {}, "metrics": before, **best_assessment}
            ]
            seen = {json.dumps({}, sort_keys=True)}

            def evaluate(params):
                nonlocal best, best_metrics, best_assessment
                key = json.dumps(params, sort_keys=True)
                if key in seen or len(candidates) >= 25:
                    return
                seen.add(key)
                metrics = core.render(original, params)
                check = assess(
                    metrics, before, targets, params, intent.get("preserve", [])
                )
                item = {
                    "index": len(candidates),
                    "params": dict(params),
                    "metrics": metrics,
                    **check,
                }
                candidates.append(item)
                event(
                    "evaluate_candidate",
                    index=item["index"],
                    params=params,
                    loss=check["loss"],
                    accepted=check["accepted"],
                    violations=check["violations"],
                )
                if check["accepted"] and check["loss"] < best_assessment["loss"] - 1e-6:
                    best = dict(params)
                    best_metrics = metrics
                    best_assessment = check

            for p in candidate_params(terms, before):
                evaluate(p)
            # Refine one parameter at a time; always render from original, never cumulatively.
            keys = list(dict.fromkeys(k for t in terms for k in t["seed"]))
            if "natural" in ids:
                keys += ["exposure", "highlights"]
            for step_scale in (0.12, 0.06):
                previous = best_assessment["loss"]
                for key in keys:
                    base = best.get(
                        key, 1 if key in ("contrast", "saturation", "gamma") else 0
                    )
                    for direction in (-1, 1):
                        p = dict(best)
                        lo, hi = BOUNDS[key]
                        p[key] = round(
                            min(hi, max(lo, base + direction * step_scale)), 5
                        )
                        evaluate(p)
                if previous - best_assessment["loss"] < 1e-5:
                    break
            core.render(original, best, out / "after.jpg", out / "preview.imgarg")
            results = []
            for t in targets:
                deviation = abs(best_metrics[t["metric"]] - t["target"]) / t["scale"]
                results.append(
                    {
                        **t,
                        "actual": best_metrics[t["metric"]],
                        "normalized_error": deviation,
                        "status": "近似达到工程目标"
                        if deviation <= 0.35
                        else "在本次受限搜索内未达到；不是数学不可达证明",
                    }
                )
            outcome = "adjusted" if best else "original_retained"
            notes = list(diagnosis["notes"])
            if not best:
                notes.append(
                    "候选没有在约束内优于原图，已保留原图；不会为了达标强行修改。"
                )
            notes.append(
                "预览指标不等于全尺寸结果；导出时会再次检查约束。未使用多模态模型，不评判人物、天空或主观审美。"
            )
            report = {
                "version": "0.1.0",
                "prompt": prompt,
                "source_name": Path(source).name,
                "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                "intent": intent,
                "terms": terms,
                "references": refs,
                "literature": literature,
                "rag_context": context,
                "diagnostics": diagnosis,
                "before": before,
                "after": best_metrics,
                "params": best,
                "targets": results,
                "candidates": candidates,
                "trace": trace,
                "notes": notes,
                "outcome": outcome,
                "device": core.device,
                "metric_contract": {
                    "domain": "IBP v1.1 decoded linear sRGB; most summary statistics measured after sRGB encoding",
                    "white_point": "D65",
                    "range": "Core-specific; see FORMAT.md",
                    "noise": "spatial proxy, not calibrated ISO noise",
                    "dynamicRangeEV": "percentile span, not sensor dynamic range",
                    "colorfulness": "Hasler formula on 0..1 encoded RGB; multiply by 255 to compare 0..255 implementations",
                    "preview_reduction": "linear-light block mean after half-size RAW decoding",
                    "thresholds": "engineering heuristics",
                },
                "preview_size": [original.shape[1], original.shape[0]],
                "seconds": round(time.monotonic() - start, 2),
                "evaluation_budget": 25,
            }
        event("complete", outcome=outcome, params=best)
        (out / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report

    def export(self, source, report, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if (
            hashlib.sha256(Path(source).read_bytes()).hexdigest()
            != report["source_sha256"]
        ):
            raise ValueError("原图已改变，请重新分析")
        with Core(self.bin_dir) as core:
            full = core.load(source, 0)
            before = core.render(full, {})
            after = core.render(full, report["params"])
            check = assess(
                after,
                before,
                targets_for(report["intent"]["intents"], before),
                report["params"],
                report["intent"].get("preserve", []),
            )
            if not check["accepted"]:
                return {
                    "exported": False,
                    "reason": "全尺寸重新检查未通过，未导出；请降低调整强度。",
                    "assessment": check,
                    "before": before,
                    "after": after,
                }
            name = "edited-" + str(time.time_ns())
            core.render(
                full, report["params"], out / (name + ".tif"), out / (name + ".imgarg")
            )
            result = {
                "exported": True,
                "image": name + ".tif",
                "recipe": name + ".imgarg",
                "before": before,
                "after": after,
                "assessment": check,
            }
            (out / (name + ".json")).write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return result
