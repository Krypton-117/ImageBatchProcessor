"""Measure -> predict -> render -> constrain -> human review, with provenance."""
import copy
import importlib.metadata
import json
import time
from pathlib import Path
from .dataset import sha256
from .engine import PhotoAgent, ClarificationNeeded
from .native import Core, validate_params
from .parameter_model import DEFAULT_PARAMS, policy_input
from .policy import assess, diagnostics, targets_for, RAW_EXT
from .rag import model_context
from .settings import atomic_json


def metadata(source):
    result = {"available": False, "values": {}, "missing": []}
    keys = ("Make", "Model", "ExposureTime", "FNumber", "ISOSpeedRatings", "PhotographicSensitivity", "FocalLength", "ExposureBiasValue", "WhiteBalance", "DateTimeOriginal", "LensModel")
    try:
        from PIL import Image, ExifTags
        with Image.open(source) as image:
            exif = dict(image.getexif())
            try:
                exif.update(image.getexif().get_ifd(34665))
            except (KeyError, ValueError):
                pass
            for key, value in exif.items():
                name = ExifTags.TAGS.get(key)
                if name in keys:
                    result["values"][name] = str(value)[:150]
        result["available"] = bool(result["values"])
    except (OSError, ValueError):
        pass
    result["missing"] = [x for x in keys if x not in result["values"]]
    return result


def observe(core, source, original, out):
    detail = core.render(original, {}, out / "before.jpg", detailed=True)
    diagnosis = diagnostics(source, detail["metrics"])
    detail.update({"sensor": diagnosis["sensor"], "exif": metadata(source),
                   "preview_size": [original.shape[1], original.shape[0]],
                   "scope": "1280-edge preview; RAW half decode; linear-light block reduction"})
    if Path(source).suffix.lower() in RAW_EXT:
        try:
            import rawpy
            with rawpy.imread(str(source)) as raw:
                detail["raw"] = {"sizes": list(raw.sizes), "camera_whitebalance": list(raw.camera_whitebalance),
                    "daylight_whitebalance": list(raw.daylight_whitebalance), "color_desc": raw.color_desc.decode(errors="replace"),
                    "rgb_xyz_matrix": raw.rgb_xyz_matrix.tolist(), "raw_type": str(raw.raw_type)}
        except Exception as e:
            detail["raw"] = {"available": False, "error_type": type(e).__name__}
    return detail, diagnosis


class WorkflowAgent(PhotoAgent):
    def run(self, source, prompt, out_dir, progress=lambda x: None, predictor=None, role="student", parent=None, manual=None):
        start = time.monotonic()
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        trace = []
        def event(tool, **data):
            trace.append({"tool": tool, **data})
            progress(trace[-1])
        if parent:
            intent = copy.deepcopy(parent["intent"])
            literature, context = parent["literature"], parent["rag_context"]
        else:
            literature = self.knowledge.search_papers(prompt)
            context = model_context(literature, self.knowledge.retrieve(prompt, limit=4))
            event("retrieve_paper_passages", status="检索", citations=[h["citation"] for h in literature["hits"]])
            event("translate_intent", status="解析需求")
            intent = self.model.parse(prompt, evidence=context)
        ids = intent["intents"]
        if intent["unsupported"] or intent["clarification"] or not ids:
            raise ClarificationNeeded("需求含当前不支持或无法确定的编辑，请修改命令")
        if any(a in ids and b in ids for a, b in (("brighten", "darken"), ("warm", "cool"), ("vivid", "muted"), ("contrast", "soft"))):
            raise ClarificationNeeded("调整方向冲突")
        terms = self.knowledge.get(ids)
        event("inspect_image", status="测量原图")
        with Core(self.bin_dir) as core:
            original = core.load(source, 1280)
            observation, diagnosis = observe(core, source, original, out)
            before = observation["metrics"]
            targets = targets_for(ids, before)
            if parent:
                observation["full_resolution"] = parent["observation"].get("full_resolution")
            else:
                full = core.load(source, 0)
                observation["full_resolution"] = core.render(full, {}, detailed=True)
                observation["full_resolution"]["size"] = [full.shape[1], full.shape[0]]
                del full
            payload = policy_input(prompt, intent, observation, targets, context)
            atomic_json(out / "input.json", {"source_sha256": sha256(source), "observation": observation,
                        "policy_input": payload, "role": role, "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            event("predict_parameters", status="生成参数")
            if parent:
                prediction = copy.deepcopy(parent["prediction"])
                proposed = validate_params(manual)
            else:
                prediction = predictor.propose(payload)
                proposed = prediction["params"]
            atomic_json(out / "prediction.json", prediction)
            candidates = []
            applied, after, check = {}, before, None
            # Keep teacher's direction; select the largest safe step, never substitute another style.
            for factor in ([1.0] if parent else [1.0, 0.75, 0.5, 0.25, 0.0]):
                params = {k: round(DEFAULT_PARAMS[k] + (v - DEFAULT_PARAMS[k]) * factor, 6) for k, v in proposed.items()}
                metrics = core.render(original, params)
                check = assess(metrics, before, targets, params, intent.get("preserve", []))
                candidates.append({"index": len(candidates), "factor": factor, "params": params, "metrics": metrics, **check})
                event("evaluate_candidate", status="渲染与检查", factor=factor, accepted=check["accepted"])
                if check["accepted"]:
                    applied, after = params, metrics
                    break
                if parent:
                    raise ValueError("参数违反保护条件：" + "；".join(check["violations"]))
            if check is None or not check["accepted"]:
                raise ValueError("原图保护检查失败")
            after_detail = core.render(original, applied, out / "after.jpg", out / "preview.imgarg", detailed=True)
            results = []
            for target in targets:
                deviation = abs(after[target["metric"]] - target["target"]) / target["scale"]
                results.append({**target, "actual": after[target["metric"]], "normalized_error": deviation,
                                "status": "接近" if deviation <= .35 else "未达到"})
            report = {"version": "0.2.0", "prompt": prompt, "source_name": Path(source).name,
                "source_sha256": sha256(source), "role": role, "intent": intent,
                "terms": terms, "references": self.knowledge.references(terms), "literature": literature, "rag_context": context,
                "diagnostics": diagnosis, "observation": observation, "before": before, "after": after,
                "after_observation": after_detail, "prediction": prediction, "policy_input": payload,
                "proposed_params": proposed, "params": applied, "applied_parameters": {**DEFAULT_PARAMS, **applied},
                "manual_correction": bool(parent), "targets": results, "candidates": candidates,
                "trace": trace, "notes": diagnosis["notes"], "outcome": "adjusted" if any(abs(v - DEFAULT_PARAMS[k]) > 1e-6 for k, v in applied.items()) else "unchanged", "device": core.device,
                "quality_check": check, "metric_contract": {
                    "engine": "IBP Core 1.1.0 / 3edbc74", "domain": "decoded linear sRGB; summary statistics in encoded sRGB",
                    "white_point": "D65", "thresholds": "engineering heuristics, not universal paper thresholds",
                    "noise": "spatial proxy", "dynamicRangeEV": "percentile span, not sensor calibration",
                    "preview_reduction": observation["scope"], "histogram": "Y/R/G/B; 256 bins; native IBP sample counts",
                    "libraries": {x: importlib.metadata.version(x) for x in ("numpy", "pillow", "rawpy")}},
                "preview_size": observation["preview_size"], "seconds": round(time.monotonic() - start, 2)}
        event("complete", status="等待标注", params=applied)
        atomic_json(out / "report.json", report)
        return report
