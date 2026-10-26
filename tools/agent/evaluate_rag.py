"""Small declared retrieval regression, latency and real text-model evidence test."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent"))
from ibp_agent.knowledge import Knowledge
from ibp_agent.rag import model_context
from ibp_agent.runtime import locations, ModelProcess
from ibp_agent.model import LocalModel

CASES = [
    ("颗粒太多，干净一点", "foi2008"),
    ("对比度 RMS", "peli1990"),
    ("颜色淡，鲜艳一点", "hasler2003"),
    ("语言引导图像编辑规划", "planning2021"),
    ("GIER language editing benchmark", "gier2021"),
    ("recovering radiance response exposure HDR", "debevec1997"),
    ("曝光和色调映射", "reinhard2002"),
    ("structural similarity SSIM luminance contrast", "wang2004"),
    ("deep features perceptual metric LPIPS", "zhang2018"),
    ("用户偏好与可接受的调整范围", "acceptable2015"),
    ("FiveK database photographers tonal adjustment", "fivek2011"),
    ("Exposure white box post processing framework", "exposure2018"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-model", action="store_true")
    args = parser.parse_args()
    paths = locations()
    k = Knowledge(paths["knowledge"], paths["data"] / "knowledge.sqlite")
    out = ROOT / ".local" / ("rag-evaluation-" + time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir()
    rows, uncached, cached = [], [], []
    for query, expected in CASES:
        result = k.search_papers(query)
        ids = [h["source_id"] for h in result["hits"]]
        rows.append(
            {
                "query": query,
                "expected": expected,
                "hit_at_5": expected in ids,
                "result": result,
            }
        )
        uncached.append(result["elapsed_ms"])
        for _ in range(5):
            cached.append(k.search_papers(query)["elapsed_ms"])
    result = {
        "index": k.papers.status(),
        "cases": rows,
        "hits_at_5": sum(r["hit_at_5"] for r in rows),
        "case_count": len(rows),
        "test_kind": "declared small regression; not independent relevance benchmark",
        "uncached_query_median_ms": statistics.median(uncached),
        "uncached_query_max_ms": max(uncached),
        "cached_query_median_ms": statistics.median(cached),
        "cached_query_max_ms": max(cached),
    }
    if args.real_model:
        runtime = ModelProcess(paths)
        try:
            runtime.start()
            model = LocalModel(runtime.endpoint, runtime.key)
            result["real_model"] = []
            for prompt, expected in [
                ("亮一点", "brighten"),
                ("降噪", "denoise"),
                ("通透一点，但不要更艳", "contrast"),
            ]:
                evidence = k.search_papers(prompt)
                parsed = model.parse(
                    prompt,
                    evidence=model_context(evidence, k.retrieve(prompt, limit=4)),
                )
                result["real_model"].append(
                    {
                        "prompt": prompt,
                        "result": parsed,
                        "pass": expected in parsed["intents"]
                        and bool(parsed["rag_context_ids"])
                        and not parsed["clarification"],
                    }
                )
        finally:
            runtime.close()
    (out / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(out)
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "hits_at_5",
                    "case_count",
                    "uncached_query_median_ms",
                    "cached_query_median_ms",
                ]
            }
        )
    )
    if args.real_model:
        print("real_model_pass", sum(r["pass"] for r in result["real_model"]))


if __name__ == "__main__":
    main()
