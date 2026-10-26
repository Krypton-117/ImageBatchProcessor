"""Verify local PDF identities and incrementally index physical pages for RAG."""

import argparse
import json
import logging
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent"))
from ibp_agent.rag import PaperIndex, VERSION, digest
from ibp_agent.runtime import user_root
from pypdf import PdfReader


def words(value):
    value = unicodedata.normalize("NFKD", value).replace("ﬁ", "fi").replace("ﬀ", "ff")
    return re.findall("[a-z0-9]+", value.lower())


def identify(first_pages, sha256, sources):
    content = set(words(first_pages[:2500]))
    normalized = " ".join(words(first_pages[:3000]))
    ranked = []
    for source in sources:
        if source.get("type") != "research_paper":
            continue
        tokens = set(words(source["title"])) - {
            "a",
            "an",
            "the",
            "of",
            "and",
            "for",
            "with",
            "in",
            "to",
        }
        score = len(tokens & content) / max(1, len(tokens))
        # Previously verified content identity can disambiguate a title.
        exact_hash = source.get("sha256") == sha256
        phrase = " ".join(words(source["title"])) in normalized
        ranked.append((3.0 if exact_hash else 2.0 if phrase else score, source))
    ranked.sort(key=lambda x: x[0], reverse=True)
    if not ranked:
        return None, {"reason": "no_research_sources"}
    best = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    if (
        best[0] < 0.75
        or best[0] == second_score
        or (best[0] < 2 and best[0] - second_score < 0.12)
    ):
        return None, {
            "best_title": best[1]["title"],
            "score": best[0],
            "second_score": second_score,
        }
    return best[1], {
        "method": "verified_sha256"
        if best[0] == 3
        else "title_phrase_first_pages"
        if best[0] == 2
        else "title_token_overlap_first_two_pages",
        "score": min(best[0], 1.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", type=Path, default=ROOT / "docs/research/papers")
    parser.add_argument("--db", type=Path, default=user_root() / "knowledge.sqlite")
    args = parser.parse_args()
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    sources_path = ROOT / "docs/research/sources.json"
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    index = PaperIndex(args.db)
    existing = {d["id"]: d for d in index.documents()}
    report = {
        "db": str(args.db.resolve()),
        "matched": [],
        "unmatched": [],
        "errors": [],
    }
    seen = set()
    started = time.perf_counter()
    for path in sorted(args.papers.glob("*.pdf")):
        try:
            sha = digest(path)
            reader = PdfReader(path)
            first = [p.extract_text() or "" for p in reader.pages[:2]]
            source, evidence = identify(" ".join(first)[:12000], sha, sources)
            if source is None:
                report["unmatched"].append({"file": path.name, **evidence})
                continue
            if source["id"] in seen:
                report["unmatched"].append(
                    {
                        "file": path.name,
                        "reason": "duplicate_source",
                        "id": source["id"],
                    }
                )
                continue
            previous = existing.get(source["id"], {})
            unchanged = (
                previous.get("sha256") == sha
                and previous.get("version") == VERSION
                and previous.get("path") == str(path.resolve())
            )
            if unchanged:
                chunks = previous["chunks"]
                empty_pages = []
            else:
                pages = first + [p.extract_text() or "" for p in reader.pages[2:]]
                empty_pages = [
                    i + 1 for i, t in enumerate(pages) if len(t.strip()) < 60
                ]
                chunks = index.add(source, path, pages, sha)
            seen.add(source["id"])
            source.update(
                local_path=str(path.resolve().relative_to(ROOT))
                if path.resolve().is_relative_to(ROOT)
                else str(path.resolve()),
                sha256=sha,
                download_status="available_local",
                access="Local PDF identity verified and page text indexed; local research use only",
                pdf_pages=len(reader.pages),
                bytes=path.stat().st_size,
                checked_at=time.strftime("%Y-%m-%d"),
                missing_reason=None,
                download_error=None,
                rag_index={"version": VERSION, "chunks": chunks, "identity": evidence},
            )
            report["matched"].append(
                {
                    "file": path.name,
                    "id": source["id"],
                    "sha256": sha,
                    "pages": len(reader.pages),
                    "chunks": chunks,
                    "cached": unchanged,
                    "short_text_pages": empty_pages,
                    **evidence,
                }
            )
        except Exception as e:
            report["errors"].append({"file": path.name, "error": str(e)})
    missing = [
        s for s in sources if s.get("type") == "research_paper" and s["id"] not in seen
    ]
    report["missing"] = [s["id"] for s in missing]
    report["seconds"] = round(time.perf_counter() - started, 3)
    report["index"] = index.status()
    for target in (sources_path, ROOT / "agent/knowledge/sources.json"):
        target.write_text(
            json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    out = ROOT / ".local/rag-ingestion.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 本地文献核验与补充位置",
        "",
        "核验日期："
        + time.strftime("%Y-%m-%d")
        + "。按 PDF 题名或已验证 SHA-256 关联；保留用户文件名。",
        "",
        f"已关联 {len(seen)} 篇；未匹配 {len(report['unmatched'])} 个文件；读取错误 {len(report['errors'])} 个。",
        "",
    ]
    if not missing:
        lines.append(
            "书目中的 12 篇研究论文均已放置并建立全文索引。CIE 术语来源为网页，未作为缺失论文 PDF。"
        )
    else:
        lines += [
            f"- {s['id']}: {ROOT / 'docs/research/papers' / (s['id'] + '.pdf')}"
            for s in missing
        ]
    lines += [
        "",
        "详细文件映射、哈希、页数与抽取异常：`.local/rag-ingestion.json`。PDF 及全文数据库不进入 Git 或公开安装包。",
    ]
    (ROOT / "docs/research/MISSING_PAPERS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "matched": len(seen),
                "missing": report["missing"],
                "unmatched": report["unmatched"],
                "errors": report["errors"],
                "index": report["index"],
                "seconds": report["seconds"],
            },
            ensure_ascii=False,
        )
    )
    if report["errors"] or report["unmatched"] or report["missing"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
