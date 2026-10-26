"""Local, inspectable SQLite terminology and literature index."""

import json, sqlite3
from contextlib import closing
from pathlib import Path


class Knowledge:
    def __init__(self, root, db):
        from .rag import PaperIndex

        self.papers = PaperIndex(db)
        self.root = Path(root)
        self.db = Path(db)
        self.db.parent.mkdir(parents=True, exist_ok=True)
        self.terms = json.loads(
            (self.root / "terms.json").read_text(encoding="utf-8-sig")
        )
        source_path = self.root / "sources.json"
        self.sources = (
            json.loads(source_path.read_text(encoding="utf-8-sig"))
            if source_path.exists()
            else []
        )
        with closing(sqlite3.connect(self.db)) as c:
            c.executescript(
                'CREATE TABLE IF NOT EXISTS terms(id TEXT PRIMARY KEY, data TEXT NOT NULL); CREATE TABLE IF NOT EXISTS sources(id TEXT PRIMARY KEY, data TEXT NOT NULL); CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(id UNINDEXED, text, tokenize="unicode61");'
            )
            c.execute("DELETE FROM terms")
            c.execute("DELETE FROM sources")
            c.execute("DELETE FROM search")
            for t in self.terms:
                text = " ".join(
                    [t["label"], t["english"], *t["aliases"], t["definition"]]
                )
                # Explicitly index Chinese unigrams/bigrams; default FTS does not segment Chinese.
                chars = "".join(x for x in text if "\u4e00" <= x <= "\u9fff")
                tokens = (
                    " ".join(chars)
                    + " "
                    + " ".join(chars[i : i + 2] for i in range(len(chars) - 1))
                )
                c.execute(
                    "INSERT INTO terms VALUES(?,?)",
                    (t["id"], json.dumps(t, ensure_ascii=False)),
                )
                c.execute(
                    "INSERT INTO search VALUES(?,?)", (t["id"], text + " " + tokens)
                )
            for s in self.sources:
                c.execute(
                    "INSERT INTO sources VALUES(?,?)",
                    (s["id"], json.dumps(s, ensure_ascii=False)),
                )
            c.commit()

    def get(self, ids):
        return [t for t in self.terms if t["id"] in ids]

    def retrieve(self, text, limit=6):
        scored = []
        for t in self.terms:
            score = sum(
                len(a)
                for a in [t["label"], t["english"], *t["aliases"]]
                if a.lower() in text.lower()
            )
            if score:
                scored.append((score, t))
        if scored:
            return [t for _, t in sorted(scored, key=lambda x: -x[0])[:limit]]
        tokens = [x for x in text.split() if x.isalnum()]
        chars = [x for x in text if "\u4e00" <= x <= "\u9fff"]
        tokens += ["".join(chars[i : i + 2]) for i in range(len(chars) - 1)]
        if not tokens:
            return []
        query = " OR ".join('"' + x.replace('"', '""') + '"' for x in tokens[:24])
        with closing(sqlite3.connect(self.db)) as c:
            ids = [
                r[0]
                for r in c.execute(
                    "SELECT id FROM search WHERE search MATCH ? ORDER BY rank LIMIT ?",
                    (query, limit),
                )
            ]
        return self.get(ids)

    def references(self, terms):
        ids = {s for t in terms for s in t["sources"]}
        return [s for s in self.sources if s["id"] in ids]

    def search_papers(self, query, limit=5):
        return self.papers.search(query, self.retrieve(query, limit=4), limit)
