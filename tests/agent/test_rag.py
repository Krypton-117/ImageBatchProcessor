import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent"))
from ibp_agent.rag import PaperIndex, model_context, page_chunks
from ibp_agent.model import LocalModel

sys.path.insert(0, str(ROOT / "tools/agent"))
from index_papers import identify


class RagTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.index = PaperIndex(self.root / "knowledge.sqlite")
        self.source = {
            "id": "noise",
            "title": "Raw noise variance model",
            "url": "https://example.invalid/paper",
        }
        self.index.add(
            self.source,
            self.root / "arbitrary-name.pdf",
            [
                "First page about camera response. " * 8,
                "Poissonian Gaussian noise variance models raw camera signal. " * 12,
            ],
            "hash1",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_chinese_query_returns_body_and_correct_physical_page(self):
        result = self.index.search("噪声方差")
        self.assertEqual(result["hits"][0]["page"], 2)
        self.assertIn("Poissonian", result["hits"][0]["text"])
        self.assertEqual(result["hits"][0]["sha256"], "hash1")
        self.assertTrue(self.index.search("噪声方差")["cache_hit"])

    def test_replacement_invalidates_cache_and_removes_old_text(self):
        self.index.search("噪声")
        self.index.add(
            self.source,
            self.root / "new.pdf",
            ["Chromaticity white balance calibration. " * 10],
            "hash2",
        )
        result = self.index.search("poissonian")
        self.assertFalse(result["cache_hit"])
        self.assertEqual(result["hits"], [])
        self.assertEqual(self.index.status()["documents"], 1)

    def test_empty_pdf_does_not_replace_existing_index(self):
        with self.assertRaises(ValueError):
            self.index.add(self.source, self.root / "bad.pdf", [""], "bad")
        self.assertEqual(self.index.documents()[0]["sha256"], "hash1")

    def test_unknown_query_and_fts_syntax_are_data(self):
        self.assertEqual(self.index.search('" OR NEAR(x) * : xyzzy')["hits"], [])
        with self.assertRaises(ValueError):
            self.index.search("")
        self.assertIsNone(self.index.pdf("../../secret"))

    def test_context_is_bounded_and_sent_in_actual_model_request(self):
        result = self.index.search("噪声")
        context = model_context(result, [])
        self.assertTrue(all(len(h["text"]) <= 500 for h in context["excerpts"]))
        captured = []
        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "intents": ["denoise"],
                                "unsupported": False,
                                "clarification": False,
                                "preserve": [],
                            }
                        )
                    }
                }
            ]
        }
        import io

        class Opener:
            def open(self, request, timeout):
                captured.append(json.loads(request.data))
                return io.BytesIO(json.dumps(response).encode())

        with patch("urllib.request.build_opener", return_value=Opener()):
            parsed = LocalModel().parse("降噪", evidence=context)
        messages = captured[0]["messages"]
        self.assertTrue(any("Poissonian" in m["content"] for m in messages))
        self.assertIn("noise:p2:c0", parsed["rag_context_ids"])
        self.assertEqual(parsed["intents"], ["denoise"])

    def test_chunks_retain_text_and_no_oversized_single_page(self):
        chunks = list(page_chunks("A camera noise model. " * 200))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 1100 for c in chunks))


class PaperIdentityTests(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {
                "id": "peli",
                "type": "research_paper",
                "title": "Contrast in complex images",
            },
            {
                "id": "wang",
                "type": "research_paper",
                "title": "Image quality assessment: from error visibility to structural similarity",
            },
        ]

    def test_specific_title_beats_incidental_short_title_tokens(self):
        text = self.sources[1]["title"] + " Complex camera images have contrast."
        source, evidence = identify(text, "new-hash", self.sources)
        self.assertEqual(source["id"], "wang")
        self.assertEqual(evidence["method"], "title_phrase_first_pages")

    def test_verified_hash_disambiguates_quoted_other_title(self):
        self.sources[1]["sha256"] = "verified"
        source, evidence = identify(self.sources[0]["title"], "verified", self.sources)
        self.assertEqual(source["id"], "wang")
        self.assertEqual(evidence["method"], "verified_sha256")

    def test_ambiguous_and_unrelated_titles_are_not_guessed(self):
        text = " ".join(source["title"] for source in self.sources)
        self.assertIsNone(identify(text, "new", self.sources)[0])
        self.assertIsNone(
            identify("A completely unrelated research topic", "new", self.sources)[0]
        )
        self.assertIsNone(identify(text, "new", [])[0])
        self.assertEqual(
            identify(self.sources[0]["title"], "new", self.sources[:1])[0]["id"], "peli"
        )


if __name__ == "__main__":
    unittest.main()
