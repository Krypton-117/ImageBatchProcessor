import json, sys, tempfile, threading, unittest, urllib.error, urllib.request
from pathlib import Path
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent"))
from ibp_agent.server import make_handler
from ibp_agent.knowledge import Knowledge


class Dummy:
    token = "test-token"
    jobs = {}
    uploads = {}
    ready = False
    busy = False
    error = None


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = Dummy()
        self.app.paths = {"data": Path(self.temp.name), "web": ROOT / "agent/web"}
        self.app.knowledge = Knowledge(
            ROOT / "agent/knowledge", Path(self.temp.name) / "knowledge.sqlite"
        )
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.app))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def test_token_and_cross_origin_are_enforced(self):
        for h in [
            {},
            {"X-IBP-Token": "wrong"},
            {"X-IBP-Token": "test-token", "Origin": "https://evil.example"},
        ]:
            with self.assertRaises(urllib.error.HTTPError) as cm:
                self.opener.open(
                    urllib.request.Request(self.url + "/api/status", headers=h)
                )
            self.assertEqual(cm.exception.code, 403)

    def test_status_and_static_ui(self):
        with self.opener.open(
            urllib.request.Request(
                self.url + "/api/status", headers={"X-IBP-Token": "test-token"}
            )
        ) as r:
            self.assertFalse(json.load(r)["ready"])
        with self.opener.open(self.url) as r:
            self.assertIn(b"PHOTO AGENT", r.read())
            self.assertIn(
                "frame-ancestors 'none'", r.headers["Content-Security-Policy"]
            )

    def test_rag_routes_keep_authentication_and_reject_invalid_query(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.opener.open(self.url + "/api/rag/search?q=noise")
        self.assertEqual(cm.exception.code, 403)
        for path, status in [
            ("/api/rag/search", 400),
            ("/api/rag/papers/unknown", 404),
        ]:
            with self.assertRaises(urllib.error.HTTPError) as cm:
                self.opener.open(
                    urllib.request.Request(
                        self.url + path, headers={"X-IBP-Token": "test-token"}
                    )
                )
            self.assertEqual(cm.exception.code, status)
        with self.opener.open(
            urllib.request.Request(
                self.url + "/api/rag/search?q=noise",
                headers={"X-IBP-Token": "test-token"},
            )
        ) as r:
            self.assertEqual(json.load(r)["hits"], [])

    def test_path_traversal_and_unrecognized_job(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.opener.open(
                urllib.request.Request(
                    self.url + "/files/unknown/../../secret",
                    headers={"X-IBP-Token": "test-token"},
                )
            )
        self.assertEqual(cm.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
