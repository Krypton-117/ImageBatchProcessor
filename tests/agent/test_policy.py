import sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent"))
from ibp_agent.native import validate_params
from ibp_agent.policy import assess, targets_for
from ibp_agent.knowledge import Knowledge
from ibp_agent.model import LocalModel


class PolicyTests(unittest.TestCase):
    def test_model_cannot_inject_unknown_parameters(self):
        for p in [
            {"shell": 1},
            {"exposure": float("nan")},
            {"exposure": 99},
            {"denoise": True},
        ]:
            with self.assertRaises(ValueError):
                validate_params(p)

    def test_cloud_endpoint_rejected(self):
        for u in ["https://api.example.com", "http://evil.test", "file:///tmp/x"]:
            with self.assertRaises(ValueError):
                LocalModel(u)

    def baseline(self):
        return dict(
            median=0.3,
            p95=0.8,
            p05=0.1,
            rmsContrast=0.15,
            labB=3,
            colorfulness=0.2,
            edgeEnergy=0.05,
            noiseEstimate=0.002,
            whiteClip=0.001,
            blackClip=0.01,
        )

    def test_better_target_cannot_override_clipping_gate(self):
        b = self.baseline()
        m = dict(b, median=0.4, whiteClip=0.1)
        result = assess(m, b, targets_for(["brighten"], b), {"exposure": 0.5})
        self.assertLess(result["target_loss"], 0.001)
        self.assertFalse(result["accepted"])

    def test_denoise_cannot_destroy_edges_to_game_noise(self):
        b = self.baseline()
        m = dict(b, noiseEstimate=0, edgeEnergy=0.001)
        self.assertFalse(
            assess(m, b, targets_for(["denoise"], b), {"denoise": 0.3})["accepted"]
        )

    def test_identity_is_accepted(self):
        b = self.baseline()
        self.assertTrue(assess(b, b, targets_for(["brighten"], b), {})["accepted"])

    def test_fts_chinese_retrieval_and_parameter_binding(self):
        with tempfile.TemporaryDirectory() as d:
            k = Knowledge(ROOT / "agent/knowledge", Path(d) / "index.db")
            self.assertIn(
                "contrast", [x["id"] for x in k.retrieve("照片灰蒙蒙，想通透")]
            )
            for t in k.terms:
                validate_params(t["seed"])
                self.assertTrue(t["sources"])

    def test_extreme_baselines_do_not_reverse_requested_direction(self):
        for value in (0.01, 0.9):
            b = dict(
                self.baseline(), median=value, p95=value, p05=value, rmsContrast=value
            )
            for intent, metric, sign in [
                ("brighten", "median", 1),
                ("darken", "median", -1),
                ("highlights", "p95", -1),
                ("shadows", "p05", 1),
                ("contrast", "rmsContrast", 1),
            ]:
                t = targets_for([intent], b)[0]
                self.assertGreaterEqual(sign * (t["target"] - b[metric]), 0)


if __name__ == "__main__":
    unittest.main()
