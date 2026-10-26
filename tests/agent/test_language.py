import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent"))
from ibp_agent.language import guard_intent


class LanguageGuardTests(unittest.TestCase):
    def raw(self):
        return dict(
            intents=["vivid"], preserve=[], unsupported=False, clarification=False
        )

    def test_unknown_semantic_recolor_cannot_become_global_edit(self):
        r = guard_intent("请把红色衣服变成蓝色", self.raw())
        self.assertTrue(r["unsupported"])

    def test_aesthetic_ambiguity_requires_clarification(self):
        self.assertTrue(guard_intent("给我高级一些", self.raw())["clarification"])

    def test_negation_and_preservation_override_model_and_remain_auditable(self):
        raw = self.raw()
        r = guard_intent("通透一点，但不要更艳", raw)
        self.assertEqual(r["intents"], ["contrast"])
        self.assertIn("color", r["preserve"])
        self.assertEqual(r["raw_model_result"], raw)

    def test_unmatched_clause_does_not_silently_disappear(self):
        self.assertTrue(
            guard_intent("亮一点，增加伦勃朗光", self.raw())["clarification"]
        )

    def test_reduce_contrast_negates_strength_not_the_requested_reduction(self):
        r = guard_intent("反差别那么强，降低对比", self.raw())
        self.assertEqual(r["intents"], ["soft"])

    def test_exact_exposure_amount_is_not_silently_approximated(self):
        self.assertTrue(guard_intent("增加曝光半档", self.raw())["clarification"])


if __name__ == "__main__":
    unittest.main()
