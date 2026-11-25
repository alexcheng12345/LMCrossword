import unittest
from unittest.mock import Mock

from main import cumulative_response_metrics
from utils.model import LLaMA


class LLaMARetryTest(unittest.TestCase):
    @staticmethod
    def make_model(responses, max_retries):
        model = LLaMA.__new__(LLaMA)
        model.max_retries = max_retries
        model.cost = 0
        model.retry_stats = {
            "total_turns": 0,
            "total_attempts": 0,
            "total_retries": 0,
        }
        model.length_violation_stats = {
            "raw_length_violation_count": 0,
            "raw_length_violation_turns": 0,
        }
        model._generate_once = Mock(side_effect=responses)
        return model

    def test_retries_wrong_length_answer(self):
        model = self.make_model([["TOOLONG"], ["ROCK"]], max_retries=1)
        prompt = [
            {"role": "system", "content": "Solve the clue."},
            {"role": "user", "content": "Clue: STONE (4)// Answer: "},
        ]

        self.assertEqual(model.infer(prompt), ["rock"])
        self.assertEqual(model._generate_once.call_count, 2)
        retry_prompt = model._generate_once.call_args.args[0]
        self.assertIn("TOOLONG", retry_prompt[-1]["content"])
        self.assertIn("exactly 4", retry_prompt[-1]["content"])
        metrics = model.get_metrics()
        self.assertEqual(metrics["total_retries"], 1)
        self.assertEqual(metrics["raw_length_violation_count"], 1)
        self.assertEqual(metrics["raw_length_violation_turns"], 1)

    def test_zero_retries_returns_original_length_violation(self):
        model = self.make_model([["TOOLONG"]], max_retries=0)
        prompt = [
            {"role": "user", "content": "Clue: STONE (4)// Answer: "},
        ]

        self.assertEqual(model.infer(prompt), ["toolong"])
        self.assertEqual(model._generate_once.call_count, 1)
        self.assertEqual(model.retry_stats["total_retries"], 0)

    def test_metrics_include_retry_efficiency_and_violation_rate(self):
        metrics = cumulative_response_metrics(
            {},
            {
                "total_turns": 2,
                "total_attempts": 3,
                "total_retries": 1,
                "raw_length_violation_count": 1,
            },
        )

        self.assertAlmostEqual(metrics["average_retries_per_turn"], 0.5)
        self.assertAlmostEqual(metrics["turn_efficiency"], 2 / 3)
        self.assertAlmostEqual(metrics["raw_length_violation_rate"], 100 / 3)


if __name__ == "__main__":
    unittest.main()
