import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evaluate_outputs import (
    answer_embedding,
    classify_api_call_error,
    cosine_similarity,
    load_call_metadata,
    normalized_semantic_accuracy,
    summarize_api_call_outcomes,
)


class FakeEmbeddings:
    key_to_index = {
        "same": 0,
        "orthogonal": 1,
        "opposite": 2,
        "new": 3,
        "york": 4,
    }
    vectors = {
        "same": np.array([1.0, 0.0]),
        "orthogonal": np.array([0.0, 1.0]),
        "opposite": np.array([-1.0, 0.0]),
        "new": np.array([1.0, 0.0]),
        "york": np.array([0.0, 1.0]),
    }

    def __getitem__(self, token):
        return self.vectors[token]


class SemanticScoreTests(unittest.TestCase):
    def test_cosine_similarity_is_unscaled(self):
        vectors = FakeEmbeddings()

        self.assertEqual(
            cosine_similarity(vectors["same"], vectors["same"]),
            1.0,
        )
        self.assertEqual(
            cosine_similarity(vectors["same"], vectors["orthogonal"]),
            0.0,
        )
        self.assertEqual(
            cosine_similarity(vectors["same"], vectors["opposite"]),
            -1.0,
        )

    def test_normalizes_plain_cosine_sum_to_zero_to_one_hundred(self):
        self.assertEqual(normalized_semantic_accuracy(2.0, 2), 100.0)
        self.assertEqual(normalized_semantic_accuracy(0.0, 2), 50.0)
        self.assertEqual(normalized_semantic_accuracy(-2.0, 2), 0.0)
        self.assertIsNone(normalized_semantic_accuracy(0.0, 0))

    def test_answer_embedding_ignores_out_of_vocabulary_tokens(self):
        vector = answer_embedding("same unknown", FakeEmbeddings())

        np.testing.assert_array_equal(vector, np.array([1.0, 0.0]))

    def test_answer_embedding_returns_none_when_no_tokens_are_known(self):
        self.assertIsNone(answer_embedding("unknown", FakeEmbeddings()))

    @patch("evaluate_outputs.wordninja.split", return_value=["new", "york"])
    def test_answer_embedding_segments_concatenated_phrases(self, split):
        vector = answer_embedding("newyork", FakeEmbeddings())

        np.testing.assert_array_equal(vector, np.array([0.5, 0.5]))
        split.assert_called_once_with("newyork")


class ApiCallOutcomeTests(unittest.TestCase):
    def test_classifies_success_and_failure_categories(self):
        self.assertEqual(classify_api_call_error(None), "success")
        self.assertEqual(
            classify_api_call_error("RateLimitError: Error code: 429"),
            "rate_limit_429",
        )
        self.assertEqual(
            classify_api_call_error("AuthenticationError: 401 invalid token"),
            "authentication_401",
        )
        self.assertEqual(
            classify_api_call_error("InternalServerError: Error code: 500"),
            "other_failure",
        )

    def test_outcome_counts_are_mutually_exclusive(self):
        entries = [
            {"error": None},
            {"error": "HTTP 429 rate limit"},
            {"error": "401 invalid token"},
            {"error": "HTTP 500"},
        ]

        outcomes = summarize_api_call_outcomes(entries)

        self.assertEqual(outcomes["successful_calls"], 1)
        self.assertEqual(outcomes["failed_calls"], 3)
        self.assertEqual(outcomes["rate_limit_calls_429"], 1)
        self.assertEqual(outcomes["authentication_failures_401"], 1)
        self.assertEqual(outcomes["other_api_failures"], 1)

    def test_call_metadata_uses_categories_from_raw_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            call_log = log_dir / "sample_calls.jsonl"
            call_log.write_text(
                "\n".join(
                    json.dumps(entry)
                    for entry in (
                        {"error": None, "clue_id": "a"},
                        {"error": "RateLimitError: 429", "clue_id": "b"},
                        {"error": "AuthenticationError: 401", "clue_id": "c"},
                        {"error": "InternalServerError: 500", "clue_id": "d"},
                    )
                ),
                encoding="utf-8",
            )

            metadata = load_call_metadata(log_dir / "sample.json", log_dir)

        self.assertEqual(metadata["number_of_calls"], 4)
        self.assertEqual(metadata["successful_calls"], 1)
        self.assertEqual(metadata["failed_calls"], 3)
        self.assertEqual(metadata["rate_limit_calls_429"], 1)
        self.assertEqual(metadata["authentication_failures_401"], 1)
        self.assertEqual(metadata["other_api_failures"], 1)


if __name__ == "__main__":
    unittest.main()
