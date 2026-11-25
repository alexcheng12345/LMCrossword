import unittest
from unittest.mock import patch

import numpy as np

from evaluate_outputs import answer_embedding, normalized_cosine_score


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
    def test_cosine_is_mapped_to_zero_to_one_hundred(self):
        vectors = FakeEmbeddings()

        self.assertEqual(
            normalized_cosine_score(vectors["same"], vectors["same"]),
            100.0,
        )
        self.assertEqual(
            normalized_cosine_score(vectors["same"], vectors["orthogonal"]),
            50.0,
        )
        self.assertEqual(
            normalized_cosine_score(vectors["same"], vectors["opposite"]),
            0.0,
        )

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


if __name__ == "__main__":
    unittest.main()
