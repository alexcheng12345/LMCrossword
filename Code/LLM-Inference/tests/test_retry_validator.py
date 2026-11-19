import unittest

from src.retry_validator import (
    extract_required_length,
    is_valid_answer,
    normalize_answer,
)


class RetryValidatorTest(unittest.TestCase):
    def test_extracts_length_from_latest_clue(self):
        prompt = [{"role": "user", "content": "Clue: Daily update (4)// Answer:"}]
        self.assertEqual(extract_required_length(prompt), 4)

    def test_normalizes_answer_to_letters(self):
        self.assertEqual(normalize_answer("Answer: COFFEE"), "COFFEE")
        self.assertEqual(normalize_answer("ice-cream"), "icecream")

    def test_rejects_and_accepts_character_lengths(self):
        self.assertTrue(is_valid_answer("CAT", 3))
        self.assertFalse(is_valid_answer("CATS!", 3))


if __name__ == "__main__":
    unittest.main()
