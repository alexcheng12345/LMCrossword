import unittest

from utils.primitives import Crossword


class StubClue:
    def __init__(self, length):
        self.answer = "x" * length

    def __len__(self):
        return len(self.answer)


class LengthPruneTest(unittest.TestCase):
    def test_keeps_only_exact_lengths_without_truncating(self):
        crossword = Crossword.__new__(Crossword)
        crossword.current_answers = {
            "1_a": "cat",
            "2_d": "elephant",
            "3_a": "four",
        }
        crossword.clues = {
            "1_a": StubClue(3),
            "2_d": StubClue(4),
            "3_a": StubClue(4),
        }

        crossword._length_prune()

        self.assertEqual(crossword.current_answers, {
            "1_a": "cat",
            "3_a": "four",
        })


if __name__ == "__main__":
    unittest.main()
