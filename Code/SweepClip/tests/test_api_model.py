import unittest

from utils.model import APIModel


class StubTextPart:
    def __init__(self, text):
        self.text = text


class StubMessage:
    def __init__(self, content=None, reasoning_content=None, model_extra=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.model_extra = model_extra


class APIModelResponseTest(unittest.TestCase):
    def test_extracts_plain_text(self):
        self.assertEqual(
            APIModel._message_content(StubMessage(content="Answer: ROCKY")),
            "Answer: ROCKY",
        )

    def test_extracts_structured_text(self):
        message = StubMessage(content=[StubTextPart("RO"), StubTextPart("CKY")])
        self.assertEqual(APIModel._message_content(message), "ROCKY")

    def test_does_not_use_reasoning_as_final_answer(self):
        message = StubMessage(reasoning_content="Answer: ROCKY")
        self.assertEqual(APIModel._message_content(message), "")

    def test_normalizes_answer(self):
        self.assertEqual(APIModel._process("Answer: ROCKY\nextra"), "rocky")

    def test_preserves_length_violation_for_no_retry_pruning(self):
        self.assertEqual(APIModel._process("weneedanswer for clue"), "weneedanswerforclue")

    def test_empty_final_content_is_rejected_as_empty_answer(self):
        message = StubMessage(content=None, reasoning_content="I reasoned about this")
        self.assertEqual(APIModel._message_content(message), "")


if __name__ == "__main__":
    unittest.main()
