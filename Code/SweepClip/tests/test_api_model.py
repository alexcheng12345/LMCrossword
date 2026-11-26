import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from utils.model import APIModel, config


class StubTextPart:
    def __init__(self, text):
        self.text = text


class StubMessage:
    def __init__(self, content=None, reasoning_content=None, model_extra=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.model_extra = model_extra


class StubInternalServerError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


class APIModelResponseTest(unittest.TestCase):
    @staticmethod
    def make_model(responses, max_retries=1):
        model = APIModel.__new__(APIModel)
        model.max_retries = max_retries
        model.model = "deepseek-flash"
        model.input_price = 0
        model.output_price = 0
        model.cost = 0
        model.total_cost = 0
        model.retry_stats = {
            "total_turns": 0,
            "total_attempts": 0,
            "total_retries": 0,
            "api_overload_retries": 0,
        }
        model.length_violation_stats = {
            "raw_length_violation_count": 0,
            "raw_length_violation_turns": 0,
        }
        model.response_diagnostics = {
            "final_content_turn_count": 0,
            "reasoning_only_turn_count": 0,
            "empty_response_turn_count": 0,
        }
        model.client = Mock()
        model.client.chat.completions.create.side_effect = [
            SimpleNamespace(
                choices=[SimpleNamespace(message=StubMessage(content=answer))],
                usage=None,
            )
            for answer in responses
        ]
        return model

    @staticmethod
    def target_prompt():
        return [
            {"role": "system", "content": "Answer with one word."},
            {"role": "user", "content": "Clue: STONE (4)// Answer: "},
        ]

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

    def test_api_alias_uses_its_configured_base_url(self):
        ollama_url = "http://localhost:11434/v1"
        with patch("utils.model.OpenAI") as openai_client:
            with patch.dict(
                config,
                {"api_model_base_urls": {"ollama_llama3_70b": ollama_url}},
            ):
                with patch.dict("os.environ", {"OPENAI_API_KEY": "ollama"}):
                    APIModel(model_name="ollama_llama3_70b")

        self.assertEqual(
            openai_client.call_args.kwargs["base_url"],
            ollama_url,
        )

    def test_retries_wrong_length_answer_and_returns_valid_retry(self):
        model = self.make_model(["TOOLONG", "ROCK"])

        self.assertEqual(model.infer(self.target_prompt()), ["rock"])
        self.assertEqual(model.retry_stats["total_attempts"], 2)
        self.assertEqual(model.retry_stats["total_retries"], 1)
        self.assertEqual(model.length_violation_stats["raw_length_violation_count"], 1)
        self.assertEqual(model.length_violation_stats["raw_length_violation_turns"], 1)

        retry_messages = model.client.chat.completions.create.call_args_list[1].kwargs[
            "messages"
        ]
        self.assertEqual(retry_messages[-2]["content"], "TOOLONG")
        self.assertIn("exactly 4", retry_messages[-1]["content"])

    def test_multiple_bad_attempts_count_one_violation_turn(self):
        model = self.make_model(["TOOLONG", "STILLTOOLONG"])

        self.assertEqual(model.infer(self.target_prompt()), ["stilltoolong"])
        self.assertEqual(model.retry_stats["total_attempts"], 2)
        self.assertEqual(model.retry_stats["total_retries"], 1)
        self.assertEqual(model.length_violation_stats["raw_length_violation_count"], 2)
        self.assertEqual(model.length_violation_stats["raw_length_violation_turns"], 1)

    def test_zero_retry_mode_returns_first_answer_without_retrying(self):
        model = self.make_model(["TOOLONG"], max_retries=0)

        self.assertEqual(model.infer(self.target_prompt()), ["toolong"])
        self.assertEqual(model.retry_stats["total_attempts"], 1)
        self.assertEqual(model.retry_stats["total_retries"], 0)

    def test_cpu_overload_503_waits_and_retries_until_success(self):
        model = self.make_model(["ROCK"])
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=StubMessage(content="ROCK"))],
            usage=None,
        )
        model.client.chat.completions.create.side_effect = [
            StubInternalServerError(
                "Error code: 503 - system cpu overloaded",
                status_code=503,
            ),
            completion,
        ]

        with patch("utils.model.InternalServerError", StubInternalServerError):
            with patch("utils.model.time.sleep") as sleep:
                self.assertEqual(model.infer(self.target_prompt()), ["rock"])

        self.assertEqual(
            model.client.chat.completions.create.call_count,
            2,
        )
        self.assertEqual(sleep.call_args.args[0], 5)
        self.assertEqual(model.retry_stats["api_overload_retries"], 1)

    def test_non_overload_server_error_is_not_retried(self):
        model = self.make_model(["ROCK"])
        model.client.chat.completions.create.side_effect = StubInternalServerError(
            "Error code: 503 - service unavailable",
            status_code=503,
        )

        with patch("utils.model.InternalServerError", StubInternalServerError):
            with patch("utils.model.time.sleep") as sleep:
                with self.assertRaises(StubInternalServerError):
                    model.infer(self.target_prompt())

        self.assertEqual(
            model.client.chat.completions.create.call_count,
            1,
        )
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
