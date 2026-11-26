import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from run_subtoken_experiment import (
    COUNT_ONLY_SYSTEM_PROMPT,
    build_counting_messages,
    category_for_id,
    evaluate_predictions,
    get_provider_settings,
    parse_predicted_count,
    run_count_request,
    validate_experiment_design,
)


class SubtokenExperimentTests(unittest.TestCase):
    def test_extracts_the_final_count_from_response(self):
        self.assertEqual(parse_predicted_count("Answer: 7"), 7)
        self.assertEqual(parse_predicted_count("3\n9\n7\n"), 7)
        self.assertIsNone(parse_predicted_count("unclear"))

    def test_maps_existing_dataset_categories(self):
        self.assertEqual(category_for_id("mid_12"), "mid")
        self.assertEqual(category_for_id("gibberish_12"), "gibberish")
        with self.assertRaises(ValueError):
            category_for_id("other_12")

    def test_configures_st4011_openai_compatible_api(self):
        settings = get_provider_settings("st4011")

        self.assertEqual(settings["base_url"], "https://st4011.loping151.net/v1")
        self.assertEqual(settings["api_key_env"], "ST4011_API_KEY")
        self.assertEqual(settings["model"], "deepseek-flash")

    def test_builds_prompt_that_requests_only_a_number(self):
        messages = build_counting_messages(
            [
                {"role": "system", "content": "Old instruction"},
                {"role": "user", "content": "Word: boat // 4\nWord: tree // "},
            ]
        )

        self.assertEqual(messages[0]["content"], COUNT_ONLY_SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"], "Word: boat // 4\nWord: tree // ")

    def test_request_uses_configured_token_budget_and_marks_empty_content(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=""))
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=128),
        )

        entry = run_count_request(
            client,
            "deepseek-flash",
            "common_0",
            [{"role": "user", "content": "Word: boat // "}],
            128,
        )

        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["max_tokens"], 128)
        self.assertEqual(request["temperature"], 1)
        self.assertEqual(request["messages"][0]["content"], COUNT_ONLY_SYSTEM_PROMPT)
        self.assertEqual(entry["response"], "")
        self.assertTrue(entry["error"].startswith("EmptyResponse:"))

    def test_reports_category_and_word_length_accuracy(self):
        dataset = {
            "common_0": "cat",
            "vocab_0": "book",
            "gibberish_0": "xy",
        }
        predictions = {
            "common_0": "3",
            "vocab_0": "5",
        }

        details, summary = evaluate_predictions(dataset, predictions)

        self.assertEqual([row["correct"] for row in details], [True, False, False])
        summary_by_group = {
            (row["group_type"], row["group"]): row for row in summary
        }
        self.assertEqual(summary_by_group[("overall", "all")]["accuracy_pct"], 33.33)
        self.assertEqual(summary_by_group[("word_frequency", "common")]["correct"], 1)
        self.assertEqual(summary_by_group[("word_type", "gibberish")]["items"], 1)
        self.assertEqual(summary_by_group[("word_length", "4")]["items"], 1)

    def test_warns_when_data_does_not_match_planned_sample_design(self):
        warnings = validate_experiment_design(
            {
                "common_0": "cat",
                "mid_0": "boat",
                "rare_0": "word",
                "vocab_0": "cat",
                "gibberish_0": "xy",
            }
        )

        self.assertEqual(len(warnings), 4)
        self.assertTrue(any("not the planned 1,000" in warning for warning in warnings))
        self.assertTrue(any("matching word-length" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
