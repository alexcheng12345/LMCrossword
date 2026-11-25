import unittest
from threading import Event
from unittest.mock import patch

import main_b3
import main_b2
import src.api_llm_modified as api_llm_modified


class FakeDictionary:
    def __init__(self, candidates):
        self.candidates = candidates

    def retrieve_candidates(self, clue, search_terms, required_length, limit):
        self.search = (clue, search_terms, required_length, limit)
        return self.candidates


PROMPT = [
    {"role": "system", "content": "Solve the clue."},
    {"role": "user", "content": "Clue: Sea mammal (4)// Answer: "},
]
CANDIDATES = [
    {"word": "seal", "definitions": ["An aquatic mammal"], "score": 2.0},
    {"word": "orca", "definitions": ["A large marine mammal"], "score": 1.0},
]


class B3Tests(unittest.TestCase):
    def test_extracts_only_the_final_target_clue_and_length(self):
        prompt = [
            {"role": "system", "content": "System"},
            {
                "role": "user",
                "content": (
                    "Clue: Port (4)// Answer: COLE\n"
                    "Clue: Sea mammal (4)// Answer: "
                ),
            },
        ]
        self.assertEqual(main_b3.extract_clue_and_length(prompt), ("Sea mammal", 4))

    @patch("main_b3.run_inference")
    def test_retries_selection_after_no_valid_candidate(self, run_inference):
        run_inference.side_effect = [
            ('{"search_terms": ["marine mammal", "sea animal"]}', {}, None, 0.1),
            ("NO_VALID_CANDIDATE", {}, None, 0.1),
            ("C02", {}, None, 0.1),
        ]
        dictionary = FakeDictionary(CANDIDATES)
        progress_events = []

        clue_id, answer, calls, stopped = main_b3.run_b3_for_clue(
            object(),
            "id",
            PROMPT,
            "Sea mammal",
            4,
            dictionary,
            4,
            progress_callback=progress_events.append,
        )

        self.assertEqual((clue_id, answer), ("id", "ORCA"))
        self.assertFalse(stopped)
        self.assertEqual(
            [call["stage"] for call in calls],
            [
                "query_expansion",
                "candidate_selection",
                "candidate_selection_retry",
            ],
        )
        self.assertFalse(calls[-1]["b3_decision"]["b2_fallback_used"])
        self.assertEqual(dictionary.search[2:], (4, 30))
        self.assertEqual(
            [event["event"] for event in progress_events],
            ["start", "finish"] * 3,
        )
        self.assertEqual(
            [event["stage"] for event in progress_events if event["event"] == "start"],
            [
                "query_expansion",
                "candidate_selection",
                "candidate_selection_retry",
            ],
        )

    @patch("main_b3.run_b2_for_clue")
    @patch("main_b3.run_inference")
    def test_falls_back_to_b2_when_dictionary_has_no_candidates(
        self, run_inference, run_b2
    ):
        run_inference.return_value = (
            '{"search_terms": ["marine mammal"]}',
            {},
            None,
            0.1,
        )
        run_b2.return_value = ("id", "SEAL", [])

        clue_id, answer, calls, stopped = main_b3.run_b3_for_clue(
            object(),
            "id",
            PROMPT,
            "Sea mammal",
            4,
            FakeDictionary([]),
            4,
        )

        self.assertEqual((clue_id, answer), ("id", "SEAL"))
        self.assertFalse(stopped)
        self.assertTrue(calls[-1]["b3_decision"]["b2_fallback_used"])
        self.assertEqual(
            calls[-1]["b3_decision"]["fallback_reason"],
            "no_length_matched_dictionary_candidates",
        )

    @patch("main_b3.run_b2_for_clue")
    @patch("main_b3.run_inference")
    def test_query_expansion_rate_limit_stops_without_fallback(
        self, run_inference, run_b2
    ):
        run_inference.return_value = (
            "",
            {},
            "RateLimitError: Error code: 429 - rate limit exceeded",
            0.1,
        )
        stop_event = Event()
        progress_events = []

        clue_id, answer, calls, stopped = main_b3.run_b3_for_clue(
            object(),
            "id",
            PROMPT,
            "Sea mammal",
            4,
            FakeDictionary(CANDIDATES),
            4,
            stop_event,
            progress_events.append,
        )

        self.assertEqual((clue_id, answer), ("id", None))
        self.assertTrue(stopped)
        self.assertTrue(stop_event.is_set())
        self.assertEqual(len(calls), 1)
        run_b2.assert_not_called()
        run_inference.assert_called_once()
        self.assertEqual(
            [event["event"] for event in progress_events],
            ["start", "finish"],
        )
        self.assertEqual(progress_events[-1]["entry"]["error"], calls[0]["error"])

    @patch("main_b3.run_b2_for_clue")
    @patch("main_b3.run_inference")
    def test_selection_rate_limit_stops_without_retry_or_fallback(
        self, run_inference, run_b2
    ):
        run_inference.side_effect = [
            ('{"search_terms": ["marine mammal"]}', {}, None, 0.1),
            (
                "",
                {},
                "RateLimitError: Error code: 429 - rate limit exceeded",
                0.1,
            ),
        ]
        stop_event = Event()

        clue_id, answer, calls, stopped = main_b3.run_b3_for_clue(
            object(),
            "id",
            PROMPT,
            "Sea mammal",
            4,
            FakeDictionary(CANDIDATES),
            4,
            stop_event,
        )

        self.assertEqual((clue_id, answer), ("id", None))
        self.assertTrue(stopped)
        self.assertTrue(stop_event.is_set())
        self.assertEqual(len(calls), 2)
        run_b2.assert_not_called()
        self.assertEqual(run_inference.call_count, 2)

    @patch("main_b2.run_inference")
    def test_b2_fallback_stops_on_rate_limit(self, run_inference):
        run_inference.return_value = (
            "",
            {},
            "RateLimitError: Error code: 429 - rate limit exceeded",
            0.1,
        )
        stop_event = Event()

        clue_id, answer, attempts = main_b2.run_b2_for_clue(
            object(),
            "id",
            PROMPT,
            4,
            4,
            stop_on_rate_limit=True,
            stop_event=stop_event,
        )

        self.assertEqual((clue_id, answer), ("id", None))
        self.assertTrue(stop_event.is_set())
        self.assertEqual(len(attempts), 1)

    @patch("main_b2.run_inference")
    def test_b2_fallback_reports_each_api_call(self, run_inference):
        run_inference.return_value = ("CAT", {}, None, 0.1)
        progress_events = []

        main_b2.run_b2_for_clue(
            object(),
            "id",
            PROMPT,
            3,
            1,
            progress_callback=progress_events.append,
        )

        self.assertEqual(
            [event["event"] for event in progress_events],
            ["start", "finish"],
        )
        self.assertEqual(progress_events[0]["stage"], "b2_fallback")
        self.assertEqual(progress_events[-1]["entry"]["answer"], "CAT")

    @patch("src.api_llm_modified.OpenAI")
    def test_api_client_can_disable_sdk_retries(self, openai_client):
        settings = api_llm_modified.config["omnirouters"]
        with patch.dict("os.environ", {settings["api_key_env"]: "test-key"}):
            api_llm_modified.OmniRoutersGPT(max_retries=0, timeout=90)

        self.assertEqual(openai_client.call_args.kwargs["max_retries"], 0)
        self.assertEqual(openai_client.call_args.kwargs["timeout"], 90)

    @patch("main_b3.run_inference")
    def test_already_stopped_worker_makes_no_api_call(self, run_inference):
        stop_event = Event()
        stop_event.set()

        result = main_b3.run_b3_for_clue(
            object(),
            "id",
            PROMPT,
            "Sea mammal",
            4,
            FakeDictionary(CANDIDATES),
            4,
            stop_event,
        )

        self.assertEqual(result, ("id", None, [], True))
        run_inference.assert_not_called()


if __name__ == "__main__":
    unittest.main()
