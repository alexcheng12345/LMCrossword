import unittest

from analyze_b3_candidate_recall import analyze_candidate_recall


class B3CandidateRecallTests(unittest.TestCase):
    def test_counts_hits_and_rank_buckets_once_per_clue(self):
        query = {
            "hit": {"clue": "Uncertain", "answer": "rocky"},
            "miss": {"clue": "Sea mammal", "answer": "seal"},
            "no_candidates": {"clue": "Odd clue", "answer": "answer"},
        }
        call_entries = [
            {"clue_id": "hit", "stage": "query_expansion"},
            {
                "clue_id": "hit",
                "stage": "candidate_selection",
                "candidate_words": ["ROCKY", "wrong"],
                "candidate_count": 2,
            },
            {
                "clue_id": "hit",
                "stage": "candidate_selection_retry",
                "candidate_words": ["ROCKY", "wrong"],
                "candidate_count": 2,
            },
            {"clue_id": "hit", "stage": "query_expansion"},
            {
                "clue_id": "hit",
                "stage": "candidate_selection",
                "candidate_words": ["wrong", "ROCKY"],
                "candidate_count": 2,
            },
            {
                "clue_id": "miss",
                "stage": "candidate_selection",
                "candidate_words": ["orca", "whale"],
                "candidate_count": 2,
            },
            {
                "clue_id": "no_candidates",
                "stage": "b2_fallback",
                "candidate_count": 0,
                "fallback_reason": "no_length_matched_dictionary_candidates",
            },
        ]

        summary, details = analyze_candidate_recall(query, call_entries)

        self.assertEqual(summary["dataset_clues"], 3)
        self.assertEqual(summary["candidate_lists_found"], 2)
        self.assertEqual(summary["clues_without_candidates"], 1)
        self.assertEqual(summary["candidate_hits"], 1)
        self.assertEqual(summary["candidate_misses_with_list"], 1)
        self.assertEqual(summary["overall_recall_at_30_pct"], 33.33)
        self.assertEqual(
            summary["recall_at_30_when_candidates_exist_pct"], 50.0
        )
        self.assertEqual(summary["hits_at_rank_1"], 0)
        self.assertEqual(summary["hits_in_top_5"], 1)
        self.assertEqual(details[0]["candidate_rank"], 2)
        self.assertEqual(
            summary["no_candidate_fallback_reasons"],
            {"no_length_matched_dictionary_candidates": 1},
        )

    def test_rejects_candidate_logs_for_unknown_clues(self):
        with self.assertRaisesRegex(ValueError, "unknown clue ID"):
            analyze_candidate_recall(
                {"known": {"answer": "word"}},
                [{"clue_id": "unknown", "candidate_words": ["word"]}],
            )

    def test_rejects_inconsistent_candidate_lists_for_same_clue(self):
        with self.assertRaisesRegex(ValueError, "inconsistent candidate lists"):
            analyze_candidate_recall(
                {"clue": {"answer": "word"}},
                [
                    {"clue_id": "clue", "candidate_words": ["word"]},
                    {"clue_id": "clue", "candidate_words": ["other"]},
                ],
            )


if __name__ == "__main__":
    unittest.main()
