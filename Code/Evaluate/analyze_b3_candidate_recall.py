import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUERY_PATH = ROOT / "Data" / "Datasets" / "NYT-Clues" / "NYT_query.json"
DEFAULT_CALL_LOG = (
    ROOT
    / "Code"
    / "LLM-Inference"
    / "logs"
    / "omnirouters_gpt_B3_(NYT__fewShot_5)_calls.jsonl"
)
DEFAULT_OUTPUT_DIR = ROOT / "Result"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Measure whether B3's WordNet candidate lists contain NYT answers."
    )
    parser.add_argument(
        "--query",
        type=Path,
        default=DEFAULT_QUERY_PATH,
        help="NYT query JSON containing clue IDs and correct answers",
    )
    parser.add_argument(
        "--call-log",
        type=Path,
        default=DEFAULT_CALL_LOG,
        help="B3 JSONL call log containing candidate_words",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for summary and per-clue CSV/JSON results",
    )
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_answer(answer):
    return "".join(char for char in str(answer or "").strip().lower() if char.isalpha())


def analyze_candidate_recall(query, call_entries):
    latest_attempts = {}

    for entry in call_entries:
        clue_id = entry.get("clue_id")
        if clue_id not in query:
            raise ValueError(f"B3 call log contains unknown clue ID: {clue_id}")

        attempt = latest_attempts.setdefault(
            clue_id,
            {
                "candidate_words": [],
                "fallback_reason": None,
            },
        )
        if entry.get("stage") == "query_expansion":
            attempt = {
                "candidate_words": [],
                "fallback_reason": None,
            }
            latest_attempts[clue_id] = attempt

        candidates = entry.get("candidate_words")
        if candidates is not None:
            if not isinstance(candidates, list):
                raise ValueError(
                    f"candidate_words for clue {clue_id} must be a JSON list."
                )
            candidate_words = [str(word) for word in candidates]
            if candidate_words:
                previous = attempt["candidate_words"]
                if previous and previous != candidate_words:
                    raise ValueError(
                        "B3 logged inconsistent candidate lists within the latest "
                        f"attempt for clue {clue_id}."
                    )
                attempt["candidate_words"] = candidate_words

        decision = entry.get("b3_decision")
        if isinstance(decision, dict):
            attempt["fallback_reason"] = decision.get("fallback_reason")
        elif entry.get("fallback_reason") is not None:
            attempt["fallback_reason"] = entry["fallback_reason"]

    detail_rows = []
    hits = 0
    candidate_total = 0
    rank_counts = Counter()
    no_candidate_reasons = Counter()

    for clue_id, record in query.items():
        answer = normalize_answer(record.get("answer"))
        attempt = latest_attempts.get(
            clue_id,
            {"candidate_words": [], "fallback_reason": None},
        )
        candidates = attempt["candidate_words"]
        normalized_candidates = [normalize_answer(word) for word in candidates]
        try:
            candidate_rank = normalized_candidates.index(answer) + 1
        except ValueError:
            candidate_rank = None

        has_candidates = bool(candidates)
        hit = candidate_rank is not None
        if has_candidates:
            candidate_total += len(candidates)
            if hit:
                hits += 1
                rank_counts[candidate_rank] += 1
        else:
            reason = attempt["fallback_reason"] or "no_candidate_list_logged"
            no_candidate_reasons[reason] += 1

        detail_rows.append(
            {
                "clue_id": clue_id,
                "clue": record.get("clue", ""),
                "correct_answer": record.get("answer", ""),
                "candidate_list_size": len(candidates),
                "answer_in_candidates": hit,
                "candidate_rank": candidate_rank or "",
                "fallback_reason": attempt["fallback_reason"] or "",
                "candidate_words": ";".join(candidates),
            }
        )

    total_clues = len(query)
    listed_clues = sum(
        bool(latest_attempts.get(clue_id, {}).get("candidate_words"))
        for clue_id in query
    )
    no_candidate_clues = total_clues - listed_clues
    summary = {
        "dataset_clues": total_clues,
        "candidate_lists_found": listed_clues,
        "clues_without_candidates": no_candidate_clues,
        "candidate_hits": hits,
        "candidate_misses_with_list": listed_clues - hits,
        "overall_recall_at_30_pct": round(hits / total_clues * 100, 2)
        if total_clues
        else None,
        "recall_at_30_when_candidates_exist_pct": round(hits / listed_clues * 100, 2)
        if listed_clues
        else None,
        "mean_candidates_per_list": round(candidate_total / listed_clues, 2)
        if listed_clues
        else None,
        "hits_at_rank_1": rank_counts[1],
        "hits_in_top_5": sum(count for rank, count in rank_counts.items() if rank <= 5),
        "hits_in_top_10": sum(
            count for rank, count in rank_counts.items() if rank <= 10
        ),
        "hits_in_top_30": hits,
        "no_candidate_fallback_reasons": dict(sorted(no_candidate_reasons.items())),
    }
    return summary, detail_rows


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_arguments()
    if not args.query.is_file():
        raise FileNotFoundError(args.query)
    if not args.call_log.is_file():
        raise FileNotFoundError(args.call_log)

    query = load_json(args.query)
    call_entries = load_jsonl(args.call_log)
    if not isinstance(query, dict) or not query:
        raise ValueError("Query JSON must contain a non-empty object.")

    summary, detail_rows = analyze_candidate_recall(query, call_entries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "b3_candidate_recall_nyt"
    summary_json_path = args.output_dir / f"{stem}_summary.json"
    summary_csv_path = args.output_dir / f"{stem}_summary.csv"
    details_csv_path = args.output_dir / f"{stem}_by_clue.csv"

    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_csv(summary_csv_path, [summary], list(summary))
    write_csv(
        details_csv_path,
        detail_rows,
        [
            "clue_id",
            "clue",
            "correct_answer",
            "candidate_list_size",
            "answer_in_candidates",
            "candidate_rank",
            "fallback_reason",
            "candidate_words",
        ],
    )

    print(json.dumps(summary, indent=2))
    print(f"Summary JSON: {summary_json_path}")
    print(f"Per-clue results: {details_csv_path}")


if __name__ == "__main__":
    main()
