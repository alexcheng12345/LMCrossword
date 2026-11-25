import argparse
import csv
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import toml
from tqdm import tqdm

from main_modified import calculate_estimated_cost, config, run_inference, write_json
from src.api_llm_modified import OmniRoutersGPT


ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "nyt": ROOT / "Data" / "Datasets" / "NYT-Clues" / "NYT_query.json",
    "cryp": ROOT / "Data" / "Datasets" / "Cryptonite-Clues" / "cryptonite_query.json",
    "init": ROOT / "Data" / "Datasets" / "Init-Clues" / "init_query.json",
}
CONFIG_PATH = Path(__file__).resolve().with_name("config.toml")
config = toml.load(CONFIG_PATH)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run B1: repeated identical prompts without feedback, selecting "
            "the first length-compliant answer."
        )
    )
    parser.add_argument("--input", required=True, help="Few-shot prompt JSON file")
    parser.add_argument(
        "--llm",
        default="omnirouters_gpt",
        help="Configured model key (default: omnirouters_gpt)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of clues",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Maximum concurrent clues (default: 4)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=4,
        help="Maximum identical, no-feedback calls per clue (default: 4)",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Reuse completed B1 clue outputs and append to existing logs",
    )
    return parser.parse_args()


def normalize_answer(answer):
    answer = str(answer or "").strip().lower()
    answer = answer.replace("clue: ", "").replace("answer: ", "")
    answer = re.sub(r"<(.*)>", "\n", answer)
    lines = [line for line in answer.split("\n") if line]
    answer = "" if not lines else lines[0]
    return "".join(char for char in answer if char.isalpha())


def get_dataset_name(input_path):
    match = re.search(r"(NYT|Cryp|Init)__fewShot_", input_path.stem, re.IGNORECASE)
    if not match:
        raise ValueError(
            "Input filename must identify the dataset, e.g. NYT__fewShot_5.json."
        )
    return match.group(1).lower()


def run_b1_for_clue(model, clue_id, prompt, expected_length, max_attempts):
    attempts = []
    selected_index = None
    last_success_index = None

    for attempt_number in range(1, max_attempts + 1):
        answer, usage, error, latency = run_inference(model, prompt)
        normalized = normalize_answer(answer)
        length_compliant = error is None and len(normalized) == expected_length
        attempt = {
            "clue_id": clue_id,
            "attempt_number": attempt_number,
            "answer": answer,
            "normalized_answer": normalized,
            "observed_length": len(normalized),
            "expected_length": expected_length,
            "length_compliant": length_compliant,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "estimated_usd": calculate_estimated_cost(usage),
            "latency_seconds": round(latency, 4),
            "error": error,
            "selected_as_final": False,
        }
        attempts.append(attempt)

        if error is None:
            last_success_index = len(attempts) - 1
        if length_compliant:
            selected_index = len(attempts) - 1
            break

    if selected_index is None:
        selected_index = last_success_index

    if selected_index is not None:
        attempts[selected_index]["selected_as_final"] = True
        final_answer = attempts[selected_index]["answer"]
    else:
        final_answer = None

    return clue_id, final_answer, attempts


def main():
    args = parse_arguments()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.llm != "omnirouters_gpt":
        raise ValueError("main_b1.py currently supports omnirouters_gpt only.")

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    dataset_name = get_dataset_name(input_path)
    query_path = DATASETS[dataset_name]

    with input_path.open("r", encoding="utf-8") as handle:
        prompts = json.load(handle)
    with query_path.open("r", encoding="utf-8") as handle:
        query = json.load(handle)

    if args.limit is not None:
        prompts = dict(list(prompts.items())[:args.limit])

    missing_query_ids = [clue_id for clue_id in prompts if clue_id not in query]
    if missing_query_ids:
        raise ValueError(
            f"{len(missing_query_ids)} prompt IDs are missing from {query_path}."
        )

    output_dir = Path(__file__).resolve().with_name("outputs")
    log_dir = Path(__file__).resolve().with_name("logs")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    output_stem = f"{args.llm}_B1_({input_path.stem})"
    output_path = output_dir / f"{output_stem}.json"
    log_path = log_dir / f"{output_stem}_calls.jsonl"
    summary_path = log_dir / f"{output_stem}_summary.json"
    csv_path = log_dir / f"{output_stem}_summary.csv"

    results = {}
    if args.cache and output_path.is_file():
        with output_path.open("r", encoding="utf-8") as handle:
            results = json.load(handle)

    previous_summary = {}
    if args.cache and summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as handle:
            previous_summary = json.load(handle)

    previous_entries = []
    if args.cache and log_path.is_file():
        with log_path.open("r", encoding="utf-8") as handle:
            previous_entries = [json.loads(line) for line in handle if line.strip()]

    pending = [
        (clue_id, prompt, len(normalize_answer(query[clue_id]["answer"])))
        for clue_id, prompt in prompts.items()
        if not (args.cache and clue_id in results)
    ]

    model = OmniRoutersGPT(args.llm) if pending else None
    start_time = time.monotonic()
    log_mode = "a" if previous_entries else "w"

    with log_path.open(log_mode, encoding="utf-8") as log_handle:
        if pending:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        run_b1_for_clue,
                        model,
                        clue_id,
                        prompt,
                        expected_length,
                        args.max_attempts,
                    ): clue_id
                    for clue_id, prompt, expected_length in pending
                }

                next_call_number = len(previous_entries) + 1
                for completed_clues, future in enumerate(
                    tqdm(
                        as_completed(futures),
                        desc="Running B1 prompts",
                        total=len(futures),
                    ),
                    start=1,
                ):
                    clue_id, final_answer, attempts = future.result()
                    if final_answer is not None:
                        results[clue_id] = final_answer

                    for attempt in attempts:
                        attempt.update(
                            {
                                "model": config["llm"][args.llm],
                                "provider": "omnirouters",
                                "call_number": next_call_number,
                            }
                        )
                        next_call_number += 1
                        log_handle.write(json.dumps(attempt) + "\n")
                    log_handle.flush()

                    if completed_clues % 100 == 0:
                        write_json(output_path, results)

    write_json(output_path, results)

    with log_path.open("r", encoding="utf-8") as handle:
        all_entries = [json.loads(line) for line in handle if line.strip()]

    calls_by_clue = Counter(entry["clue_id"] for entry in all_entries)
    retry_calls = sum(max(0, count - 1) for count in calls_by_clue.values())
    valid_final_answers = sum(
        1
        for clue_id, answer in results.items()
        if len(normalize_answer(answer))
        == len(normalize_answer(query[clue_id]["answer"]))
    )

    cost_config = config.get("omnirouters", {}).get("cost", {})
    pricing_configured = (
        cost_config.get("input_usd_per_million", -1.0) > 0
        and cost_config.get("output_usd_per_million", -1.0) > 0
    )
    available_costs = [
        entry["estimated_usd"]
        for entry in all_entries
        if entry.get("estimated_usd") is not None
    ]
    all_costs_available = len(available_costs) == len(all_entries)
    total_cost = sum(available_costs) if available_costs else None
    input_tokens = sum(entry.get("input_tokens") or 0 for entry in all_entries)
    output_tokens = sum(entry.get("output_tokens") or 0 for entry in all_entries)
    failed_calls = sum(1 for entry in all_entries if entry.get("error"))
    successful_calls = len(all_entries) - failed_calls
    previous_calls = len(previous_entries)
    call_count = len(all_entries)
    cumulative_wall_time = (
        previous_summary.get("wall_clock_seconds", 0)
        + time.monotonic()
        - start_time
    )

    summary = {
        "condition": "B1",
        "model_key": args.llm,
        "model": config["llm"][args.llm],
        "provider": "omnirouters",
        "base_url": config["omnirouters"]["base_url"],
        "dataset": dataset_name,
        "max_attempts_per_clue": args.max_attempts,
        "clues_requested": len(prompts),
        "completed_outputs": len(results),
        "missing_outputs": max(0, len(prompts) - len(results)),
        "length_compliant_final_answers": valid_final_answers,
        "length_unresolved_final_answers": len(results) - valid_final_answers,
        "number_of_calls": call_count,
        "last_run_calls": call_count - previous_calls,
        "successful_calls": successful_calls,
        "failed_calls": failed_calls,
        "retry_calls": retry_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_api_cost_usd": (
            round(total_cost, 8)
            if pricing_configured and total_cost is not None
            else None
        ),
        "cost_status": (
            "estimated"
            if pricing_configured and all_costs_available
            else "partial_estimate_failed_calls_missing_usage"
            if pricing_configured and available_costs
            else "unavailable_pricing_not_configured"
        ),
        "wall_clock_seconds": round(cumulative_wall_time, 4),
        "output_file": str(output_path),
        "call_log_file": str(log_path),
    }
    write_json(summary_path, summary)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
