import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import toml
from tqdm import tqdm

from src.api_llm_modified import OmniRoutersGPT


CONFIG_PATH = Path(__file__).resolve().with_name("config.toml")
config = toml.load(CONFIG_PATH)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run and measure LLM inference with call and cost logging."
    )
    parser.add_argument("--input", required=True, help="Prompt JSON file")
    parser.add_argument(
        "--llm",
        required=True,
        help="Configured model key, currently omnirouters_gpt",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of clues",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Reuse existing successful result entries",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Maximum concurrent API requests (default: 4; use 1 for sequential)",
    )
    return parser.parse_args()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)


def calculate_estimated_cost(usage):
    cost_config = config.get("omnirouters", {}).get("cost", {})
    input_price = cost_config.get("input_usd_per_million", -1.0)
    output_price = cost_config.get("output_usd_per_million", -1.0)

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")

    if (
        input_price <= 0
        or output_price <= 0
        or input_tokens is None
        or output_tokens is None
    ):
        return None

    return (
        input_tokens * input_price / 1_000_000
        + output_tokens * output_price / 1_000_000
    )


def is_rate_limit_error(error):
    if not error:
        return False
    error_text = str(error).lower()
    return (
        "ratelimiterror" in error_text
        or "rate limit" in error_text
        or "429" in error_text
    )


def run_inference(model, prompt):
    call_start = time.monotonic()
    try:
        answers, usage = model(prompt)
        answer = answers[0] if answers else ""
        return answer, usage, None, time.monotonic() - call_start
    except Exception as exc:
        return (
            "",
            {},
            f"{type(exc).__name__}: {exc}",
            time.monotonic() - call_start,
        )


def main():
    args = parse_arguments()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if args.llm not in config["llm"]:
        raise ValueError(
            f"{args.llm} is not configured. Available: {list(config['llm'])}"
        )
    if args.llm != "omnirouters_gpt":
        raise ValueError(
            "main_modified.py currently supports omnirouters_gpt only."
        )

    with input_path.open("r", encoding="utf-8") as handle:
        prompts = json.load(handle)

    if args.limit is not None:
        prompts = dict(list(prompts.items())[:args.limit])

    output_dir = Path(__file__).resolve().with_name("outputs")
    log_dir = Path(__file__).resolve().with_name("logs")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"{args.llm}_({input_path.stem}).json"
    output_path = output_dir / output_name
    log_path = log_dir / f"{args.llm}_({input_path.stem})_calls.jsonl"
    summary_path = log_dir / f"{args.llm}_({input_path.stem})_summary.json"
    csv_path = log_dir / f"{args.llm}_({input_path.stem})_summary.csv"

    results = {}
    if args.cache and output_path.is_file():
        with output_path.open("r", encoding="utf-8") as handle:
            results = json.load(handle)

    previous_summary = {}
    previous_call_entries = []
    if args.cache and summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as handle:
            previous_summary = json.load(handle)
    if args.cache and log_path.is_file():
        with log_path.open("r", encoding="utf-8") as handle:
            previous_call_entries = [
                json.loads(line) for line in handle if line.strip()
            ]

    pending = [
        (clue_id, prompt)
        for clue_id, prompt in prompts.items()
        if not (args.cache and clue_id in results)
    ]
    new_calls = len(pending)
    successful_calls = 0
    failed_calls = 0
    start_time = time.monotonic()

    model = OmniRoutersGPT(args.llm)

    log_mode = "a" if previous_call_entries else "w"
    with log_path.open(log_mode, encoding="utf-8") as log_handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_inference, model, prompt): (
                    len(previous_call_entries) + call_number,
                    clue_id,
                )
                for call_number, (clue_id, prompt) in enumerate(pending, start=1)
            }

            for completed_calls, future in enumerate(
                tqdm(
                    as_completed(futures),
                    desc="Generating measured outputs",
                    total=len(futures),
                ),
                start=1,
            ):
                call_number, clue_id = futures[future]
                answer, usage, call_error, latency = future.result()
                if call_error is None:
                    results[clue_id] = answer
                    successful_calls += 1
                else:
                    failed_calls += 1

                call_input_tokens = usage.get("input_tokens")
                call_output_tokens = usage.get("output_tokens")
                call_cost = calculate_estimated_cost(usage)

                log_entry = {
                    "clue_id": clue_id,
                    "model": config["llm"][args.llm],
                    "provider": "omnirouters",
                    "call_number": call_number,
                    "answer": answer,
                    "input_tokens": call_input_tokens,
                    "output_tokens": call_output_tokens,
                    "estimated_usd": call_cost,
                    "latency_seconds": round(latency, 4),
                    "error": call_error,
                }
                log_handle.write(json.dumps(log_entry) + "\n")
                log_handle.flush()

                if completed_calls % 100 == 0:
                    write_json(output_path, results)

    write_json(output_path, results)

    with log_path.open("r", encoding="utf-8") as handle:
        all_call_entries = [json.loads(line) for line in handle if line.strip()]

    cost_config = config.get("omnirouters", {}).get("cost", {})
    pricing_configured = (
        cost_config.get("input_usd_per_million", -1.0) > 0
        and cost_config.get("output_usd_per_million", -1.0) > 0
    )
    logged_costs = [
        entry["estimated_usd"]
        for entry in all_call_entries
        if entry.get("estimated_usd") is not None
    ]
    all_costs_available = len(logged_costs) == len(all_call_entries)
    total_estimated_cost = sum(logged_costs) if logged_costs else None
    all_input_tokens = sum(
        entry.get("input_tokens") or 0 for entry in all_call_entries
    )
    all_output_tokens = sum(
        entry.get("output_tokens") or 0 for entry in all_call_entries
    )
    cumulative_wall_time = (
        previous_summary.get("wall_clock_seconds", 0)
        + time.monotonic() - start_time
    )
    summary = {
        "model_key": args.llm,
        "model": config["llm"][args.llm],
        "provider": "omnirouters",
        "base_url": config["omnirouters"]["base_url"],
        "clues_requested": len(prompts),
        "completed_outputs": len(results),
        "missing_outputs": max(0, len(prompts) - len(results)),
        "number_of_calls": len(all_call_entries) if all_call_entries else new_calls,
        "successful_calls": sum(
            1 for entry in all_call_entries if not entry.get("error")
        )
        if all_call_entries
        else successful_calls,
        "failed_calls": sum(
            1 for entry in all_call_entries if entry.get("error")
        )
        if all_call_entries
        else failed_calls,
        "input_tokens": all_input_tokens,
        "output_tokens": all_output_tokens,
        "total_tokens": all_input_tokens + all_output_tokens,
        "estimated_api_cost_usd": (
            round(total_estimated_cost, 8)
            if pricing_configured and total_estimated_cost is not None
            else None
        ),
        "cost_status": (
            "estimated"
            if pricing_configured and all_costs_available
            else "partial_estimate_failed_calls_missing_usage"
            if pricing_configured and logged_costs
            else "unavailable_pricing_not_configured"
        ),
        "wall_clock_seconds": round(cumulative_wall_time, 4),
        "last_run_calls": new_calls,
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
