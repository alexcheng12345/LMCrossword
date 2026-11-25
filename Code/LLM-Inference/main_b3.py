import argparse
import csv
import json
import math
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Event, Lock

import toml
from tqdm import tqdm

from main_b2 import normalize_answer, run_b2_for_clue
from main_modified import (
    calculate_estimated_cost,
    is_rate_limit_error,
    run_inference,
    write_json,
)
from src.api_llm_modified import OmniRoutersGPT
from src.wordnet_dictionary import WordNetDictionary


CONFIG_PATH = Path(__file__).resolve().with_name("config.toml")
config = toml.load(CONFIG_PATH)
MAX_SELECTION_ATTEMPTS = 2
MAX_SEARCH_TERMS = 8
MAX_CANDIDATES = 30


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run B3: retrieve length-matched WordNet candidates and ask the "
            "model to select one, with a bounded retry and B2 fallback."
        )
    )
    parser.add_argument("--input", required=True, help="Few-shot prompt JSON file")
    parser.add_argument(
        "--llm",
        default="omnirouters_gpt",
        help="Configured model key (default: omnirouters_gpt)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Maximum concurrent clues (default: 4)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=4,
        help="Maximum B2 fallback calls per clue (default: 4)",
    )
    parser.add_argument(
        "--dictionary-dir",
        type=Path,
        default=None,
        help="Optional NLTK data directory containing the local WordNet corpus",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Reuse completed outputs and append new calls to existing logs",
    )
    return parser.parse_args()


def get_dataset_name(input_path):
    match = re.search(r"(NYT|Cryp|Init)__fewShot_", input_path.stem, re.IGNORECASE)
    if not match:
        raise ValueError(
            "Input filename must identify the dataset, e.g. NYT__fewShot_5.json."
        )
    return match.group(1).lower()


def extract_clue_and_length(prompt):
    user_messages = [
        message.get("content", "")
        for message in prompt
        if message.get("role") == "user"
    ]
    pattern = re.compile(
        r"Clue:\s*(.*?)\s*\((\d+)\)\s*//\s*Answer:\s*",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for content in reversed(user_messages):
        markers = list(re.finditer(r"\bClue:", content, flags=re.IGNORECASE))
        for marker in reversed(markers):
            match = pattern.fullmatch(content[marker.start() :].strip())
            if match:
                return match.group(1).strip(), int(match.group(2))
    raise ValueError(
        "Could not read the target clue and numeric enumeration from the "
        "prompt's final 'Clue: ... (N)// Answer:' line."
    )


def build_search_prompt(clue, required_length):
    return [
        {
            "role": "system",
            "content": (
                "You are a crossword clue query-expansion assistant. Do not "
                "solve the clue or suggest answer candidates. Return only a "
                "JSON object with a `search_terms` array of 3 to 8 short "
                "synonyms, definitions, or paraphrases that can be used to "
                "search a dictionary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Clue: {clue}\nRequired answer length: {required_length} "
                "letters. Return useful dictionary search terms."
            ),
        },
    ]


def parse_search_terms(response):
    response_text = str(response or "").strip()
    start = response_text.find("{")
    end = response_text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Search-term response did not contain a JSON object.")

    payload = json.loads(response_text[start : end + 1])
    terms = payload.get("search_terms")
    if isinstance(terms, str):
        terms = [terms]
    if not isinstance(terms, list):
        raise ValueError("Search-term JSON must contain a `search_terms` list.")

    normalized_terms = []
    seen = set()
    for term in terms:
        if not isinstance(term, str) or not term.strip():
            continue
        cleaned = re.sub(r"\s+", " ", term).strip()
        key = cleaned.casefold()
        if key not in seen:
            seen.add(key)
            normalized_terms.append(cleaned)
        if len(normalized_terms) == MAX_SEARCH_TERMS:
            break
    if not normalized_terms:
        raise ValueError("Search-term response did not contain usable terms.")
    return normalized_terms


def build_candidate_prompt(prompt, clue, required_length, candidates, retry=False):
    messages = [dict(message) for message in prompt]
    candidate_lines = []
    candidate_ids = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = f"C{index:02d}"
        candidate_ids.append(candidate_id)
        definitions = "; ".join(candidate["definitions"])
        if len(definitions) > 220:
            definitions = definitions[:217] + "..."
        candidate_lines.append(
            f"{candidate_id}: {candidate['word'].upper()} — "
            f"{definitions or 'No definition available'}"
        )

    instruction = (
        "Choose the single best candidate for the clue. Return only its ID "
        "(for example, C01). If none is suitable, return "
        "NO_VALID_CANDIDATE. Do not invent an answer."
    )
    if retry:
        instruction = (
            "This is the final selection attempt. Choose the best available "
            "candidate from the list, even if uncertain. Return only its ID. "
            "Do not return NO_VALID_CANDIDATE and do not invent an answer."
        )

    content = (
        f"Target clue: {clue} ({required_length} letters)\n\n"
        "Dictionary candidates:\n"
        + "\n".join(candidate_lines)
        + "\n\n"
        + instruction
    )
    user_message = next(
        (message for message in reversed(messages) if message.get("role") == "user"),
        None,
    )
    if user_message is None:
        messages.append({"role": "user", "content": content})
    else:
        user_message["content"] = f"{user_message.get('content', '')}\n\n{content}"
    return messages, candidate_ids


def parse_selection(response, candidate_ids):
    response_text = str(response or "").strip().upper()
    if re.search(r"NO[\s_-]*VALID[\s_-]*CANDIDATE", response_text):
        return None

    matches = re.findall(r"\bC(\d{1,2})\b", response_text)
    if not matches:
        return None
    candidate_id = f"C{int(matches[-1]):02d}"
    return candidate_id if candidate_id in candidate_ids else None


def run_b3_for_clue(
    model,
    clue_id,
    prompt,
    clue,
    required_length,
    dictionary,
    max_fallback_attempts,
    stop_event=None,
    progress_callback=None,
):
    if stop_event is None:
        stop_event = Event()

    call_entries = []
    candidates = []
    search_terms = []
    selected_candidate_id = None
    fallback_reason = None

    def request(stage, messages, attempt_number=None):
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "start",
                    "clue_id": clue_id,
                    "stage": stage,
                    "attempt_number": attempt_number,
                }
            )
        response, usage, error, latency = run_inference(model, messages)
        return response, usage, error, latency

    def complete_call(entry):
        call_entries.append(entry)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "finish",
                    "clue_id": clue_id,
                    "stage": entry["stage"],
                    "attempt_number": entry.get("attempt_number"),
                    "error": entry.get("error"),
                    "latency_seconds": entry.get("latency_seconds"),
                    "entry": dict(entry),
                }
            )

    if stop_event.is_set():
        return clue_id, None, call_entries, True

    search_response, usage, error, latency = request(
        "query_expansion",
        build_search_prompt(clue, required_length),
    )
    search_entry = {
        "clue_id": clue_id,
        "stage": "query_expansion",
        "answer": search_response,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "estimated_usd": calculate_estimated_cost(usage),
        "latency_seconds": round(latency, 4),
        "error": error,
    }
    complete_call(search_entry)

    if is_rate_limit_error(error):
        stop_event.set()
        return clue_id, None, call_entries, True
    if stop_event.is_set():
        return clue_id, None, call_entries, True
    if error:
        fallback_reason = "query_expansion_api_error"
    else:
        try:
            search_terms = parse_search_terms(search_response)
        except (json.JSONDecodeError, ValueError) as exc:
            search_entry["parse_error"] = str(exc)
            fallback_reason = "invalid_query_expansion"

    if fallback_reason is None:
        candidates = dictionary.retrieve_candidates(
            clue,
            search_terms,
            required_length,
            limit=MAX_CANDIDATES,
        )
        if not candidates:
            fallback_reason = "no_length_matched_dictionary_candidates"

    if candidates and fallback_reason is None:
        for selection_attempt in range(1, MAX_SELECTION_ATTEMPTS + 1):
            if stop_event.is_set():
                return clue_id, None, call_entries, True
            messages, candidate_ids = build_candidate_prompt(
                prompt,
                clue,
                required_length,
                candidates,
                retry=selection_attempt > 1,
            )
            selection_stage = (
                "candidate_selection"
                if selection_attempt == 1
                else "candidate_selection_retry"
            )
            response, usage, error, latency = request(
                selection_stage,
                messages,
                attempt_number=selection_attempt,
            )
            entry = {
                "clue_id": clue_id,
                "stage": (
                    "candidate_selection"
                    if selection_attempt == 1
                    else "candidate_selection_retry"
                ),
                "attempt_number": selection_attempt,
                "answer": response,
                "search_terms": search_terms,
                "candidate_words": [candidate["word"] for candidate in candidates],
                "candidate_count": len(candidates),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "estimated_usd": calculate_estimated_cost(usage),
                "latency_seconds": round(latency, 4),
                "error": error,
            }
            if is_rate_limit_error(error):
                stop_event.set()
                complete_call(entry)
                return clue_id, None, call_entries, True
            if error is None:
                selected_candidate_id = parse_selection(response, candidate_ids)
                if selected_candidate_id is not None:
                    selected_index = int(selected_candidate_id[1:]) - 1
                    selected_word = candidates[selected_index]["word"]
                    if len(normalize_answer(selected_word)) == required_length:
                        entry["selected_candidate_id"] = selected_candidate_id
                        final_answer = selected_word.upper()
                        entry["b3_decision"] = {
                            "candidate_count": len(candidates),
                            "selected_candidate_id": selected_candidate_id,
                            "b2_fallback_used": False,
                            "fallback_reason": None,
                        }
                        complete_call(entry)
                        return clue_id, final_answer, call_entries, False
            complete_call(entry)
            if stop_event.is_set():
                return clue_id, None, call_entries, True
            if selection_attempt == MAX_SELECTION_ATTEMPTS:
                fallback_reason = "candidate_selection_failed_after_retry"

    def report_fallback_progress(event):
        if event["event"] == "finish":
            event["entry"].update(
                {
                    "stage": "b2_fallback",
                    "candidate_count": len(candidates),
                    "fallback_reason": fallback_reason,
                }
            )
        if progress_callback is not None:
            progress_callback(event)

    _, fallback_answer, fallback_attempts = run_b2_for_clue(
        model,
        clue_id,
        prompt,
        required_length,
        max_fallback_attempts,
        stop_on_rate_limit=True,
        stop_event=stop_event,
        progress_callback=report_fallback_progress,
    )
    for fallback_entry in fallback_attempts:
        fallback_entry.update(
            {
                "stage": "b2_fallback",
                "candidate_count": len(candidates),
                "fallback_reason": fallback_reason,
            }
        )
        call_entries.append(fallback_entry)

    if stop_event.is_set():
        if call_entries:
            call_entries[-1]["b3_decision"] = {
                "candidate_count": len(candidates),
                "selected_candidate_id": None,
                "b2_fallback_used": True,
                "fallback_reason": fallback_reason,
            }
        return clue_id, None, call_entries, True

    if call_entries:
        call_entries[-1]["b3_decision"] = {
            "candidate_count": len(candidates),
            "selected_candidate_id": None,
            "b2_fallback_used": True,
            "fallback_reason": fallback_reason,
        }
    return clue_id, fallback_answer, call_entries, False


def main():
    args = parse_arguments()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        raise ValueError("--request-timeout must be a positive finite number")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.llm not in config["llm"]:
        raise ValueError(
            f"{args.llm} is not configured. Available: {list(config['llm'])}"
        )
    if args.llm != "omnirouters_gpt":
        raise ValueError("main_b3.py currently supports omnirouters_gpt only.")

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    dataset_name = get_dataset_name(input_path)
    with input_path.open("r", encoding="utf-8") as handle:
        prompts = json.load(handle)
    if not isinstance(prompts, dict) or not prompts:
        raise ValueError("Prompt JSON must contain at least one clue.")
    if args.limit is not None:
        prompts = dict(list(prompts.items())[: args.limit])

    parsed_prompts = {}
    for clue_id, prompt in prompts.items():
        clue, required_length = extract_clue_and_length(prompt)
        parsed_prompts[clue_id] = (prompt, clue, required_length)

    dictionary = WordNetDictionary(
        {required_length for _, _, required_length in parsed_prompts.values()},
        data_dir=args.dictionary_dir,
    )

    output_dir = Path(__file__).resolve().with_name("outputs")
    log_dir = Path(__file__).resolve().with_name("logs")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    output_stem = f"{args.llm}_B3_({input_path.stem})"
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
        (clue_id, *parsed_prompts[clue_id])
        for clue_id in parsed_prompts
        if not (args.cache and clue_id in results)
    ]
    model = (
        OmniRoutersGPT(
            args.llm,
            max_retries=0,
            timeout=args.request_timeout,
        )
        if pending
        else None
    )
    stop_event = Event()
    start_time = time.monotonic()
    log_mode = "a" if previous_entries else "w"

    with log_path.open(log_mode, encoding="utf-8") as log_handle:
        if pending:
            progress_lock = Lock()
            active_requests = set()
            failed_requests = 0
            last_request = "waiting for API calls"
            completed_clues = 0
            next_call_number = len(previous_entries) + 1
            run_decisions = {}

            with tqdm(
                total=None,
                desc="B3 API calls",
                unit="call",
                dynamic_ncols=True,
            ) as progress:

                def report_progress(event):
                    nonlocal failed_requests, last_request, next_call_number
                    request_key = (
                        event["clue_id"],
                        event["stage"],
                        event.get("attempt_number"),
                    )
                    with progress_lock:
                        if event["event"] == "start":
                            active_requests.add(request_key)
                            last_request = (
                                f'{event["clue_id"]}:{event["stage"]} '
                                "in progress"
                            )
                        else:
                            active_requests.discard(request_key)
                            progress.update(1)
                            if event.get("error"):
                                failed_requests += 1
                                last_request = (
                                    f'{event["clue_id"]}:{event["stage"]} '
                                    "failed"
                                )
                            else:
                                last_request = (
                                    f'{event["clue_id"]}:{event["stage"]} '
                                    "completed"
                                )
                            entry = event.get("entry")
                            if entry is not None:
                                entry.update(
                                    {
                                        "model": config["llm"][args.llm],
                                        "provider": "omnirouters",
                                        "call_number": next_call_number,
                                    }
                                )
                                next_call_number += 1
                                log_handle.write(json.dumps(entry) + "\n")
                                log_handle.flush()

                        progress.set_postfix(
                            clues=f"{completed_clues}/{len(pending)}",
                            active=len(active_requests),
                            last=last_request,
                            failed=failed_requests,
                            refresh=True,
                        )

                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = {
                        executor.submit(
                            run_b3_for_clue,
                            model,
                            clue_id,
                            prompt,
                            clue,
                            required_length,
                            dictionary,
                            args.max_attempts,
                            stop_event,
                            report_progress,
                        ): clue_id
                        for clue_id, prompt, clue, required_length in pending
                    }

                    for future in as_completed(futures):
                        clue_id, final_answer, entries, _ = future.result()
                        if final_answer is not None:
                            results[clue_id] = final_answer

                        for entry in entries:
                            decision = entry.get("b3_decision")
                            if decision is not None:
                                run_decisions[clue_id] = decision

                        with progress_lock:
                            completed_clues += 1
                            progress.set_postfix(
                                clues=f"{completed_clues}/{len(pending)}",
                                active=len(active_requests),
                                last=last_request,
                                failed=failed_requests,
                                refresh=True,
                            )

                        if completed_clues % 100 == 0:
                            write_json(output_path, results)

    write_json(output_path, results)
    with log_path.open("r", encoding="utf-8") as handle:
        all_entries = [json.loads(line) for line in handle if line.strip()]

    decisions_by_clue = {
        entry["clue_id"]: entry["b3_decision"]
        for entry in all_entries
        if "b3_decision" in entry
    }
    decisions_by_clue.update(run_decisions if pending else {})
    decisions = list(decisions_by_clue.values())
    calls_by_clue = Counter(entry["clue_id"] for entry in all_entries)
    retry_calls = sum(max(0, count - 1) for count in calls_by_clue.values())
    candidate_count_total = sum(
        decision["candidate_count"] for decision in decisions
    )
    clue_lengths = {
        clue_id: parsed_prompts[clue_id][2] for clue_id in parsed_prompts
    }
    length_compliant = sum(
        1
        for clue_id, answer in results.items()
        if clue_id in clue_lengths
        and len(normalize_answer(answer)) == clue_lengths[clue_id]
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
    previous_calls = len(previous_entries)
    stopped_on_rate_limit = stop_event.is_set()

    summary = {
        "condition": "B3",
        "model_key": args.llm,
        "model": config["llm"][args.llm],
        "provider": "omnirouters",
        "base_url": config["omnirouters"]["base_url"],
        "dataset": dataset_name,
        "request_timeout_seconds": args.request_timeout,
        "run_status": (
            "stopped_on_rate_limit" if stopped_on_rate_limit else "completed"
        ),
        "dictionary": "WordNet (local NLTK corpus)",
        "maximum_candidates": MAX_CANDIDATES,
        "selection_attempts": MAX_SELECTION_ATTEMPTS,
        "b2_fallback_max_attempts": args.max_attempts,
        "clues_requested": len(prompts),
        "completed_outputs": len(results),
        "missing_outputs": max(0, len(prompts) - len(results)),
        "length_compliant_final_answers": length_compliant,
        "length_unresolved_final_answers": len(results) - length_compliant,
        "number_of_calls": len(all_entries),
        "last_run_calls": len(all_entries) - previous_calls,
        "successful_calls": sum(
            1 for entry in all_entries if not entry.get("error")
        ),
        "failed_calls": sum(1 for entry in all_entries if entry.get("error")),
        "retry_calls": retry_calls,
        "query_expansion_calls": sum(
            1 for entry in all_entries if entry.get("stage") == "query_expansion"
        ),
        "selection_retry_calls": sum(
            1
            for entry in all_entries
            if entry.get("stage") == "candidate_selection_retry"
        ),
        "b2_fallback_clues": sum(
            1 for decision in decisions if decision["b2_fallback_used"]
        ),
        "b2_fallback_calls": sum(
            1 for entry in all_entries if entry.get("stage") == "b2_fallback"
        ),
        "mean_candidates_per_completed_clue": round(
            candidate_count_total / len(decisions), 2
        )
        if decisions
        else 0.0,
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
        "wall_clock_seconds": round(
            previous_summary.get("wall_clock_seconds", 0)
            + time.monotonic()
            - start_time,
            4,
        ),
        "output_file": str(output_path),
        "call_log_file": str(log_path),
    }
    write_json(summary_path, summary)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(json.dumps(summary, indent=2))
    if stopped_on_rate_limit:
        raise SystemExit(
            "B3 stopped after an API rate-limit response (HTTP 429). "
            "Completed outputs and call logs were saved; resume later with "
            "--cache and fewer workers."
        )


if __name__ == "__main__":
    main()
