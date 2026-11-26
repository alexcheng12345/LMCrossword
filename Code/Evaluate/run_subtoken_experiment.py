import argparse
import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import toml
from openai import APIError, OpenAI
from tqdm import tqdm


COUNT_ONLY_SYSTEM_PROMPT = (
    "Count the characters in the target word. Return only the count as a "
    "number. Do not include an explanation, label, punctuation, or any other text."
)
DEFAULT_MAX_TOKENS = 128
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_PATH = ROOT / "Code" / "Prompts" / "prompts" / "counting_fewShot_5.json"
DEFAULT_DATASET_PATH = (
    ROOT / "Data" / "Datasets" / "Sub-token Counting" / "counting_dataset.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "Result" / "Sub-token Counting"
INFERENCE_CONFIG_PATH = ROOT / "Code" / "LLM-Inference" / "config.toml"
WORD_CATEGORIES = {
    "common": "word_frequency",
    "mid": "word_frequency",
    "rare": "word_frequency",
    "vocab": "word_type",
    "gibberish": "word_type",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run and evaluate the sub-token character-counting experiment."
    )
    parser.add_argument(
        "--provider",
        choices=("omnirouters", "st4011"),
        default="omnirouters",
        help="OpenAI-compatible API provider (default: omnirouters)",
    )
    parser.add_argument("--model", help="Override the provider's configured model")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="JSON prompts generated for the counting experiment",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="JSON mapping prompt IDs to target words",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for raw call logs and evaluation CSV/JSON files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of words to run",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Maximum concurrent API requests (default: 4)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum completion tokens per request (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful calls in an existing matching JSONL log",
    )
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_provider_settings(provider):
    if provider == "omnirouters":
        config = toml.load(INFERENCE_CONFIG_PATH)
        settings = config["omnirouters"]
        return {
            "base_url": settings["base_url"],
            "api_key_env": settings["api_key_env"],
            "model": settings["model"],
        }
    return {
        "base_url": "https://st4011.loping151.net/v1",
        "api_key_env": "ST4011_API_KEY",
        "model": "deepseek-flash",
    }


def create_client(settings):
    api_key = os.getenv(settings["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"Set the {settings['api_key_env']} environment variable before "
            "running the sub-token experiment."
        )
    return OpenAI(api_key=api_key, base_url=settings["base_url"])


def category_for_id(item_id):
    category = item_id.split("_", 1)[0]
    if category not in WORD_CATEGORIES:
        raise ValueError(f"Unrecognized sub-token category in item ID: {item_id}")
    return category


def validate_experiment_design(dataset):
    category_counts = {category: 0 for category in WORD_CATEGORIES}
    length_distributions = {"vocab": {}, "gibberish": {}}
    for item_id, word in dataset.items():
        category = category_for_id(item_id)
        category_counts[category] += 1
        if category in length_distributions:
            length = len(word)
            length_distributions[category][length] = (
                length_distributions[category].get(length, 0) + 1
            )

    warnings = [
        f"{category} has {category_counts[category]} items, not the planned 1,000."
        for category in ("common", "mid", "rare")
        if category_counts[category] != 1000
    ]
    if length_distributions["vocab"] != length_distributions["gibberish"]:
        warnings.append(
            "Vocabulary and gibberish controls do not have matching word-length "
            "distributions."
        )
    return warnings


def parse_predicted_count(response):
    numbers = re.findall(r"\b\d+\b", str(response or ""))
    return int(numbers[-1]) if numbers else None


def build_counting_messages(prompt):
    messages = [dict(message) for message in prompt]
    for message in messages:
        if message.get("role") == "system":
            message["content"] = COUNT_ONLY_SYSTEM_PROMPT
            break
    else:
        messages.insert(
            0,
            {"role": "system", "content": COUNT_ONLY_SYSTEM_PROMPT},
        )
    return messages


def run_count_request(client, model_name, item_id, prompt, max_tokens):
    entry = {"item_id": item_id, "model": model_name}
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=build_counting_messages(prompt),
            max_tokens=max_tokens,
            temperature=1,
        )
        if not completion.choices:
            raise ValueError("The API returned no completion choices.")

        message = completion.choices[0].message
        response = message.content
        if not isinstance(response, str) or not response.strip():
            entry.update(
                {
                    "response": "",
                    "error": "EmptyResponse: API returned no text content.",
                }
            )
            return entry

        usage = getattr(completion, "usage", None)
        entry.update(
            {
                "response": response,
                "input_tokens": (
                    getattr(usage, "prompt_tokens", None)
                    if usage is not None
                    else None
                ),
                "output_tokens": (
                    getattr(usage, "completion_tokens", None)
                    if usage is not None
                    else None
                ),
            }
        )
    except APIError as exc:
        entry.update(
            {
                "response": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    except ValueError as exc:
        entry.update({"response": "", "error": f"EmptyResponse: {exc}"})
    return entry


def evaluate_predictions(dataset, predictions, errors=None):
    errors = errors or {}
    details = []
    for item_id, word in dataset.items():
        category = category_for_id(item_id)
        expected_count = len(word)
        response = predictions.get(item_id, "")
        predicted_count = parse_predicted_count(response)
        details.append(
            {
                "item_id": item_id,
                "category": category,
                "word": word,
                "word_length": expected_count,
                "expected_count": expected_count,
                "predicted_count": predicted_count,
                "correct": predicted_count == expected_count,
                "response": response,
                "error": errors.get(item_id, ""),
            }
        )

    summary_rows = []

    def add_summary(group_type, group, rows):
        correct = sum(row["correct"] for row in rows)
        total = len(rows)
        summary_rows.append(
            {
                "group_type": group_type,
                "group": group,
                "items": total,
                "correct": correct,
                "accuracy_pct": round(correct / total * 100, 2) if total else None,
            }
        )

    add_summary("overall", "all", details)
    for category, group_type in WORD_CATEGORIES.items():
        add_summary(
            group_type,
            category,
            [row for row in details if row["category"] == category],
        )
    for length in sorted({row["word_length"] for row in details}):
        add_summary(
            "word_length",
            str(length),
            [row for row in details if row["word_length"] == length],
        )
    return details, summary_rows


def read_call_log(path):
    entries = []
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            entries = [json.loads(line) for line in handle if line.strip()]
    return entries


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_arguments()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if not args.prompts.is_file():
        raise FileNotFoundError(args.prompts)
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)

    all_prompts = load_json(args.prompts)
    all_words = load_json(args.dataset)
    if not isinstance(all_prompts, dict) or not isinstance(all_words, dict):
        raise ValueError("Prompts and dataset must both be JSON objects.")
    if not set(all_prompts).issubset(all_words):
        raise ValueError("Some prompt IDs are missing from the word dataset.")
    for warning in validate_experiment_design(all_words):
        print(f"WARNING: {warning}")
    if args.limit is not None:
        all_prompts = dict(list(all_prompts.items())[: args.limit])
    words = {item_id: all_words[item_id] for item_id in all_prompts}
    for item_id in words:
        category_for_id(item_id)

    settings = get_provider_settings(args.provider)
    model_name = args.model or settings["model"]
    safe_model_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
    output_stem = f"{args.provider}_{safe_model_name}_subtoken"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    call_log_path = args.output_dir / f"{output_stem}_calls.jsonl"
    predictions_path = args.output_dir / f"{output_stem}_predictions.json"
    details_path = args.output_dir / f"{output_stem}_details.csv"
    summary_path = args.output_dir / f"{output_stem}_summary.csv"

    if call_log_path.exists() and not args.resume:
        raise FileExistsError(
            f"{call_log_path} already exists. Use --resume to continue it."
        )
    previous_entries = read_call_log(call_log_path) if args.resume else []
    predictions = {
        entry["item_id"]: entry["response"]
        for entry in previous_entries
        if (
            entry.get("item_id") in all_prompts
            and not entry.get("error")
            and isinstance(entry.get("response"), str)
            and entry["response"].strip()
        )
    }
    errors = {}
    for entry in previous_entries:
        item_id = entry.get("item_id")
        if item_id not in all_prompts or item_id in predictions:
            continue
        if entry.get("error"):
            errors[item_id] = entry["error"]
        elif not str(entry.get("response") or "").strip():
            errors[item_id] = "EmptyResponse: Previous call returned no text."

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")

    pending = [
        (item_id, prompt)
        for item_id, prompt in all_prompts.items()
        if item_id not in predictions
    ]
    completed_count = len(all_prompts) - len(pending)

    client = create_client(settings)
    log_mode = "a" if args.resume and call_log_path.exists() else "w"
    with call_log_path.open(log_mode, encoding="utf-8") as log_handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_count_request,
                    client,
                    model_name,
                    item_id,
                    prompt,
                    args.max_tokens,
                ): item_id
                for item_id, prompt in pending
            }
            for future in tqdm(
                as_completed(futures),
                total=len(all_prompts),
                initial=completed_count,
                desc="Running sub-token test",
                unit="word",
            ):
                item_id = futures[future]
                entry = future.result()
                log_handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
                log_handle.flush()
                if entry.get("error"):
                    errors[item_id] = entry["error"]
                else:
                    predictions[item_id] = entry["response"]
                    errors.pop(item_id, None)

    details, summary_rows = evaluate_predictions(words, predictions, errors)
    with predictions_path.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2, ensure_ascii=True)
    write_csv(
        details_path,
        details,
        [
            "item_id",
            "category",
            "word",
            "word_length",
            "expected_count",
            "predicted_count",
            "correct",
            "response",
            "error",
        ],
    )
    write_csv(
        summary_path,
        summary_rows,
        ["group_type", "group", "items", "correct", "accuracy_pct"],
    )
    print(f"Raw call log: {call_log_path}")
    print(f"Per-word results: {details_path}")
    print(f"Summary: {summary_path}")
    print(f"API failures: {len(errors)}")


if __name__ == "__main__":
    main()
