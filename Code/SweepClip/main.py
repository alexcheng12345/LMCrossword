import argparse
import json
import os
import re

from utils.dataset import Dataset
from utils.model import APIModel, GPT, LLaMA


def normalize_answer(answer):
    return "".join(character for character in str(answer).lower() if character.isalpha())


def load_final_answers(output_file):
    if not os.path.exists(output_file):
        return {}
    with open(output_file, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines()]

    section_indexes = [
        index for index, line in enumerate(lines)
        if line == "------ Current Answers --------"
    ]
    if not section_indexes:
        return {}

    start = section_indexes[-1] + 1
    answers = {}
    for line in lines[start:]:
        if line.startswith("-") or not line:
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        answers[key.strip()] = value.strip()
    return answers


def answer_metrics(dataset, indices, output_directory):
    exact_match_count = 0
    length_violation_count = 0
    compared_count = 0
    clue_count = 0
    indices = list(indices)

    for index in indices:
        crossword = dataset[index]
        clue_count += len(crossword.clues)
        output_file = os.path.join(output_directory, f"Crossword_{crossword.number}")
        predictions = (
            crossword.current_answers
            if crossword.current_answers
            else load_final_answers(output_file)
        )
        for clue_key, clue in crossword.clues.items():
            prediction = predictions.get(clue_key)
            if prediction is None:
                continue
            actual = normalize_answer(prediction)
            expected = normalize_answer(clue.answer)
            compared_count += 1
            exact_match_count += int(actual == expected)
            length_violation_count += int(len(actual) != len(expected))

    return {
        "evaluated_crosswords": len(indices),
        "ground_truth_clues": clue_count,
        "compared_answers": compared_count,
        "exact_match_count": exact_match_count,
        "exact_match_accuracy": (
            exact_match_count / compared_count * 100 if compared_count else 0.0
        ),
        "length_violation_count": length_violation_count,
        "length_violation_rate": (
            length_violation_count / compared_count * 100 if compared_count else 0.0
        ),
    }


def cumulative_response_metrics(previous, current_metrics):
    cumulative = dict(current_metrics)
    cumulative_fields = (
        "total_turns",
        "total_attempts",
        "total_retries",
        "api_overload_retries",
        "raw_length_violation_count",
        "raw_length_violation_turns",
        "final_content_turn_count",
        "reasoning_only_turn_count",
        "empty_response_turn_count",
    )
    for field in cumulative_fields:
        if field in previous:
            cumulative[field] = previous[field] + current_metrics.get(field, 0)

    total_turns = cumulative.get("total_turns", 0)
    total_attempts = cumulative.get("total_attempts", 0)
    total_retries = cumulative.get("total_retries", 0)
    cumulative["average_retries_per_turn"] = (
        total_retries / total_turns if total_turns else 0.0
    )
    cumulative["average_attempts_per_turn"] = (
        total_attempts / total_turns if total_turns else 0.0
    )
    cumulative["turn_efficiency"] = (
        total_turns / total_attempts if total_attempts else 0.0
    )
    raw_violations = cumulative.get("raw_length_violation_count", 0)
    cumulative["raw_length_violation_rate"] = (
        raw_violations / total_attempts * 100 if total_attempts else 0.0
    )
    cumulative["estimated_cost_usd"] = (
        previous.get("estimated_cost_usd", 0.0)
        + current_metrics.get("estimated_cost_usd", 0.0)
    )

    return cumulative


def refresh_metrics(dataset, mode_directory, previous_metrics, current_metrics):
    metrics = cumulative_response_metrics(previous_metrics, current_metrics)

    completed_indices = []
    completed_numbers = set()
    for filename in os.listdir(mode_directory):
        if not filename.startswith("Crossword_"):
            continue
        try:
            number = int(filename.split("_", 1)[1])
        except ValueError:
            continue
        completed_numbers.add(number)

    for index, crossword in enumerate(dataset.data):
        if crossword.number in completed_numbers:
            completed_indices.append(index)

    metrics.update(answer_metrics(dataset, completed_indices, mode_directory))
    metrics_file = os.path.join(mode_directory, "sweepclip_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)
    return metrics


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the SweepClip crossword algorithm.")
    parser.add_argument(
        "--model",
        choices=("gpt", "deepseek", "llama"),
        default="gpt",
        help="LLM backend to use.",
    )
    parser.add_argument(
        "--api-model",
        default=None,
        help="Provider model name or API alias for generic OpenAI-compatible API use.",
    )
    parser.add_argument(
        "--with-retry-request",
        action="store_const",
        const="with_retry_request",
        dest="retry_mode",
        help="Retry answers that violate the clue character length.",
    )
    parser.add_argument(
        "--without-retry-request",
        action="store_const",
        const="without_retry_request",
        dest="retry_mode",
        help="Run without character-length retry requests.",
    )
    parser.add_argument(
        "--crosswords",
        type=int,
        default=100,
        help="Number of crosswords to process.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based dataset index to start at.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Exclusive zero-based dataset index to stop at.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip crosswords that already have output files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Crossword_<number> files for selected puzzles.",
    )
    parser.set_defaults(retry_mode="with_retry_request")
    return parser.parse_args()


def main():
    args = parse_arguments()

    run_model = args.api_model or args.model
    model_directory = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_model).strip("_")
    mode_directory = os.path.join("outputs", model_directory, args.retry_mode)
    log_directory = os.path.join("logs", model_directory, args.retry_mode)
    os.makedirs(mode_directory, exist_ok=True)
    os.makedirs(log_directory, exist_ok=True)

    if args.model in ("gpt", "deepseek") or args.api_model:
        max_retries = None if args.retry_mode == "with_retry_request" else 0
        api_model = args.api_model or args.model
        llm = APIModel(
            model_name=api_model,
            max_retries=max_retries,
            log_directory=log_directory,
        )
    else:
        max_retries = None if args.retry_mode == "with_retry_request" else 0
        llm = LLaMA(max_retries=max_retries)

    dataset = Dataset()
    metrics_file = os.path.join(mode_directory, "sweepclip_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as file:
            cumulative_metrics = json.load(file)
    else:
        cumulative_metrics = {}
    run_snapshot = {
        field: 0
        for field in (
            "total_turns",
            "total_attempts",
            "total_retries",
            "api_overload_retries",
            "raw_length_violation_count",
            "raw_length_violation_turns",
            "final_content_turn_count",
            "reasoning_only_turn_count",
            "empty_response_turn_count",
        )
    }
    run_snapshot["estimated_cost_usd"] = 0.0
    stop = args.end if args.end is not None else min(args.start + args.crosswords, len(dataset))
    stop = min(stop, len(dataset))
    if args.start < 0 or stop < args.start:
        raise ValueError("--start must be non-negative and --end must be at least --start.")

    selected_indices = []
    for index in range(args.start, stop):
        crossword = dataset[index]
        output_file = os.path.join(mode_directory, f"Crossword_{crossword.number}")
        if args.resume and os.path.exists(output_file):
            print(f"Skipping completed crossword {crossword.number}")
            continue
        if args.overwrite:
            if os.path.exists(output_file):
                os.remove(output_file)
        selected_indices.append(index)
        crossword.output_directory = mode_directory
        print("Currently computing :", index)
        print(crossword.number)
        crossword.solve(llm)
        current_run_metrics = llm.get_metrics() if hasattr(llm, "get_metrics") else {}
        incremental_metrics = {}
        for field in run_snapshot:
            incremental_metrics[field] = (
                current_run_metrics.get(field, 0)
                - run_snapshot[field]
            )
            run_snapshot[field] = current_run_metrics.get(field, 0)
        cumulative_metrics = refresh_metrics(
            dataset,
            mode_directory,
            cumulative_metrics,
            incremental_metrics,
        )

    metrics = cumulative_metrics
    with open(metrics_file, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()
