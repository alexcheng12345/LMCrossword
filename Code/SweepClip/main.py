import argparse
import json
import os
import re

from utils.dataset import Dataset
from utils.model import APIModel, GPT, LLaMA


def normalize_answer(answer):
    return "".join(character for character in str(answer).lower() if character.isalpha())


def answer_metrics(dataset, evaluated_count):
    exact_match_count = 0
    length_violation_count = 0
    compared_count = 0
    clue_count = 0

    for index in range(evaluated_count):
        crossword = dataset[index]
        clue_count += len(crossword.clues)
        for clue_key, clue in crossword.clues.items():
            prediction = crossword.current_answers.get(clue_key)
            if prediction is None:
                continue
            actual = normalize_answer(prediction)
            expected = normalize_answer(clue.answer)
            compared_count += 1
            exact_match_count += int(actual == expected)
            length_violation_count += int(len(actual) != len(expected))

    return {
        "evaluated_crosswords": evaluated_count,
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
        help="Retry GPT answers that violate the clue character length.",
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
    parser.set_defaults(retry_mode="with_retry_request")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.model == "llama" and args.retry_mode == "with_retry_request":
        raise ValueError("Character-length retry requests are currently supported only for API models.")

    run_model = args.api_model or args.model
    model_directory = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_model).strip("_")
    mode_directory = os.path.join("outputs", model_directory, args.retry_mode)
    log_directory = os.path.join("logs", model_directory, args.retry_mode)
    os.makedirs(mode_directory, exist_ok=True)
    os.makedirs(log_directory, exist_ok=True)
    for filename in os.listdir(mode_directory):
        if filename.startswith("Crossword_"):
            os.remove(os.path.join(mode_directory, filename))
    log_file = os.path.join(log_directory, "Crossword_all.txt")
    if os.path.exists(log_file):
        os.remove(log_file)

    if args.model in ("gpt", "deepseek") or args.api_model:
        max_retries = None if args.retry_mode == "with_retry_request" else 0
        api_model = args.api_model or args.model
        llm = APIModel(
            model_name=api_model,
            max_retries=max_retries,
            log_directory=log_directory,
        )
    else:
        llm = LLaMA()

    dataset = Dataset()
    evaluated_count = min(args.crosswords, len(dataset))
    for index in range(evaluated_count):
        crossword = dataset[index]
        crossword.output_directory = mode_directory
        print("Currently computing :", index)
        print(crossword.number)
        crossword.solve(llm)

    metrics = llm.get_metrics() if hasattr(llm, "get_metrics") else {}
    metrics.update(answer_metrics(dataset, evaluated_count))
    metrics_file = os.path.join(mode_directory, "sweepclip_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()
