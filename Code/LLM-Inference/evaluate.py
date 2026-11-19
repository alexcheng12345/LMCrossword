import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from postprocess import postprocess_cot, postprocess_normal


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_answer(answer):
    if answer is None:
        return ""
    return "".join(character for character in str(answer).lower() if character.isalpha())


def evaluate(predictions, ground_truth):
    compared = []
    mismatches = []

    for clue_id, prediction in predictions.items():
        ground_truth_entry = ground_truth.get(str(clue_id))
        if not ground_truth_entry:
            continue

        expected = normalize_answer(ground_truth_entry.get("answer"))
        actual = normalize_answer(prediction)
        if not expected:
            continue

        length_matches = len(actual) == len(expected)
        exact_matches = actual == expected
        compared.append((clue_id, exact_matches, length_matches))

        if not length_matches:
            mismatches.append(
                {
                    "id": str(clue_id),
                    "predicted": actual,
                    "actual_length": len(actual),
                    "required_length": len(expected),
                }
            )

    compared_count = len(compared)
    exact_count = sum(exact_match for _, exact_match, _ in compared)
    mismatch_count = len(mismatches)

    return {
        "ground_truth_count": len(ground_truth),
        "prediction_count": len(predictions),
        "compared_count": compared_count,
        "missing_predictions": len(ground_truth) - compared_count,
        "exact_matches": exact_count,
        "accuracy": (exact_count / compared_count * 100) if compared_count else 0.0,
        "character_number_mismatches": mismatch_count,
        "character_number_mismatch_percent": (
            mismatch_count / compared_count * 100
        ) if compared_count else 0.0,
        "mismatches": mismatches,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare LLM crossword predictions with the ground-truth dataset."
    )
    parser.add_argument("--predictions", required=True, help="Path to the output JSON file")
    parser.add_argument("--ground-truth", required=True, help="Path to the query JSON file")
    parser.add_argument(
        "--postprocess",
        choices=("normal", "cot", "none"),
        default="normal",
        help="Post-process raw predictions before comparing",
    )
    parser.add_argument("--details", help="Optional path for mismatch details JSON")
    return parser.parse_args()


def main():
    args = parse_arguments()
    predictions = load_json(Path(args.predictions))
    ground_truth = load_json(Path(args.ground_truth))

    if args.postprocess == "normal":
        predictions = postprocess_normal(predictions)
    elif args.postprocess == "cot":
        predictions = postprocess_cot(predictions)

    metrics = evaluate(predictions, ground_truth)
    mismatch_details = metrics.pop("mismatches")

    print(json.dumps(metrics, indent=4))

    if args.details:
        details_path = Path(args.details)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with details_path.open("w", encoding="utf-8") as file:
            json.dump(mismatch_details, file, indent=4)
        print(f"Mismatch details: {details_path}")


if __name__ == "__main__":
    main()
