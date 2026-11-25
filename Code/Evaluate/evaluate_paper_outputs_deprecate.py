import csv
import json
import re
import sys
from pathlib import Path
from difflib import SequenceMatcher


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "Data" / "Raw Outputs" / "Section 3.2"
QUERY_FILES = {
    "NYT": ROOT / "Data" / "Datasets" / "NYT-Clues" / "NYT_query.json",
    "Cryp": ROOT / "Data" / "Datasets" / "Cryptonite-Clues" / "cryptonite_query.json",
    "Init": ROOT / "Data" / "Datasets" / "Init-Clues" / "init_query.json",
}


def normalize_answer(text):
    if text is None:
        return ""
    text = str(text).strip().lower()
    if not text:
        return ""
    text = re.sub(r"^answer:\s*", "", text)
    text = re.sub(r"^clue:\s*", "", text)
    if not text:
        return ""
    if "\n" not in text and "\r" not in text:
        lines = [text]
    else:
        lines = text.splitlines()
    if not lines:
        return ""
    text = lines[0]
    text = text.replace("answer:", "").replace("clue:", "")
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z]", "", text)
    return text


def discover_dataset(filename):
    for dataset in ("NYT", "Cryp", "Init"):
        if f"_{dataset}_" in f"_{filename}" or f"__{dataset}" in filename:
            return dataset
    return None


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def semantic_proxy(prediction, ground_truth):
    if not prediction or not ground_truth:
        return False
    return SequenceMatcher(None, prediction, ground_truth).ratio() >= 0.50


def evaluate_file(raw_path, query, dataset):
    predictions = load_json(raw_path)
    matches = 0
    violations = 0
    evaluated = 0
    missing_output = 0
    semantic_proxy_matches = 0

    for clue_id, record in query.items():
        if clue_id not in predictions:
            missing_output += 1
            continue

        ground_truth = normalize_answer(record["answer"])
        prediction = normalize_answer(predictions[clue_id])
        evaluated += 1

        if prediction == ground_truth:
            matches += 1
        if semantic_proxy(prediction, ground_truth):
            semantic_proxy_matches += 1
        if prediction and len(prediction) != len(ground_truth):
            violations += 1

    return {
        "file": raw_path.name,
        "dataset": dataset,
        "shots": 10 if "fewShot_10" in raw_path.name else 5,
        "model": raw_path.parent.name,
        "evaluated": evaluated,
        "matches": matches,
        "accuracy": round(matches / evaluated * 100, 2) if evaluated else 0.0,
        "violations": violations,
        "length_violation_rate": (
            round(violations / evaluated * 100, 2) if evaluated else 0.0
        ),
        "missing_output": missing_output,
        "mean_calls_per_clue": 1.0,
        "median_calls_per_clue": 1,
        "retry_count": 0,
        "retry_trigger_rate": 0.0,
        "mean_turn_efficiency": 1.0,
        "semantic_proxy_matches": semantic_proxy_matches,
        "semantic_proxy_accuracy": (
            round(semantic_proxy_matches / evaluated * 100, 2)
            if evaluated else 0.0
        ),
        "semantic_method": "SequenceMatcher lexical similarity >= 0.50",
        "number_of_calls": len(query),
        "api_cost_usd": None,
        "api_cost_status": "unavailable_not_in_paper_raw_outputs",
    }


def main():
    output_csv = ROOT / "Data" / "paper_output_evaluation.csv"
    rows = []

    for raw_path in sorted(RAW_DIR.rglob("*fewShot*.json")):
        name = raw_path.name
        if "NYT" in name:
            dataset = "NYT"
        elif "Cryp" in name:
            dataset = "Cryp"
        elif "Init" in name:
            dataset = "Init"
        else:
            continue

        rows.append(
            evaluate_file(
                raw_path,
                load_json(QUERY_FILES[dataset]),
                dataset,
            )
        )

    fieldnames = list(rows[0].keys()) if rows else [
        "file",
        "dataset",
        "shots",
        "model",
        "evaluated",
        "matches",
        "accuracy",
        "violations",
        "length_violation_rate",
        "missing_output",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(output_csv)
    print(f"Evaluated {len(rows)} paper raw-output files")
    for row in rows:
        print(
            f"{row['dataset']} {row['shots']}-shot {row['model']}: "
            f"{row['accuracy']:.2f}% exact; "
            f"{row['length_violation_rate']:.2f}% length violations"
        )


if __name__ == "__main__":
    main()
