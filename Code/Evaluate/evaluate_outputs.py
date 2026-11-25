import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import gensim.downloader as gensim_api
import numpy as np
import wordninja


ROOT = Path(__file__).resolve().parents[2]
PAPER_OUTPUT_DIR = ROOT / "Data" / "Raw Outputs" / "Section 3.2"
USER_OUTPUT_DIR = ROOT / "Code" / "LLM-Inference" / "outputs"
USER_LOG_DIR = ROOT / "Code" / "LLM-Inference" / "logs"
RESULT_DIR = ROOT / "Result"

DATASETS = {
    "NYT": ROOT / "Data" / "Datasets" / "NYT-Clues" / "NYT_query.json",
    "Cryp": ROOT / "Data" / "Datasets" / "Cryptonite-Clues" / "cryptonite_query.json",
    "Init": ROOT / "Data" / "Datasets" / "Init-Clues" / "init_query.json",
}

DEFAULT_EMBEDDING_MODEL = "glove-wiki-gigaword-50"


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def filter_predictions(predictions):
    """Mirror Filter() from Code/LLM-Inference/src/postprocess.py."""
    filtered = {}
    for key, prediction in predictions.items():
        answer = str(prediction or "").strip().lower()
        answer = answer.replace("clue: ", "")
        answer = answer.replace("answer: ", "")
        answer = re.sub(r"<(.*)>", "\n", answer)
        filtered[key] = answer
    return filtered


def postprocess_normal(predictions):
    """Mirror postprocess_normal() from the original postprocess.py."""
    processed = filter_predictions(predictions)
    for key, answer in processed.items():
        lines = [line for line in answer.split("\n") if line]
        answer = "" if not lines else lines[0]
        processed[key] = "".join(char for char in answer if char.isalpha())
    return processed


def normalize_answer(answer):
    return "".join(char for char in str(answer or "").strip().lower() if char.isalpha())


def answer_embedding(answer, embeddings):
    tokens = re.findall(r"[a-z]+", str(answer or "").lower())
    segmented_tokens = []
    for token in tokens:
        if token in embeddings.key_to_index:
            segmented_tokens.append(token)
        else:
            segmented_tokens.extend(wordninja.split(token))

    vectors = [
        embeddings[token]
        for token in segmented_tokens
        if token in embeddings.key_to_index
    ]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def normalized_cosine_score(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return None
    cosine = float(np.dot(left, right) / denominator)
    cosine = max(-1.0, min(1.0, cosine))
    return (cosine + 1.0) * 50.0


def normalize_dataset_name(name):
    normalized = name.strip().lower()
    aliases = {
        "nyt": "NYT",
        "cryp": "Cryp",
        "cryptonite": "Cryp",
        "init": "Init",
    }
    return aliases.get(normalized)


def dataset_from_name(name):
    match = re.search(r"(NYT|Cryp|Init)__fewShot_", name, re.IGNORECASE)
    return normalize_dataset_name(match.group(1)) if match else None


def load_call_metadata(output_path, log_dir):
    stem = output_path.stem
    summary_path = log_dir / f"{stem}_summary.json"
    call_log_path = log_dir / f"{stem}_calls.jsonl"

    summary = load_json(summary_path) if summary_path.is_file() else {}
    call_entries = []
    if call_log_path.is_file():
        with call_log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    call_entries.append(json.loads(line))
    calls_by_clue = Counter(
        entry.get("clue_id")
        for entry in call_entries
        if entry.get("clue_id") is not None
    )
    retry_calls = sum(max(0, count - 1) for count in calls_by_clue.values())

    if not summary and not call_entries:
        return None

    input_tokens = sum(entry.get("input_tokens") or 0 for entry in call_entries)
    output_tokens = sum(entry.get("output_tokens") or 0 for entry in call_entries)
    logged_costs = [entry.get("estimated_usd") for entry in call_entries]
    cost_is_available = bool(logged_costs) and all(
        value is not None for value in logged_costs
    )
    cost = sum(logged_costs) if cost_is_available else None

    if summary:
        input_tokens = summary.get("input_tokens", input_tokens)
        output_tokens = summary.get("output_tokens", output_tokens)
        cost = summary.get("estimated_api_cost_usd", cost)

    return {
        "number_of_calls": summary.get(
            "number_of_calls", len(call_entries) if call_entries else None
        ),
        "retry_calls_estimate": retry_calls if call_entries else None,
        "successful_calls": summary.get("successful_calls"),
        "failed_calls": summary.get("failed_calls"),
        "input_tokens": input_tokens if call_entries or summary else None,
        "output_tokens": output_tokens if call_entries or summary else None,
        "total_tokens": (
            input_tokens + output_tokens if call_entries or summary else None
        ),
        "api_cost_usd": cost,
        "api_cost_status": summary.get(
            "cost_status",
            "estimated_from_call_log" if cost_is_available else "unavailable",
        ),
        "call_source": "inference_summary_and_call_log",
    }


def score_output(
    output_path,
    dataset_name,
    query,
    source,
    log_dir,
    embeddings,
    embedding_model_name,
):
    raw_predictions = load_json(output_path)
    predictions = postprocess_normal(raw_predictions)
    evaluated = 0
    matches = 0
    length_violations = 0
    semantic_scores = []
    missing_output = 0

    for clue_id, record in query.items():
        if clue_id not in predictions:
            missing_output += 1
            continue

        prediction = normalize_answer(predictions[clue_id])
        ground_truth = normalize_answer(record["answer"])
        evaluated += 1

        if prediction == ground_truth:
            matches += 1
        if prediction and len(prediction) != len(ground_truth):
            length_violations += 1
        prediction_vector = answer_embedding(prediction, embeddings)
        ground_truth_vector = answer_embedding(record["answer"], embeddings)
        if prediction_vector is not None and ground_truth_vector is not None:
            score = normalized_cosine_score(
                prediction_vector,
                ground_truth_vector,
            )
            if score is not None:
                semantic_scores.append(score)

    metadata = load_call_metadata(output_path, log_dir) if source == "user" else None
    calls = None
    call_source = "not_available"
    if metadata:
        calls = metadata["number_of_calls"]
        call_source = metadata["call_source"]
    elif source == "paper":
        calls = len(raw_predictions)
        call_source = "paper_raw_outputs_assumed_one_call_per_answer"
    elif source == "user":
        calls = len(raw_predictions)
        call_source = "inferred_one_call_per_output_no_call_log"

    retries = max(0, calls - evaluated) if calls is not None else None
    if metadata and metadata["retry_calls_estimate"] is not None:
        retries = metadata["retry_calls_estimate"]
    elif source == "paper":
        retries = 0
    elif calls is not None and not metadata:
        retries = 0

    return {
        "source": source,
        "file": output_path.name,
        "dataset": dataset_name,
        "model": (
            output_path.parent.name
            if source == "paper"
            else output_path.name.split("_(", 1)[0]
        ),
        "shots": 10 if "fewShot_10" in output_path.name else 5,
        "clues_in_dataset": len(query),
        "evaluated": evaluated,
        "missing_output": missing_output,
        "exact_matches": matches,
        "exact_match_accuracy_pct": round(matches / evaluated * 100, 2)
        if evaluated
        else 0.0,
        "length_violations": length_violations,
        "length_violation_rate_pct": round(
            length_violations / evaluated * 100, 2
        )
        if evaluated
        else 0.0,
        "semantic_cosine_score_sum_0_100": round(sum(semantic_scores), 2),
        "semantic_cosine_score_0_100": round(
            sum(semantic_scores) / len(semantic_scores), 2
        )
        if semantic_scores
        else None,
        "semantic_embedding_scored_pairs": len(semantic_scores),
        "semantic_embedding_coverage_pct": round(
            len(semantic_scores) / evaluated * 100, 2
        )
        if evaluated
        else 0.0,
        "semantic_embedding_model": embedding_model_name,
        "semantic_method": (
            "Mean pre-trained word vectors; cosine mapped from [-1, 1] to [0, 100]; "
            "answer words segmented with wordninja"
        ),
        "number_of_calls": calls,
        "successful_api_calls": metadata["successful_calls"] if metadata else None,
        "failed_api_calls": metadata["failed_calls"] if metadata else None,
        "retry_calls_estimate": retries,
        "mean_calls_per_evaluated_clue": round(calls / evaluated, 4)
        if calls is not None and evaluated
        else None,
        "call_source": call_source,
        "input_tokens": metadata["input_tokens"] if metadata else None,
        "output_tokens": metadata["output_tokens"] if metadata else None,
        "total_tokens": metadata["total_tokens"] if metadata else None,
        "api_cost_usd": metadata["api_cost_usd"] if metadata else None,
        "api_cost_status": (
            metadata["api_cost_status"]
            if metadata
            else "unavailable_paper_outputs_have_no_billing_metadata"
            if source == "paper"
            else "unavailable_no_call_summary_or_log"
        ),
    }


def collect_outputs(output_dir, dataset_name, source_filter):
    is_paper_dir = (
        output_dir == PAPER_OUTPUT_DIR or PAPER_OUTPUT_DIR in output_dir.parents
    )
    source = (
        "paper"
        if source_filter == "auto" and is_paper_dir
        else "user"
        if source_filter == "auto"
        else source_filter
    )

    return [
        (path, source)
        for path in sorted(output_dir.rglob("*.json"))
        if not path.name.startswith("processed_")
        and dataset_from_name(path.name) == dataset_name
    ]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate JSON model outputs against a selected crossword dataset."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory containing model output JSON files (searched recursively)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("nyt", "cryp", "cryptonite", "init"),
        help="Dataset used to score the outputs",
    )
    parser.add_argument(
        "--name",
        default=None,
        help=(
            "Optional label used in result filenames; defaults to the "
            "output directory name"
        ),
    )
    parser.add_argument(
        "--source",
        choices=("auto", "paper", "user"),
        default="auto",
        help="Use paper or user metadata; auto detects paper raw-output folders",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=USER_LOG_DIR,
        help=(
            "Directory containing matching *_calls.jsonl and *_summary.json "
            "metadata (default: Code/LLM-Inference/logs)"
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=(
            "Pre-trained gensim word-vector model (default: "
            f"{DEFAULT_EMBEDDING_MODEL}; downloaded and cached on first use)"
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Output directory does not exist: {output_dir}")
    dataset_name = normalize_dataset_name(args.dataset)
    if dataset_name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    embeddings = gensim_api.load(args.embedding_model)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for output_path, source in collect_outputs(
        output_dir, dataset_name, args.source
    ):
        query = load_json(DATASETS[dataset_name])
        rows.append(
            score_output(
                output_path,
                dataset_name,
                query,
                source,
                args.log_dir.resolve(),
                embeddings,
                args.embedding_model,
            )
        )

    if not rows:
        raise FileNotFoundError(
            f"No JSON outputs for dataset {dataset_name} were found in {output_dir}."
        )

    result_name = args.name if args.name is not None else output_dir.name
    input_name = re.sub(r"[^A-Za-z0-9._-]+", "_", result_name).strip("._-")
    if not input_name:
        input_name = "outputs"
    result_stem = f"evaluation_{input_name}_{args.dataset}"
    result_csv = RESULT_DIR / f"{result_stem}.csv"
    result_json = RESULT_DIR / f"{result_stem}.json"
    fieldnames = list(rows[0].keys())

    with result_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with result_json.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    print(f"Evaluated {len(rows)} files for {dataset_name}.")
    print(f"CSV: {result_csv}")
    print(f"JSON: {result_json}")
    for row in rows:
        print(
            f"{row['source']} | {row['dataset']} | {row['model']} | "
            f"{row['shots']}-shot | exact={row['exact_match_accuracy_pct']:.2f}% | "
            f"length violations={row['length_violation_rate_pct']:.2f}% | "
            f"semantic cosine={row['semantic_cosine_score_0_100']} "
            f"(coverage={row['semantic_embedding_coverage_pct']:.2f}%) | "
            f"calls={row['number_of_calls']} | cost={row['api_cost_usd']}"
        )


if __name__ == "__main__":
    main()
