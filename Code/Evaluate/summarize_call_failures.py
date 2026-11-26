import argparse
import csv
import json
import re
from pathlib import Path

from evaluate_outputs import (
    RESULT_DIR,
    USER_LOG_DIR,
    dataset_from_name,
    summarize_api_call_outcomes,
)


def generate_report(log_dir=USER_LOG_DIR, output_path=None):
    log_dir = Path(log_dir)
    if output_path is None:
        output_path = RESULT_DIR / "inference_api_call_failure_breakdown.csv"
    output_path = Path(output_path)

    rows = []
    for call_log_path in sorted(log_dir.glob("*_calls.jsonl")):
        if "_B3_" in call_log_path.name or "__fewShot_5" not in call_log_path.name:
            continue
        dataset = dataset_from_name(call_log_path.name)
        if dataset is None:
            continue

        condition_match = re.search(r"_B([12])_", call_log_path.name)
        condition = f"B{condition_match.group(1)}" if condition_match else "B0"
        entries = [
            json.loads(line)
            for line in call_log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        outcomes = summarize_api_call_outcomes(entries)
        total_calls = len(entries)
        failed_calls = (
            outcomes["rate_limit_calls_429"]
            + outcomes["authentication_failures_401"]
            + outcomes["other_api_failures"]
        )
        rows.append(
            {
                "condition": condition,
                "dataset": dataset,
                "call_log": call_log_path.name,
                "total_api_calls": total_calls,
                "successful_api_calls": outcomes["successful_calls"],
                "failed_api_calls": failed_calls,
                "rate_limit_failures_429": outcomes["rate_limit_calls_429"],
                "authentication_failures_401": outcomes[
                    "authentication_failures_401"
                ],
                "other_api_failures": outcomes["other_api_failures"],
                "counts_reconcile": (
                    outcomes["successful_calls"] + failed_calls == total_calls
                ),
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"No B0-B2 five-shot call logs found in {log_dir}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path, rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Classify B0-B2 raw call-log entries without modifying the logs."
        )
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=USER_LOG_DIR,
        help="Directory containing *_calls.jsonl files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_DIR / "inference_api_call_failure_breakdown.csv",
        help="CSV report path",
    )
    args = parser.parse_args()

    output_path, rows = generate_report(args.log_dir, args.output)
    print(f"Classified {len(rows)} raw call logs.")
    print(f"CSV: {output_path}")
    for row in rows:
        print(
            f"{row['condition']} {row['dataset']}: "
            f"{row['successful_api_calls']} successful, "
            f"{row['rate_limit_failures_429']} rate-limited (429), "
            f"{row['authentication_failures_401']} authentication (401), "
            f"{row['other_api_failures']} other failures"
        )


if __name__ == "__main__":
    main()
