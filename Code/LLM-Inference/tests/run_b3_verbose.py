import argparse
import json
import sys
from pathlib import Path
from threading import Event


INFERENCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INFERENCE_DIR))

import main_b3
from src.api_llm_modified import OmniRoutersGPT
from src.wordnet_dictionary import WordNetDictionary


DEFAULT_INPUT = (
    INFERENCE_DIR.parent / "Prompts" / "prompts" / "NYT__fewShot_5.json"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run B3 sequentially and print every model response without "
            "writing or changing the normal B3 output and log files."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Prompt JSON file (default: NYT__fewShot_5.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Maximum clues to run sequentially (default: 1)",
    )
    parser.add_argument(
        "--clue-id",
        default=None,
        help="Run a specific clue ID instead of the first clues",
    )
    parser.add_argument(
        "--llm",
        default="omnirouters_gpt",
        help="Configured model key (default: omnirouters_gpt)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=4,
        help="Maximum B2 fallback calls for each clue (default: 4)",
    )
    parser.add_argument(
        "--dictionary-dir",
        type=Path,
        default=None,
        help="Optional directory containing the local WordNet corpus",
    )
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="Also print the full messages sent to the model",
    )
    return parser.parse_args()


def classify_stage(messages):
    content = "\n".join(
        str(message.get("content", "")) for message in messages
    ).lower()
    if "query-expansion assistant" in content:
        return "query expansion"
    if "dictionary candidates:" in content:
        return "candidate selection"
    return "B2 fallback"


def main():
    args = parse_arguments()
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    with args.input.open("r", encoding="utf-8") as handle:
        prompts = json.load(handle)
    if not isinstance(prompts, dict) or not prompts:
        raise ValueError("Prompt JSON must contain at least one clue.")

    if args.clue_id is not None:
        if args.clue_id not in prompts:
            raise KeyError(f"Clue ID not found in prompt file: {args.clue_id}")
        selected_prompts = {args.clue_id: prompts[args.clue_id]}
    else:
        selected_prompts = dict(list(prompts.items())[: args.limit])

    parsed_prompts = {}
    for clue_id, prompt in selected_prompts.items():
        clue, required_length = main_b3.extract_clue_and_length(prompt)
        parsed_prompts[clue_id] = (prompt, clue, required_length)

    dictionary = WordNetDictionary(
        {length for _, _, length in parsed_prompts.values()},
        data_dir=args.dictionary_dir,
    )
    model = OmniRoutersGPT(args.llm, max_retries=0)
    original_run_inference = main_b3.run_inference
    active_clue_id = None
    request_number = 0

    def run_and_print(model_instance, messages):
        nonlocal request_number
        request_number += 1
        stage = classify_stage(messages)
        print(
            f"\n[{active_clue_id}] API call {request_number}: {stage}",
            flush=True,
        )
        if args.show_prompts:
            print("Request messages:")
            print(json.dumps(messages, indent=2, ensure_ascii=False))

        answer, usage, error, latency = original_run_inference(
            model_instance,
            messages,
        )
        print("Model output:", answer, flush=True)
        if error:
            print("API error:", error, flush=True)
        print(
            "Usage:",
            json.dumps(usage, ensure_ascii=False),
            f"| latency: {latency:.2f}s",
            flush=True,
        )
        return answer, usage, error, latency

    main_b3.run_inference = run_and_print
    stop_event = Event()
    try:
        for clue_id, (prompt, clue, required_length) in parsed_prompts.items():
            active_clue_id = clue_id
            start_call = request_number
            print(
                f"\n{'=' * 72}\n"
                f"Clue {clue_id}: {clue} ({required_length})\n"
                "Running B3 with one sequential worker.",
                flush=True,
            )
            returned_id, answer, entries, stopped = main_b3.run_b3_for_clue(
                model,
                clue_id,
                prompt,
                clue,
                required_length,
                dictionary,
                args.max_attempts,
                stop_event,
            )

            print("\nClue result:")
            print(
                json.dumps(
                    {
                        "clue_id": returned_id,
                        "clue": clue,
                        "required_length": required_length,
                        "answer": answer,
                        "normalized_answer": main_b3.normalize_answer(answer),
                        "length_compliant": (
                            len(main_b3.normalize_answer(answer))
                            == required_length
                            if answer is not None
                            else False
                        ),
                        "calls_for_clue": request_number - start_call,
                        "stopped_on_rate_limit": stopped,
                        "call_details": entries,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                flush=True,
            )

            if stopped:
                print(
                    "\nStopping verbose test after a rate limit. "
                    "No B3 output or log files were written.",
                    flush=True,
                )
                return 1
    finally:
        main_b3.run_inference = original_run_inference

    print(
        f"\nVerbose B3 test finished: {len(parsed_prompts)} clue(s), "
        f"{request_number} API call(s). No output files were written.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
