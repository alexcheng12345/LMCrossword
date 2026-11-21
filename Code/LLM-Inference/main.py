from src.llm import LMFactory
import toml
import json
import sys
import os
import getopt
from tqdm import tqdm

config = toml.load("config.toml")


def parse_arguments():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_directory, llm_name = os.path.join(script_directory, "outputs"), "llama3_8Bi"
    input_file = None
    ground_truth_file = None
    retry_mode = "with_retry_request"
    argumentList = sys.argv[1:]
    long_options, options = [
        "help",
        "input=",
        "llm=",
        "ground-truth=",
        "with-retry-request",
        "without-retry-request",
    ], "hi:l:g:"
    arguments, values = getopt.getopt(argumentList, options, long_options)
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-h", "--help"):
            print(
                    "\nThis file runs inference with language models given a prompt file and "\
                    "saves the output to a json file. \nUsage: \n"\
                    "\tpython main.py --input [input JSON prompt file] --llm [llm name] "\
                    "[--with-retry-request | --without-retry-request]\n\n"\
                    "Available LLMs are - {phi3_3Bi, mistral_7Bi, llama3_8Bi, mixtral, "\
                    "llama2_70B, llama3_70B, gpt_3, gpt_4t, claude, deepseek_flash}.\n"\
            )
            exit(0)
        elif currentArgument in ("-i", "--input"):
            input_file = currentValue
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"File {input_file} does not exist")
            if "json" not in os.path.splitext(input_file)[-1]:
                raise ValueError(f"File {input_file} is not a JSON file.")
        elif currentArgument in ("-l", "--llm"):
            if currentValue not in config['llm']:
                raise ValueError(
                        f"{currentValue} is not an acceptable LLM name"\
                        "Please see 'python main.py --help'."
                )
            llm_name = currentValue
        elif currentArgument in ("-g", "--ground-truth"):
            ground_truth_file = currentValue
            if not os.path.exists(ground_truth_file):
                raise FileNotFoundError(f"Ground-truth file {ground_truth_file} does not exist")
        elif currentArgument == "--with-retry-request":
            retry_mode = "with_retry_request"
        elif currentArgument == "--without-retry-request":
            retry_mode = "without_retry_request"
        else:
            raise ValueError("Invalid input, see python main.py --help")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if input_file is None:
        raise ValueError("Input file not provided")
    input_basename = os.path.basename(input_file)
    output_file = (
        f"{llm_name}_({os.path.splitext(input_basename)[0]})"
        f"_{retry_mode}.json"
    )
    output_file = os.path.join(output_directory, output_file)
    return llm_name, input_file, output_file, retry_mode, ground_truth_file


def find_ground_truth(input_file, explicit_ground_truth=None):
    if explicit_ground_truth:
        return explicit_ground_truth

    script_directory = os.path.dirname(os.path.abspath(__file__))
    repository_root = os.path.abspath(os.path.join(script_directory, "..", ".."))
    input_name = os.path.basename(input_file).lower()
    datasets = (
        ("nyt", os.path.join(repository_root, "Data", "Datasets", "NYT-Clues", "NYT_query.json")),
        ("cryp", os.path.join(repository_root, "Data", "Datasets", "Cryptonite-Clues", "cryptonite_query.json")),
        ("init", os.path.join(repository_root, "Data", "Datasets", "Init-Clues", "init_query.json")),
    )
    for marker, dataset_path in datasets:
        if marker in input_name:
            return dataset_path
    return None


def normalize_answer(answer):
    if answer is None:
        return ""
    return "".join(character for character in str(answer).lower() if character.isalpha())


def add_answer_metrics(model_metrics, predictions, ground_truth_file):
    metrics = dict(model_metrics)
    if not ground_truth_file:
        return metrics

    with open(ground_truth_file, "r", encoding="utf-8") as file:
        ground_truth = json.load(file)

    compared_count = 0
    exact_count = 0
    length_violation_count = 0
    for clue_id, prediction in predictions.items():
        entry = ground_truth.get(str(clue_id))
        if not entry:
            continue

        expected = normalize_answer(entry.get("answer"))
        actual = normalize_answer(prediction)
        if not expected:
            continue

        compared_count += 1
        exact_count += int(actual == expected)
        length_violation_count += int(len(actual) != len(expected))

    metrics.update({
        "compared_count": compared_count,
        "exact_match_count": exact_count,
        "exact_match_accuracy": (
            exact_count / compared_count * 100 if compared_count else 0.0
        ),
        "length_violation_count": length_violation_count,
        "length_violation_rate": (
            length_violation_count / compared_count * 100 if compared_count else 0.0
        ),
    })
    return metrics

def get_results(llm_name, input_file, output_file, retry_mode, ground_truth_file):
    with open(input_file, 'r') as f:
        prompts = json.load(f)
    results = {}
    max_retries = None if retry_mode == "with_retry_request" else 0
    model = LMFactory(llm_name, max_retries=max_retries)
    for idx, (tag, prompt) in tqdm(
                                    enumerate(prompts.items()),
                                    desc = "Generating outputs",
                                    total = len(prompts.items(),
                                )
    ):
        answer = model(prompt)[0]
        results[tag] = answer
        #print(f"LLM ANSWER: {answer}")
        if idx%100 == 0 and idx>0:
            write_json(output_file, results)
            if hasattr(model, "get_metrics"):
                metrics_file = os.path.splitext(output_file)[0] + "_metrics.json"
                current_metrics = add_answer_metrics(
                    model.get_metrics(), results, ground_truth_file
                )
                write_json(metrics_file, current_metrics)
                print(json.dumps(current_metrics, indent=4))
    write_json(output_file, results)
    if hasattr(model, "get_metrics"):
        metrics_file = os.path.splitext(output_file)[0] + "_metrics.json"
        final_metrics = add_answer_metrics(model.get_metrics(), results, ground_truth_file)
        write_json(metrics_file, final_metrics)
        print(json.dumps(final_metrics, indent=4))

def write_json(file, dataset):
    with open(file, 'w') as f:
        json.dump(dataset, f, indent = 4)

def main():
    llm_name, input_file, output_file, retry_mode, ground_truth_file = parse_arguments()
    ground_truth_file = find_ground_truth(input_file, ground_truth_file)
    get_results(llm_name, input_file, output_file, retry_mode, ground_truth_file)

if __name__ == "__main__":
    main()
