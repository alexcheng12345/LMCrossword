# Code for LLM Inference.

This folder containts code to run inference with language models given a prompt file and saves the output to a json file. 

## Instructions.

  - Make sure requirements are installed. Please see ```../README.md```

  - Make sure to edit the configuration file ```config.toml``` to provide the correct path to the local LLM, or
  the name of the model in case you want to download from Huggingface.

  - For API based LLMs please add your API key. This can be done by adding the following lines to your environment.
```
export OPENAI_API_KEY='your api key'
export ANTHROPIC_API_KEY='your api key'
```

  - Run with the following command.

```
python main.py --input [input JSON prompt file] --llm [llm name]

```
  - Outputs are saved in ```./outputs/{model_name}_{filename}.json```
  - Post-processing can be performed with the file ```./src/postprocess.py```.
  - Post-processing script assumes that the raw LLM outputs are in ```./outputs```
  and the file names are unchanged.
  - The script may be run with - 

```
python src/postprocess.py

```

## Measured OmniRouters inference

The original `main.py` remains unchanged. Use `main_modified.py` for
OpenAI-compatible OmniRouters calls with JSONL call logs and cost summaries.

Set the API key without storing it in source or config files:

```powershell
$env:OMNIROUTERS_API_KEY="your-key"
```

Run on a prompt file:

```powershell
python main_modified.py --input <prompt-json> --llm omnirouters_gpt
```

Optional arguments:

```powershell
python main_modified.py --input <prompt-json> --llm omnirouters_gpt --limit 100 --workers 4
python main_modified.py --input <prompt-json> --llm omnirouters_gpt --workers 1
python main_modified.py --input <prompt-json> --llm omnirouters_gpt --cache --workers 4
```

`--workers` controls the maximum number of concurrent requests (default: 4).
Use `--workers 1` for sequential inference. Lower the worker count if the
provider returns rate-limit errors; higher concurrency can increase throughput
but does not reduce token-based API cost.

To resume a partial run without repeating successful clues, use `--cache`.
This appends new attempts to the existing call log and updates the cumulative
summary; successful prior outputs and their usage records are retained.

## B1: repeated sampling without feedback

`main_b1.py` reuses the configured OmniRouters model and the same prompt, but
calls the model repeatedly for each clue without sending corrective feedback.
It stops at the first length-compliant answer, up to four attempts per clue.
If none is length-compliant, it selects the last successful response.

```powershell
python main_b1.py --input ..\Prompts\prompts\NYT__fewShot_5.json --llm omnirouters_gpt --workers 4
```

Use `--max-attempts 4` to set the B1 call budget, `--limit 20` for a small test,
and `--cache` to resume completed clues without repeating their calls. B1
results and logs use a `_B1_` filename prefix and do not overwrite B0 files.

## B2: length feedback

`main_b2.py` reuses the same model and few-shot prompt, checks the normalized
answer length, and sends the previous answer plus the expected and observed
lengths as feedback after a wrong-length response. It stops at the first
length-compliant answer, up to four total calls per clue. If no answer meets
the length, it selects the last successful response.

```powershell
python main_b2.py --input ..\Prompts\prompts\NYT__fewShot_5.json --limit 20 --workers 4
```

Remove `--limit 20` for the full dataset. Use `--cache` to resume completed
clues without repeating their calls. B2 results and logs use a `_B2_` filename
prefix, separate from both B0 and B1.
For this NYT prompt, the result is
`outputs/omnirouters_gpt_B2_(NYT__fewShot_5).json`; the matching call log and
summary are written under `logs/` with `_calls.jsonl` and `_summary.json`
suffixes.

## B3: local WordNet candidates

`main_b3.py` first asks the model for dictionary search terms, retrieves up to
30 exact-length candidates from a local WordNet corpus, and asks the model to
choose a candidate ID. If the model returns `NO_VALID_CANDIDATE` (or an
unparseable selection), it makes one more selection call using the same
shortlist. If that still fails, or no candidates are retrieved, it falls back
to B2. The required answer length is parsed from the prompt's final clue
enumeration; B3 does not read the hidden query answer to filter candidates.

Install the declared dependency and the WordNet corpus once:

```powershell
python -m pip install -r ..\requirements.txt
python -m nltk.downloader wordnet
```

Run a small test from `Code\LLM-Inference`:

```powershell
python main_b3.py --input ..\Prompts\prompts\NYT__fewShot_5.json --limit 20 --workers 4
```

Omit `--limit 20` for the full input. Use `--dictionary-dir` if the WordNet
corpus is stored outside NLTK's default data paths, and `--cache` to resume
completed clues. B3 outputs and call logs use a `_B3_` filename prefix.
The B3 client disables automatic SDK retries. On the first HTTP 429, it logs
the failed call, stops scheduling additional model calls, saves completed
outputs and the partial summary, and exits with an error. Requests already in
flight may finish. Resume later with `--cache` and fewer workers.

Outputs are written to:

- `outputs/omnirouters_gpt_(<prompt-name>).json`
- `logs/omnirouters_gpt_(<prompt-name>)_calls.jsonl`
- `logs/omnirouters_gpt_(<prompt-name>)_summary.json`
- `logs/omnirouters_gpt_(<prompt-name>)_summary.csv`

Set `[omnirouters.cost]` in `config.toml` to the account's actual
input/output prices per million tokens. Until these values are set,
the summary reports `estimated_api_cost_usd = null`.

