# SweepClip

This folder contains code for the SweepClip algorithm.


## Instructions

  - Make sure requirements are installed. Please see ```../README.md```
  - Make sure to edit the configuration file ```config.toml``` to provide the correct path to the dataset
  and local LLaMA 3 70B model.
  - For GPT 4 Turbo please add your API key. This can be done by adding the following line to your environment (bashrc).

```
export OPENAI_API_KEY='your api key'
```
  - Run with the following command.

```
python main.py
```
  - Use the API mode options:

```
python main.py --model gpt --with-retry-request --crosswords 100
python main.py --model gpt --without-retry-request --crosswords 100
python main.py --model deepseek --with-retry-request --crosswords 100
python main.py --api-model "provider-model-name" --with-retry-request --crosswords 100
python main.py --model llama --with-retry-request --crosswords 100
python main.py --model llama --without-retry-request --crosswords 100
```

For local LLaMA runs, `--with-retry-request` retries answers that do not match
the clue's character length. The retry limit is set by `[validation].max_retries`
in `config.toml`. `--without-retry-request` disables those extra generation
attempts; SweepClip's normal length pruning still applies.

### Run LLaMA 3 70B with Ollama on Windows

Install Ollama, then download the model and set the API key expected by the
OpenAI-compatible client (Ollama does not validate this placeholder):

```powershell
ollama pull llama3:70b
$env:OPENAI_API_KEY = "ollama"
```

Run these commands from the `Code\SweepClip` directory while Ollama is running:

```powershell
python main.py --api-model ollama_llama3_70b --with-retry-request --crosswords 100
python main.py --api-model ollama_llama3_70b --without-retry-request --crosswords 100
```

The `ollama_llama3_70b` alias and its local endpoint are configured in
`config.toml`. `--with-retry-request` retries answers that violate the clue
length, up to `[validation].max_retries`; without it, SweepClip still applies
its normal length pruning. Ollama runs use separate output folders for the two
modes. The existing `[api].base_url` remains unchanged for other API providers.

Use ranges and resume instead of rerunning the full dataset:

```
python main.py --api-model deepseekv4pro --without-retry-request --start 0 --end 20 --overwrite
python main.py --api-model deepseekv4pro --without-retry-request --start 20 --end 40 --resume
python main.py --api-model deepseekv4pro --without-retry-request --start 40 --end 60 --resume
```

`--start` and `--end` are zero-based dataset indices. `--end` is exclusive.
`--resume` skips crosswords that already have a `Crossword_<number>` file.
`--overwrite` reruns selected crosswords and removes their existing output files.

API aliases and provider model names are configured under `[api_models]`.

The metrics distinguish API response channels:

- `final_content_turn_count` — API calls that returned `message.content`
- `reasoning_only_turn_count` — calls that exposed reasoning without final content
- `empty_response_turn_count` — calls with neither usable channel

  - The API base URL, API key environment variable, and retry limit can be configured in `config.toml`.
  - Results are stored separately in `outputs/with_retry_request/` and `outputs/without_retry_request/`.
  - Each run writes `sweepclip_metrics.json` with retry, exact-match, length-violation, and cost metrics.
  - The LLM call logs (queries and outputs) are stored in ```logs``` and the
  generated answers are stored in ```outputs```.

  - For analysis, i.e. calculation of metrics, etc run - 
```
python analysis.py
```
  - Analyze a specific retry mode:

```
python analysis.py --output-directory "outputs/with_retry_request"
python analysis.py --output-directory "outputs/without_retry_request"
```
