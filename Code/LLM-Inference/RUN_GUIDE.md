# LLM Inference Run Guide

## 1. Create the environment

Run from the repository root on Windows PowerShell:

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r Code\requirements.txt
```

Install the API client if you only use an OpenAI-compatible API:

```powershell
python -m pip install openai
```

## 2. Generate prompts

Update `Code\Prompts\config.toml` if its dataset paths differ from this repository.

```powershell
cd Code\Prompts
New-Item -ItemType Directory -Force prompts | Out-Null
python Prompt.py
python HintedPrompt.py
python CountingPrompt.py
```

The generated files appear in `Code\Prompts\prompts`.

## 3. Configure API or local inference

Return to `Code\LLM-Inference` and edit `config.toml`.

For an OpenAI-compatible API:

```toml
[api]
base_url = "https://your-provider.example/v1"
openai_models = ["deepseek_flash"]
max_new = 4096
```

Set the credential through the environment instead of saving it in `config.toml`:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
```

Add a model alias and provider model name to the `[llm]` section:

```toml
deepseek_flash = "deepseek-flash"
```

For a local Hugging Face model, set the corresponding `[llm]` value to a local directory or Hugging Face model name.

## 4. Run inference

The input path should be absolute or relative to `Code\LLM-Inference`.

```powershell
cd Code\LLM-Inference
python main.py --input "..\Prompts\prompts\NYT__fewShot_5.json" --llm deepseek_flash --with-retry-request
```

To run the original behavior without retry requests:

```powershell
python main.py --input "..\Prompts\prompts\NYT__fewShot_5.json" --llm deepseek_flash --without-retry-request
```

The output files are distinct:

```text
outputs\<llm_name>_(<prompt_file_name>)_with_retry_request.json
outputs\<llm_name>_(<prompt_file_name>)_without_retry_request.json
```

Each corresponding `_metrics.json` file includes retry metrics plus:

- `exact_match_accuracy` — percentage of predictions that exactly match the ground truth
- `length_violation_rate` — percentage whose normalized character length differs from the ground truth
- `exact_match_count`
- `length_violation_count`
- `compared_count`

Available API/local names are the keys under `[llm]` in `config.toml`.

Output is written to:

```text
Code\LLM-Inference\outputs\<llm_name>_(<prompt_file_name>).json
```

## 5. Post-process output

From `Code\LLM-Inference`:

```powershell
python src\postprocess.py
```

This creates `processed_<original-file-name>.json` files in `outputs`.

## 6. Compare predictions with ground truth

Example for the NYT dataset:

```powershell
python evaluate.py `
  --predictions "outputs\deepseek_flash_(NYT__fewShot_5).json" `
  --ground-truth "..\..\Data\Datasets\NYT-Clues\NYT_query.json" `
  --postprocess normal `
  --details "outputs\length-mismatches.json"
```

The report prints:

- Prediction and ground-truth counts
- Exact-match accuracy
- Character-number mismatch percentage
- Missing prediction count

Accuracy is calculated over IDs shared by both files. A prediction is an exact match only when it equals the ground-truth answer after case and non-letter normalization.

Character mismatch percentage is the percentage of compared predictions whose normalized character count differs from the ground-truth answer.

Use `--postprocess cot` for Chain-of-Thought files or `--postprocess none` for already normalized predictions.
