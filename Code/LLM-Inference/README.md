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
  - API inference rejects answers whose alphabetic character count does not match
  the `(<length>)//` value in the prompt. It retries up to
  `validation.max_retries` times (3 by default).
  - Retry metrics are written beside the output as
  `{model_name}_({prompt_file_name})_metrics.json` and include average retries
  per turn and turn efficiency.
  - Post-processing can be performed with the file ```./src/postprocess.py```.
  - Post-processing script assumes that the raw LLM outputs are in ```./outputs```
  and the file names are unchanged.
  - The script may be run with - 

```
python src/postprocess.py

```

