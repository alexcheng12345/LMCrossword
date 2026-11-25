import os
from pathlib import Path

import toml
from openai import OpenAI


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.toml"
config = toml.load(CONFIG_PATH)


class OmniRoutersGPT:
    def __init__(
        self,
        name="omnirouters_gpt",
        max_retries=2,
        timeout=None,
    ):
        settings = config["omnirouters"]
        api_key = os.getenv(settings["api_key_env"])
        if not api_key:
            raise RuntimeError(
                f"Set the {settings['api_key_env']} environment variable "
                "before running OmniRouters inference."
            )

        self.name = name
        self.model = config["llm"][name]
        if timeout is None:
            self.client = OpenAI(
                api_key=api_key,
                base_url=settings["base_url"],
                max_retries=max_retries,
            )
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url=settings["base_url"],
                max_retries=max_retries,
                timeout=timeout,
            )

    def __call__(self, prompt):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            max_tokens=config["gen_config"]["max_new"],
            n=config["gen_config"]["seq_num"],
            temperature=config["gen_config"]["temperature"],
            top_p=None,
        )

        usage = getattr(completion, "usage", None)
        usage_record = {}
        if usage is not None:
            usage_record = {
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }

        return [choice.message.content for choice in completion.choices], usage_record
