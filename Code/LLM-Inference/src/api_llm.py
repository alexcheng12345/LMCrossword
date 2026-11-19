import toml
import os
from time import sleep

config = toml.load("config.toml")


class GPT:
    def __init__(self, name = "gpt_3"):
        from openai import OpenAI
        self.name = name
        self.model = config['llm'][name]
        api_config = config.get('api', {})
        self.client = OpenAI(
            base_url=api_config.get('base_url'),
            api_key=api_config.get('api_key') or os.getenv('OPENAI_API_KEY'),
        )

    def __call__(self, prompt:list[dict]) -> list[str]:
        """ 
        Calls the OpenAI API and returns completions. The prompt must be in
        the following format:

        prompt = [
              {"role": "system", "content": config['prompt']['system']},
              {"role": "user", "content": prompt},
        ]
        Args:
            list[dict] : See docstring of infer method for explanation.
        Returns:
            list[str]: Completions by GPT.
        """
        max_new = config.get('api', {}).get('max_new', config['gen_config']['max_new'])
        completion = self.client.chat.completions.create(
                  model = self.model,
                  messages= prompt,
                  max_tokens = max_new,
                  n = config['gen_config']['seq_num'],
                  temperature = config['gen_config']['temperature'],
        )
        results = []
        for choice in completion.choices:
            message = choice.message
            content = getattr(message, "content", None)
            if isinstance(content, list):
                content = "".join(
                    getattr(part, "text", "")
                    if hasattr(part, "text")
                    else str(part.get("text", "")) if isinstance(part, dict)
                    else ""
                    for part in content
                )
            if content:
                results.append(content)
                continue

            reasoning_content = getattr(message, "reasoning_content", None)
            if reasoning_content is None:
                extra = getattr(message, "model_extra", None)
                if isinstance(extra, dict):
                    reasoning_content = extra.get("reasoning_content")
            raise RuntimeError(
                f"The API returned no final answer in message.content "
                f"(reasoning_content={'present' if reasoning_content else 'absent'}, "
                f"finish_reason={choice.finish_reason})."
            )
        return results

class Claude:
    def __init__(self, name = "claude"):
        from anthropic import Anthropic
        self.name = name
        self.model = config['llm'][name]
        self.client = Anthropic()

    def __call__(self, prompt:list[dict]) -> list[str]:
        """ 
        Calls the OpenAI API and returns completions. The prompt must be in
        the following format:

        prompt = [
              {"role": "system", "content": config['prompt']['system']},
              {"role": "user", "content": prompt},
        ]
        Args:
            list[dict] : See docstring of infer method for explanation.
        Returns:
            list[str]: Completions by GPT.
        """
        results = []
        for i in range(config['gen_config']['seq_num']):
            completion = self.client.messages.create(
                    model = self.model,
                    max_tokens = config['gen_config']['max_new'],
                    temperature = config['gen_config']['temperature'],
                    system = prompt[0]['content'],
                    messages = [prompt[-1]]
            )
            results.append(completion.content[0].text)
        return results if results else [""]
