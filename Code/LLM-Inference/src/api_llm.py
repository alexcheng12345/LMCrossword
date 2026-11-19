import toml
import os
from time import sleep
from src.retry_validator import extract_required_length, is_valid_answer, retry_message

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
        self.retry_stats = {
            "total_turns": 0,
            "total_attempts": 0,
            "total_retries": 0,
        }

    def get_metrics(self):
        total_turns = self.retry_stats["total_turns"]
        total_attempts = self.retry_stats["total_attempts"]
        total_retries = self.retry_stats["total_retries"]
        return {
            **self.retry_stats,
            "average_retries_per_turn": total_retries / total_turns if total_turns else 0.0,
            "average_attempts_per_turn": total_attempts / total_turns if total_turns else 0.0,
            "turn_efficiency": total_turns / total_attempts if total_attempts else 0.0,
        }

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
        max_retries = config.get('validation', {}).get('max_retries', 3)
        required_length = extract_required_length(prompt)
        retry_prompt = list(prompt)
        retries = 0

        self.retry_stats["total_turns"] += 1

        while True:
            self.retry_stats["total_attempts"] += 1
            completion = self.client.chat.completions.create(
                      model = self.model,
                      messages= retry_prompt,
                      max_tokens = max_new,
                      n = config['gen_config']['seq_num'],
                      temperature = config['gen_config']['temperature'],
            )
            results = []
            invalid_answers = []
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
                    if not is_valid_answer(content, required_length):
                        invalid_answers.append(content)
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

            if required_length is None or not invalid_answers:
                return results
            if retries >= max_retries:
                return results

            retry_prompt = retry_prompt + [
                {"role": "assistant", "content": invalid_answers[0]},
                {"role": "user", "content": retry_message(required_length, invalid_answers[0])},
            ]
            retries += 1
            self.retry_stats["total_retries"] += 1

class Claude:
    def __init__(self, name = "claude"):
        from anthropic import Anthropic
        self.name = name
        self.model = config['llm'][name]
        self.client = Anthropic()
        self.retry_stats = {
            "total_turns": 0,
            "total_attempts": 0,
            "total_retries": 0,
        }

    def get_metrics(self):
        total_turns = self.retry_stats["total_turns"]
        total_attempts = self.retry_stats["total_attempts"]
        total_retries = self.retry_stats["total_retries"]
        return {
            **self.retry_stats,
            "average_retries_per_turn": total_retries / total_turns if total_turns else 0.0,
            "average_attempts_per_turn": total_attempts / total_turns if total_turns else 0.0,
            "turn_efficiency": total_turns / total_attempts if total_attempts else 0.0,
        }

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
        max_retries = config.get('validation', {}).get('max_retries', 3)
        required_length = extract_required_length(prompt)
        results = []
        for i in range(config['gen_config']['seq_num']):
            self.retry_stats["total_turns"] += 1
            retry_messages = [prompt[-1]]
            retries = 0

            while True:
                self.retry_stats["total_attempts"] += 1
                completion = self.client.messages.create(
                        model = self.model,
                        max_tokens = config['gen_config']['max_new'],
                        temperature = config['gen_config']['temperature'],
                        system = prompt[0]['content'],
                        messages = retry_messages
                )
                answer = completion.content[0].text
                if required_length is None or is_valid_answer(answer, required_length):
                    results.append(answer)
                    break
                if retries >= max_retries:
                    results.append(answer)
                    break

                retry_messages = retry_messages + [
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": retry_message(required_length, answer)},
                ]
                retries += 1
                self.retry_stats["total_retries"] += 1
        return results if results else [""]
