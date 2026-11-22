from openai import OpenAI
import warnings
import torch
import json
import os
import accelerate
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import toml


config = toml.load("config.toml")


class LLM:
    def __init__(self, model, log_directory="./logs"):
        if model not in ('gpt4', 'llama3') and model not in config:
            raise ValueError(f"Unknown model configuration: {model}")
        self.cost = 0
        if model in config:
            self.cost_threshold = config[model].get('budget', float('inf'))
        else:
            self.cost_threshold = float('inf')
        self.log_directory = log_directory
        os.makedirs(self.log_directory, exist_ok=True)

    @property
    def budget_exceeded(self):
        return (self.cost > self.cost_threshold)

    def reset_cost(self):
        self.cost = 0

    def process(self, pred):
        ans = pred.strip().lower()
        ans = ans.replace("clue: ","")
        ans = ans.replace("answer: ","")
        ans = [i for i in ans.split("\n") if i]
        ans = "" if not ans else ans[0]
        ans = "".join([i for i in ans if i.isalpha()])
        return ans

    def infer(self, prompt):
        raise NotImplementedError

    def __call__(self, prompt:list[dict]) -> list[str]:
        """ 
        Calls the LLM and returns completions. The prompt must be in
        the following format:

        prompt = [
              {"role": "system", "content": config['prompt']['system']},
              {"role": "user", "content": prompt},
        ]
        Args:
            list[dict] : See docstring of infer method for explanation.
        Returns:
            list[str]: Completions by GPT.
            bool: Budget exceeded.
        """
        if self.budget_exceeded:
            return [""], True
        response = self.infer(prompt)
        log = f"{prompt[-1]['content'].split("\n")[-1]} {response[0]}\n"
        with open(os.path.join(self.log_directory, "Crossword_all.txt"), "a", encoding="utf-8") as f:
            f.write(log)
        return response, self.budget_exceeded


class APIModel(LLM):
    def __init__(self, model_name="gpt", max_retries=None, log_directory="./logs"):
        super().__init__('gpt4', log_directory=log_directory)
        api_models = config.get('api_models', {})
        if model_name in api_models:
            self.model_name = model_name
            self.model = api_models[model_name]
        else:
            self.model_name = model_name
            self.model = model_name
        api_config = config.get('api', {})
        self.client = OpenAI(
            base_url=api_config.get('base_url'),
            api_key=api_config.get('api_key') or os.getenv('OPENAI_API_KEY'),
        )
        pricing = config.get('api_pricing', {})
        self.input_price = pricing.get(f"{self.model_name}_input", 0)
        self.output_price = pricing.get(f"{self.model_name}_output", 0)
        self.max_retries = (
            config.get('validation', {}).get('max_retries', 3)
            if max_retries is None
            else max_retries
        )
        self.retry_stats = {
            "total_turns": 0,
            "total_attempts": 0,
            "total_retries": 0,
        }
        self.length_violation_stats = {
            "raw_length_violation_count": 0,
            "raw_length_violation_turns": 0,
        }
        self.response_diagnostics = {
            "final_content_turn_count": 0,
            "reasoning_only_turn_count": 0,
            "empty_response_turn_count": 0,
        }
        self.total_cost = 0.0

    def get_metrics(self):
        total_turns = self.retry_stats["total_turns"]
        total_attempts = self.retry_stats["total_attempts"]
        total_retries = self.retry_stats["total_retries"]
        raw_violation_count = self.length_violation_stats["raw_length_violation_count"]
        return {
            **self.retry_stats,
            "average_retries_per_turn": (
                total_retries / total_turns if total_turns else 0.0
            ),
            "average_attempts_per_turn": (
                total_attempts / total_turns if total_turns else 0.0
            ),
            "turn_efficiency": (
                total_turns / total_attempts if total_attempts else 0.0
            ),
            "estimated_cost_usd": round(self.total_cost, 6),
            **self.length_violation_stats,
            **self.response_diagnostics,
            "raw_length_violation_rate": (
                raw_violation_count / total_attempts * 100
                if total_attempts else 0.0
            ),
        }

    @staticmethod
    def _required_length(prompt):
        import re
        for message in reversed(prompt):
            matches = re.findall(r"\((\d+)\)", str(message.get("content", "")))
            if matches:
                return int(matches[-1])
        return None

    @staticmethod
    def _process(raw_answer):
        if raw_answer is None:
            return ""
        ans = raw_answer.strip().lower()
        ans = ans.replace("clue: ", "")
        ans = ans.replace("answer: ", "")
        lines = [line for line in ans.split("\n") if line]
        candidate = lines[0] if lines else ""
        return "".join(character for character in candidate if character.isalpha())

    @staticmethod
    def _message_content(message):
        content = getattr(message, "content", None)
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if hasattr(part, "text"):
                    text_parts.append(str(part.text or ""))
                elif isinstance(part, dict):
                    text_parts.append(str(part.get("text", "") or ""))
                else:
                    text_parts.append(str(part))
            content = "".join(text_parts)
        if content:
            return content
        return ""

    def infer(self, prompt):
        required_length = self._required_length(prompt)
        retry_prompt = list(prompt)
        retries = 0
        self.retry_stats["total_turns"] += 1

        while True:
            self.retry_stats["total_attempts"] += 1
            cmpl = self.client.chat.completions.create(
                      model = self.model,
                      messages= retry_prompt,
                      max_tokens = config.get('api', {}).get(
                          'max_new',
                          config['model']['max_new'],
                      ),
                      n = config['model']['seq_num'],
                      temperature = config['model']['temperature'],
            )
            usage = getattr(cmpl, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            request_cost = (
                prompt_tokens * self.input_price
                + completion_tokens * self.output_price
            ) / 1_000_000
            self.cost += request_cost
            self.total_cost += request_cost
            choice_contents = [
                self._message_content(choice.message)
                for choice in cmpl.choices
            ]
            if not choice_contents:
                choice_contents = [""]
            final_content_count = sum(
                1
                for choice in cmpl.choices
                if getattr(choice.message, "content", None)
            )
            reasoning_content_count = sum(
                1
                for choice in cmpl.choices
                if (
                    getattr(choice.message, "reasoning_content", None)
                    or (
                        isinstance(getattr(choice.message, "model_extra", None), dict)
                        and getattr(choice.message, "model_extra", {}).get("reasoning_content")
                    )
                )
            )
            if final_content_count:
                self.response_diagnostics["final_content_turn_count"] += 1
            elif reasoning_content_count:
                self.response_diagnostics["reasoning_only_turn_count"] += 1
            else:
                self.response_diagnostics["empty_response_turn_count"] += 1
            results = [self._process(content) for content in choice_contents]
            raw_length_violations = sum(
                1
                for answer in results
                if required_length is not None
                and len(answer) != required_length
            )
            if raw_length_violations:
                self.length_violation_stats["raw_length_violation_count"] += raw_length_violations
                self.length_violation_stats["raw_length_violation_turns"] += 1

            if required_length is None or len(results[0]) == required_length:
                return results
            if retries >= self.max_retries:
                return results

            retry_prompt = retry_prompt + [
                {"role": "assistant", "content": choice_contents[0]},
                {
                    "role": "user",
                    "content": (
                        f"Your previous answer was rejected because it has "
                        f"{len(results[0])} alphabetic characters, but exactly "
                        f"{required_length} are required. Reply with only the "
                        f"answer word containing exactly {required_length} "
                        f"alphabetic characters."
                    ),
                },
            ]
            retries += 1
            self.retry_stats["total_retries"] += 1


class GPT(APIModel):
    def __init__(self, max_retries=None, log_directory="./logs"):
        super().__init__(
            model_name="gpt",
            max_retries=max_retries,
            log_directory=log_directory,
        )


class LLaMA(LLM):
    """ Base LLM class implements inference """
    def __init__(self):
        super().__init__('llama3')
        self.llm_path = config['llama3']['path']
        self.loaded = False
        self.load_tokenizer()
        self._init_config()

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path, trust_remote_code = True)
        self.tokenizer.unk_token = "<notrequired>" 
        self.tokenizer.sep_token = "<notrequired>"
        self.tokenizer.pad_token = "<notrequired>"
        self.tokenizer.cls_token = "<notrequired>"
        self.tokenizer.mask_token = "<notrequired>"
        self.tokenizer.chat_template = config['llama3']['template']

    def load_model(self):
        if self.loaded:
            return
        self.model = AutoModelForCausalLM.from_pretrained(
                    self.llm_path,
                    torch_dtype = torch.bfloat16,
                    device_map = "auto",
                    trust_remote_code = True,
                )
        self.loaded = True

    def _init_config(self):
        """ Inititalize generation config. """
        self.generation_config = transformers.GenerationConfig(
                max_new_tokens = config['model']['max_new'],
                temperature=config['model']['temperature'],
                top_p = None,
                do_sample = True,
                num_return_sequences = config['model']['seq_num'],
                pad_token_id = self.tokenizer.eos_token_id,
                eos_token_id = self.tokenizer.eos_token_id,
        )


    def infer(self, prompt):
        self.load_model()
        torch.cuda.empty_cache()
        self.model.eval()
        inputs = self.tokenizer.apply_chat_template(prompt, tokenize=True, return_tensors="pt")
        prompt_length = len(self.tokenizer.decode(inputs[0], skip_special_tokens=False))
        with torch.inference_mode():
            results = self.model.generate(
                    input_ids = inputs.to("cuda"),
                    generation_config = self.generation_config,
            )
        results = [self.tokenizer.decode(result, skip_special_tokens=False) for result in results]
        results = [result[prompt_length:] for result in results]
        self.cost += 1
        return [self.process(result) for result in results]
