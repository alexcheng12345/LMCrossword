import toml
import os
config = toml.load("config.toml")


def LMFactory(name: str, max_retries=None):
    if name not in config['llm']:
        raise ValueError(
                f"{name} is not a valid LLM name. "
                f"Please choose from {list(config['llm'].keys())}."
        )
    api_config = config.get('api', {})
    openai_models = api_config.get('openai_models', [])
    if name in openai_models or 'gpt' in name:
        from src.api_llm import GPT
        return GPT(name, max_retries=max_retries)
    elif 'claude' in name:
        from src.api_llm import Claude
        return Claude(name, max_retries=max_retries)
    else:
        from src.local_llm import Dummy, LLM
        if not os.path.exists(config['llm'][name]):
            return Dummy(name)
        return LLM(name)
