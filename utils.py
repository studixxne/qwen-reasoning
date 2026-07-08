import functools
from dataclasses import dataclass
from transformers import HfArgumentParser

def get_config(config_class: dataclass):
    parser = HfArgumentParser((config_class,))
    config, *_ = parser.parse_args_into_dataclasses()
    return config

def get_prompt(question):
    SYSTEM_PROMPT = f"""You are a helpful and expert math assistant. Your task is to solve mathematical problems logically.

You MUST strictly follow this output format:
1. First, think step-by-step and write your logical thinking process inside <reasoning> and </reasoning> tags.
2. Then, write your clear explanation for the user and final answer inside <answer> and </answer> tags. 
3. Inside the <answer> tag, you MUST write the final numerical answer at the last text, formatted exactly like '#### [Answer]'. (e.g. #### 42)
4. Do not output anything else outside of these tags."""
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

def extract_content(func):
    @functools.wraps(func)
    def _wrapper(completions, **kwargs):
        flat_completions = [c[0]['content'].strip() for c in completions]
        return func(completions=flat_completions, **kwargs)
    return _wrapper