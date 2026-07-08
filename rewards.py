import re
import json

from utils import extract_content

@extract_content
def format_reward(completions, **kwargs):
    pattern = re.compile(r"^<reasoning>.*?</reasoning>\s*<answer>.*?</answer>$", re.DOTALL)
    return [0.2 if pattern.match(pred) else 0.0 for pred in completions]

@extract_content
def answer_reward(completions, answer, **kwargs):
    pattern = re.compile(r"####\s*(\d+)")

    rewards = []
    for pred, a in zip(completions, answer):
        pred_match = pattern.findall(pred)
        ans_match = pattern.findall(a)

        if (pred_match and ans_match) and int(pred_match[-1]) == int(ans_match[-1]):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards

# 단순 대화 기록 logging
@extract_content
def logging_reward(completions, answer, **kwargs):
    with open("all_completions_log.jsonl", "a", encoding="utf-8") as f:
        for c, a in zip(completions, answer):
            log_data = {"completion": c, "answer": a}
            f.write(json.dumps(log_data, ensure_ascii=False) + '\n')

    return [0.0] * len(completions)