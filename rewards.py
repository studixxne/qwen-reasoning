import math
from utils import extract_content, parse_format, parse_answer, parse_tag

@extract_content
def format_reward(completions, **kwargs):
    return [0.2 if parse_format(pred) else 0.0 for pred in completions]

@extract_content
def answer_reward(completions, answer, **kwargs):
    rewards = []
    for c, a in zip(completions, answer):
        pred = parse_answer(c)
        ans = parse_answer(a)

        if pred and pred == ans:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

@extract_content
def answer_length_reward(completions, **kwargs):
    rewards = []
    target_length = 200
    max_reward = 0.5

    for c in completions:
        _, answer = parse_tag(c)
        reward = 0
        if answer:
            answer_length = len(answer)
            if answer_length >= target_length:
                reward = max_reward
            else:
                reward = max_reward * math.sin((answer_length / target_length) * (math.pi / 2))
        rewards.append(reward)
    
    return rewards

@extract_content
def answer_length_penalty(completions, **kwargs):
    penalties = []
    target_length = 200
    max_penalty = -0.5

    for c in completions:
        _, answer = parse_tag(c)
        penalty = max_penalty
        if answer:
            answer_length = len(answer)
            if answer_length >= target_length:
                penalty = 0.0
            else:
                penalty = max_penalty * (1 - math.sin((answer_length / target_length) * (math.pi / 2)))

        penalties.append(penalty)

    return penalties