from utils import extract_content, parse_format, parse_answer

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