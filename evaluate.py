from datasets import load_dataset
from transformers import AutoTokenizer
from dataclasses import dataclass
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from utils import get_prompt, get_config, parse_answer

@dataclass
class EvalConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    peft_path: str = None
    dataset_name: str = "openai/gsm8k"

def get_prompts_and_answers(dataset_name: str, tokenizer):
    dataset = load_dataset(dataset_name, 'main')['test']
    prompts = []
    ground_truths = []

    for item in dataset:
        question, answer = item['question'], item['answer']

        prompt = tokenizer.apply_chat_template(get_prompt(question), tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        ground_truths.append(parse_answer(answer))

    return prompts, ground_truths
        
def main(config: EvalConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    use_lora = config.peft_path is not None

    llm = LLM(
        model=config.base_model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        enable_lora=use_lora,
        max_lora_rank=16 if use_lora else None
    )

    lora_request = None
    if use_lora:
        lora_request = LoRARequest(
            lora_name="peft",
            lora_int_id=1,
            lora_path=config.peft_path
        )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
        stop_token_ids=[tokenizer.eos_token_id]
    )

    prompts, ground_truths = get_prompts_and_answers(config.dataset_name, tokenizer)
    print(f"Generating response for {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params=sampling_params, lora_request=lora_request)

    correct_count = 0

    for response, gt in zip(outputs, ground_truths):
        generated_text = response.outputs[0].text
        pred_answer = parse_answer(generated_text)

        if pred_answer and pred_answer == gt:
            correct_count += 1

    acc = (correct_count / len(prompts)) * 100
    print(f"Final Accuracy on {config.dataset_name} Test: {acc:.2f}% ({correct_count}/{len(prompts)})")

if __name__ == "__main__":
    config = get_config(EvalConfig)
    main(config)