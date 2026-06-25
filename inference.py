import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM_PROMPT = f"""You are a helpful and expert math assistant. Your task is to solve mathematical problems logically.

You MUST strictly follow this output format:
1. First, think step-by-step and write your logical thinking process inside <reasoning> and </reasoning> tags.
2. Then, write your clear explanation for the user and final answer inside <answer> and </answer> tags. 
3. Inside the <answer> tag, you MUST write the final numerical answer at the last text, formatted exactly like '#### [Answer]'. (e.g. #### 42)
4. Do not output anything else outside of these tags."""

dataset = load_dataset('openai/gsm8k', 'main')['test']

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def load_model():
    device = get_device()
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct").to(device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    return model, tokenizer, device

def main():
    model, tokenizer, device = load_model()

    for step, item in enumerate(dataset):
        question, answer = item['question'], item['answer']
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer([text], return_tensors='pt').to(device)
    
        with torch.no_grad():
            generated_ids = model.generate(**input_ids, max_new_tokens=1024)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids.input_ids, generated_ids)]
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        print(responses[0])

if __name__ == "__main__":
    main()