import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModelForCausalLM
from dataclasses import dataclass

from utils import get_prompt, get_device, get_config

@dataclass
class GenConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    peft_path: str = "models/checkpoint-1800"
    device = get_device()

def load_model(config: GenConfig):
    model = AutoModelForCausalLM.from_pretrained(config.base_model, dtype=torch.float16).to(config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    if config.peft_path:
        model = PeftModelForCausalLM.from_pretrained(model, config.peft_path).to(config.device)
    
    return model, tokenizer

def main(config: GenConfig):
    model, tokenizer = load_model(config)

    print("\n\n🔖 Assistant: Can I help you? \n")

    while True:
        question = input("👤 user: ").strip()

        if not question:
            continue

        prompt = get_prompt(question)

        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(config.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=2048)   
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"\n🍃 Assistant: {response}\n")

if __name__ == "__main__":
    config = get_config(GenConfig)
    main(config)