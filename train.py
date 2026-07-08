import os

import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from dataclasses import dataclass

from utils import get_prompt, get_config
import rewards

@dataclass
class TrainConfig():
    # ==========================================
    # Basic Setting
    # ==========================================
    project_name: str = "Qwen-GRPO"
    run_name: str = None
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_name: str = "openai/gsm8k"
    output_dir: str = "models"

    # ==========================================     
    # Training Setting
    # ==========================================
    is_cuda: bool = torch.cuda.is_available()
    use_vllm: bool = is_cuda
    bf16: bool = is_cuda
    pin_memory: bool = is_cuda
    gradient_checkpointing: bool = is_cuda

    save_strategy: str = "steps"
    save_steps: int = 300
    save_total_limit: int = 10

    # ==========================================
    # Hyper Parameter Setting
    # ==========================================
    learning_rate: float = 5e-6
    beta: float = 0.00
    max_grad_norm: float = 1.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_generations: int = 8
    generation_batch_size: int = 16
    max_completion_length: int = 2048
    num_train_epochs: int = 1

    # ==========================================
    # Logging Setting
    # ==========================================
    report_to: str = 'wandb'
    logging_steps: int = 10
    log_completions: bool = True

def grpo_train(config: TrainConfig):
    training_args = GRPOConfig(
        output_dir = config.output_dir,

        learning_rate = config.learning_rate,
        max_grad_norm = config.max_grad_norm,
        beta = config.beta,
        per_device_train_batch_size = config.per_device_train_batch_size,
        gradient_accumulation_steps= config.gradient_accumulation_steps,
        num_generations = config.num_generations,
        generation_batch_size = config.generation_batch_size,
        max_completion_length = config.max_completion_length,
        num_train_epochs = config.num_train_epochs,

        use_vllm = config.use_vllm,
        bf16 = config.bf16,
        dataloader_pin_memory = config.pin_memory,
        gradient_checkpointing = config.gradient_checkpointing,

        save_strategy = config.save_strategy,
        save_steps = config.save_steps,
        save_total_limit = config.save_total_limit,

        logging_steps = config.logging_steps,
        report_to = config.report_to,
        log_completions = config.log_completions
    )

    dataset = load_dataset(config.dataset_name, "main")["train"]
    dataset = dataset.map(lambda example: {"prompt": get_prompt(example['question'])})
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    trainer = GRPOTrainer(
        model=config.base_model,
        peft_config=peft_config,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[rewards.format_reward, rewards.answer_reward, rewards.logging_reward],
        processing_class=tokenizer
    )

    trainer.train()

    final_save_dir = os.path.join(config.output_dir, "final_model")

    print(f"Saving final model to {final_save_dir}...")
    trainer.save_model(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)

    if config.report_to == 'wandb':
        wandb.finish()

if __name__ == '__main__':
    config = get_config(TrainConfig)

    if config.report_to == 'wandb':
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=vars(config)
        )

    grpo_train(config)