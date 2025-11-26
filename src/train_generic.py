import os
import sys
import torch
import json
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
import argparse
from src.data_loader import load_tool_calling_dataset

def main():
    parser = argparse.ArgumentParser(description="Train an SLM for tool calling")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per device train batch size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device ID")
    
    args = parser.parse_args()
    
    print(f"Starting training for {args.model_id} on GPU {args.device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Model and Tokenizer
    model_name = args.model_id
    output_dir = args.output_dir

    # Quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="auto",
    )

    # LoRA Config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Dataset
    dataset = load_tool_calling_dataset(start=0, end=10000)

    def format_instruction(examples):
        prompts = []
        for i in range(len(examples['query'])):
            query = examples['query'][i]
            tools = json.loads(examples['tools'][i])
            answers = json.loads(examples['answers'][i])
            
            # Tool Defs
            tool_defs = []
            for tool in tools:
                func_name = tool.get('name')
                params = tool.get('parameters', {}).get('properties', {})
                param_defs = [f"{pname}: {p.get('type') if isinstance(p, dict) else p}" for pname, p in params.items()]
                tool_defs.append(f"def {func_name}({', '.join(param_defs)}):")
            tool_defs_str = "\n".join(tool_defs)
            
            # Tool Calls
            tool_calls = []
            for answer in answers:
                func_name = answer.get('name')
                args = answer.get('arguments', {})
                tool_calls.append(f"{func_name}(**{json.dumps(args)})")
            tool_calls_str = "\n".join(tool_calls)
            
            prompt = (
                f"<|system|>\nYou are an expert AI assistant with access to a suite of tools. Use them to answer the user's question.\n"
                f"Tools:\n{tool_defs_str}<|end|>\n"
                f"<|user|>\n{query}<|end|>\n"
                f"<|assistant|>\n<|tool_code|>\n{tool_calls_str}<|end|>"
            )
            prompts.append(prompt)
        return tokenizer(prompts, truncation=True, padding="max_length", max_length=1024)

    train_dataset = dataset.map(format_instruction, batched=True)

    # Training Config
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="no",
        max_length=1024,          # Using max_length in SFTConfig
        dataset_text_field="text", # Using dataset_text_field in SFTConfig
        packing=False,             # Using packing in SFTConfig
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # peft_config=peft_config, # Already applied manually
        processing_class=tokenizer, # Correct arg for trl 0.25.1
    )

    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(f"{output_dir}-final")
    print("Training complete!")

if __name__ == "__main__":
    main()
