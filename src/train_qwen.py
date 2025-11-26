import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from src.data_loader import load_tool_calling_dataset
import json
from trl import SFTConfig, SFTTrainer

# Default to GPU 2 (Shared with SmolLM if needed)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def preprocess_data(examples, tokenizer):
    """
    Preprocesses the data into a structured prompt format.
    """
    processed_examples = {"text": []}
    for i in range(len(examples['query'])):
        query = examples['query'][i]
        tools = json.loads(examples['tools'][i])
        answers = json.loads(examples['answers'][i])
        
        # Create a simplified tool definition
        tool_defs = []
        for tool in tools:
            func_name = tool.get('name')
            params = tool.get('parameters', {}).get('properties', {})
            param_defs = [f"{pname}: {p.get('type') if isinstance(p, dict) else p}" for pname, p in params.items()]
            tool_defs.append(f"def {func_name}({', '.join(param_defs)}):")

        tool_defs_str = "\n".join(tool_defs)
        
        # Create the tool calls
        tool_calls = []
        for answer in answers:
            func_name = answer.get('name')
            args = answer.get('arguments', {})
            tool_calls.append(f"{func_name}(**{json.dumps(args)})")
            
        tool_calls_str = "\n".join(tool_calls)

        # Build the prompt
        prompt = (
            f"<|system|>\nYou are an expert AI assistant with access to a suite of tools. Use them to answer the user's question.\n"
            f"Tools:\n{tool_defs_str}<|end|>\n"
            f"<|user|>\n{query}<|end|>\n"
            f"<|assistant|>\n<|tool_code|>\n{tool_calls_str}<|end|>"
        )
        processed_examples["text"].append(prompt)

    return tokenizer(processed_examples["text"], truncation=True, padding="max_length", max_length=1024)

def main():
    # Configuration
    model_name = "Qwen/Qwen2.5-0.5B-Instruct" # 0.5B parameters, very strong
    output_dir = "./models/qwen2.5-0.5b-tool-calling"
    
    print(f"Training model: {model_name}")
    print(f"Output dir: {output_dir}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_tool_calling_dataset()
    if not dataset:
        print("Failed to load dataset. Exiting.")
        return

    # Use a reasonable dataset size
    print("Selecting subset of 10,000 samples...")
    try:
        dataset = dataset.select(range(10000))
    except IndexError:
        print("Dataset smaller than 10000, using full dataset.")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Preprocess and tokenize the dataset
    print("Preprocessing dataset...")
    tokenized_dataset = dataset.map(lambda examples: preprocess_data(examples, tokenizer), batched=True)

    # Load model with quantization
    print("Loading model...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    # LoRA configuration
    # Qwen uses same modules as Llama/Mistral
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3, 
        per_device_train_batch_size=2, # Reduced from 8 to fit in memory
        gradient_accumulation_steps=4, # Increased to keep effective batch size similar
        learning_rate=2e-4,
        logging_steps=50,
        save_steps=1000,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,
        remove_unused_columns=False,
        dataset_text_field="text",
        max_length=1024,
        bf16=True,
    )

    # Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training finished.")
    
    # Save the model
    trainer.save_model(f"{output_dir}-final")
    print("Model saved successfully.")

if __name__ == "__main__":
    main()

