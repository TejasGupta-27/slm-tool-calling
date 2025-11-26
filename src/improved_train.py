import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from src.data_loader import load_tool_calling_dataset
import json
from trl import SFTConfig, SFTTrainer

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
        # Note: We ensure the parts we want to mask (user input) are clearly separated
        prompt = (
            f"<|system|>\nYou are an expert AI assistant with access to a suite of tools. Use them to answer the user's question.\n"
            f"Tools:\n{tool_defs_str}<|end|>\n"
            f"<|user|>\n{query}<|end|>\n"
            f"<|assistant|>\n<|tool_code|>\n{tool_calls_str}<|end|>"
        )
        processed_examples["text"].append(prompt)

    return tokenizer(processed_examples["text"], truncation=True, padding="max_length", max_length=1024)

def main():
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # Model and tokenizer names
    model_name = "microsoft/Phi-3-medium-4k-instruct"
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_tool_calling_dataset()
    if not dataset:
        print("Failed to load dataset. Exiting.")
        return

    # INCREASED DATASET SIZE: 6000 -> 10000
    # Using more data helps learn diverse schemas
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
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)

    # Training arguments
    # INCREASED EPOCHS: 1 -> 2
    training_args = SFTConfig(
        output_dir="./models/phi3-medium-tool-calling-improved",
        num_train_epochs=2, 
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2, # Effective batch size 8
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
        # data_collator=collator, # Removing collator as it caused import issues
    )

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training finished.")
    
    # Save the model
    trainer.save_model("./models/phi3-medium-tool-calling-final-improved")
    print("Model saved successfully.")

if __name__ == "__main__":
    main()

