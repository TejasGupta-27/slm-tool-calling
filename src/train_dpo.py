import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model
from trl import DPOConfig, DPOTrainer

def main():
    # Set CUDA device (User can override)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        
    model_name = "microsoft/Phi-3-medium-4k-instruct"
    # Path to the SFT model (Original one, for testing DPO now)
    sft_adapter_path = "./models/phi3-medium-tool-calling-final"
    output_dir = "./models/phi3-medium-tool-calling-dpo"
    
    print(f"Loading DPO dataset...")
    try:
        dataset = load_from_disk("./data/dpo_tool_calling")
    except Exception:
        print("Please run src/prepare_dpo_data.py first!")
        return

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading SFT Model (Base + Adapter)...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    
    # Load the SFT adapter
    try:
        model = PeftModel.from_pretrained(base_model, sft_adapter_path)
        # For DPO with LoRA, we generally want to train on top of the SFT policy.
        # We can either merge (if not 4bit) or just use peft model directly.
        # With 4bit, we cannot merge easily. 
        # TRL's DPOTrainer handles PEFT models automatically if we pass peft_config.
        # Ideally, we treat the SFT model as the "Reference". 
        # But usually DPO requires a reference model AND a model to train.
        # If we use PEFT, the reference model is implicit (the frozen base model + frozen adapter state initially).
        print("SFT Adapter loaded.")
    except Exception as e:
        print(f"Could not load SFT adapter from {sft_adapter_path}. Error: {e}")
        print("Using base model only (Not recommended for DPO)")
        model = base_model

    # LoRA configuration for DPO
    # We want to fine-tune the weights further.
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = DPOConfig(
        output_dir=output_dir,
        beta=0.1, # Strength of KL penalty
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5, # Lower LR for DPO
        logging_steps=10,
        save_steps=100,
        num_train_epochs=1,
        max_length=1024,
        max_prompt_length=512,
        remove_unused_columns=False,
        bf16=True,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None, # None because we use PEFT; base model is ref
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer, # Replaces 'tokenizer' in newer TRL versions
        peft_config=peft_config,
    )

    print("Starting DPO training...")
    trainer.train()
    print("DPO Training finished.")
    
    trainer.save_model(f"{output_dir}-final")
    print("DPO Model saved.")

if __name__ == "__main__":
    main()

