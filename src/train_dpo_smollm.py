import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from trl import DPOConfig, DPOTrainer

def main():
    # Set CUDA device to 1 (Shared with Llama)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        
    model_name = "HuggingFaceTB/SmolLM-360M-Instruct"
    sft_adapter_path = "./models/smollm-360m-tool-calling-final"
    output_dir = "./models/smollm-360m-tool-calling-dpo"
    
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

    print("Loading Base Model (4-bit)...")
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
    
    # Load the SFT adapter and make it TRAINABLE
    print(f"Loading SFT Adapter from {sft_adapter_path} (Trainable)...")
    model = PeftModel.from_pretrained(base_model, sft_adapter_path, is_trainable=True)
    model.print_trainable_parameters()

    training_args = DPOConfig(
        output_dir=output_dir,
        beta=0.1, 
        per_device_train_batch_size=4, # Can handle larger batch size for SmolLM
        gradient_accumulation_steps=4,
        learning_rate=1e-5, 
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
        ref_model=None, 
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting DPO training for SmolLM...")
    trainer.train()
    print("DPO Training finished.")
    
    trainer.save_model(f"{output_dir}-final")
    print("DPO Model saved.")

if __name__ == "__main__":
    main()

