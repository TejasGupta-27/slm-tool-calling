import os
import sys
import json
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
import re

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_tool_calling_dataset
from src.evaluate import format_prompt_for_eval, parse_tool_calls

def test_model_outputs(num_samples=3):
    """
    Test script to inspect outputs from both base and fine-tuned models.
    Sequential execution to save memory.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    base_model_name = "microsoft/Phi-3-medium-4k-instruct"
    peft_model_path = "./models/phi3-medium-tool-calling-final"
    
    print("=" * 80)
    print("Loading dataset...")
    eval_dataset = load_tool_calling_dataset(start=6000, end=6000 + num_samples)
    if not eval_dataset:
        print("Failed to load dataset")
        return
    
    print(f"Loaded {len(eval_dataset)} samples")
    
    # Pre-compute prompts
    samples_data = []
    for idx, sample in enumerate(eval_dataset):
        prompt = format_prompt_for_eval(sample)
        samples_data.append({
            "sample": sample,
            "prompt": prompt,
            "base_output": None,
            "ft_output": None
        })
    
    print("=" * 80)
    
    # Load tokenizer
    print("\n[1] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ---------------------------------------------------------
    # 1. Base Model Generation
    # ---------------------------------------------------------
    print("\n[2] Generating with BASE model...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="auto",
    )
    
    base_pipe = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    
    for i, data in enumerate(samples_data):
        print(f"  Generating sample {i+1}/{num_samples} (Base)...")
        outputs = base_pipe(
            data["prompt"],
            max_new_tokens=128,
            pad_token_id=tokenizer.pad_token_id,
            return_full_text=False
        )
        data["base_output"] = outputs[0]["generated_text"]
        
    # Cleanup Base Model
    print("  Cleaning up base model...")
    del base_pipe, base_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # ---------------------------------------------------------
    # 2. Fine-Tuned Model Generation
    # ---------------------------------------------------------
    print("\n[3] Generating with FINE-TUNED model...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="auto",
    )
    
    ft_model = PeftModel.from_pretrained(base_model, peft_model_path)
    # No merge for 4-bit
    
    ft_pipe = pipeline(
        "text-generation",
        model=ft_model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    
    for i, data in enumerate(samples_data):
        print(f"  Generating sample {i+1}/{num_samples} (Fine-Tuned)...")
        outputs = ft_pipe(
            data["prompt"],
            max_new_tokens=128,
            pad_token_id=tokenizer.pad_token_id,
            return_full_text=False
        )
        data["ft_output"] = outputs[0]["generated_text"]

    # Cleanup FT Model
    print("  Cleaning up fine-tuned model...")
    del ft_pipe, ft_model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # ---------------------------------------------------------
    # 3. Comparison Display
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    for i, data in enumerate(samples_data):
        print(f"\nSAMPLE {i+1}/{num_samples}")
        print("-" * 40)
        
        query = data["sample"]["query"]
        ground_truths = json.loads(data["sample"]["answers"])
        
        print(f"[Query]: {query}")
        print(f"[Ground Truth]:\n{json.dumps(ground_truths, indent=2)}")
        
        # Base Output Processing
        base_raw = data["base_output"]
        if "<|assistant|>" in base_raw:
            base_pred_str = base_raw.split("<|assistant|>", 1)[1].strip()
        else:
            base_pred_str = base_raw.strip()
        
        base_preds = parse_tool_calls(base_pred_str)
        
        print(f"\n[Base Model Output]:")
        print(f"Raw: {base_pred_str[:200]}..." if len(base_pred_str) > 200 else f"Raw: {base_pred_str}")
        print(f"Parsed: {json.dumps(base_preds, indent=2)}")
        
        # FT Output Processing
        ft_raw = data["ft_output"]
        if "<|assistant|>" in ft_raw:
            ft_pred_str = ft_raw.split("<|assistant|>", 1)[1].strip()
        else:
            ft_pred_str = ft_raw.strip()
            
        ft_preds = parse_tool_calls(ft_pred_str)
        
        print(f"\n[Fine-Tuned Model Output]:")
        print(f"Raw: {ft_pred_str[:200]}..." if len(ft_pred_str) > 200 else f"Raw: {ft_pred_str}")
        print(f"Parsed: {json.dumps(ft_preds, indent=2)}")
        
        print("-" * 80)

if __name__ == "__main__":
    test_model_outputs(num_samples=3)
