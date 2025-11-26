import os
import json
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
from src.data_loader import load_tool_calling_dataset
from src.evaluate import format_prompt_for_eval

def generate_for_model(model_id, adapter_path, dataset, device="0"):
    print(f"\n=== Processing {model_id} (Adapter: {adapter_path}) ===")
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    print("Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="auto",
    )
    
    results = []
    
    print(f"Loading Adapter from {adapter_path}...")
    try:
        ft_model = PeftModel.from_pretrained(base_model, adapter_path)
        ft_pipe = pipeline("text-generation", model=ft_model, tokenizer=tokenizer, device_map="auto")
        
        for i, sample in enumerate(dataset):
            prompt = format_prompt_for_eval(sample)
            print(f"\n--- Sample {i+1} Prompt ---\n{prompt[:200]}...") # Print start of prompt
            
            ft_out = ft_pipe(prompt, max_new_tokens=128, pad_token_id=tokenizer.pad_token_id, return_full_text=False)
            ft_text = ft_out[0]["generated_text"]
            
            print(f"--- Raw Output ---\n{ft_text}\n------------------")
            
            results.append({
                "query": sample["query"],
                "raw_output": ft_text
            })
            
        del ft_pipe, ft_model
    except Exception as e:
        print(f"Error loading adapter {adapter_path}: {e}")
            
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return results

def main():
    dataset = load_tool_calling_dataset(start=6000, end=6003) 
    
    # Test Phi-3 DPO
    generate_for_model(
        "microsoft/Phi-3-medium-4k-instruct", 
        "./models/phi3-medium-tool-calling-dpo-final", 
        dataset, 
        device="2"
    )

if __name__ == "__main__":
    main()

