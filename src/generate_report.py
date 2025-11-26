import os
import json
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
from src.data_loader import load_tool_calling_dataset
from src.evaluate import format_prompt_for_eval

def generate_for_model(model_id, adapter_path, dataset, device="0"):
    print(f"\n=== Processing {model_id} ===")
    print(f"Adapter: {adapter_path}")
    
    # Check if adapter exists before doing anything expensive
    if adapter_path and not os.path.exists(adapter_path):
        print(f"Warning: Adapter path {adapter_path} does not exist. Skipping.")
        return [{
            "query": sample["query"],
            "ground_truth": json.loads(sample["answers"]),
            "base_output": "SKIPPED",
            "ft_output": "ADAPTER_NOT_FOUND"
        } for sample in dataset]

    os.environ["CUDA_VISIBLE_DEVICES"] = device
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    # 1. Generate with BASE Model
    print("Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="auto",
    )
    
    base_pipe = pipeline("text-generation", model=base_model, tokenizer=tokenizer, device_map="auto")
    
    results = []
    
    for sample in dataset:
        prompt = format_prompt_for_eval(sample)
        
        # Base Generation
        try:
            base_out = base_pipe(prompt, max_new_tokens=128, pad_token_id=tokenizer.pad_token_id, return_full_text=False)
            base_text = base_out[0]["generated_text"]
            if "<|assistant|>" in base_text:
                base_text = base_text.split("<|assistant|>", 1)[1].strip()
            else:
                base_text = base_text.strip()
        except Exception as e:
            base_text = f"ERROR: {e}"
            
        results.append({
            "query": sample["query"],
            "ground_truth": json.loads(sample["answers"]),
            "base_output": base_text,
            "ft_output": "N/A (Baseline Only)" if not adapter_path else None # Placeholder
        })
        
    del base_pipe, base_model
    gc.collect()
    torch.cuda.empty_cache()
    
    if not adapter_path:
        return results

    # 2. Generate with FINE-TUNED Model
    print("Loading Fine-Tuned Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="auto",
    )
    
    try:
        ft_model = PeftModel.from_pretrained(base_model, adapter_path)
        ft_pipe = pipeline("text-generation", model=ft_model, tokenizer=tokenizer, device_map="auto")
        
        for i, sample in enumerate(dataset):
            prompt = format_prompt_for_eval(sample)
            try:
                ft_out = ft_pipe(prompt, max_new_tokens=128, pad_token_id=tokenizer.pad_token_id, return_full_text=False)
                ft_text = ft_out[0]["generated_text"]
                
                if "<|assistant|>" in ft_text:
                    ft_text = ft_text.split("<|assistant|>", 1)[1].strip()
                else:
                    ft_text = ft_text.strip()
            except Exception as e:
                ft_text = f"ERROR: {e}"
                
            results[i]["ft_output"] = ft_text
            
        del ft_pipe, ft_model
    except Exception as e:
        print(f"Error loading adapter {adapter_path}: {e}")
        for res in results:
            res["ft_output"] = f"ERROR_LOADING_ADAPTER: {e}"
            
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return results

def main():
    # Load 3 representative samples
    # Indices chosen to show variety
    dataset = load_tool_calling_dataset(start=6000, end=6003) 
    
    report_data = {}
    
    # Define all models to test
    models_to_test = [
        {
            "name": "Llama-3.2-1B (SFT)",
            "id": "meta-llama/Llama-3.2-1B-Instruct",
            "adapter": "./models/llama-3.2-1b-tool-calling-final"
        },
        {
            "name": "SmolLM-360M (SFT)",
            "id": "HuggingFaceTB/SmolLM-360M-Instruct",
            "adapter": "./models/smollm-360m-tool-calling-final"
        },
        {
            "name": "SmolLM-360M (DPO)",
            "id": "HuggingFaceTB/SmolLM-360M-Instruct",
            "adapter": "./models/smollm-360m-tool-calling-dpo-final"
        },
        {
            "name": "Qwen2.5-0.5B (SFT)",
            "id": "Qwen/Qwen2.5-0.5B-Instruct",
            "adapter": "./models/qwen2.5-0.5b-tool-calling-final"
        },
        {
            "name": "Qwen2.5-1.5B (SFT)",
            "id": "Qwen/Qwen2.5-1.5B-Instruct",
            "adapter": "./models/qwen2.5-1.5b-tool-calling-final"
        },
        {
            "name": "Phi-3-Medium (SFT)",
            "id": "microsoft/Phi-3-medium-4k-instruct",
            "adapter": "./models/phi3-medium-tool-calling-final"
        },
        {
            "name": "Phi-3-Medium (DPO)",
            "id": "microsoft/Phi-3-medium-4k-instruct",
            "adapter": "./models/phi3-medium-tool-calling-dpo-final"
        },
        {
            "name": "Mistral-7B-Baseline",
            "id": "mistralai/Mistral-7B-Instruct-v0.3",
            "adapter": None
        }
    ]
    
    # Using GPU 1 as it is currently free
    device_id = "1"
    
    for m in models_to_test:
        try:
            report_data[m["name"]] = generate_for_model(
                m["id"], 
                m["adapter"], 
                dataset, 
                device=device_id
            )
        except Exception as e:
            print(f"Failed to process {m['name']}: {e}")
            report_data[m["name"]] = {"error": str(e)}
        
    # Save to JSON
    output_file = "report_outputs.json"
    with open(output_file, "w") as f:
        json.dump(report_data, f, indent=2)
        
    print(f"\nReport generated: {output_file}")

if __name__ == "__main__":
    main()
