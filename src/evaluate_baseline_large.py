import os
import torch
import argparse
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from src.evaluate import format_prompt_for_eval, parse_tool_calls, calculate_metrics, calculate_hallucination_rate
from src.data_loader import load_tool_calling_dataset

def main():
    parser = argparse.ArgumentParser(description="Evaluate a large baseline model")
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="HuggingFace model ID")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    print(f"Device set to use cuda:{args.device}")

    # Load Dataset
    print("Loading dataset...")
    # Load range 6000:7000 as used in other evaluations
    dataset = load_tool_calling_dataset(start=6000, end=7000)

    # Load Tokenizer
    print(f"Loading Tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model (4-bit)
    print(f"Loading Model: {args.model_id}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="eager" # Use eager to avoid flash-attn issues if not installed
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        pad_token_id=tokenizer.pad_token_id,
        return_full_text=False,
        device_map="auto"
    )

    print("Generating and Calculating Metrics...")
    
    # Initialize metric accumulators
    total_exact_match = 0
    precision_list = []
    recall_list = []
    f1_list = []
    name_f1_list = []
    hallucination_rate_list = []

    for i, sample in enumerate(dataset):
        if i % 50 == 0:
            print(f"Processing {i}/{len(dataset)}")
            
        prompt = format_prompt_for_eval(sample)
        ground_truth = json.loads(sample["answers"])
        available_tools = json.loads(sample["tools"])
        
        try:
            outputs = pipe(prompt)
            generated_text = outputs[0]["generated_text"]
            
            # Strip <|assistant|> if present
            if "<|assistant|>" in generated_text:
                generated_text = generated_text.split("<|assistant|>", 1)[1].strip()
            
            # Parse
            parsed = parse_tool_calls(generated_text)
            
            # Calculate metrics for this sample
            from src.evaluate import calculate_hallucination_rate # Import here if needed or at top
            
            # calculate_metrics in src/evaluate.py expects (predictions: List[Dict], ground_truths: List[Dict])
            p, r, f1, em, name_f1 = calculate_metrics(parsed, ground_truth)
            hr = calculate_hallucination_rate(parsed, available_tools)
            
            total_exact_match += em
            precision_list.append(p)
            recall_list.append(r)
            f1_list.append(f1)
            name_f1_list.append(name_f1)
            hallucination_rate_list.append(hr)
            
        except Exception as e:
            print(f"Error at index {i}: {e}")
            # Penalize for error
            precision_list.append(0.0)
            recall_list.append(0.0)
            f1_list.append(0.0)
            name_f1_list.append(0.0)
            hallucination_rate_list.append(0.0) # Or 1.0? If it crashed, maybe 0 hallucination but 0 recall.

    print("Calculating average metrics...")
    num_samples = len(dataset)
    avg_precision = sum(precision_list) / num_samples
    avg_recall = sum(recall_list) / num_samples
    avg_f1 = sum(f1_list) / num_samples
    avg_name_f1 = sum(name_f1_list) / num_samples
    avg_exact_match = total_exact_match / num_samples
    avg_hallucination = sum(hallucination_rate_list) / num_samples

    print(f"\n========================================")
    print(f"BASELINE RESULTS: {args.model_id}")
    print(f"========================================")
    print(f"Exact Match:    {avg_exact_match:.4f}")
    print(f"Precision:      {avg_precision:.4f}")
    print(f"Recall:         {avg_recall:.4f}")
    print(f"F1 Score:       {avg_f1:.4f}")
    print(f"Tool Select F1: {avg_name_f1:.4f}")
    print(f"Hallucination:  {avg_hallucination:.4f}")
    print(f"========================================")

if __name__ == "__main__":
    main()

