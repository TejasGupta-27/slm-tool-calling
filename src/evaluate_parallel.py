import os
import gc
import math
import json
import re
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    logging as hf_logging,
    BitsAndBytesConfig,
)
from peft import PeftModel
from tqdm import tqdm
from src.data_loader import load_tool_calling_dataset

hf_logging.set_verbosity_info()

def parse_tool_calls(prediction_str: str):
    """
    Parses the model's prediction string to extract tool calls.
    Handles both training format (**kwargs) and standard kwargs.
    """
    # Remove tool_code tags if present
    if "<|tool_code|>" in prediction_str:
        prediction_str = prediction_str.split("<|tool_code|>", 1)[1]
    if "<|end|>" in prediction_str:
        prediction_str = prediction_str.split("<|end|>", 1)[0]
    
    pattern = re.compile(r"(\w+)\((.*?)\)")
    matches = pattern.findall(prediction_str)

    calls = []
    for func_name, args_str in matches:
        if func_name == "print": continue

        try:
            args_str = args_str.strip()
            # Handle training format: **{"arg": "value"}
            if args_str.startswith("**"):
                json_str = args_str[2:].strip()
                if json_str.startswith("{") and json_str.endswith("}"):
                    args_dict = json.loads(json_str)
                    calls.append({"name": func_name, "arguments": args_dict})
                    continue
            
            # Handle keyword args format
            args_dict = eval(f"dict({args_str})", {"__builtins__": None}, {})
            calls.append({"name": func_name, "arguments": args_dict})
        except Exception:
            continue
    return calls

def calculate_metrics(predictions, ground_truths):
    """
    Calculates precision, recall, F1, Exact Match, and Tool Selection F1.
    """
    # 1. Exact Match
    pred_set = {json.dumps(p, sort_keys=True) for p in predictions}
    gt_set = {json.dumps(g, sort_keys=True) for g in ground_truths}
    
    correct_calls = len(pred_set.intersection(gt_set))
    precision = correct_calls / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = correct_calls / len(gt_set) if len(gt_set) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = 1 if pred_set == gt_set else 0

    # 2. Tool Selection Match
    pred_names = {p.get("name") for p in predictions}
    gt_names = {g.get("name") for g in ground_truths}
    correct_names = len(pred_names.intersection(gt_names))
    
    name_p = correct_names / len(pred_names) if len(pred_names) > 0 else 0.0
    name_r = correct_names / len(gt_names) if len(gt_names) > 0 else 0.0
    name_f1 = 2 * (name_p * name_r) / (name_p + name_r) if (name_p + name_r) > 0 else 0.0

    return precision, recall, f1, exact_match, name_f1

def calculate_hallucination_rate(predictions, available_tools):
    if not predictions: return 0.0
    
    available_tool_map = {}
    for tool in available_tools:
        name = tool.get("name")
        params = tool.get("parameters", {}).get("properties", {})
        available_tool_map[name] = set(params.keys())

    hallucinated_calls = 0
    for pred in predictions:
        tool_name = pred.get("name")
        args = pred.get("arguments", {}) or {}

        if tool_name not in available_tool_map:
            hallucinated_calls += 1
            continue
        
        valid_args = available_tool_map[tool_name]
        for arg_name in args:
            if arg_name not in valid_args:
                hallucinated_calls += 1
                break
    return hallucinated_calls / len(predictions)

def format_prompt(sample):
    query = sample["query"]
    tools = json.loads(sample["tools"])
    tool_defs = []
    for tool in tools:
        func_name = tool.get("name")
        params = tool.get("parameters", {}).get("properties", {})
        param_defs = [f"{pname}: {p.get('type') if isinstance(p, dict) else p}" for pname, p in params.items()]
        tool_defs.append(f"def {func_name}({', '.join(param_defs)}):")
    tool_defs_str = "\n".join(tool_defs)
    
    return (
        f"<|system|>\nYou are an expert AI assistant with access to a suite of tools. Use them to answer the user's question.\n"
        f"Tools:\n{tool_defs_str}<|end|>\n"
        f"<|user|>\n{query}<|end|>\n"
        f"<|assistant|>\n"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Base model HuggingFace ID")
    parser.add_argument("--peft_path", type=str, required=True, help="Path to fine-tuned adapter")
    parser.add_argument("--device", type=str, default="0", help="CUDA device ID")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    print(f"Loading dataset...")
    # Evaluating on the same 50 samples for quick comparison first
    # Change to full range(6000, 7000) for full eval
    eval_dataset = load_tool_calling_dataset(start=6000, end=6050) 
    prompts = [format_prompt(s) for s in eval_dataset]
    
    print(f"Loading Tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading Model: {args.base_model}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="auto",
    )
    
    print(f"Loading Adapter: {args.peft_path}")
    model = PeftModel.from_pretrained(base_model, args.peft_path)
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    
    print("Generating...")
    # Batch generation
    batch_size = 8
    all_outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        outs = pipe(batch, max_new_tokens=128, pad_token_id=tokenizer.pad_token_id)
        all_outputs.extend(outs)
        
    print("Calculating metrics...")
    results = {
        "exact_matches": 0,
        "precision": [], "recall": [], "f1": [], "name_f1": [], "hallucination_rate": []
    }
    
    for i, sample in enumerate(eval_dataset):
        ground_truths = json.loads(sample["answers"])
        available_tools = json.loads(sample["tools"])
        raw_output = all_outputs[i][0]["generated_text"]
        
        if "<|assistant|>" in raw_output:
            pred_str = raw_output.split("<|assistant|>", 1)[1].strip()
        else:
            pred_str = raw_output.strip()
            
        preds = parse_tool_calls(pred_str)
        p, r, f1, em, nf1 = calculate_metrics(preds, ground_truths)
        hr = calculate_hallucination_rate(preds, available_tools)
        
        results["exact_matches"] += em
        results["precision"].append(p)
        results["recall"].append(r)
        results["f1"].append(f1)
        results["name_f1"].append(nf1)
        results["hallucination_rate"].append(hr)
        
    n = len(eval_dataset)
    print("\n" + "="*40)
    print(f"RESULTS: {args.peft_path}")
    print("="*40)
    print(f"Exact Match:    {results['exact_matches'] / n:.4f}")
    print(f"Precision:      {sum(results['precision']) / n:.4f}")
    print(f"Recall:         {sum(results['recall']) / n:.4f}")
    print(f"F1 Score:       {sum(results['f1']) / n:.4f}")
    print(f"Tool Select F1: {sum(results['name_f1']) / n:.4f}")
    print(f"Hallucination:  {sum(results['hallucination_rate']) / n:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()

