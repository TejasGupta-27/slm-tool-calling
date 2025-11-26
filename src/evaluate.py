import os
import gc
import math
import json
import re

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

hf_logging.set_verbosity_info()  # more verbose transformers logs


def load_models_and_tokenizer(
    base_model_name="microsoft/Phi-3-medium-4k-instruct",
    peft_model_path="./models/phi3-medium-tool-calling-final",
):
    """
    Loads the base model, fine-tuned PEFT model, and tokenizer.
    (Not used in main(), but kept in case you want it for separate scripts.)
    """
    # Load tokenizer
    print("[LOAD] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[LOAD] Tokenizer loaded. pad_token_id =", tokenizer.pad_token_id)

    # Load base model
    print("[LOAD] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    print("[LOAD] Base model loaded.")

    # Load fine-tuned model
    print(f"[LOAD] Loading PEFT adapter from: {peft_model_path}")
    ft_model = PeftModel.from_pretrained(base_model, peft_model_path)
    print("[LOAD] PEFT adapter loaded, merging and unloading LoRA layers...")
    ft_model = ft_model.merge_and_unload()  # Merge LoRA layers
    print("[LOAD] LoRA layers merged. Fine-tuned model ready.")

    print("[LOAD] Models and tokenizer loaded successfully.")
    return base_model, ft_model, tokenizer


def generate_prediction(model, tokenizer, prompt):
    """
    Generates a prediction for a given prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        pad_token_id=tokenizer.pad_token_id,
    )

    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction


def format_prompt_for_eval(sample):
    """
    Formats a sample into the same prompt structure used for training.
    Matches the training format exactly.
    """
    query = sample["query"]
    tools = json.loads(sample["tools"])

    tool_defs = []
    for tool in tools:
        func_name = tool.get("name")
        params = tool.get("parameters", {}).get("properties", {})
        param_defs = [
            f"{pname}: {p.get('type') if isinstance(p, dict) else p}"
            for pname, p in params.items()
        ]
        tool_defs.append(f"def {func_name}({', '.join(param_defs)}):")

    tool_defs_str = "\n".join(tool_defs)

    # Match training format exactly: <|system|> for instructions, <|user|> for query
    prompt = (
        f"<|system|>\nYou are an expert AI assistant with access to a suite of tools. Use them to answer the user's question.\n"
        f"Tools:\n{tool_defs_str}<|end|>\n"
        f"<|user|>\n{query}<|end|>\n"
        f"<|assistant|>\n"
    )
    return prompt


def parse_tool_calls(prediction_str: str):
    """
    Parses the model's prediction string to extract tool calls using a regex.

    Handles multiple formats:
    1. Training format: function_name(**{"arg1": "value1", "arg2": "value2"})
    2. Keyword args format: function_name(arg1=value1, arg2="value2")
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
        # Don't parse the print function
        if func_name == "print":
            continue

        try:
            args_str = args_str.strip()
            
            # Handle training format: **{"arg": "value"}
            if args_str.startswith("**"):
                # Extract the JSON part after **
                json_str = args_str[2:].strip()
                if json_str.startswith("{") and json_str.endswith("}"):
                    args_dict = json.loads(json_str)
                    calls.append({"name": func_name, "arguments": args_dict})
                    continue
            
            # Handle keyword args format: arg1=value1, arg2="value2"
            # Parse Python-like keyword args safely (no builtins)
            args_dict = eval(f"dict({args_str})", {"__builtins__": None}, {})
            calls.append({"name": func_name, "arguments": args_dict})
        except Exception as e:
            # If we can't parse, skip this call
            continue

    return calls


def calculate_metrics(predictions, ground_truths):
    """
    Calculates precision, recall, and F1 score for tool calls (Exact Match).
    Also calculates "Tool Selection" metrics (ignoring arguments).
    """
    # 1. Exact Match (Function Name + Arguments)
    pred_set = {json.dumps(p, sort_keys=True) for p in predictions}
    gt_set = {json.dumps(g, sort_keys=True) for g in ground_truths}

    correct_calls = len(pred_set.intersection(gt_set))

    precision = correct_calls / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = correct_calls / len(gt_set) if len(gt_set) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    exact_match = 1 if pred_set == gt_set else 0

    # 2. Tool Selection Match (Function Name Only)
    # Using counts to handle multiple calls to same tool
    pred_names = [p.get("name") for p in predictions]
    gt_names = [g.get("name") for g in ground_truths]
    
    # Simple set intersection for unique tools (easier metric)
    pred_name_set = set(pred_names)
    gt_name_set = set(gt_names)
    
    correct_names = len(pred_name_set.intersection(gt_name_set))
    
    name_precision = correct_names / len(pred_name_set) if len(pred_name_set) > 0 else 0.0
    name_recall = correct_names / len(gt_name_set) if len(gt_name_set) > 0 else 0.0
    name_f1 = (
        2 * (name_precision * name_recall) / (name_precision + name_recall)
        if (name_precision + name_recall) > 0
        else 0.0
    )

    return precision, recall, f1, exact_match, name_f1


def calculate_hallucination_rate(predictions, available_tools):
    """
    Calculates the hallucination rate of the model:
    fraction of predicted calls which either:
      - use a non-existent tool, or
      - use at least one non-existent argument.
    """
    if not predictions:
        return 0.0

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
            continue  # no need to check args

        valid_args = available_tool_map[tool_name]
        for arg_name in args:
            if arg_name not in valid_args:
                hallucinated_calls += 1
                break

    return hallucinated_calls / len(predictions)


def batched_generate(pipe, prompts, batch_size=8, desc="Generating"):
    """
    Generate in batches with clear logs so we can see progress and where it might hang.
    """
    all_outputs = []
    n = len(prompts)
    num_batches = math.ceil(n / batch_size)
    print(
        f"[GEN] Starting generation: {n} prompts, "
        f"batch_size={batch_size}, num_batches={num_batches}"
    )

    for b_idx, start in enumerate(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        batch_prompts = prompts[start:end]

        print(
            f"[GEN] Batch {b_idx + 1}/{num_batches} "
            f"(indices {start}..{end - 1}) -> calling pipeline"
        )
        outputs = pipe(
            batch_prompts,
            max_new_tokens=128,
            pad_token_id=pipe.tokenizer.pad_token_id,
        )
        print(f"[GEN] Batch {b_idx + 1}/{num_batches} done")
        all_outputs.extend(outputs)

    print("[GEN] All batches finished")
    return all_outputs


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("[MAIN] CUDA_VISIBLE_DEVICES set to 0")

    base_model_name = "microsoft/Phi-3-medium-4k-instruct"
    peft_model_path = "./models/phi3-medium-tool-calling-final"

    print("[MAIN] Loading evaluation dataset...")
    eval_dataset = load_tool_calling_dataset(start=6000, end=7000)
    # eval_dataset = load_tool_calling_dataset(start=6000, end=6050) # 50 samples
    if not eval_dataset:
        print("[ERROR] Failed to load evaluation dataset. Exiting.")
        return

    print(f"[MAIN] Eval dataset loaded: {len(eval_dataset)} samples")

    print("[MAIN] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[MAIN] Tokenizer loaded. pad_token_id =", tokenizer.pad_token_id)

    print("[MAIN] Formatting prompts...")
    prompts = [format_prompt_for_eval(sample) for sample in eval_dataset]
    print(f"[MAIN] Formatted {len(prompts)} prompts")
    if len(prompts) > 0:
        print("[DEBUG] First prompt snippet:\n", prompts[0][:500], "\n--- END SNIPPET ---")

    results = {
        "fine_tuned": {
            "exact_matches": 0,
            "precision": [],
            "recall": [],
            "f1": [],
            "name_f1": [],
            "hallucination_rate": [],
        },
        "base": {
            "exact_matches": 0,
            "precision": [],
            "recall": [],
            "f1": [],
            "name_f1": [],
            "hallucination_rate": [],
        },
    }

    # --- Evaluate Fine-Tuned Model ---
    print("\n[FT] Loading and evaluating fine-tuned model...")
    print("[FT] Loading base model (for PEFT) ...")
    
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
    print("[FT] Base model loaded")

    print(f"[FT] Loading PEFT adapter from: {peft_model_path}")
    ft_model = PeftModel.from_pretrained(base_model, peft_model_path)
    # Do NOT merge_and_unload when using 4bit/8bit quantization as it's not supported easily
    # ft_model = ft_model.merge_and_unload() 
    print("[FT] PEFT adapter loaded. Model ready.")

    print("[FT] Creating text-generation pipeline...")
    ft_pipe = pipeline(
        "text-generation",
        model=ft_model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    print("[FT] Pipeline created. Starting generation...")

    ft_outputs = batched_generate(
        ft_pipe, prompts, batch_size=8, desc="Fine-tuned generation"
    )

    print("[FT] Generation done. Computing metrics...")
    for i, sample in tqdm(
        enumerate(eval_dataset),
        desc="Metrics (Fine-Tuned)",
        total=len(eval_dataset),
    ):
        if i % 50 == 0:
            print(f"[FT] Computing metrics for sample {i}/{len(eval_dataset)}")

        ground_truths = json.loads(sample["answers"])
        available_tools = json.loads(sample["tools"])
        ft_raw_output = ft_outputs[i][0]["generated_text"]

        if "<|assistant|>" in ft_raw_output:
            ft_pred_str = ft_raw_output.split("<|assistant|>", 1)[1].strip()
        else:
            ft_pred_str = ft_raw_output.strip()

        ft_preds = parse_tool_calls(ft_pred_str)
        p, r, f1, em, name_f1 = calculate_metrics(ft_preds, ground_truths)
        hr = calculate_hallucination_rate(ft_preds, available_tools)

        results["fine_tuned"]["exact_matches"] += em
        results["fine_tuned"]["precision"].append(p)
        results["fine_tuned"]["recall"].append(r)
        results["fine_tuned"]["f1"].append(f1)
        results["fine_tuned"]["name_f1"].append(name_f1)
        results["fine_tuned"]["hallucination_rate"].append(hr)

    print("[FT] Metrics computed. Cleaning up fine-tuned model...")
    del base_model, ft_model, ft_pipe
    gc.collect()
    torch.cuda.empty_cache()
    print("[FT] Cleanup done.")

    # --- Evaluate Base Model ---
    print("\n[BASE] Loading and evaluating base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="auto",
    )
    print("[BASE] Base model loaded")

    print("[BASE] Creating text-generation pipeline...")
    base_pipe = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    print("[BASE] Pipeline created. Starting generation...")

    base_outputs = batched_generate(
        base_pipe, prompts, batch_size=8, desc="Base generation"
    )

    print("[BASE] Generation done. Computing metrics...")
    for i, sample in tqdm(
        enumerate(eval_dataset), desc="Metrics (Base)", total=len(eval_dataset)
    ):
        if i % 50 == 0:
            print(f"[BASE] Computing metrics for sample {i}/{len(eval_dataset)}")

        ground_truths = json.loads(sample["answers"])
        available_tools = json.loads(sample["tools"])
        base_raw_output = base_outputs[i][0]["generated_text"]

        if "<|assistant|>" in base_raw_output:
            base_pred_str = base_raw_output.split("<|assistant|>", 1)[1].strip()
        else:
            base_pred_str = base_raw_output.strip()

        base_preds = parse_tool_calls(base_pred_str)
        p, r, f1, em, name_f1 = calculate_metrics(base_preds, ground_truths)
        hr = calculate_hallucination_rate(base_preds, available_tools)

        results["base"]["exact_matches"] += em
        results["base"]["precision"].append(p)
        results["base"]["recall"].append(r)
        results["base"]["f1"].append(f1)
        results["base"]["name_f1"].append(name_f1)
        results["base"]["hallucination_rate"].append(hr)

    print("[BASE] Metrics computed. Cleaning up base model...")
    del base_model, base_pipe
    gc.collect()
    torch.cuda.empty_cache()
    print("[BASE] Cleanup done.")

    # --- Print results ---
    print("\n--- Evaluation Results ---")
    num_samples = len(eval_dataset)
    for model_name, res in results.items():
        avg_precision = sum(res["precision"]) / num_samples
        avg_recall = sum(res["recall"]) / num_samples
        avg_f1 = sum(res["f1"]) / num_samples
        avg_name_f1 = sum(res["name_f1"]) / num_samples
        exact_match_rate = res["exact_matches"] / num_samples
        avg_hallucination_rate = sum(res["hallucination_rate"]) / num_samples

        print(f"\nModel: {model_name.replace('_', ' ').title()}")
        print(f"  Exact Match Rate: {exact_match_rate:.4f}")
        print(f"  Precision:        {avg_precision:.4f}")
        print(f"  Recall:           {avg_recall:.4f}")
        print(f"  F1 Score:         {avg_f1:.4f}")
        print(f"  Tool Select F1:   {avg_name_f1:.4f}")
        print(f"  Hallucination Rt: {avg_hallucination_rate:.4f}")


if __name__ == "__main__":
    main()
