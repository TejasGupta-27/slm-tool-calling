import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from src.data_loader import load_tool_calling_dataset
import json
from tqdm import tqdm
import re
import ast
import gc
import os

def load_models_and_tokenizer(base_model_name="microsoft/Phi-3-medium-4k-instruct", peft_model_path="./models/phi3-medium-tool-calling-final"):
    """
    Loads the base model, fine-tuned PEFT model, and tokenizer.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    # Load fine-tuned model
    print("Loading fine-tuned model...")
    ft_model = PeftModel.from_pretrained(base_model, peft_model_path)
    ft_model = ft_model.merge_and_unload() # Merge LoRA layers

    print("Models and tokenizer loaded successfully.")
    return base_model, ft_model, tokenizer

def generate_prediction(model, tokenizer, prompt):
    """
    Generates a prediction for a given prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        pad_token_id=tokenizer.pad_token_id
    )
    
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction

def format_prompt_for_eval(sample):
    """
    Formats a sample into the same prompt structure used for training.
    """
    query = sample['query']
    tools = json.loads(sample['tools'])
    
    tool_defs = []
    for tool in tools:
        func_name = tool.get('name')
        params = tool.get('parameters', {}).get('properties', {})
        param_defs = [f"{pname}: {p.get('type') if isinstance(p, dict) else p}" for pname, p in params.items()]
        tool_defs.append(f"def {func_name}({', '.join(param_defs)}):")

    tool_defs_str = "\n".join(tool_defs)

    prompt = (
        f"<|user|>\nYou are an expert AI assistant with access to a suite of tools. Use them to answer the user's question.\n"
        f"Tools:\n{tool_defs_str}<|end|>\n"
        f"<|user|>\n{query}<|end|>\n"
        f"<|assistant|>\n"
    )
    return prompt

def parse_tool_calls(prediction_str: str):
    """
    Parses the model's prediction string to extract tool calls using a more flexible regex.
    """
    # Regex to find function calls like `function_name(arg1=value1, arg2="value2")`
    # It handles string arguments, integer/float arguments, and lists/dicts.
    pattern = re.compile(r"(\w+)\((.*?)\)")
    matches = pattern.findall(prediction_str)
    
    calls = []
    for match in matches:
        func_name = match[0]
        args_str = match[1]

        # Don't parse the print function
        if func_name == 'print':
            continue

        try:
            # A bit of a hack to parse Python-like arguments
            # We create a dummy function definition and parse the arguments
            args_dict = eval(f"dict({args_str})", {"__builtins__": None}, {})
            calls.append({"name": func_name, "arguments": args_dict})
        except Exception:
            # Fallback for simple, non-keyword arguments if the above fails
            try:
                # This is very basic and might not cover all cases
                # e.g., get_user_follower_list('user123', 100)
                # We need to know the argument names for this to be robust.
                # For this dataset, let's assume simple cases can be mapped.
                # This part will need improvement for a general solution.
                pass # Sticking to keyword-based parsing for now.
            except Exception:
                continue
    return calls

def calculate_metrics(predictions, ground_truths):
    """
    Calculates precision, recall, and F1 score for tool calls.
    """
    pred_set = {json.dumps(p, sort_keys=True) for p in predictions}
    gt_set = {json.dumps(g, sort_keys=True) for g in ground_truths}
    
    correct_calls = len(pred_set.intersection(gt_set))
    
    precision = correct_calls / len(pred_set) if len(pred_set) > 0 else 0
    recall = correct_calls / len(gt_set) if len(gt_set) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    exact_match = 1 if pred_set == gt_set else 0

    return precision, recall, f1, exact_match

def calculate_hallucination_rate(predictions, available_tools):
    """
    Calculates the hallucination rate of the model.
    """
    if not predictions:
        return 0
        
    available_tool_map = {tool['name']: tool['parameters']['properties'].keys() for tool in available_tools}
    hallucinated_calls = 0
    
    for pred in predictions:
        if pred['name'] not in available_tool_map:
            hallucinated_calls += 1
            continue # If the tool itself is hallucinated, no need to check args
        
        valid_args = available_tool_map[pred['name']]
        for arg_name in pred['arguments']:
            if arg_name not in valid_args:
                hallucinated_calls += 1
                break # Move to the next predicted call

    return hallucinated_calls / len(predictions)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    base_model_name = "microsoft/Phi-3-medium-4k-instruct"
    peft_model_path = "./models/phi3-medium-tool-calling-final"
    
    eval_dataset = load_tool_calling_dataset(start=6000, end=7000)
    if not eval_dataset:
        print("Failed to load evaluation dataset. Exiting.")
        return

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    prompts = [format_prompt_for_eval(sample) for sample in eval_dataset]

    results = {
        "fine_tuned": {"exact_matches": 0, "precision": [], "recall": [], "f1": [], "hallucination_rate": []},
        "base": {"exact_matches": 0, "precision": [], "recall": [], "f1": [], "hallucination_rate": []}
    }

    # --- Evaluate Fine-Tuned Model ---
    print("Loading and evaluating fine-tuned model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=torch.bfloat16, device_map="auto"
    )
    ft_model = PeftModel.from_pretrained(base_model, peft_model_path)
    ft_model = ft_model.merge_and_unload()
    
    ft_pipe = pipeline("text-generation", model=ft_model, tokenizer=tokenizer, device_map="auto")
    ft_outputs = ft_pipe(prompts, max_new_tokens=128, pad_token_id=tokenizer.pad_token_id, batch_size=8)

    for i, sample in tqdm(enumerate(eval_dataset), desc="Metrics (Fine-Tuned)", total=len(eval_dataset)):
        ground_truths = json.loads(sample['answers'])
        available_tools = json.loads(sample['tools'])
        ft_raw_output = ft_outputs[i][0]['generated_text']
        ft_pred_str = ft_raw_output.split('<|assistant|>')[1].strip() if '<|assistant|>' in ft_raw_output else ""
        ft_preds = parse_tool_calls(ft_pred_str)
        p, r, f1, em = calculate_metrics(ft_preds, ground_truths)
        hr = calculate_hallucination_rate(ft_preds, available_tools)
        results["fine_tuned"]["exact_matches"] += em
        results["fine_tuned"]["precision"].append(p)
        results["fine_tuned"]["recall"].append(r)
        results["fine_tuned"]["f1"].append(f1)
        results["fine_tuned"]["hallucination_rate"].append(hr)

    del base_model, ft_model, ft_pipe
    gc.collect()
    torch.cuda.empty_cache()

    # --- Evaluate Base Model ---
    print("\nLoading and evaluating base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=torch.bfloat16, device_map="auto"
    )
    base_pipe = pipeline("text-generation", model=base_model, tokenizer=tokenizer, device_map="auto")
    base_outputs = base_pipe(prompts, max_new_tokens=128, pad_token_id=tokenizer.pad_token_id, batch_size=8)
    
    for i, sample in tqdm(enumerate(eval_dataset), desc="Metrics (Base)", total=len(eval_dataset)):
        ground_truths = json.loads(sample['answers'])
        available_tools = json.loads(sample['tools'])
        base_raw_output = base_outputs[i][0]['generated_text']
        base_pred_str = base_raw_output.split('<|assistant|>')[1].strip() if '<|assistant|>' in base_raw_output else ""
        base_preds = parse_tool_calls(base_pred_str)
        p, r, f1, em = calculate_metrics(base_preds, ground_truths)
        hr = calculate_hallucination_rate(base_preds, available_tools)
        results["base"]["exact_matches"] += em
        results["base"]["precision"].append(p)
        results["base"]["recall"].append(r)
        results["base"]["f1"].append(f1)
        results["base"]["hallucination_rate"].append(hr)
    
    # Print results
    print("\n--- Evaluation Results ---")
    for model_name, res in results.items():
        num_samples = len(eval_dataset)
        avg_precision = sum(res['precision']) / num_samples
        avg_recall = sum(res['recall']) / num_samples
        avg_f1 = sum(res['f1']) / num_samples
        exact_match_rate = res['exact_matches'] / num_samples
        avg_hallucination_rate = sum(res['hallucination_rate']) / num_samples
        
        print(f"\nModel: {model_name.replace('_', ' ').title()}")
        print(f"  Exact Match Rate: {exact_match_rate:.4f}")
        print(f"  Precision: {avg_precision:.4f}")
        print(f"  Recall: {avg_recall:.4f}")
        print(f"  F1 Score: {avg_f1:.4f}")
        print(f"  Hallucination Rate: {avg_hallucination_rate:.4f}")

if __name__ == "__main__":
    main()
