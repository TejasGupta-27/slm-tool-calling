import json
import random
import copy
from datasets import Dataset
from src.data_loader import load_tool_calling_dataset

def perturb_tool_call(tool_calls, tools_def):
    """
    Creates a 'rejected' version of the tool call by introducing errors.
    Errors modeled:
    1. Wrong parameter name (hallucination) - most common error
    2. Missing parameter
    3. Wrong tool name
    """
    rejected_calls = copy.deepcopy(tool_calls)
    
    if not rejected_calls:
        # If empty, just add a garbage call
        return [{"name": "unknown_tool", "arguments": {}}]
    
    # Pick one call to corrupt
    target_call = random.choice(rejected_calls)
    error_type = random.choice(["param_name", "missing_param", "tool_name"])
    
    if error_type == "param_name" and target_call.get("arguments"):
        # Rename a key to a synonym-like hallucination
        args = target_call["arguments"]
        key = random.choice(list(args.keys()))
        val = args.pop(key)
        
        # Common hallucinations mappings could be used, or just random strings
        fake_key = key + "_id" if "id" not in key else key.replace("id", "")
        if fake_key == key: fake_key = key + "_value"
        
        args[fake_key] = val
        
    elif error_type == "missing_param" and target_call.get("arguments"):
        # Delete a random argument
        args = target_call["arguments"]
        key = random.choice(list(args.keys()))
        del args[key]
        
    elif error_type == "tool_name":
        # Change tool name
        target_call["name"] = target_call["name"] + "_tool"
        
    return rejected_calls

def format_dpo_entry(sample):
    query = sample["query"]
    tools = json.loads(sample["tools"])
    answers = json.loads(sample["answers"]) # Ground truth (Chosen)
    
    # Create Tool Definitions String (System Prompt)
    tool_defs = []
    for tool in tools:
        func_name = tool.get('name')
        params = tool.get('parameters', {}).get('properties', {})
        param_defs = [f"{pname}: {p.get('type') if isinstance(p, dict) else p}" for pname, p in params.items()]
        tool_defs.append(f"def {func_name}({', '.join(param_defs)}):")
    tool_defs_str = "\n".join(tool_defs)
    
    system_prompt = (
        f"<|system|>\nYou are an expert AI assistant with access to a suite of tools. Use them to answer the user's question.\n"
        f"Tools:\n{tool_defs_str}<|end|>\n"
        f"<|user|>\n{query}<|end|>\n"
        f"<|assistant|>\n"
    )
    
    # Format Chosen (Correct)
    chosen_lines = []
    for ans in answers:
        chosen_lines.append(f"{ans['name']}(**{json.dumps(ans['arguments'])})")
    chosen_text = "<|tool_code|>\n" + "\n".join(chosen_lines) + "<|end|>"
    
    # Format Rejected (Incorrect)
    rejected_data = perturb_tool_call(answers, tools)
    rejected_lines = []
    for ans in rejected_data:
        rejected_lines.append(f"{ans['name']}(**{json.dumps(ans['arguments'])})")
    rejected_text = "<|tool_code|>\n" + "\n".join(rejected_lines) + "<|end|>"
    
    return {
        "prompt": system_prompt,
        "chosen": chosen_text,
        "rejected": rejected_text
    }

def main():
    print("Loading dataset for DPO preparation...")
    dataset = load_tool_calling_dataset(start=0, end=10000) # Use same range as SFT or new
    if not dataset: return

    print("Generating preference pairs (Chosen vs Rejected)...")
    dpo_data = []
    for sample in dataset:
        dpo_data.append(format_dpo_entry(sample))
        
    dpo_dataset = Dataset.from_list(dpo_data)
    print(f"Created DPO dataset with {len(dpo_dataset)} samples.")
    
    # Save to disk for the training script to load
    dpo_dataset.save_to_disk("./data/dpo_tool_calling")
    print("Saved DPO dataset to ./data/dpo_tool_calling")

if __name__ == "__main__":
    main()

