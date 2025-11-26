import os
import json
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_tool_calling_dataset
from src.evaluate import format_prompt_for_eval, parse_tool_calls

def test_parsing():
    """
    Simple test to check if parsing works with expected formats.
    """
    print("=" * 80)
    print("Testing Parsing Logic")
    print("=" * 80)
    
    # Load one sample
    eval_dataset = load_tool_calling_dataset(start=6000, end=6001)
    if not eval_dataset:
        print("Failed to load dataset")
        return
    
    sample = eval_dataset[0]
    ground_truths = json.loads(sample["answers"])
    query = sample["query"]
    
    print(f"\n[Query]: {query[:200]}...")
    print(f"\n[Ground Truth]: {json.dumps(ground_truths, indent=2)}")
    
    # Show the prompt format
    prompt = format_prompt_for_eval(sample)
    print(f"\n[Prompt Format]:\n{prompt[:800]}...")
    
    # Test different output formats that the model might generate
    print("\n" + "=" * 80)
    print("Testing Parser with Different Formats")
    print("=" * 80)
    
    # Format 1: Training format (what model was trained on)
    # Training uses: function_name(**{"arg1": "value1", "arg2": "value2"})
    test_output_1 = 'get_team_info(**{"teamid": "team567"})\nget_team_info(**{"teamid": "team890"})'
    print(f"\n[Test Output 1 - Training Format]:\n{test_output_1}")
    parsed_1 = parse_tool_calls(test_output_1)
    print(f"[Parsed]: {json.dumps(parsed_1, indent=2)}")
    
    # Format 2: Keyword args format (what parser expects)
    test_output_2 = 'get_team_info(teamid="team567")\nget_team_info(teamid="team890")'
    print(f"\n[Test Output 2 - Keyword Args Format]:\n{test_output_2}")
    parsed_2 = parse_tool_calls(test_output_2)
    print(f"[Parsed]: {json.dumps(parsed_2, indent=2)}")
    
    # Format 3: With tool_code tags (training format)
    test_output_3 = '<|tool_code|>\nget_team_info(**{"teamid": "team567"})\nget_team_info(**{"teamid": "team890"})<|end|>'
    print(f"\n[Test Output 3 - With Tags]:\n{test_output_3}")
    parsed_3 = parse_tool_calls(test_output_3)
    print(f"[Parsed]: {json.dumps(parsed_3, indent=2)}")
    
    # Format 4: What ground truth looks like when converted
    print(f"\n[Ground Truth Format]:")
    for gt in ground_truths:
        gt_str = json.dumps(gt, sort_keys=True)
        print(f"  {gt_str}")
    
    # Check if parsed formats match ground truth
    print("\n" + "=" * 80)
    print("Format Comparison")
    print("=" * 80)
    
    # Convert ground truth to the format used for comparison
    gt_set = {json.dumps(g, sort_keys=True) for g in ground_truths}
    print(f"\n[Ground Truth Set]: {gt_set}")
    
    if parsed_1:
        pred_set_1 = {json.dumps(p, sort_keys=True) for p in parsed_1}
        print(f"[Parsed 1 Set]: {pred_set_1}")
        print(f"[Match?]: {pred_set_1 == gt_set}")
    
    if parsed_2:
        pred_set_2 = {json.dumps(p, sort_keys=True) for p in parsed_2}
        print(f"[Parsed 2 Set]: {pred_set_2}")
        print(f"[Match?]: {pred_set_2 == gt_set}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print("\nThe parser expects: function_name(arg1=value1, arg2=value2)")
    print("Training format uses: function_name(**{json.dumps(args)})")
    print("\nThis mismatch is likely why metrics are zero!")
    print("The model was trained to output the **{...} format,")
    print("but the parser only recognizes keyword argument format.")

if __name__ == "__main__":
    test_parsing()

