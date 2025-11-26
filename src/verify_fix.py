import sys
import os
import json
import re

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import parse_tool_calls, format_prompt_for_eval

def verify_fixes():
    print("=" * 80)
    print("VERIFYING FIXES FOR ZERO METRICS")
    print("=" * 80)

    # 1. Verify Parser Fix
    print("\n[1] Verifying Parser Logic...")
    
    # This is the format the model was trained on (which was failing before)
    training_format_output = """<|tool_code|>
get_team_info(**{"teamid": "team567"})
get_team_info(**{"teamid": "team890"})<|end|>"""
    
    print(f"Input String:\n{training_format_output}")
    
    parsed = parse_tool_calls(training_format_output)
    print(f"\nParsed Result: {json.dumps(parsed, indent=2)}")
    
    expected = [
        {"name": "get_team_info", "arguments": {"teamid": "team567"}},
        {"name": "get_team_info", "arguments": {"teamid": "team890"}}
    ]
    
    # Check equality (ignoring order for robustness)
    parsed_set = {json.dumps(p, sort_keys=True) for p in parsed}
    expected_set = {json.dumps(e, sort_keys=True) for e in expected}
    
    if parsed_set == expected_set:
        print("\n✅ SUCCESS: Parser now correctly handles the training format!")
    else:
        print("\n❌ FAILURE: Parser still cannot handle the training format.")
        print(f"Expected: {expected}")
        print(f"Got:      {parsed}")

    # 2. Verify Prompt Fix
    print("\n" + "-" * 80)
    print("[2] Verifying Prompt Format...")
    
    # Mock sample
    sample = {
        "query": "Test Query",
        "tools": json.dumps([
            {
                "name": "test_tool",
                "parameters": {"properties": {"arg1": {"type": "string"}}}
            }
        ])
    }
    
    prompt = format_prompt_for_eval(sample)
    print(f"Generated Prompt:\n{prompt}")
    
    if "<|system|>" in prompt and "<|user|>" in prompt:
        print("\n✅ SUCCESS: Prompt now uses the correct <|system|> tag matching training data.")
    else:
        print("\n❌ FAILURE: Prompt might still be using the wrong format (e.g., <|user|> at start).")

    print("\n" + "=" * 80)
    if parsed_set == expected_set and "<|system|>" in prompt:
        print("CONCLUSION: The logic causing zero metrics has been fixed.")
        print("You can now run the main evaluation script: python -m src.evaluate")
    else:
        print("CONCLUSION: Some fixes are still missing.")

if __name__ == "__main__":
    verify_fixes()

