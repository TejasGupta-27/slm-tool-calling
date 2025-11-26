import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    print(f"Merging {args.adapter_path} into {args.base_model}...")

    # Load Base Model in 16-bit (not 4-bit) to allow merging
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("Loading adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    print("Merging adapter...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Merge complete.")

if __name__ == "__main__":
    main()

