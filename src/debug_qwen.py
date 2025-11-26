import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "Qwen/Qwen2.5-7B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print("Loading model...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="cuda:1",
        attn_implementation="eager"
    )
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt").to("cuda:1")

print("Generating...")
try:
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0]))
    print("Success!")
except Exception as e:
    print(f"Generation failed: {e}")

