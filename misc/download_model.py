from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_name = "Qwen/Qwen2.5-Math-1.5B"

# Create models directory if it doesn't exist
CACHE = os.environ.get("CACHE", "~/.cache/")  # 默认值保留原路径
os.makedirs(os.path.join(CACHE, f"hf_models/{model_name}"), exist_ok=True)

print(f"Downloading {model_name} model and tokenizer...")

# Download and save the model
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Save to local directory
local_path = os.path.join(CACHE, f"hf_models/{model_name}")
print(f"Saving model and tokenizer to {local_path}")
model.save_pretrained(local_path)
tokenizer.save_pretrained(local_path)

print("Download completed!") 