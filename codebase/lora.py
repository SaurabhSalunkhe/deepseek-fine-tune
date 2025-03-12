import torch

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
auth_token = "hf_YourAuthTokenHere"  # remove if not needed

# Load the model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=auth_token,
    trust_remote_code=True,
    load_in_8bit=True,
    torch_dtype="auto",
    device_map="auto"
)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Wrap model with LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Fixed target modules
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, lora_config)
print("Model wrapped with LoRA.")
