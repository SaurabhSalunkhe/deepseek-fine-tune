# Import necessary modules
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set your model name.
# Replace this with the DeepSeek model you wish to use. For example:
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# If the model is public you might not need an auth token. Otherwise, insert your token below.
auth_token = "hf_YourAuthTokenHere"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_auth_token=auth_token,  # Omit this parameter if not required.
    trust_remote_code=True       # If the model uses custom code.
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# Test the model with a quick inference
input_text = "Hello, how are you today?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

