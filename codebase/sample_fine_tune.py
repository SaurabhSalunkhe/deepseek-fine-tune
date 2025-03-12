import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Define model name and your Hugging Face authentication token
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
auth_token = ""  # Replace with your actual token

# Set up quantization for memory efficiency (8-bit loading)
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf8",
    bnb_8bit_compute_dtype=torch.bfloat16
)

# Load the base model with quantization
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=auth_token,
    trust_remote_code=True,
    quantization_config=quant_config,
    torch_dtype="auto",
    device_map="auto"
)

# Load the tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load your patient discharge summaries dataset
sample_data = [
    {"text": "Patient admitted with chest pain. Discharged after 3 days with stable vitals."},
    {"text": "65-year-old male with pneumonia. Treated with antibiotics, discharged after 5 days."},
    {"text": "Patient presented with sprained ankle. Given pain medication and discharged same day."},
    {"text": "Diabetic patient admitted for monitoring. Discharged with adjusted insulin dosage."}
]
# Convert the list of dictionaries to a Dataset object
dataset = Dataset.from_list(sample_data)

# Preprocess the dataset (tokenize the text)
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Set up data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Configure LoRA (Low-Rank Adaptation)
lora_config = LoraConfig(
    r=8,              # Rank of the adaptation
    lora_alpha=32,    # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Transformer modules to adapt
    lora_dropout=0.1, # Dropout for regularization
    bias="none"       # No bias adaptation
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
print("Model wrapped with LoRA.")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_deepseek",      # Where to save the model
    num_train_epochs=3,                      # Number of training epochs
    per_device_train_batch_size=4,           # Batch size per device
    gradient_accumulation_steps=2,           # Accumulate gradients for larger effective batch size
    logging_steps=10,                        # Log every 10 steps
    save_steps=100,                          # Save checkpoint every 100 steps
    fp16=True if torch.cuda.is_available() else False,  # Use mixed precision if GPU is available
    learning_rate=5e-5,                      # Learning rate
    report_to="none"                         # Disable external reporting
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Start fine-tuning
print("Starting fine-tuning...")
start_train = time.time()
trainer.train()
end_train = time.time()
print(f"Fine-tuning completed in {end_train - start_train:.2f} seconds.")

# Save the fine-tuned model
trainer.save_model("./fine_tuned_deepseek")
print("Fine-tuned model saved.")
