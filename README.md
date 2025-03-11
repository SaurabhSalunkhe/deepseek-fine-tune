Fine-Tuning DeepSeek-R1-Distill-Qwen-1.5B for Healthcare Discharge Summaries

Introduction

This project outlines high-level steps to fine-tune the DeepSeek-R1-Distill-Qwen-1.5B large language model specifically to generate patient discharge summaries tailored for healthcare settings. Fine-tuning significantly improves the model's accuracy and relevance compared to methods like Retrieval-Augmented Generation (RAG). Key concepts covered include model installation, quantization, parameter-efficient fine-tuning using LoRA, and critical hyperparameter tuning.

Sample Dataset Overview

The provided sample dataset consists of artificially generated patient records, each featuring a concise clinical note paired with a detailed discharge summary. This represents a sequence-to-sequence task where the model converts brief, technical clinical notes into structured, clear summaries.

Example:

Clinical Note:

"36 y/o female, adm 2025-03-07, c/o headache x 1 week. Dx: appendicitis. PMH: hyperlipidemia. Tx: azithromycin 500 mg x 1 then 250 mg QD x 4 days, albuterol PRN. CT scan: 90% RCA occlusion. Labs: CRP 15 mg/L. D/c 2025-03-07."

Discharge Summary:

"Ms. Joseph Morton, 36, was admitted on 2025-03-07 with headache for 1 week, diagnosed with appendicitis. History of hyperlipidemia. Treated with azithromycin 500 mg x 1 then 250 mg QD x 4 days, albuterol PRN. CT scan revealed 90% RCA occlusion. Labs indicated CRP 15 mg/L. Discharged on 2025-03-07. Follow up with your cardiologist in 1 week."

Step-by-Step Fine-Tuning Guide

1. Installing and Loading the Model

Begin by installing necessary libraries (transformers, torch, peft, and bitsandbytes) and load the DeepSeek-R1-Distill-Qwen-1.5B model and tokenizer from Hugging Face. Ensure to configure trust settings appropriately based on the model’s documentation.

2. Quantization

Quantization is used to reduce the memory footprint of the model and enhance inference speed by converting weights from high precision (32-bit floats) to lower precision (8-bit integers). Benefits include:

Reduced memory usage: Ideal for deployment in resource-constrained environments.

Faster inference: Essential for real-time healthcare applications.

Energy efficiency: Lowers computational costs and improves sustainability.

3. Implementing LoRA for Efficient Fine-Tuning

Low-Rank Adaptation (LoRA) fine-tunes the model efficiently by adding low-rank matrices to specific layers while keeping original weights frozen, significantly reducing computational and memory requirements. Important parameters:

Rank (r): Typically set to 8 or 16.

Alpha: Usually starts around 32, adjustable based on task complexity.

Target Modules: Focus primarily on attention layers such as q_proj and v_proj.

4. Hyperparameter Tuning

Critical hyperparameters to fine-tune for optimal performance include:

Learning Rate: Recommended start at 2e-5, reduce if overfitting occurs.

Batch Size: Typically 4 or 8; use gradient accumulation if memory constraints arise.

Epochs: Usually 3–5, monitoring validation loss closely.

5. Generating Discharge Summaries

After fine-tuning, the model will be capable of converting clinical notes into comprehensive discharge summaries. Evaluation and inference should be performed carefully to ensure the generated summaries meet clinical and documentation standards.

Conclusion

Fine-tuning DeepSeek-R1-Distill-Qwen-1.5B with quantization and LoRA provides an effective solution for creating accurate and compliant patient discharge summaries. This approach improves upon traditional methods by internalizing medical knowledge, understanding clinical jargon, and ensuring security and compliance. Experimentation with the outlined techniques and dataset customization will optimize the model for specific healthcare documentation needs.

