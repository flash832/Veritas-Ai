# train_model.py
# Run this script to train the Veritas 13.0 Neural Brain (10 Classes)
# Ideally run on a machine with a GPU (e.g., Google Colab)

import torch
from datasets import load_dataset, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import shutil
import os

print("🧠 Generating AI Fingerprint Dataset from Real-World Data...")

import random

print("🧠 Fetching Real-World Intelligence (Enterprise AI-vs-Human Dataset)...")

try:
    # 1. Load a guaranteed native Parquet dataset (no scripts, no missing configs)
    dataset = load_dataset("andythetechnerd03/AI-human-text", split="train")
    
    # 2. This dataset is huge (400k+ rows). Let's shuffle and take 4,000 for local PC training.
    dataset = dataset.shuffle(seed=42).select(range(4000))
    
    # 3. The dataset columns are 'text' and 'generated' (0=Human, 1=AI).
    # The Hugging Face Trainer strictly expects the target column to be named 'label'.
    dataset = dataset.rename_column("generated", "label")
    
    # 4. Split into train/test for evaluation
    dataset = dataset.train_test_split(test_size=0.1)

    print(f"✅ Successfully loaded {len(dataset['train'])} REAL training examples.")

except Exception as e:
    print(f"❌ CRITICAL FAILURE loading dataset: {e}")
    exit() # Force the script to stop if it fails, no more synthetic fallbacks!

# --- 3. TRAIN THE MULTI-CLASS MODEL ---
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=10) 

training_args = TrainingArguments(
    output_dir="./veritas_model_10x",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    num_train_epochs=3, 
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

print("🚀 Starting Training...")
trainer.train()

# --- 4. SAVE ---
print("💾 Saving Veritas 13.0 Brain...")
model.save_pretrained("./veritas_model")
tokenizer.save_pretrained("./veritas_model")
