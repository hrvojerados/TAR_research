import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
  AutoTokenizer, 
  AutoModelForSequenceClassification, 
  TrainingArguments, 
  Trainer,
  DataCollatorWithPadding
)
import evaluate

# ================= CONFIGURATION =================
# Choose which experiment to run: 
# 1. "raw_text" 
# 2. "ner_masked_text" (Named Entities)
# 3. "noun_masked_text" (Entities + Nouns)
TEXT_COLUMN = "ner_masked_text" 

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 512   # Max for BERT
BATCH_SIZE = 8     # Small batch to prevent Colab OOM
ACCUMULATION = 2   # Effective batch size = 8 * 2 = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
# =================================================
def prepare_loco_subset(df, target_topic=None, n_samples=None, random_seed=42):
  """
  Filters by topic, samples the data, and converts labels to 0/1.
  """
  # 1. Map labels to integers
  label_map = {'mainstream': 0, 'conspiracy': 1}
  df['label'] = df['subcorpus'].map(label_map)
  
  # 2. Filter by Topic (if specified)
  if target_topic:
    # Case-insensitive filtering
    df = df[df['topic'].str.lower() == target_topic.lower()].copy()

  # 3. Fix the Size (Sampling)
  if n_samples and n_samples < len(df):
    df = df.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
  elif n_samples and n_samples > len(df):
    print(f"Warning: Requested {n_samples} but only {len(df)} available for topic '{target_topic}'")
  return df

# 1. Load your raw files
df_train_raw = pd.read_csv("../data/finalDataset/train.csv")
df_val_raw   = pd.read_csv("../data/finalDataset/val.csv")
df_test_raw  = pd.read_csv("../data/finalDataset/test.csv")

# 2. Filter them
TOPIC = None
SIZE_train = 200
SIZE_val = None
SIZE_test = None

train_df = prepare_loco_subset(df_train_raw, target_topic=TOPIC, n_samples=SIZE_train)
val_df   = prepare_loco_subset(df_val_raw,   target_topic=TOPIC, n_samples=SIZE_val) # Keep all dev for better val
test_df  = prepare_loco_subset(df_test_raw,  target_topic=TOPIC, n_samples=SIZE_test)

# 3. Convert to HuggingFace format
dataset = DatasetDict({
  "train": Dataset.from_pandas(train_df),
  "validation": Dataset.from_pandas(val_df),
  "test": Dataset.from_pandas(test_df)
})

# 4. Now run the training loop (using the script from the previous message)
# Use 'raw text', 'ner_masked_text', or 'noun_masked_text' as TEXT_COLUMN

# 2. Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
  # This tokenizes the specific column chosen in CONFIG
  return tokenizer(
    examples[TEXT_COLUMN], 
    truncation=True, 
    max_length=MAX_LENGTH
  )

# Map tokenization across all splits
tokenized_datasets = dataset.map(tokenize_fn, batched=True)

# Data collator handles dynamic padding for efficiency
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 3. Metrics (Accuracy and F1)
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)
  acc = acc_metric.compute(predictions=predictions, references=labels)
  f1 = f1_metric.compute(predictions=predictions, references=labels)
  return {**acc, **f1}

# 4. Model Setup
model = AutoModelForSequenceClassification.from_pretrained(
  MODEL_NAME, 
  num_labels=2
)

# 5. Training Arguments (Optimized for Colab)
training_args = TrainingArguments(
  output_dir=f"./results_{TEXT_COLUMN}_{TOPIC}",
  eval_strategy="epoch",
  save_strategy="epoch",
  learning_rate=LEARNING_RATE,
  per_device_train_batch_size=BATCH_SIZE,
  per_device_eval_batch_size=BATCH_SIZE,
  gradient_accumulation_steps=ACCUMULATION, # Tricks for memory
  num_train_epochs=EPOCHS,
  weight_decay=0.01,
  load_best_model_at_end=True,
  metric_for_best_model="f1",
  fp16=True,               # Use Mixed Precision (faster/less memory)
  report_to="none"
)

# 6. Initialize Trainer
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_datasets["train"],
  eval_dataset=tokenized_datasets["validation"],
  processing_class=tokenizer,
  data_collator=data_collator,
  compute_metrics=compute_metrics,
)

# 7. Train
trainer.train()

# 8. Final Evaluation on Test Set
print(f"\n FINAL TEST RESULTS FOR: {TEXT_COLUMN} {TOPIC}")
test_results = trainer.evaluate(tokenized_datasets["test"])
print(test_results)
