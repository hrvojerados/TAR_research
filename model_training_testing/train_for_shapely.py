import os
import pandas as pd
import numpy as np
import torch
import gc
import evaluate
from torch import nn
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

# ================= CONFIGURATION =================
MODEL_NAME = "bert-base-uncased"
TOPICS = ["climate_change", "illuminati", "vaccine"]
TEXT_TYPES = ["raw_text", "ner_masked_text", "noun_masked_text"]

MAX_LENGTH = 512
BATCH_SIZE = 16
ACCUMULATION = 2
EPOCHS = 20           # Increased epochs, but EarlyStopping will cut it short
LEARNING_RATE = 2e-5
DATA_DIR = "../data/finalDataset" 
os.makedirs("models", exist_ok=True)

# 1. Custom Trainer to boost F1 (Weighted CrossEntropy)
class WeightedTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    labels = inputs.get("labels")
    outputs = model(**inputs)
    logits = outputs.get("logits")
    # Boost the conspiracy class (1) to improve Recall/F1
    # [Weight for Mainstream, Weight for Conspiracy]
    weights = torch.tensor([1.0, 2.0]).to(model.device) 
    loss_fct = nn.CrossEntropyLoss(weight=weights)
    loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    return (loss, outputs) if return_outputs else loss

# 2. Simplified Loading (Uses 100% of your provided files)
def load_full_df(path, text_col):
  df = pd.read_csv(path)
  label_map = {'mainstream': 0, 'conspiracy': 1}
  if 'subcorpus' in df.columns:
    df['label'] = df['subcorpus'].map(label_map)
  else:
    # Fallback if label column is named differently
    df['label'] = df['label'].map(label_map)
  
  df = df[[text_col, 'label']].rename(columns={text_col: "text"})
  return df

acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)
  return {
    "accuracy": acc_metric.compute(predictions=predictions, references=labels)["accuracy"],
    "f1": f1_metric.compute(predictions=predictions, references=labels)["f1"]
  }

# ... (Keep all imports, WeightedTrainer class, and load_full_df function from your previous script) ...

# --- PHASE 2: START COMBINED TRAINING LOOP ---
print(f"\n{'#'*40}\nPHASE 2: TRAINING COMBINED MODELS\n{'#'*40}")

for text_type in TEXT_TYPES:
  run_name = f"COMBINED_{text_type}"
  output_dir = f"./models/{run_name}"
  print(f"\n{'='*30}\nRUNNING: {run_name}\n{'='*30}")

  # 1. Load and Merge Data across all TOPICS
  train_dfs, val_dfs, test_dfs = [], [], []

  for topic in TOPICS:
    train_dfs.append(load_full_df(f"{DATA_DIR}/train_{topic}.csv", text_type))
    val_dfs.append(load_full_df(f"{DATA_DIR}/val_{topic}.csv", text_type))
    test_dfs.append(load_full_df(f"{DATA_DIR}/test_{topic}.csv", text_type))

  # Concatenate and Shuffle
  combined_train_df = pd.concat(train_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
  combined_val_df   = pd.concat(val_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
  combined_test_df  = pd.concat(test_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

  print(f"Combined Training Set Size: {len(combined_train_df)} rows")

  # 2. Convert to HuggingFace Dataset
  ds = DatasetDict({
    "train": Dataset.from_pandas(combined_train_df),
    "validation": Dataset.from_pandas(combined_val_df),
    "test": Dataset.from_pandas(combined_test_df)
  })

  # 3. Tokenization
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  tokenized_ds = ds.map(lambda x: tokenizer(x["text"], truncation=True, max_length=MAX_LENGTH), 
                        batched=True, remove_columns=["text"])

  # 4. Model Setup
  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

  # 5. Training Arguments (Using your existing settings)
  args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=ACCUMULATION,
    num_train_epochs=EPOCHS,
    weight_decay=0.1,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True, 
    report_to="none",
    save_total_limit=1
  )

  # 6. Initialize Weighted Trainer
  trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
  )

  # 7. Train & Evaluate
  trainer.train()

  print(f"\nFinal Test Results for {run_name}:")
  print(trainer.evaluate(tokenized_ds["test"]))

  # 8. Cleanup
  del model, trainer
  gc.collect()
  torch.cuda.empty_cache()
