import os
import pandas as pd
import torch
import gc
import evaluate
import glob  # Added this to find the checkpoint folders
from datasets import Dataset
from transformers import (
  AutoTokenizer, 
  AutoModelForSequenceClassification, 
  Trainer, 
  TrainingArguments,
  DataCollatorWithPadding
)

# ================= CONFIGURATION =================
MODEL_DIR = "./models"
DATA_DIR = "../data/finalDataset"
TOPICS = ["climate_change", "illuminati", "vaccine"]
TEXT_TYPES = ["raw_text", "ner_masked_text", "noun_masked_text"]
MODEL_NAME = "bert-base-uncased" 
MAX_LENGTH = 512
# =================================================

def load_test_df(path, text_col):
  if not os.path.exists(path):
    print(f"Warning: File not found {path}")
    return None
  df = pd.read_csv(path)
  label_map = {'mainstream': 0, 'conspiracy': 1}
  df['label'] = df['subcorpus'].map(label_map)
  df = df[[text_col, 'label']].rename(columns={text_col: "text"})
  return df

acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
  logits, labels = eval_pred
  predictions = torch.argmax(torch.tensor(logits), dim=-1)
  return {
    "accuracy": acc_metric.compute(predictions=predictions, references=labels)["accuracy"],
    "f1": f1_metric.compute(predictions=predictions, references=labels)["f1"]
  }

results = []

for text_type in TEXT_TYPES:
  for train_topic in TOPICS:
    # Base path
    base_path = f"{MODEL_DIR}/{train_topic}_{text_type}"
    
    # --- NEW LOGIC TO FIND THE CHECKPOINT ---
    # Look for config.json in the root. If not there, look in checkpoint folders.
    if os.path.exists(os.path.join(base_path, "config.json")):
      actual_model_path = base_path
    else:
      checkpoints = glob.glob(os.path.join(base_path, "checkpoint-*"))
      if checkpoints:
        # Get the latest checkpoint folder
        actual_model_path = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
      else:
        print(f"Skipping: No model found in {base_path}")
        continue
    # ----------------------------------------

    print(f"\n>>> LOADING MODEL FROM: {actual_model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(actual_model_path)
    
    for test_topic in TOPICS:
      test_file = f"{DATA_DIR}/test_{test_topic}.csv"
      test_df = load_test_df(test_file, text_type)
      if test_df is None: continue

      test_ds = Dataset.from_pandas(test_df)
      tokenized_test = test_ds.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=MAX_LENGTH), 
        batched=True,
        remove_columns=["text"]
      )

      trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
      )

      print(f"Evaluating {train_topic} model on {test_topic} data...")
      metrics = trainer.evaluate(tokenized_test)

      results.append({
        "Model_Type": text_type,
        "Trained_On": train_topic,
        "Tested_On": test_topic,
        "Accuracy": metrics["eval_accuracy"],
        "F1": metrics["eval_f1"]
      })

    del model
    gc.collect()
    torch.cuda.empty_cache()

# Save and show results
results_df = pd.DataFrame(results)
results_df.to_csv("cross_topic_results.csv", index=False)
print("\nDONE! Results saved to 'cross_topic_results.csv'")
print(results_df)
