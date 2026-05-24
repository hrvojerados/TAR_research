import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
  AutoTokenizer, 
  AutoModelForSequenceClassification, 
  Trainer,
  TrainingArguments,
  DataCollatorWithPadding
)
import evaluate
import gc

# ================= CONFIGURATION =================
TEXT_COLUMNS = ["raw_text", "ner_masked_text", "noun_masked_text"]
MODEL_BASE_NAME = "bert-base-uncased" 
MAX_LENGTH = 512
BATCH_SIZE = 16 
RESULTS_BASE_DIR = "./" 
# =================================================

def prepare_test_data(df, topic, text_col):
  """Prepares the test set. If topic is 'None', returns the whole set."""
  label_map = {'mainstream': 0, 'conspiracy': 1}
  
  if topic == "None" or topic is None:
    subset = df.copy()
  else:
    subset = df[df['topic'] == topic].copy()
  
  if len(subset) == 0:
    return pd.DataFrame()

  subset['label'] = subset['subcorpus'].map(label_map)
  subset = subset.rename(columns={text_col: "text"})
  return subset[['text', 'label']]

# 1. Load Data
df_test_raw = pd.read_csv("../data/finalDataset/test.csv")

# Identify individual topics in the data
individual_topics = sorted(df_test_raw['topic'].dropna().unique().tolist())
# We want to test on individual topics AND the aggregate ("None")
test_topics = ["None"] + individual_topics

# Your models were trained on the whole dataset, so training topic is just "None"
train_topics = ["None"]

# 2. Setup Metrics
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)
  return {
    "accuracy": acc_metric.compute(predictions=predictions, references=labels)["accuracy"],
    "f1": f1_metric.compute(predictions=predictions, references=labels)["f1"]
  }

results_list = []
tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE_NAME)

# 3. Main Evaluation Loop
for text_col in TEXT_COLUMNS:
  for train_topic in train_topics:
    model_path = os.path.join(RESULTS_BASE_DIR, f"results_{text_col}_{train_topic}")
    
    if not os.path.exists(model_path):
      print(f"⚠️ Model not found at {model_path}, skipping...")
      continue
        
    print(f"\n--- Loading Model: {text_col} (Generalist) ---")
    
    try:
      model = AutoModelForSequenceClassification.from_pretrained(model_path)
      model.to("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
      print(f"❌ Error loading model: {e}")
      continue

    trainer = Trainer(
      model=model,
      args=TrainingArguments(output_dir="./tmp_eval", per_device_eval_batch_size=BATCH_SIZE, report_to="none"),
      processing_class=tokenizer,
      data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
      compute_metrics=compute_metrics
    )

    for test_topic in test_topics:
      test_sub = prepare_test_data(df_test_raw, test_topic, text_col)
      if test_sub.empty: continue
      
      test_ds = Dataset.from_pandas(test_sub)
      def tokenize_fn(ex): return tokenizer(ex["text"], truncation=True, max_length=MAX_LENGTH)
      tokenized_test = test_ds.map(tokenize_fn, batched=True, remove_columns=["text"], desc=f"Tokenizing {test_topic}")

      metrics = trainer.evaluate(tokenized_test)
      
      results_list.append({
        "text_type": text_col,
        "trained_on": train_topic,
        "tested_on": test_topic,
        "f1": metrics["eval_f1"],
        "accuracy": metrics["eval_accuracy"]
      })
      print(f"   Tested on {test_topic:12}: F1={metrics['eval_f1']:.4f}")

    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

# 4. Save Results
if results_list:
  results_df = pd.DataFrame(results_list)
  results_df.to_csv("generalist_performance_results.csv", index=False)
  
  # 5. Visualization
  import seaborn as sns
  import matplotlib.pyplot as plt

  for text_col in results_df['text_type'].unique():
    plt.figure(figsize=(12, 4))
    subset = results_df[results_df['text_type'] == text_col]
    
    # We pivot to create a single-row heatmap
    matrix = subset.pivot(index='trained_on', columns='tested_on', values='f1')
    
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap='Blues', cbar_kws={'label': 'F1 Score'})
    plt.title(f"Generalist Model Performance across Topics ({text_col})")
    plt.savefig(f"heatmap_{text_col}.png", bbox_inches='tight', dpi=300)
    plt.show()
else:
  print("❌ No results to visualize.")
