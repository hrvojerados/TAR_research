import os
import glob
import json
import pandas as pd
import torch
import gc
import shap
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from collections import defaultdict

# ================= CONFIGURATION =================
MODEL_DIR = "./models"  # Your models folder
DATA_DIR = "../data/finalDataset"
TEXT_TYPES = ["raw_text", "ner_masked_text", "noun_masked_text"]
TOPICS = ["climate_change", "illuminati", "vaccine"]

MAX_SAMPLES = 100       # How many test rows to explain? (SHAP is slow, start with 100)
DEVICE = 0 if torch.cuda.is_available() else -1
# =================================================

def get_latest_checkpoint(path):
  checkpoints = glob.glob(os.path.join(path, "checkpoint-*"))
  if not checkpoints:
    return path if os.path.exists(os.path.join(path, "config.json")) else None
  return sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]

def process_shap_for_type(text_type):
  print(f"\n{'='*40}\nANALYZING: {text_type}\n{'='*40}")
  
  # 1. Load the Combined Model for this text type
  model_path = get_latest_checkpoint(f"../model_training_testing/models/COMBINED_{text_type}")
  if not model_path:
    print(f"Skipping {text_type}: No model found.")
    return
  
  model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda" if DEVICE == 0 else "cpu")
  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

  # 2. Setup Pipeline
  pred_pipeline = pipeline(
    "text-classification", 
    model=model, 
    tokenizer=tokenizer, 
    device=DEVICE, 
    top_k=None
  )

  # 3. Prepare Test Data (Merging all topics for the combined test)
  test_dfs = [pd.read_csv(f"{DATA_DIR}/test_{topic}.csv") for topic in TOPICS]
  test_df = pd.concat(test_dfs).sample(n=MAX_SAMPLES, random_state=42)
  texts = test_df[text_type].tolist()

  def truncate_to_bert_limit(text, max_tokens=500):
    # We tokenize, slice the first 510 tokens, and decode back to string
    # We use 510 to leave room for [CLS] and [SEP] tokens
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens, add_special_tokens=False)
    return tokenizer.decode(tokens)

  print("Truncating long texts...")
  texts = [truncate_to_bert_limit(t) for t in texts]

  # 4. Run SHAP
  # We use a custom masker to handle BERT subwords correctly
  explainer = shap.Explainer(pred_pipeline)
  shap_values = explainer(texts)

  # 5. Aggregate token contributions
  token_stats = defaultdict(lambda: {"total_shap": 0.0, "count": 0})

  # shap_values.values shape: [samples, tokens, classes]
  # We want index 1 (Conspiracy)
  for i in range(len(shap_values)):
    tokens = shap_values.data[i]
    scores = shap_values.values[i][:, 1] # Class 1 (Conspiracy)
    
    for token, score in zip(tokens, scores):
      clean_token = token.strip().lower()
      if clean_token == "" or clean_token in ["[cls]", "[sep]", "[pad]"]:
        continue
      
      token_stats[clean_token]["total_shap"] += score
      token_stats[clean_token]["count"] += 1

  # 6. Calculate Averages and Save
  final_list = []
  for token, stats in token_stats.items():
    final_list.append({
      "token": token,
      "avg_shap": stats["total_shap"] / stats["count"],
      "frequency": stats["count"]
    })

  df_results = pd.DataFrame(final_list).sort_values(by="avg_shap", ascending=False)
  output_file = f"shap_analysis_{text_type}.csv"
  df_results.to_csv(output_file, index=False)
  print(f"Saved top 20 tokens for {text_type}:")
  print(df_results.head(20))

  # Cleanup memory
  del model, pred_pipeline, explainer, shap_values
  gc.collect()
  torch.cuda.empty_cache()

if __name__ == "__main__":
  for t_type in TEXT_TYPES:
    process_shap_for_type(t_type)
