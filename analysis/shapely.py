import torch
import shap
import numpy as np
import json
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk # If you saved it, otherwise just use your existing df
import pandas as pd
from collections import defaultdict

# ================= CONFIGURATION =================
MODEL_PATH = "../model_training_testing/results_ner_masked_text/checkpoint-6"
# 1. "raw_text" 
# 2. "ner_masked_text" (Named Entities)
# 3. "noun_masked_text" (Entities + Nouns)
TEXT_COLUMN = "ner_masked_text"
DEVICE = -1 # -1 for CPU, 0 for GPU
MAX_SAMPLES = 50 # Start small! SHAP is very slow on CPU.
# =================================================

def calculate_shap_contributions():
  # 1. Load Model and Tokenizer
  print(f"Loading model from {MODEL_PATH}...")
  model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
  tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

  # 2. Create a Pipeline
  # SHAP works best with a transformers pipeline
  pred_pipeline = pipeline(
    "text-classification", 
    model=model, 
    tokenizer=tokenizer, 
    device=DEVICE,
    top_k=None # Returns probabilities for all classes
  )

  # 3. Prepare Data
  # Assuming you have your test_df from the previous script
  # For this example, let's load a subset of the text
  test_df = pd.read_csv("../data/finalDataset/test.csv") # Adjust path
  # Filter for the topic you trained on

  # test_df = test_df[test_df['topic'].str.lower() == "illuminati"].head(MAX_SAMPLES)
  
  raw_texts = test_df[TEXT_COLUMN].tolist()
  def truncate_to_bert_limit(text, max_tokens=510):
    # We tokenize, slice the first 510 tokens, and decode back to string
    # We use 510 to leave room for [CLS] and [SEP] tokens
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens, add_special_tokens=False)
    return tokenizer.decode(tokens)

  print("Truncating long texts...")
  texts = [truncate_to_bert_limit(t) for t in raw_texts]
  # 4. Initialize SHAP Explainer
  # This explainer is optimized for NLP models
  explainer = shap.Explainer(pred_pipeline)

  print(f"Calculating SHAP values for {len(texts)} samples... (This will take time)")
  shap_values = explainer(texts)

  # 5. Aggregate Values
  # shap_values.values is an array of [samples, tokens, classes]
  # We want class 1 (Conspiracy)
  # If the model output is [Mainstream, Conspiracy], class 1 is Conspiracy.
  
  token_importance = defaultdict(float)
  token_counts = defaultdict(int)

  # Iterate through each document in the SHAP output
  for i in range(len(shap_values)):
    doc_tokens = shap_values.data[i]     # The actual strings (tokens)
    doc_values = shap_values.values[i]   # The SHAP values for each token
    
    # doc_values has shape (num_tokens, 2)
    # We take index 1 for the 'Conspiracy' class
    conspiracy_scores = doc_values[:, 1]

    for token, score in zip(doc_tokens, conspiracy_scores):
      token = token.strip()
      if token == "": continue
      
      # We use absolute value to find "most influential" 
      # or raw value to find "most predictive of conspiracy"
      token_importance[token] += score 
      token_counts[token] += 1

  # 6. Save results
  result_data = {
    "importance": dict(token_importance),
    "counts": dict(token_counts)
  }
  
  with open("shap_token_results.json", "w") as f:
    json.dump(result_data, f)
  
  print("Done! Results saved to shap_token_results.json")

if __name__ == "__main__":
  calculate_shap_contributions()
