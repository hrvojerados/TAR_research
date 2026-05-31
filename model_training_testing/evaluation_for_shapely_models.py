import os
import pandas as pd
import numpy as np
import torch
import evaluate
import gc
import glob
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)

# ================= CONFIGURATION =================
MODEL_NAME = "bert-base-uncased" 
TOPICS = ["climate_change", "illuminati", "vaccine"]
TEXT_TYPES = ["raw_text", "ner_masked_text", "noun_masked_text"]

# PATHS: Based on your screenshot
DATA_DIR = "../data/finalDataset" 
MODELS_DIR = "models"  # Removed the '../' because you are running from model_training_testing

MAX_LENGTH = 512
BATCH_SIZE = 16

def get_best_model_path(base_path):
    """
    Checks if config.json is in the base folder. 
    If not, looks for the highest checkpoint folder.
    """
    if os.path.exists(os.path.join(base_path, "config.json")):
        return base_path
    
    checkpoints = glob.glob(os.path.join(base_path, "checkpoint-*"))
    if checkpoints:
        # Returns the checkpoint with the highest number
        return max(checkpoints, key=os.path.getmtime)
    return base_path

# Load metrics
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": acc_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=predictions, references=labels)["f1"]
    }

def load_full_df(path, text_col):
    if not os.path.exists(path):
        print(f"!!! Error: Data file not found at {path}")
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    label_map = {'mainstream': 0, 'conspiracy': 1}
    col = 'subcorpus' if 'subcorpus' in df.columns else 'label'
    df['label'] = df[col].map(label_map)
    df = df[[text_col, 'label']].rename(columns={text_col: "text"})
    return df

# ================= EVALUATION LOOP =================
results = []
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"\n{'#'*60}\nSTARTING EVALUATION ON COMBINED TEST SET\n{'#'*60}")

for text_type in TEXT_TYPES:
    base_folder = f"{MODELS_DIR}/COMBINED_{text_type}"
    
    if not os.path.exists(base_folder):
        print(f"Skipping {text_type}: Folder not found at {base_folder}")
        continue

    # Automatically find the path (handles the 'checkpoint-582' folder)
    model_path = get_best_model_path(base_folder)
    print(f"\n>>> Loading Model from: {model_path}")
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Could not load model at {model_path}: {e}")
        continue
    
    eval_args = TrainingArguments(
        output_dir="./temp_eval",
        per_device_eval_batch_size=BATCH_SIZE,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    # Load and Merge Test Data
    test_dfs = []
    for topic in TOPICS:
        df = load_full_df(f"{DATA_DIR}/test_{topic}.csv", text_type)
        if not df.empty:
            test_dfs.append(df)
    
    if not test_dfs:
        print(f"No test data found for {text_type}. Check DATA_DIR.")
        continue

    combined_test_df = pd.concat(test_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    combined_ds = Dataset.from_pandas(combined_test_df)
    tokenized_ds = combined_ds.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=MAX_LENGTH), 
        batched=True
    )
    
    print(f"Evaluating {text_type} on {len(combined_test_df)} samples...")
    metrics = trainer.evaluate(tokenized_ds)
    
    results.append({
        "text_type": text_type,
        "test_accuracy": metrics["eval_accuracy"],
        "test_f1": metrics["eval_f1"]
    })

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

# ================= SUMMARY =================
if results:
    results_df = pd.DataFrame(results)
    print(f"\n{'='*40}\nFINAL COMBINED RESULTS\n{'='*40}")
    print(results_df.to_string(index=False))
    results_df.to_csv("combined_evaluation_results.csv", index=False)
else:
    print("\nNo results generated. Please check folder paths.")
