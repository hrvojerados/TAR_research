import os
import pandas as pd
from sklearn.model_selection import train_test_split


def create_balanced_splits(filepath, train_size=0.80, val_size=0.10, test_size=0.10):
  # 1. Load the dataset
  df = pd.read_csv(filepath)

  # Check if dataset is too small to split safely
  min_required_samples = 3  # Need at least some rows per group to split
  group_counts = df.groupby(["topic", "subcorpus"]).size()

  print("--- Current Group Counts ---")
  print(group_counts)
  print("-" * 30)

  # Temporary fallback for testing your 20-row dataset
  if df.shape[0] < 30:
    print(
      "\n⚠️ WARNING: Dataset is currently too small for true stratification."
    )
    print("Running a basic random split just so your pipeline doesn't break.")
    # Simple random split for your 20 rows just to test execution
    train, temp = train_test_split(df, test_size=0.20, random_state=42)
    val, test = train_test_split(temp, test_size=0.50, random_state=42)
    return train, val, test

  # 2. Advanced Split logic for when your full data arrives
  train_subs = []
  val_subs = []
  test_subs = []

  # Group by both columns to isolate each block (e.g., vaccine + conspiracy)
  grouped = df.groupby(["topic", "subcorpus"])

  for name, group in grouped:
    # Check if the specific sub-group has enough rows to split
    if len(group) < 3:
      # Too small to split proportionally; assign randomly to prevent crashes
      train_g, temp_g = train_test_split(group, test_size=0.20, random_state=42)
      val_g, test_g = train_test_split(temp_g, test_size=0.50, random_state=42)
    else:
      # Split the group into Train (80%) and Temp (20%)
      train_g, temp_g = train_test_split(
        group, test_size=(val_size + test_size), random_state=42
      )
      # Split Temp into Val (50% of temp) and Test (50% of temp)
      val_g, test_g = train_test_split(
        temp_g,
        test_size=(test_size / (val_size + test_size)),
        random_state=42,
      )

    train_subs.append(train_g)
    val_subs.append(val_g)
    test_subs.append(test_g)

  # Combine all the balanced pieces back together
  train_df = pd.concat(train_subs).sample(frac=1, random_state=42).reset_index(drop=True)
  val_df = pd.concat(val_subs).sample(frac=1, random_state=42).reset_index(drop=True)
  test_df = pd.concat(test_subs).sample(frac=1, random_state=42).reset_index(drop=True)

  return train_df, val_df, test_df


# --- EXECUTION ---
# Replace with your actual filename
csv_filename = "finalDataset/final_experimental_dataset.csv"

train_set, val_set, test_set = create_balanced_splits(csv_filename)

# Save the subsets
train_set.to_csv("finalDataset/train.csv", index=False)
val_set.to_csv("finalDataset/val.csv", index=False)
test_set.to_csv("finalDataset/test.csv", index=False)

print(f"\nDone! Saved {len(train_set)} train, {len(val_set)} val, and {len(test_set)} test rows.")
