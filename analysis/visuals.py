import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TOPIC1 = "something"
TOPIC2 = "something"
def plot_shap_results(input_file="shap_token_results.json", top_n=100):
  with open(input_file, "r") as f:
    data = json.load(f)
  
  importance = data["importance"]
  counts = data["counts"]

  # Convert to DataFrame
  df = pd.DataFrame({
    "token": list(importance.keys()),
    "total_shap": list(importance.values()),
    "occurrence": list(counts.values())
  })

  # Average SHAP per occurrence to avoid high-frequency common words (like 'the') 
  # from dominating just by appearing often.
  df['avg_shap'] = df['total_shap'] / df['occurrence']

  # Sort by absolute total impact
  df['abs_shap'] = df['total_shap'].abs()
  df = df.sort_values(by="abs_shap", ascending=False).head(top_n)

  # Plot
  plt.figure(figsize=(12, 20))
  sns.barplot(
    data=df, 
    y="token", 
    x="total_shap", 
    palette="vlag"
  )
  plt.title(f"Top {top_n} Tokens by Total SHAP Contribution (Conspiracy Class)")
  plt.xlabel("Total SHAP Value (Positive = Predictive of Conspiracy, Negative = Mainstream)")
  plt.grid(axis='x', linestyle='--', alpha=0.7)
  plt.tight_layout()
  plt.savefig(f"top_tokens_shap_{TOPIC1}_{TOPIC2}.png")
  plt.show()

if __name__ == "__main__":
  plot_shap_results()
