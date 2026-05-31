import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TOPIC1 = "something"
TOPIC2 = "something"

def plot_shap_results(input_file="shap_analysis_noun_masked_text.csv", top_n=30):
    # 1. Load the CSV file
    try:
        df = pd.read_csv(f"{input_file}.csv")
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found. Please check the path.")
        return

    # 2. Sort by absolute average impact to catch strongest predictors (both positive and negative)
    df['abs_avg_shap'] = df['avg_shap'].abs()
    df = df.sort_values(by="abs_avg_shap", ascending=False).head(top_n)
    
    # 3. Sort again by actual avg_shap so the plot flows beautifully from positive to negative
    df = df.sort_values(by="avg_shap", ascending=False)

    # 4. Dynamic figure height based on top_n to prevent text squishing
    fig_height = max(6, int(top_n * 0.35))
    plt.figure(figsize=(12, fig_height))
    
    # 5. Plot using Seaborn
    sns.barplot(
        data=df, 
        y="token", 
        x="avg_shap", 
        hue="token",
        palette="vlag",
        legend=False
    )
    
    # Add a solid center line at 0 to separate classes visually
    plt.axvline(0, color='black', linewidth=1, linestyle='-')
    
    # Labels and Styling
    plt.title(f"Top {top_n} Tokens by Average SHAP Contribution\n({TOPIC1} vs {TOPIC2})", fontsize=14, pad=15)
    plt.xlabel("Average SHAP Value (Positive = Conspiracy, Negative = Mainstream)", fontsize=12)
    plt.ylabel("Tokens", fontsize=12)
    
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # 6. Save and show
    output_png = f"{input_file}.png"
    plt.savefig(output_png, dpi=300)
    print(f"Plot successfully saved to {output_png}")
    plt.show()

if __name__ == "__main__":
    # You can change this to "shap_analysis_ner_masked_text.csv" or "shap_analysis_noun_masked_text.csv"
    for name in ["shap_analysis_raw_text", "shap_analysis_ner_masked_text", "shap_analysis_noun_masked_text"]:
      plot_shap_results(name, top_n=30)
