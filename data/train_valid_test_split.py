import os
import pandas as pd

def create_specific_topic_splits(filepath, output_dir="finalDataset"):
  # 1. Učitavanje podataka
  df = pd.read_csv(filepath)
  os.makedirs(output_dir, exist_ok=True)

  # Definiramo ciljane brojeve (prema tvojoj slici)
  # Za Illuminati koristimo prosjek ostalih jer ga nema na slici
  targets = {
    "vaccine": {
      "train": {"mainstream": 796, "conspiracy": 268},
      "val":   {"mainstream": 104, "conspiracy": 29},
      "test":  {"mainstream": 100, "conspiracy": 33}
    },
    "climate change": {
      "train": {"mainstream": 799, "conspiracy": 265},
      "val":   {"mainstream": 99,  "conspiracy": 34},
      "test":  {"mainstream": 102, "conspiracy": 31}
    },
    "illuminati": { # Procjena bazirana na ostalima
      "train": {"mainstream": 700, "conspiracy": 260},
      "val":   {"mainstream": 62, "conspiracy": 35},
      "test":  {"mainstream": 61, "conspiracy": 35}
    }
  }

  topics_to_process = ["vaccine", "climate change", "illuminati"]

  for topic in topics_to_process:
    print(f"\n--- Obrađujem topic: {topic.upper()} ---")
    
    # Filtriraj podatke samo za taj topic
    topic_df = df[df['topic'].str.lower() == topic.lower()]
    
    if topic_df.empty:
      print(f"⚠️ Topic '{topic}' nije pronađen u datasetu. Preskačem...")
      continue

    # Razdvoji na mainstream i conspiracy
    # Provjeri točne nazive u svom CSV-u (možda su s velikim početnim slovom)
    mainstream_pool = topic_df[topic_df['subcorpus'].str.lower() == 'mainstream'].sample(frac=1, random_state=42)
    conspiracy_pool = topic_df[topic_df['subcorpus'].str.lower() == 'conspiracy'].sample(frac=1, random_state=42)

    splits = ["train", "val", "test"]
    
    for split in splits:
      n_main = targets[topic][split]["mainstream"]
      n_cons = targets[topic][split]["conspiracy"]

      # Provjera imamo li dovoljno podataka, ako nemamo uzmi sve dostupno
      if len(mainstream_pool) < n_main:
        print(f"  ! Manjak mainstream podataka za {split} ({len(mainstream_pool)}/{n_main})")
        n_main = len(mainstream_pool)
      
      if len(conspiracy_pool) < n_cons:
        print(f"  ! Manjak conspiracy podataka za {split} ({len(conspiracy_pool)}/{n_cons})")
        n_cons = len(conspiracy_pool)

      # Uzorkovanje
      m_part = mainstream_pool.iloc[:n_main]
      c_part = conspiracy_pool.iloc[:n_cons]

      # Micanje iskorištenih redaka iz "poola" da se ne ponavljaju u splitovima
      mainstream_pool = mainstream_pool.iloc[n_main:]
      conspiracy_pool = conspiracy_pool.iloc[n_cons:]

      # Spajanje i spremanje
      final_split_df = pd.concat([m_part, c_part]).sample(frac=1, random_state=42)
      
      filename = f"{split}_{topic.replace(' ', '_')}.csv"
      save_path = os.path.join(output_dir, filename)
      final_split_df.to_csv(save_path, index=False)
      
      print(f"  ✅ Spremljeno: {filename} (M: {len(m_part)}, C: {len(c_part)})")

# --- IZVRŠAVANJE ---
csv_path = "finalDataset/final_experimental_dataset.csv"
create_specific_topic_splits(csv_path)
