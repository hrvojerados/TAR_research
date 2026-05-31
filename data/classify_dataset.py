import os
import json
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm # Recommended for a progress bar: pip install tqdm

# --- CONFIGURATION ---
INPUT_FILE = "osfstorage-archive/data/LOCO.json"
OUTPUT_FILE = "classified/classified_loco.csv"
MAX_WORKERS = 10  # Number of simultaneous API calls. Adjust based on your Tier.
TARGET_TOPICS = ['vaccine', 'climate change', 'pizza gate', 'flat earth', 'bigfoot', 'illuminati']

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def classify_text(idx, raw_text, subcorpus):
  """Function to handle a single API call."""
  try:
    response = client.chat.completions.create(
      model="gpt-4o-mini",
      response_format={"type": "json_object"},
      messages=[
        {
          "role": "system",
          "content": (
            "You are an objective research assistant classifying text into exactly one broad domain topic: "
            "['vaccine', 'climate change', 'pizzagate', 'flat earth', 'bigfoot', 'illuminati', 'other'].\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "You MUST classify BOTH mainstream/scientific texts and alternative/conspiracy texts into the same buckets based on these broad domain definitions:\n\n"
            "- 'vaccine': All texts regarding healthcare, medicine, pandemics, viral outbreaks, viruses, and immunization.\n"
            "- 'climate change': All texts regarding the environment, weather, global warming, carbon emissions, nature, and ecology.\n"
            "'pizzagate': ONLY texts explicitly regarding underground child trafficking rings run by elites, coded political emails (e.g., Wikileaks/Podesta), or the Comet Ping Pong rumors"
            "- 'flat earth': All texts regarding geography, geology, space, astronomy, NASA, mapping, or planetary shapes.\n"
            "- 'bigfoot': All texts regarding wildlife, zoology, primates (monkeys, chimpanzees, apes), cryptids, or woodland creatures.\n"
            "- 'illuminati': All texts regarding secret societies, globalist organizations, shadow governments, and high-level institutional secrecy.\n"
            "- 'other': Only use this if the text absolutely does not fit into any of the broad domains above.\n\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\"topic\": \"value\"}"
          )
        },
        {"role": "user", "content": raw_text[:25000]}  
      ],
      timeout=50 # Don't let one hang-up stop the whole script
    )
    
    result = json.loads(response.choices[0].message.content)
    predicted_topic = result.get("topic", "other").strip().lower()
    
    return {
      'original_index': idx,
      'raw_text': raw_text,
      'topic': predicted_topic,
      'subcorpus': subcorpus
    }
  except Exception as e:
    return {"error": f"Row {idx} failed: {e}"}

# --- MAIN EXECUTION ---
def main():
  os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

  # 1. Load data and determine progress
  # 2. Load the raw dataset
  print("Loading raw LOCO dataset...")
  df_raw = pd.read_json(INPUT_FILE)

  # Separate the pools
  conspiracy_pool = df_raw[df_raw['subcorpus'] == 'conspiracy'].copy()
  mainstream_pool = df_raw[df_raw['subcorpus'] == 'mainstream'].copy()

  # Assign an alternating rank to each row (0, 1, 2, 3...) within its own group
  conspiracy_pool['interleave_rank'] = range(len(conspiracy_pool))
  mainstream_pool['interleave_rank'] = range(len(mainstream_pool))

  # Combine them
  # Sorting by 'interleave_rank' puts rank 0 from both groups first, then rank 1, etc.
  # Sorting by 'subcorpus' within the rank ensures the order is consistent (e.g., Conspiracy then Mainstream)
  df = pd.concat([conspiracy_pool, mainstream_pool])
  df = df.sort_values(by=['interleave_rank', 'subcorpus']).drop(columns=['interleave_rank'])

  print(f"Dataset interleaved. Total rows: {len(df)}")
  print(f"Initial mix: {df['subcorpus'].head(10).tolist()}") # Verify it alternates
    
  if os.path.exists(OUTPUT_FILE):
    processed_indices = set(pd.read_csv(OUTPUT_FILE)['original_index'].unique())
    print(f"Resuming. Already processed {len(processed_indices)} rows.")
  else:
    processed_indices = set()
    pd.DataFrame(columns=['original_index', 'raw_text', 'topic', 'subcorpus']).to_csv(OUTPUT_FILE, index=False)

  # Filter out rows already done
  to_process = df[~df.index.isin(processed_indices)]
  
  # Optional: limit for testing
  to_process = to_process.head(30000) 

  print(f"Starting classification on {len(to_process)} rows...")

  # 2. Use ThreadPoolExecutor for speed
  with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Map tasks
    futures = {
      executor.submit(classify_text, idx, row['txt'], row['subcorpus']): idx 
      for idx, row in to_process.iterrows()
    }


    # 3. Save as they finish (Real-time progress)
    for future in tqdm(as_completed(futures), total=len(futures)):
      res = future.result()
      
      if "error" in res:
        print(res["error"])
        continue
      
      # Immediate Write-to-Disk (Crash proofing)
      new_row = pd.DataFrame([res])
      new_row = new_row[['original_index', 'raw_text', 'topic', 'subcorpus']]
      new_row.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

if __name__ == "__main__":
  main()
