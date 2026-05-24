import pandas as pd
import spacy
from tqdm import tqdm
import re

CLASSIFIED_FILE = "classified/classified_loco.csv"
FINAL_EXPERIMENTAL_FILE = "finalDataset/final_experimental_dataset.csv"

print("Initializing local spaCy pipeline...")
nlp = spacy.load("en_core_web_sm")

if not pd.io.common.file_exists(CLASSIFIED_FILE):
  raise FileNotFoundError(f"Could not find {CLASSIFIED_FILE}.")

df = pd.read_csv(CLASSIFIED_FILE)

# --- 1. CLEANING FUNCTION ---
def clean_artifacts(text):
  if not isinstance(text, str):
    return ""
  # Replace Middle Dots (·), Bullets (•), and other dot-like artifacts with a space
  # This targets the characters seen in your VisiData screenshot
  text = re.sub(r'[·•\u00b7\u2022\u2023\u2024\u2027]+', ' ', text)
  
  # Normalize curly apostrophes to straight ones so "n't" is recognized
  text = text.replace("’", "'").replace("‘", "'")
  
  # Replace multiple spaces with a single space
  text = re.sub(r'\s+', ' ', text)
  return text.strip()

print("Cleaning raw_text and updating CSV data...")
df['raw_text'] = df['raw_text'].apply(clean_artifacts)

# --- 2. MASKING LOGIC ---
def generate_masked_texts(text_content):
  doc = nlp(text_content)
  
  # Helper to protect negations and contractions
  def is_negation(token):
    return (
      token.text.lower() in ["n't", "not", "'t"] or 
      token.norm_ == "not" or 
      token.dep_ == "neg"
    )

  # --- NER Mask ---
  ner_tokens = []
  for token in doc:
    if is_negation(token):
      ner_tokens.append(token.text_with_ws)
    elif token.ent_type_:
      if token.ent_iob_ == "B":
        ner_tokens.append(f"[{token.ent_type_}]{token.whitespace_}")
      else:
        if ner_tokens:
          ner_tokens[-1] = ner_tokens[-1].rstrip() + token.whitespace_
    else:
      ner_tokens.append(token.text_with_ws)
  
  # --- Noun Mask ---
  noun_tokens = []
  for token in doc:
    # Priority 1: Keep negations
    if is_negation(token):
      noun_tokens.append(token.text_with_ws)
    # Priority 2: Named Entities
    elif token.ent_type_:
      if token.ent_iob_ == "B":
        noun_tokens.append(f"[{token.ent_type_}]{token.whitespace_}")
      else:
        if noun_tokens:
          noun_tokens[-1] = noun_tokens[-1].rstrip() + token.whitespace_
    # Priority 3: Mask Nouns
    elif token.pos_ in ["NOUN", "PROPN"]:
      noun_tokens.append(f"[NOUN]{token.whitespace_}")
    # Priority 4: Everything else
    else:
      noun_tokens.append(token.text_with_ws)
        
  return "".join(ner_tokens), "".join(noun_tokens)

# Process the cleaned text
ner_masked_texts = []
noun_masked_texts = []

for text in tqdm(df['raw_text'], desc="Processing Masks"):
  ner_res, noun_res = generate_masked_texts(text)
  ner_masked_texts.append(ner_res)
  noun_masked_texts.append(noun_res)

df['ner_masked_text'] = ner_masked_texts
df['noun_masked_text'] = noun_masked_texts

# Save the file (raw_text is now cleaned in the output)
df.to_csv(FINAL_EXPERIMENTAL_FILE, index=False)
print(f"\nProcessing complete! File saved at: {FINAL_EXPERIMENTAL_FILE}")
