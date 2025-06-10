# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("nikhilnayak123/5-million-song-lyrics-dataset")

# print("Path to dataset files:", path)
import pandas as pd
import random

# Use the full path to the actual CSV file
csv_path = '~/.cache/kagglehub/datasets/nikhilnayak123/5-million-song-lyrics-dataset/versions/3/ds2.csv'

# df = pd.read_csv(csv_path)

# print(df.head(10))
# df = pd.read_csv(
#     csv_path,
#     low_memory=False,          # disables dtype inference chunking
#     engine='python',           # more tolerant than the default 'c' engine
#     error_bad_lines=False      # skip bad rows (deprecated warning may appear)
# )
# use_cols = ['title', 'artist', 'lyrics', 'language']  # adjust as needed
# df = pd.read_csv(csv_path, usecols=use_cols, low_memory=False, engine='python')

# columns = pd.read_csv(csv_path, nrows=0)
# print(columns.columns.tolist())


# sample_size = 50000
# total_rows = 5_000_000  # adjust to match your file

# # Randomly select rows to keep (excluding header)
# keep_rows = set(random.sample(range(1, total_rows), sample_size))

# df_sampled = pd.read_csv(
#     csv_path,
#     skiprows=lambda i: i > 0 and i not in keep_rows,
# )

# print(df_sampled.head())

import re
from langdetect import detect, DetectorFactory
from tqdm import tqdm


def clean_lyrics(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\[.*?\]', '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"
    

# Ensure consistent language detection
DetectorFactory.seed = 0

output_path = "sample_set_with_language.csv"
sample_size = 50000
estimated_total_rows = 5_000_000

rows_to_keep = set(random.sample(range(1, estimated_total_rows), sample_size))

print("Sampling rows...")
df = pd.read_csv(
    csv_path,
    skiprows=lambda i: i > 0 and i not in rows_to_keep,
    usecols=['title', 'tag', 'artist', 'year', 'views', 'features', 'lyrics', 'id']
)

print("Cleaning lyrics...")
df['clean_lyrics'] = df['lyrics'].apply(clean_lyrics)

tqdm.pandas()
df['language'] = df['clean_lyrics'].progress_apply(detect_language)

df.to_csv(output_path, index=False)
print("Save complete")


