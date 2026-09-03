import pandas as pd
from pathlib import Path
import csv

SCRIPT_DIR = Path(__file__).resolve().parent.parent

dataframe = pd.read_csv(f'{SCRIPT_DIR}/data/LJSpeech-1.1/metadata.csv', sep="|", header=None, names=["filename", "text", "text_normalized"], quoting=csv.QUOTE_NONE)

print(dataframe['text_normalized'].isna().sum())

with open(f'{SCRIPT_DIR}/tokenizer_data.txt', "w") as file:
    file.write("\n".join(dataframe['text_normalized']))