import nltk 
import json
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nlp_pipeline import *

def mscoco_read_json(file_path):
  """
  Read the mscoco dataset.

  Input:
    file_path: path to the raw data, a string
  Output:
    sentence_sets: a list of paraphrase lists for respective images
  """
  print("Reading mscoco raw data .. ")
  print("  data path: %s" % file_path)
  with open(file_path, "r") as fd:
    data = json.load(fd)

  print("%d sentences in total" % len(data["annotations"]))
  
  # aggregate all sentences of the same images
  image_idx = set([d["image_id"] for d in data["annotations"]])
  paraphrases = {}
  for im in image_idx: paraphrases[im] = []
  for d in tqdm(data["annotations"]):
    im = d["image_id"]
    sent = d["caption"]
    paraphrases[im].append(sent)

  sentence_sets = [paraphrases[im] for im in paraphrases]

  return sentence_sets

def mscoco_to_csv(sentence_sets, output_file_name):
  """
  Turn a list of sentence paraphrases into a CSV file.

  Input:
    sentence_sets: a list of paraphrase lists for respective images
    output_file_name: the name of the output csv file
  Output:
    mscoco_csv: a csv file where each row contains paraphrases of one image
  """
  mscoco_data = {}
  for i in range (0, 2):
    mscoco_data["sentence" + str(i)] = [subset[i] for subset in sentence_sets]
  mscoco_df = pd.DataFrame(data=mscoco_data)
  mscoco_csv = mscoco_df.to_csv(output_file_name, index=False)
  return mscoco_csv

train_sets = mscoco_read_json("paraphrase_data/mscoco/captions_train2017.json")
mscoco_to_csv(train_sets, "mscoco_train.csv")
val_sets = mscoco_read_json("paraphrase_data/mscoco/captions_val2017.json")
mscoco_to_csv(val_sets, "mscoco_val.csv")