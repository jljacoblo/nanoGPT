"""
From a pre-trained transformer model ( char-based )
Cut out 1 layer from the model
Generate text from the model
and prepare the dataset to train a new model based on a smaller parameter size
"""
# %%
import os
import pickle
import numpy as np
import torch
from collections import defaultdict


def get_tokens(s: str) -> dict[chr, int]:
  tokens = dict()
  for r in s:
    this_article_tokens = dict()
    for c in r:
      if c not in this_article_tokens:
        this_article_tokens[c] = 0
      this_article_tokens[c] += 1
    for k, v in this_article_tokens.items():
      if k not in tokens:
        tokens[k] = 0
      tokens[k] += v
  return tokens




# %% 
with open('/media/j/wdata/git/PYTHON_IMPORT/nanoGPT/data/downsize/output_layer_0.txt', 'r') as f:
  data = f.read()

token_counts = get_tokens(data)
token_counts = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
torch.save(token_counts, '/media/j/wdata/git/PYTHON_IMPORT/nanoGPT/data/downsize/token_counts.pkl')


# get all the unique characters that occur in this text
chars = list(set([t[0] for t in token_counts[:127]] + ['℗']))
chars = sorted(chars)
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
stoi = defaultdict(lambda: vocab_size-1, stoi) # default value for unknown characters
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]
print(f"train has {len(train_data):,} characters")
del data

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
del train_data, val_data

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
stoi = { ch:i for i,ch in enumerate(chars) }
vocab_size = 128
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)


# %%
