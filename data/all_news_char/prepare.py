"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
# %%
import os
import pickle
import requests
import numpy as np
import torch


# %%
with open('article.csv', 'r') as f:
  data = f.read()

print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
sorted_tokens = torch.load('/media/j/wdata/git/PYTHON_IMPORT/nanoGPT/data/all_news_char/all_news_tokens.pt')
sorted_tokens_set = set([t[0] for t in sorted_tokens[:100]])
top_100_tokens_by_ascii = sorted(list(sorted_tokens_set))

# create a mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(top_100_tokens_by_ascii)}
stoi['℗'] = len(stoi)
itos = {i:s for s,i in stoi.items()} # inverse mapping

vocab_size = len(top_100_tokens_by_ascii)
print("all the unique characters:", ''.join(top_100_tokens_by_ascii))
print(f"vocab size: {vocab_size:,}")


def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]
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
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)


# %%
