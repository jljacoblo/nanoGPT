# %%
import torch
from torch import nn
from torch.nn import functional as F
from nanoGPT.model import GPT, GPTConfig
import pickle, math, time
from collections import defaultdict, OrderedDict
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

from jacAI.makemore.gen_names import track_activations
from jacAI.util.plot_graphs import PlotMultiMatrix


# %%



with open('/media/j/wdata/git/PYTHON_IMPORT/nanoGPT/data/all_news_char/meta_span_allnews.pkl', 'rb') as f:
  meta = pickle.load(f)
stoi, itos, vocab_size = meta['stoi'], meta['itos'], meta['vocab_size']
encode = lambda s: [stoi[c] for c in s]
stoi = defaultdict(lambda: vocab_size-1, stoi) # default value for unknown characters
itos[127] = '℗'
decode = lambda l: ''.join([itos[i] for i in l])



# %%
text = ' '
start_time = time.time()

for l in range(3,4):
  output = []

  checkpoint = torch.load('/media/j/wdata/git/PYTHON_IMPORT/nanoGPT/out-all-news-char/ckpt_layer4_head8_embd128_block1024.pt', map_location='cuda')
  gptconf = GPTConfig(**checkpoint['model_args'])
  model = GPT(gptconf)
  state_dict = checkpoint['model']
  unwanted_prefix = '_orig_mod.'
  for k,v in list(state_dict.items()):
      if k.startswith(unwanted_prefix):
          state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
  model.load_state_dict(state_dict, strict=False)
  model = model.to('cuda')
  
  model.encode = encode
  model.decode = decode

  tmp_block = model.transformer.h[l]

  # Delete all layers except the one we want to visualize
  for _ in range(4):
     del model.transformer.h[0]
  
  # Add the layer we want to back to the model
  model.transformer.h.append(tmp_block)


  start_ids = encode(text)
  x = torch.tensor([start_ids] * 30, dtype=torch.long, device='cuda')

  max_new_tokens = 1020 # number of tokens generated in each sample
  temperature = 1.2 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
  top_k = max_new_tokens # retain only the top_k most likely tokens, clamp others to have 0 probability

  for k in range(601):
    with torch.no_grad():
      y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
      for j in range(y.shape[0]):
        output.append(decode(y[j].tolist()))
        print(f'Layer{l} {j} Duration{k} Time : {time.time() - start_time} {decode(y[j][:10].tolist())}')

    if k % 50 == 0:
      torch.save(output, f'out/output_layer_{l}.pkl')
