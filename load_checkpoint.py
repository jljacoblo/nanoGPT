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


def add_all_weights_grid_to_tensorboard(weights: torch.Tensor):

  # Group every 9 cell into a square and find max of each 9 cells
  weights = weights.view(-1, weights.shape[-1])

  weights = pad_to_nearest(weights, 9)

  weights = weights.reshape(weights.shape[0] // 3, weights.shape[1] // 3, 9)
  weights, _ = weights.max(-1)

  fig, ax = plt.subplots(figsize=(20,10))

  colors = ['white', 'black']
  cmap = mcolors.LinearSegmentedColormap.from_list("", colors)

  im = ax.imshow(weights, vmin=weights.min(), vmax=weights.max(), cmap=cmap)

  cbar = fig.colorbar(im, orientation='horizontal', pad=0.05)
  cbar.ax.set_xlabel('Weights')
  return fig



def pad_to_nearest(x, base):
  '''base must be a perfect square'''
  one_dim = math.sqrt(base)
  diff_r = int((one_dim  - (x.shape[-1] % one_dim)) % one_dim)
  diff_c = int((one_dim  - (x.shape[-2] % one_dim)) % one_dim)
  return F.pad(x, (0, diff_r, 0, diff_c), 'constant', 0)



# %%
checkpoint = torch.load('/media/j/wdata/git/PYTHON_IMPORT/nanoGPT/out-all-news-char/ckpt.pt', map_location='cuda')
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
  if k.startswith(unwanted_prefix):
    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)
model = model.to('cuda')

writer = SummaryWriter()

activations, _, _ = track_activations(model, [
  'transformer.h.1.mlp.gelu'
  # 'transformer.wpe'
  # 'transformer.h.0.ln_1'
  ])



with open('/media/j/wdata/git/PYTHON_IMPORT/nanoGPT/data/all_news_char/meta_span_allnews.pkl', 'rb') as f:
  meta = pickle.load(f)
stoi, itos, vocab_size = meta['stoi'], meta['itos'], meta['vocab_size']
encode = lambda s: [stoi[c] for c in s]
stoi = defaultdict(lambda: vocab_size-1, stoi) # default value for unknown characters
itos[127] = '℗'
decode = lambda l: ''.join([itos[i] for i in l])

model.encode = encode
model.decode = decode


# %%
for name, param in model.named_parameters():
  print(name, param.shape)

# %%
# texts = [''.join(['h' for _ in range(200)])]
texts = ['News report said ']

for i in range(1):
  activations_multi = OrderedDict()
  for text in texts:
    start_ids = encode(text)
    x = torch.tensor([start_ids] * 1, dtype=torch.long, device='cuda')


    max_new_tokens = 4096 # number of tokens generated in each sample
    temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = max_new_tokens # retain only the top_k most likely tokens, clamp others to have 0 probability


    with torch.no_grad():
      y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
      for j in range(y.shape[0]):
        print(f'{i} {j} {decode(y[j].tolist())}')

    for layer_name, param in activations.items():
    #   add_all_weights_grid_to_tensorboard(writer, name, param, i)
      activations_multi[(layer_name, text)] = param

writer.close()




# %% Plot sums of activations between all vocab_size
activations_multi = sorted(activations_multi.items(), key=lambda x: x[0][0])
layers_multi = {}
for (layer_name, text), param in activations_multi:

  for c in range(param.shape[1]):

    if len(param.shape) == 3:
      weights = param[:,c,:].view(-1)
    elif len(param.shape) == 2:
      weights = param[c,:].view(-1)
    
    # weights = weights.sum(1).view(-1)

    # normalize between 0 and 1
    weights = (weights - weights.min()) / (weights.max() - weights.min())

    pad_n = math.ceil(math.sqrt(weights.shape[0]))
    weights = F.pad(weights, (0, pad_n **2 - weights.shape[0]), 'constant', 0)
    weights = weights.view(pad_n, pad_n)

    char = text[c] if c < len(text) else 'last'

    layers_multi.setdefault(layer_name, {})
    layers_multi[layer_name][f'{char}  {c}'] = weights


for layer_name, weights_multi in layers_multi.items():
  pmm = PlotMultiMatrix(layer_name, weights_multi)
  ani = pmm.plot_multi_matrix()

  # ani.save(f'animation.mp4', writer='ffmpeg', fps=1, bitrate=24)

  # plt.show()

# %%

# %%
