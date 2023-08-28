
# %%
import torch

from model import GPTConfig, GPT


# %%
checkpoint = torch.load('/media/j/wdata/git/PYTHON_IMPORT/nanoGPT/out-all-news-char/ckpt_layer4_head8_embd128_block1024.pt')
checkpoint_model_args = checkpoint['model_args']


# %% Modify architecture
checkpoint_model_args['block_size'] = 4096
checkpoint['config']['block_size'] = 4096


# %% create the model
gptconf = GPTConfig(**checkpoint_model_args)
model = GPT(gptconf)
state_dict = checkpoint['model']
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
  if k.startswith(unwanted_prefix):
    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)


# %%
orig_wpe = state_dict['transformer.wpe.weight']

# %%
state_dict['transformer.wpe.weight'] = torch.zeros(4096, 128)
state_dict['transformer.wpe.weight'][:1024, :] = orig_wpe.detach()
state_dict['transformer.wpe.weight'][1024:2048, :] = orig_wpe.detach()
state_dict['transformer.wpe.weight'][2048:3072, :] = orig_wpe.detach()
state_dict['transformer.wpe.weight'][3072:, :] = orig_wpe.detach()


# %%
state_dict.keys()

# %%
state_dict['transformer.wpe.weight'].shape

# %%
model.load_state_dict(state_dict, strict=False)





# %%
checkpoint['model'] = model.state_dict()
checkpoint
torch.save(checkpoint, '/media/j/wdata/git/PYTHON_IMPORT/nanoGPT/out-all-news-char/ckpt.pt')
# %%
