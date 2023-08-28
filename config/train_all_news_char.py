# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-all-news-char'
dataset = 'all_news_char'

dtype = 'float32'
init_from = 'scratch'

eval_interval = 25 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 5 # don't print too too often
compile = False
bias = True

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = False # override via command line if you like


gradient_accumulation_steps = 128
batch_size = 10
block_size = 512
n_embd = 128

# baby GPT model :)
n_layer = 4
n_head = 8
dropout = 0.2

learning_rate = 6e-3 # with baby networks can afford to go a bit higher
max_iters = 10000
lr_decay_iters = 10000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
use_fused = False

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
