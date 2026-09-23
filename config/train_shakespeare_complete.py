# train a character-level GPT on the FULL Project Gutenberg Shakespeare corpus (#100)
#
# data:   data/shakespeare_complete/  -> 4,808,687 train tokens, vocab 79
# model:  n_layer=8 / n_head=8 / n_embd=512 / block_size=512  ->  25,522,688 params
#         params per train token = 5.31   (the 10.7M model on tiny shakespeare was 10.73)
#
# why 8x512 and not the old 6x384:
#   the vocabulary is irrelevant to sizing (wte is only 79*512 = 40,448 params, 0.16%);
#   what matters is the token budget. Unique tokens went 1.0M -> 4.8M (x4.79), and
#   optimal model size grows roughly as sqrt(data) -> ~x2.2 -> ~23M. Round up to 8/8/512.
#   This is an empirical extrapolation, not a hard optimum: always_save_checkpoint=False
#   means the run keeps the best-val checkpoint, so overshooting costs only wall clock.
#
# nothing in train.py needs editing: data_dir = os.path.join('data', dataset)
# and vocab_size is read out of meta.pkl.

out_dir = 'out-shakespeare-complete'
eval_interval = 500
eval_iters = 200
log_interval = 10

# val loss still plateaus eventually, so only checkpoint on improvement
always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-complete'
wandb_run_name = 'gpt-8x512'

dataset = 'shakespeare_complete'

# --- model ---
n_layer = 8
n_head = 8
n_embd = 512
block_size = 512       # 256 chars is only ~2-3 lines of dialogue; 512 sees a full speech turn
bias = True
dropout = 0.2          # params/token is still >5, so keep the regularisation
                       # (try 0.1 once you confirm val loss is no longer diverging)

# --- optimisation ---
# 16 seqs x 512 = 8192 tokens per micro-batch, x2 accum = 16,384 tokens per step,
# i.e. the same token budget per step as the old 64x256 config.
# Peak activation memory is set by the micro-batch, so 16x512@accum2 fits where
# 32x512 might not. If you have headroom, use batch_size=32 and drop accum to 1.
batch_size = 16
gradient_accumulation_steps = 2

learning_rate = 1e-3   # if loss spikes, fall back to 6e-4
min_lr = 1e-4
# 6000 steps = 98.3M tokens ~= 20 epochs. Repetition returns die out around 4-16
# epochs, so this deliberately overshoots and relies on best-val checkpointing.
max_iters = 6000
lr_decay_iters = 6000
warmup_iters = 200
beta2 = 0.99           # higher beta2 because the per-step token count is small
weight_decay = 1e-1
grad_clip = 1.0
decay_lr = True

# --- variants ---
# stay comparable to the old run:        n_layer=6, n_head=6, n_embd=384, block_size=256
#                                        batch_size=64, gradient_accumulation_steps=1  (10.78M)
# safer middle option:                   n_layer=8, n_head=8, n_embd=448, block_size=512 (19.58M)
# too big for this data, will memorise:  n_layer=10+, n_embd=640+ (49M+, params/token > 10)
# longer context once this works:        block_size=1024 (+524k params in wpe)
