# nanoGPT config for an ARM CPU / Termux box (no GPU, ~4 GB RAM)
#
# Data is unchanged (data/shakespeare_complete, 4,808,687 train tokens, vocab 79).
# What changes is everything about compute: the model shrinks, the batch shrinks,
# and max_iters must be derived from YOUR measured throughput, not from this file.
#
# MANDATORY command-line overrides - default dtype/device/compile are all wrong on ARM:
#   python train.py config/train_termux_cpu.py \
#       --device=cpu --compile=False --dtype=float32
#
# Recommended environment:
#   export OMP_NUM_THREADS=4
#   export OPENBLAS_NUM_THREADS=4
#   termux-wake-lock                      # stop Android from freezing it
#
# First calibrate, then set max_iters yourself (see README notes below script):
#   python train.py config/train_termux_cpu.py --device=cpu --compile=False --dtype=float32 \
#       --max_iters=40 --eval_iters=10 --log_interval=1
#   # read "Xms" per iter  ->  tokens/s = 8192 / (X/1000)
#   # max_iters = time_budget_seconds * tokens_per_s / 8192

out_dir = 'out-termux-cpu'
eval_interval = 300     # checkpoint more often: Android may reclaim the process
eval_iters = 25         # a full eval costs 200 forward passes, wasteful on CPU
log_interval = 10
always_save_checkpoint = False
init_from = 'scratch'   # change to 'resume' to continue after Android kills it

wandb_log = False
dataset = 'shakespeare_complete'

# --- model: 4 layers x 256 wide = 3,245,312 params ---
# The 8x512 config needs ~1.5e16 FLOPs, which is weeks on 4 Cortex-A76 cores.
# This one is ~19 MFLOPs/token, so a whole night buys you ~10 epochs of the corpus.
n_layer = 4
n_head = 4
n_embd = 256
block_size = 256        # 512 would add ~25% attention FLOPs and ~4x attention memory
dropout = 0.1           # params/token here is only 0.67, so we are under-fitting, not over
bias = True

# --- optimisation: micro-batch 16 x 256 = 4096 tokens, x2 accumulation = 8192 tokens/step ---
# Memory is the reason to accumulate instead of raising batch_size outright:
# attention alone holds batch*n_head*T*T floats per layer, and you have ~1.5-2 GB usable.
batch_size = 16
gradient_accumulation_steps = 2

learning_rate = 1e-3
min_lr = 1e-4
max_iters = 4000        # ~38M tokens ~= 8 epochs; REPLACE with your own calculation
lr_decay_iters = 4000
warmup_iters = 100
beta2 = 0.99
weight_decay = 1e-1
grad_clip = 1.0
decay_lr = True

# --- if you have more patience than RAM allows above, use 6x384 instead: ---
#   n_layer=6, n_head=6, n_embd=384, batch_size=8, gradient_accumulation_steps=2
#   10,776,192 params, ~69 MFLOPs/token -> roughly 3.4x slower than this config
# Expected val loss, rough extrapolation from a 1.47 anchor:
#   4x256  ~1.54    6x384  ~1.42    8x512  ~1.34   (the last one belongs on a GPU)
