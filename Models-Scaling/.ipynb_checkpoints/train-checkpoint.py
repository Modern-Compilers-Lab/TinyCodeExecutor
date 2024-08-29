# train.py
import os
import time
import datetime
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import numpy as np
import pandas as pd
from tqdm import tqdm

from model import GPT

DATA_DIR = '/scratch/mn3620/Data-Scaling/data/48M'
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 365000 
miles = [int(max_iters * m) for m in [0.7, 0.8, 0.9]]
eval_interval = 10000
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 500
dropout = 0
compile = True
# wandb_log = True
# wandb_project = 'TinyLanguageModel'

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}

torch.manual_seed(1337)

train_data = np.memmap(os.path.join(DATA_DIR, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(DATA_DIR, 'val.bin'), dtype=np.uint16, mode='r')

meta_path = os.path.join(DATA_DIR, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def human_readable(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

# List of n_embd values
n_embd_values = [36, 84, 120, 150, 252, 372, 456, 522]

for n_embd in n_embd_values:
    # Model
    model = GPT(n_embd=n_embd)
    model = model.to(device)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_parameters_hr = human_readable(num_parameters)
    print(f'The model has {num_parameters_hr} trainable parameters')


    # Get current date and hour
    now = datetime.datetime.now()
    date_hour = now.strftime("%Y-%m-%d_%H-%M")

    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)
        del unoptimized_model

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=eval_interval, gamma=0.5)
    start_time = time.time()

    # Training loop
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"iter {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # if wandb_log:
            #     import wandb
            #     wandb.log({'train_loss': losses['train'], 'val_loss': losses['val'], 'iter': iter})
        X, Y = get_batch('train')
        logits, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()



    # Save model
    model_name = f"{num_parameters_hr}_{n_embd}_{date_hour}.pth"
    model_path = os.path.join(MODELS_DIR, model_name)
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    print(f"Total training time for {num_parameters_hr} Model: {datetime.timedelta(seconds=int(time.time() - start_time))}")
