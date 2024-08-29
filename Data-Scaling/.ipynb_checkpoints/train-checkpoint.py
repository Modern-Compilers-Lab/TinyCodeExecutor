import os
import time
import datetime
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from tqdm import tqdm
from model import GPT


# Configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
compile = True
# wandb_log = True
# wandb_project = 'TinyLanguageModel'

DATA_SIZES = ["2M", "4M", "8M", "16M", "32M", "48M", "64M", "96M"]
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')



batch_size = 64
block_size = 256
num_epochs = 1.77
eval_interval = 10000
learning_rate = 1e-3
eval_iters = 1000
dropout = 0


# Helper functions
meta_file = '/scratch/mn3620/Data-Scaling/data/96M/meta.pkl'
# Load meta information
with open(meta_file, 'rb') as f:
    meta = pickle.load(f)
    stoi = meta['stoi']
    itos = meta['itos']

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
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
        
def calculate_iterations(data_file):
    with open(data_file, 'r') as f:
        data = f.read()
        iters_per_epoch = len(data) // (batch_size * block_size)
        num_iters = num_epochs * iters_per_epoch
        return int(num_iters)

def human_readable(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

# Training Loop
for size in DATA_SIZES:
    
    data_dir = os.path.join('data', size)
    train_file = os.path.join(data_dir, 'train.txt')
    val_file = os.path.join(data_dir, 'val.txt')
    num_iters = calculate_iterations(train_file)
    print(f"Data size {size}: {num_iters} iterations")

    miles = [int( num_iters * m) for m in [0.7, 0.8, 0.9]]

    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    model = GPT()
    model.to(device)
    if compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=miles, gamma=0.1)

    # Calculate the number of parameters
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_parameters_hr = human_readable(num_parameters)

    # Training loop
    start_time = time.time()
    for iter in range(num_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f'iter {iter:5d} | train loss {losses["train"]:.4f} | val loss {losses["val"]:.4f}')
            # if wandb_log:
            #     import wandb
            #     wandb.init(project=wandb_project, name=f'TLM_RUN_{num_parameters_hr}_{size}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}', config=locals())
            #     wandb.log({
            #         "iter": iter,
            #         "train/loss": losses['train'],
            #         "val/loss": losses['val'],
            #         "lr": scheduler.get_last_lr()[0],
            #     })

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

    end_time = time.time()
    print(f'Training time for {size}: {(end_time - start_time) / 60} min')

    # Save the model
    model_filename = f"{num_parameters_hr}_{size}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.pth"
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, model_filename))
