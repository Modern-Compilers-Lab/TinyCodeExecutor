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

# Set random seed
torch.manual_seed(1337)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DATA_DIR =  'data/ALL'
DATA_DIR = '/scratch/mn3620/Data-Scaling/data/48M'
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Hyperparameters
batch_size = 64  
block_size = 256 
eval_interval = 10000
learning_rate = 1e-3 # 1e-3 , 1e-2 , 3e-4
eval_iters = 1000
dropout = 0
compile = True 

# Define epochs list
epochs_list = [0.7, 1, 1.77, 2, 3, 4, 5, 6, 7, 8]



# wandb 
wandb_log = False 
wandb_project = 'TinyLanguageModel'
# For logging purposes
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}



# Calculate max_iters based on epochs list
def calculate_iterations(data_length, batch_size, block_size, num_epochs):
    iters_per_epoch = data_length // (batch_size * block_size)
    num_iters = num_epochs * iters_per_epoch
    return int(num_iters)



# Function to get a batch of data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Estimate loss on train and test splits
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

# Define human_readable function
def human_readable(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


# Load data
train_data = np.memmap(os.path.join(DATA_DIR, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(DATA_DIR, 'val.bin'), dtype=np.uint16, mode='r')

# Estimate data length
data_length = len(train_data)

# Loop through each epoch value
for num_epochs in epochs_list:
    # Calculate max_iters for the current number of epochs
    max_iters = calculate_iterations(data_length, batch_size, block_size, num_epochs)
    
    # Miles (scheduler milestones) based on max_iters
    miles = [int(max_iters * m) for m in [0.7, 0.8, 0.9]]

    # Model
    model = GPT()
    m = model.to(device)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    num_parameters_hr = human_readable(num_parameters)
    print(f'The model has {num_parameters_hr} trainable parameters')

    # Get current date and hour
    now = datetime.datetime.now()
    date_hour = now.strftime("%Y-%m-%d_%H-%M")

    # Construct wandb_run_name
    wandb_run_name = f'TLM_RUN_{num_parameters_hr}_{num_epochs}epochs_{date_hour}'

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=miles, gamma=0.1)

    # Train
    if wandb_log:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    # Calculate training time 
    start_time = time.time()

    for iter in range(max_iters):
        # Evaluate the model on the train and val splits and log the losses
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f'iter {iter:5d} | train loss {losses["train"]:.4f} | val loss {losses["val"]:.4f}')
            if wandb_log:
                wandb.log({
                    "iter": iter,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": scheduler.get_last_lr()[0],
                })

        # Train the model for one iteration
        xb, yb = get_batch('train')

        # Forward pass
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Step the scheduler
        scheduler.step()

    end_time = time.time()
    print(f'Training time: {(end_time - start_time) / 60}  min')

    torch.save(model.state_dict(), f"{MODELS_DIR}/{num_parameters_hr}_{num_epochs}epochs_{date_hour}.pth")
