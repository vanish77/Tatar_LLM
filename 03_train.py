# -*- coding: utf-8 -*-
"""
Script for training GPT model on Tatar language
Includes optimizations: Mixed Precision, Gradient Checkpointing, MPS support
"""

import os
import time
import math
import json
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tokenizers import Tokenizer

from model import GPT, GPTConfig


class TextDataset(Dataset):
    """Dataset for language modeling"""
    
    def __init__(self, data_path: str, tokenizer: Tokenizer, block_size: int, pad_token_id: int = 0):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        
        # Load text
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"Loaded file: {data_path}")
        print(f"Text size: {len(text):,} characters")
        
        # Tokenize entire text
        encoded = tokenizer.encode(text)
        self.tokens = torch.tensor(encoded.ids, dtype=torch.long)
        
        print(f"Number of tokens: {len(self.tokens):,}")
    
    def __len__(self):
        # Ensure we return at least 0
        length = len(self.tokens) - self.block_size
        return max(0, length)
    
    def __getitem__(self, idx):
        # Ensure idx is valid
        max_idx = len(self.tokens) - self.block_size - 1
        if max_idx < 0:
            # Dataset is too small, return padding
            x = torch.full((self.block_size,), self.pad_token_id, dtype=torch.long)
            y = torch.full((self.block_size,), self.pad_token_id, dtype=torch.long)
            return x, y
        
        idx = min(idx, max_idx)
        # Take block of tokens of size block_size + 1
        chunk = self.tokens[idx:idx + self.block_size + 1]
        
        # Ensure chunk has correct size (pad if necessary)
        if len(chunk) < self.block_size + 1:
            padding = torch.full((self.block_size + 1 - len(chunk),), self.pad_token_id, dtype=torch.long)
            chunk = torch.cat([chunk, padding])
        
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def get_batch(data_loader_iter, device):
    """Get batch of data"""
    try:
        x, y = next(data_loader_iter)
    except StopIteration:
        return None, None
    
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def estimate_loss(model, val_loader, eval_iters, device, ctx):
    """Estimate loss on validation set"""
    model.eval()
    losses = []
    val_iter = iter(val_loader)
    
    # Handle case when dataset is too small
    try:
        max_iters = min(eval_iters, len(val_loader))
    except (ValueError, TypeError):
        max_iters = eval_iters
    
    for _ in range(max_iters):
        x, y = get_batch(val_iter, device)
        if x is None:
            break
        
        with ctx:
            logits, loss = model(x, y)
        losses.append(loss.item())
    
    model.train()
    return sum(losses) / len(losses) if losses else float('inf')


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """Learning rate schedule with warmup and cosine decay"""
    # Warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # Cosine decay
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def train():
    """Main training function"""
    
    # ==================== Hyperparameters ====================
    
    # Data
    data_dir = "data/processed"
    tokenizer_path = "tokenizer"
    out_dir = "models"
    
    # Model
    n_layer = 6
    n_head = 8
    n_embd = 512
    block_size = 256
    dropout = 0.1
    use_rotary = True
    use_flash_attention = False  # Set True if you want to try
    
    # Training
    batch_size = 32  # Batch size (can increase/decrease depending on memory)
    gradient_accumulation_steps = 4  # Effective batch_size = batch_size * gradient_accumulation_steps
    max_iters = 10000  # Maximum number of iterations
    eval_interval = 500  # How often to evaluate loss
    eval_iters = 100  # Number of batches for evaluation
    log_interval = 10  # How often to output logs
    
    # Optimizer
    learning_rate = 3e-4
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0  # Gradient clipping
    
    # LR schedule
    warmup_iters = 100
    lr_decay_iters = max_iters
    min_lr = learning_rate / 10
    
    # System
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    
    print(f"Using device: {device}")
    
    # Mixed precision
    use_amp = device == 'cuda'  # AMP works best on CUDA
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'mps' else torch.amp.autocast(device_type=device, dtype=ptdtype)
    
    # Gradient checkpointing for memory savings
    use_gradient_checkpointing = False  # Set True if running out of memory
    
    # ==================== Preparation ====================
    
    # Create directory for saving
    os.makedirs(out_dir, exist_ok=True)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = Tokenizer.from_file(str(Path(tokenizer_path) / "tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    # Load tokenizer configuration
    with open(Path(tokenizer_path) / "config.json", 'r') as f:
        tok_config = json.load(f)
    pad_token_id = tok_config['pad_token_id']
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = TextDataset(
        f"{data_dir}/train.txt",
        tokenizer,
        block_size,
        pad_token_id=pad_token_id
    )
    val_dataset = TextDataset(
        f"{data_dir}/val.txt",
        tokenizer,
        block_size,
        pad_token_id=pad_token_id
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Better to use 0 on MPS
        pin_memory=(device == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == 'cuda')
    )
    
    # Create model
    print("\nCreating model...")
    model_config = GPTConfig(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        dropout=dropout,
        use_rotary=use_rotary,
        use_flash_attention=use_flash_attention,
    )
    
    model = GPT(model_config)
    model.to(device)
    
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Optimizer
    optimizer = model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        betas=(beta1, beta2),
        device_type=device
    )
    
    # GradScaler for mixed precision (only for CUDA)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # ==================== Training ====================
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Maximum iterations: {max_iters}")
    print(f"Approximate number of epochs: {max_iters * batch_size * gradient_accumulation_steps // len(train_dataset)}")
    print("="*60 + "\n")
    
    train_iter = iter(train_loader)
    best_val_loss = float('inf')
    t0 = time.time()
    
    for iter_num in range(max_iters):
        # Determine learning rate
        lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluation on validation
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            # Check if validation dataset is available
            if len(val_loader) > 0:
                val_loss = estimate_loss(model, val_loader, eval_iters, device, ctx)
                print(f"\nStep {iter_num}: train loss -, val loss {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': model_config.__dict__,
                    }
                    print(f"Saving best model (val_loss: {val_loss:.4f})...")
                    torch.save(checkpoint, f"{out_dir}/best_model.pt")
            else:
                # If validation dataset is too small, skip validation
                print(f"\nStep {iter_num}: train loss -, val loss N/A (validation dataset too small)")
                # Save model anyway periodically
                if iter_num % (eval_interval * 2) == 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'best_val_loss': None,
                        'config': model_config.__dict__,
                    }
                    print(f"Saving checkpoint (iter {iter_num})...")
                    torch.save(checkpoint, f"{out_dir}/checkpoint_iter_{iter_num}.pt")
        
        # Gradient accumulation loop
        optimizer.zero_grad(set_to_none=True)
        loss_values = []  # Store full losses for averaging
        
        for micro_step in range(gradient_accumulation_steps):
            x, y = get_batch(train_iter, device)
            
            # If data is exhausted, start new epoch
            if x is None:
                train_iter = iter(train_loader)
                x, y = get_batch(train_iter, device)
                if x is None:  # Dataset is empty
                    print("Warning: Dataset is empty, skipping iteration")
                    continue
            
            # Ensure we have valid data
            if x is None or y is None:
                print("Warning: Got None batch, skipping iteration")
                continue
            
            with ctx:
                # Convert pad_token_id to -1 in targets for loss computation (to ignore padding)
                y_for_loss = y.clone()
                y_for_loss[y == pad_token_id] = -1
                
                logits, loss = model(x, y_for_loss)
                # Store full loss for logging (before normalization)
                loss_full = loss.item()
                loss_values.append(loss_full)
                
                # Normalize loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
            
            # Backward pass
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        
        # Compute average loss across all micro steps
        loss_accum = sum(loss_values) / len(loss_values) if loss_values else 0.0
        
        # Gradient clipping
        if grad_clip != 0.0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Optimizer step
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        # Logging
        if iter_num % log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tokens_per_sec = (batch_size * gradient_accumulation_steps * block_size * log_interval) / dt
            print(f"iter {iter_num}: loss {loss_accum:.4f}, lr {lr:.2e}, time {dt*1000:.2f}ms, tok/sec {tokens_per_sec:.0f}")
    
    # Final save
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_num': max_iters,
        'best_val_loss': best_val_loss if best_val_loss != float('inf') else None,
        'config': model_config.__dict__,
    }
    torch.save(checkpoint, f"{out_dir}/final_model.pt")
    print(f"Final model saved to {out_dir}/final_model.pt")
    if best_val_loss != float('inf'):
        print(f"Best val_loss: {best_val_loss:.4f}")
    else:
        print("Note: Validation dataset was too small, no val_loss available")


if __name__ == "__main__":
    train()
