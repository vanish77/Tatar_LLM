# -*- coding: utf-8 -*-
"""
GPT model architecture (decoder-only) with modern optimizations
Includes: Rotary Embeddings, Flash Attention support, LayerNorm, GELU
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GPTConfig:
    """GPT model configuration"""
    vocab_size: int = 8192
    n_layer: int = 6  # Number of transformer layers
    n_head: int = 8   # Number of attention heads
    n_embd: int = 512  # Embedding dimension
    block_size: int = 256  # Maximum context length
    dropout: float = 0.1
    bias: bool = True  # Whether to use bias in Linear and LayerNorm
    use_rotary: bool = True  # Use Rotary Position Embeddings
    use_flash_attention: bool = False  # Flash Attention (if available)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE)
    Modern approach to positional embeddings from RoFormer paper
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache cos and sin for speed
        self._seq_len_cached = max_seq_len
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor, seq_len: int):
        """
        x: (batch_size, n_heads, seq_len, head_dim)
        """
        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.device),
            self.sin_cached[:, :, :seq_len, ...].to(x.device),
        )


def rotate_half(x: torch.Tensor):
    """Rotate half of hidden dimensions"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary embeddings to query and key"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with Rotary Embeddings support"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_rotary = config.use_rotary
        self.use_flash_attention = config.use_flash_attention
        
        # Projections for Q, K, V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Rotary embeddings
        if self.use_rotary:
            self.rotary = RotaryPositionalEmbedding(
                self.head_dim, 
                max_seq_len=config.block_size
            )
        
        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        )
    
    def forward(self, x: torch.Tensor):
        B, T, C = x.size()  # batch, sequence length, embedding dim
        
        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Split into heads
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Apply Rotary Embeddings to Q and K
        if self.use_rotary:
            cos, sin = self.rotary(q, T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled Dot-Product Attention
        # Check Flash Attention availability via scaled_dot_product_attention
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch 2.0+ Flash Attention
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Standard attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        # Concatenate heads back
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-Forward Network"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer Block"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    
    def forward(self, x):
        # Pre-normalization (modern approach)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT model (decoder-only)"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
            wpe=nn.Embedding(config.block_size, config.n_embd) if not config.use_rotary else None,  # position embeddings (if not using RoPE)
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying (share weights between embedding and lm_head)
        self.transformer.wte.weight = self.lm_head.weight
        
        # Weight initialization
        self.apply(self._init_weights)
        
        # Special initialization for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        # Parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    def _init_weights(self, module):
        """Weight initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        """
        idx: (B, T) token indices
        targets: (B, T) target tokens for training (optional)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block_size {self.config.block_size}"
        
        # Token embeddings
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        
        # Position embeddings (if not using Rotary)
        if not self.config.use_rotary:
            pos = torch.arange(0, t, dtype=torch.long, device=device)  # (T,)
            pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)
        
        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        
        # Logits
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Use -1 as ignore_index to ignore padding tokens
            # Note: pad_token_id should be converted to -1 in targets for loss computation
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens
        idx: (B, T) initial sequence
        max_new_tokens: number of new tokens to generate
        temperature: sampling temperature
        top_k: top-k sampling
        """
        for _ in range(max_new_tokens):
            # Truncate sequence if it exceeds block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Take logits of last token and apply temperature
            logits = logits[:, -1, :] / temperature
            
            # Optionally apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure optimizer with weight decay only for certain parameters
        """
        # Separate parameters into those that need weight decay and those that don't
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        # Get all parameter names first
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn
                
                # Skip if parameter doesn't exist (due to weight tying)
                if fpn not in param_dict:
                    continue
                
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        
        # RoPE parameters also don't decay
        # Handle wildcard pattern for RoPE
        for pn in param_dict.keys():
            if 'rotary' in pn and 'inv_freq' in pn:
                no_decay.add(pn)
        
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay)) if pn in param_dict], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict], "weight_decay": 0.0},
        ]
        
        # Use AdamW with fused operations for speed
        use_fused = (device_type == 'cuda') and ('fused' in torch.optim.AdamW.__doc__)
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        
        return optimizer


def count_parameters(model):
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test run
    config = GPTConfig(
        vocab_size=8192,
        n_layer=6,
        n_head=8,
        n_embd=512,
        block_size=256,
        dropout=0.1,
        use_rotary=True,
    )
    
    model = GPT(config)
    print(f"\nModel configuration:")
    print(f"  Layers: {config.n_layer}")
    print(f"  Heads: {config.n_head}")
    print(f"  Dimension: {config.n_embd}")
    print(f"  Context: {config.block_size}")
    print(f"  Rotary Embeddings: {config.use_rotary}")
    print(f"\nTotal parameters: {count_parameters(model):,}")
    
    # Test input
    x = torch.randint(0, config.vocab_size, (2, 64))
    logits, loss = model(x, x)
    print(f"\nTest forward pass:")
    print(f"  Input: {x.shape}")
    print(f"  Output (logits): {logits.shape}")

