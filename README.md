# Tatar LLM — Train a Decoder-Only Model from Scratch

This repository contains the full training pipeline for a compact GPT-style language model trained entirely from scratch on a Tatar corpus. All stages are implemented locally: data collection, tokenizer training, model training, and inference demo.

see QUICKSTART.md for starting.

## Project Overview

- **Language:** Tatar (low-resource, ~5 million speakers)
- **Architecture:** Decoder-only Transformer (GPT)
- **Parameter count:** ~23M
- **Context length:** 256 tokens
- **Tokenizer:** Byte-Pair Encoding (vocab size 8192, trained from scratch)
- **Hardware target:** Apple MacBook Pro M1 Pro (MPS acceleration)

## Key Features

- Rotary Position Embeddings (RoPE)
- Optional Flash Attention via `torch.nn.functional.scaled_dot_product_attention`
- Mixed precision ready (AMP on CUDA) and hooks for gradient checkpointing
- Weight tying between token embeddings and LM head
- Cosine LR schedule with warmup plus gradient clipping
- Robust data download/validation with automatic fallbacks

## Model Artifacts

- Trained checkpoints (`best_model.pt`, `final_model.pt`) and tokenizer files are available on Hugging Face: [ivankadchenko/tatar-gpt](https://huggingface.co/ivankadchenko/tatar-gpt)
- Folder structure matches this repo (`models/`, `tokenizer/`) so you can download and drop them back into place if you want to skip training.

## Data Source

- Leipzig Wortschatz Tatar corpora — <https://wortschatz.uni-leipzig.de/en/download/tat>
