# Tatar LLM — Train a Decoder-Only Model from Scratch

This repository contains the full training pipeline for a compact GPT-style language model trained entirely from scratch on a Tatar corpus. All stages are implemented locally: data collection, tokenizer training, model training, and inference demo.

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

## End-to-End Pipeline

1. **Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate           # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Data preparation** — download and clean the Leipzig Wortschatz corpus:
   ```bash
   python 01_prepare_data.py
   ```
   - Fetches `tat_mixed_2015_1M.tar.gz` (~192?MB, 1?000?000 sentences)
   - Validates archives, extracts sentences, cleans text
   - Produces `data/processed/train.txt` (?950?k sentences) and `val.txt` (?50?k)

3. **Tokenizer training**
   ```bash
   python 02_train_tokenizer.py --vocab_size 8192 --min_frequency 2
   ```
   Saves `tokenizer/tokenizer.json` and `tokenizer/config.json`.

4. **Model training**
   ```bash
   python 03_train.py
   ```
   Default hyperparameters: `n_layer=6`, `n_head=8`, `n_embd=512`, `batch_size=32`,
   `gradient_accumulation_steps=4`, `max_iters=10_000`, `learning_rate=3e-4`.

   The script automatically:
   - Selects `mps` (or `cuda` / `cpu`) device
   - Uses cosine LR with 100 warmup iterations and weight decay 0.1
   - Performs gradient clipping at 1.0
   - Saves checkpoints to `models/`

5. **Inference demo**
   - Quick CLI smoke test: `python quick_test.py`
   - Rich notebook demo: `demo_inference.ipynb`


## Data Source

- Leipzig Wortschatz Tatar corpora — <https://wortschatz.uni-leipzig.de/en/download/tat>
