# Tatar LLM — Train a Decoder-Only Model from Scratch

This repository contains the full training pipeline for a compact GPT-style language model trained entirely from scratch on a Tatar corpus. All stages are implemented locally: data collection, tokenizer training, model training, and inference demo.

## Project Overview

- **Language:** Tatar (low-resource, ~5?million speakers)
- **Architecture:** Decoder-only Transformer (GPT)
- **Parameter count:** ~23?M
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

## Training Snapshot (M1 Pro run)

- Effective batch size: 128 tokens (32 ? grad accumulation 4)
- Training time: ~6?hours for 10?000 iterations (?14 epochs over corpus)
- Best validation loss: **8.53** (step 500)
- Throughput: ~11–12?k tokens/sec on MPS during steady state

> Loss remains relatively high because of limited model capacity and the small hardware budget. Longer training or a larger model will improve perplexity.

## Sample Generations

Using `models/best_model.pt` and the trained tokenizer:

```
Prompt: Кояшлы к?нд? Казан урамнары турында с?йл?п бир.
? Output: Ко яш лы к?нд? Казан урам нары турында с?йл?п бир . . ??м

Prompt: Татар язучысы Габдулла Тукай турында н?рс? бел?се??
? Output: Татар язучы сы Габдулла Тукай турында н?рс? бел? се? ? . , .

Prompt: Татарларны? милли ашлары нинди?
? Output: Татар ларны? милли аш лары нинди ? ? . . ел
```

The tokenizer uses subword pieces, so outputs may contain inserted spaces. At the current training stage the model mostly echoes short fragments and inserts frequent morphemes; coherent sentences appear rarely. Treat these samples as proof that the pipeline runs end-to-end rather than evidence of fluent Tatar generation. Longer training or instruction-style fine-tuning is required for meaningful answers.

## Repository Layout

```
01_prepare_data.py      # Corpus download / cleaning
02_train_tokenizer.py   # BPE tokenizer training
03_train.py             # GPT training loop
model.py                # Architecture & optimizer helpers
utils.py                # Loading, generation utilities
quick_test.py           # CLI smoke test
demo_inference.ipynb    # Notebook for inference demo
run_pipeline.sh         # Helper script to run all stages
restart_training.sh     # Convenience script for restarting training
/
??? data/processed/     # Train/val text datasets
??? models/             # Checkpoints (best_model.pt, final_model.pt)
??? tokenizer/          # Tokenizer JSON + config
```

## Inference from Python

```python
import torch
from utils import load_model, load_tokenizer, generate_text

model, _ = load_model("models/best_model.pt")
tokenizer, tok_config = load_tokenizer("tokenizer")

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
model.to(device)

prompt = "Казан турында бернич? ??мл? яз."
output = generate_text(
    model,
    tokenizer,
    tok_config,
    prompt,
    max_new_tokens=80,
    temperature=0.8,
    top_k=40,
    device=device,
)
print(output)
```

## Deployment & Sharing

- Upload `models/best_model.pt`, `models/final_model.pt`, and the `tokenizer/` directory to a storage service or the Hugging Face Hub.
- `app.py` contains a Gradio skeleton for building a simple web demo.
- See `DEPLOYMENT.md` and `QUICKSTART.md` for concise setup and sharing guidance.

## Data Source

- Leipzig Wortschatz Tatar corpora — <https://wortschatz.uni-leipzig.de/en/download/tat>

## Suggested Next Steps

- Experiment with larger vocabularies or SentencePiece unigram tokenizers
- Add automatic evaluation (perplexity, BLEU on held-out prompts)
- Fine-tune with instruction/chat-style data for QA tasks
- Extend the architecture (MoE blocks, SWA, rotary-aware flash attention)
- Publish the final notebook and checkpoints to Hugging Face for wider access

Happy hacking! Let me know if you build on this work or uncover interesting results.
