"""Utility helpers for loading models, tokenizers, and running inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


def _resolve_device(device: str) -> str:
    """Return a concrete device string."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_path: str, device: str = "auto") -> Tuple[GPT, GPTConfig]:
    """Load a trained GPT checkpoint and move it to the requested device."""
    device = _resolve_device(device)
    print(f"Loading model on device: {device}")

    checkpoint = torch.load(model_path, map_location=device)
    model_config = GPTConfig(**checkpoint["config"])

    model = GPT(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    best_val = checkpoint.get("best_val_loss")
    print(f"Checkpoint loaded. Best val loss: {best_val if best_val is not None else 'N/A'}")

    return model, model_config


def load_tokenizer(tokenizer_path: str = "tokenizer") -> Tuple[Tokenizer, dict]:
    """Load the trained tokenizer and its configuration."""
    tokenizer_dir = Path(tokenizer_path)

    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    with open(tokenizer_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"Tokenizer loaded. Vocabulary size: {tokenizer.get_vocab_size()}")
    return tokenizer, config


def generate_text(
    model: GPT,
    tokenizer: Tokenizer,
    tok_config: dict,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = "cpu",
) -> str:
    """Generate text continuation given a prompt."""
    device = _resolve_device(device)
    model.to(device)

    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    generated_ids = generated[0].tolist()
    special_tokens = {
        tok_config.get("bos_token_id"),
        tok_config.get("eos_token_id"),
        tok_config.get("pad_token_id"),
    }
    filtered_ids = [token for token in generated_ids if token not in special_tokens]

    text = tokenizer.decode(filtered_ids)
    return text.strip()


def calculate_perplexity(
    model: GPT,
    tokenizer: Tokenizer,
    text: str,
    device: str = "cpu",
) -> float:
    """Compute perplexity for a given text sample."""
    import math

    device = _resolve_device(device)
    encoded = tokenizer.encode(text)
    tokens = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    if tokens.size(1) <= 1:
        return float("inf")

    with torch.no_grad():
        _, loss = model(tokens[:, :-1], tokens[:, 1:])

    return math.exp(loss.item())


def interactive_generation(
    model_path: str = "models/best_model.pt",
    tokenizer_path: str = "tokenizer",
) -> None:
    """Interactive CLI loop for manual prompting."""
    print("=" * 60)
    print("Interactive generation demo")
    print("=" * 60)

    model, _ = load_model(model_path, device="auto")
    tokenizer, tok_config = load_tokenizer(tokenizer_path)
    device = _resolve_device("auto")

    print("Type 'quit' to exit. Prompts should be in Tatar.")
    print()

    while True:
        prompt = input("Prompt: ").strip()
        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        try:
            print("\nGenerating...\n")
            text = generate_text(
                model,
                tokenizer,
                tok_config,
                prompt,
                max_new_tokens=80,
                temperature=0.8,
                top_k=40,
                device=device,
            )
            print(text)
            print("-" * 60)
        except Exception as exc:  # pragma: no cover - interactive helper
            print(f"Error: {exc}")
            print("-" * 60)


if __name__ == "__main__":
    interactive_generation()


