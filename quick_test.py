"""Quick smoke test for the trained Tatar GPT model."""

from __future__ import annotations

import torch

from utils import generate_text, load_model, load_tokenizer


TEST_PROMPTS = [
    "Казан турында сөйлә.",
    "Габдулла Тукай кем?",
    "Татарларның милли ашлары",
    "Татарча технологияләр турында"
]


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    print("=" * 60)
    print("Quick Test — Tatar LLM")
    print("=" * 60)

    device = resolve_device()
    try:
        model, _ = load_model("models/best_model.pt", device)
        tokenizer, tok_config = load_tokenizer("tokenizer")
    except FileNotFoundError as exc:
        print(f"❌ Missing artifact: {exc}")
        print("Run the full training pipeline first (03_train.py).")
        return

    print(f"✅ Model ready on {device}. Running {len(TEST_PROMPTS)} prompts...\n")

    for idx, prompt in enumerate(TEST_PROMPTS, 1):
        print("-" * 60)
        print(f"Prompt {idx}/{len(TEST_PROMPTS)}:\n{prompt}")
        for sample in range(2):
            text = generate_text(
                model,
                tokenizer,
                tok_config,
                prompt,
                max_new_tokens=60,
                temperature=0.8,
                top_k=40,
                device=device,
            )
            print(f"  Sample {sample + 1}: {text}")
        print()

    print("=" * 60)
    print("Done. For richer demos open demo_inference.ipynb.")


if __name__ == "__main__":
    main()


