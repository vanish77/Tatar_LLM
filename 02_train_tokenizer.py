# -*- coding: utf-8 -*-
"""
Script for training tokenizer on Tatar corpus
Uses BPE (Byte Pair Encoding) via Hugging Face tokenizers library
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from pathlib import Path
import json


def train_tokenizer(
    train_file: str,
    vocab_size: int = 8192,
    min_frequency: int = 2,
    output_dir: str = "tokenizer"
):
    """
    Train BPE tokenizer on Tatar text
    
    Args:
        train_file: path to training data file
        vocab_size: vocabulary size (number of tokens)
        min_frequency: minimum frequency to include in vocabulary
        output_dir: directory to save tokenizer
    """
    print("="*50)
    print("Training tokenizer")
    print("="*50)
    
    # Create BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    
    # Configure pre-tokenizer (split by whitespace)
    tokenizer.pre_tokenizer = Whitespace()
    
    # Special tokens
    special_tokens = [
        "<pad>",  # Padding token
        "<unk>",  # Unknown token
        "<bos>",  # Begin of sequence
        "<eos>",  # End of sequence
    ]
    
    # Configure trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )
    
    # Train on file
    print(f"\nTraining on file: {train_file}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Minimum frequency: {min_frequency}\n")
    
    tokenizer.train(files=[train_file], trainer=trainer)
    
    # Configure post-processor to add special tokens
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )
    
    # Create directory for saving
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save tokenizer
    tokenizer.save(str(output_path / "tokenizer.json"))
    
    # Save configuration
    config = {
        "vocab_size": vocab_size,
        "model_type": "BPE",
        "special_tokens": special_tokens,
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token_id": tokenizer.token_to_id("<pad>"),
        "unk_token_id": tokenizer.token_to_id("<unk>"),
        "bos_token_id": tokenizer.token_to_id("<bos>"),
        "eos_token_id": tokenizer.token_to_id("<eos>"),
    }
    
    with open(output_path / "config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*50)
    print("Tokenizer trained and saved!")
    print(f"Directory: {output_path}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print("="*50)
    
    # Test tokenizer
    test_tokenizer(tokenizer)
    
    return tokenizer


def test_tokenizer(tokenizer):
    """Test tokenizer functionality"""
    print("\nTesting tokenizer:\n")
    
    test_sentences = [
        "Kazan shehere Tatarstan Respublikasining bashkalasy.",
        "Min tatarcha oyranem.",
        "Haller nichek?",
    ]
    
    for sentence in test_sentences:
        encoded = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoded.ids)
        
        print(f"Original: {sentence}")
        print(f"Tokens: {encoded.tokens}")
        print(f"IDs: {encoded.ids}")
        print(f"Decoded: {decoded}")
        print("-" * 50)


def load_tokenizer(tokenizer_path: str = "tokenizer"):
    """Load saved tokenizer"""
    tokenizer = Tokenizer.from_file(str(Path(tokenizer_path) / "tokenizer.json"))
    return tokenizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train tokenizer")
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/processed/train.txt",
        help="Path to training data file"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=8192,
        help="Vocabulary size"
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum frequency to include in vocabulary"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tokenizer",
        help="Directory to save tokenizer"
    )
    
    args = parser.parse_args()
    
    train_tokenizer(
        train_file=args.train_file,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        output_dir=args.output_dir,
    )

