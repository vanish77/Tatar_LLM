"""Central configuration dataclasses for training and tokenizer settings."""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Default hyperparameters for training the GPT model."""

    # Paths
    data_dir: str = "data/processed"
    tokenizer_path: str = "tokenizer"
    out_dir: str = "models"

    # Model architecture
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 256
    dropout: float = 0.1
    use_rotary: bool = True
    use_flash_attention: bool = False

    # Training loop
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_iters: int = 10_000
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 10

    # Optimizer settings
    learning_rate: float = 3e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Learning rate schedule
    warmup_iters: int = 100
    lr_decay_iters: int = 10_000
    min_lr: float = 3e-5

    # Memory optimizations
    use_gradient_checkpointing: bool = False

    @classmethod
    def tiny(cls) -> "TrainingConfig":
        """Very small configuration for quick smoke tests."""
        return cls(
            n_layer=4,
            n_head=4,
            n_embd=256,
            block_size=128,
            batch_size=16,
            max_iters=2_000,
        )

    @classmethod
    def medium(cls) -> "TrainingConfig":
        """Slightly larger model for extended experiments."""
        return cls(
            n_layer=8,
            n_head=8,
            n_embd=512,
            block_size=256,
            batch_size=32,
            max_iters=15_000,
        )

    @classmethod
    def large(cls) -> "TrainingConfig":
        """Heavier configuration if you have more compute available."""
        return cls(
            n_layer=12,
            n_head=12,
            n_embd=768,
            block_size=512,
            batch_size=16,
            gradient_accumulation_steps=8,
            max_iters=25_000,
        )


@dataclass
class TokenizerConfig:
    """Default tokenizer hyperparameters."""

    vocab_size: int = 8_192
    min_frequency: int = 2

    @classmethod
    def small(cls) -> "TokenizerConfig":
        """Smaller vocabulary variant for quick experiments."""
        return cls(vocab_size=4_096)

    @classmethod
    def large(cls) -> "TokenizerConfig":
        """Larger vocabulary variant if memory allows."""
        return cls(vocab_size=16_384)


# Example usage:
# from config import TrainingConfig
# config = TrainingConfig()


