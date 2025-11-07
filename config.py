"""
���������������� ���� � ����������� ��� �������� ������
��������� ����� ������������������ � ������� ����������������
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """������������ ��������"""
    
    # ����
    data_dir: str = "data/processed"
    tokenizer_path: str = "tokenizer"
    out_dir: str = "models"
    
    # ����������� ������
    n_layer: int = 6  # ���������� �������������� ������
    n_head: int = 8   # ���������� attention heads
    n_embd: int = 512  # ����������� �����������
    block_size: int = 256  # ����� ���������
    dropout: float = 0.1
    use_rotary: bool = True  # Rotary Position Embeddings
    use_flash_attention: bool = False  # Flash Attention (����������������)
    
    # ��������
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_iters: int = 10000
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 10
    
    # �����������
    learning_rate: float = 3e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Learning rate schedule
    warmup_iters: int = 100
    lr_decay_iters: int = 10000  # ������ = max_iters
    min_lr: float = 3e-5  # learning_rate / 10
    
    # �����������
    use_gradient_checkpointing: bool = False  # �������� ���� �� ������� ������
    
    # ��� ���������/������� ������ (������������)
    @classmethod
    def tiny(cls):
        """����� ��������� ������������ ��� �������� ������������"""
        return cls(
            n_layer=4,
            n_head=4,
            n_embd=256,
            block_size=128,
            batch_size=16,
            max_iters=2000,
        )
    
    # ��� ������� ������ (������ ��������/��������)
    @classmethod
    def medium(cls):
        """������� ������������ - ������� ������"""
        return cls(
            n_layer=8,
            n_head=8,
            n_embd=512,
            block_size=256,
            batch_size=32,
            max_iters=15000,
        )
    
    # ��� ������� ������ (���� ���� ����� � �������)
    @classmethod
    def large(cls):
        """������� ������������ ��� ������� ��������"""
        return cls(
            n_layer=12,
            n_head=12,
            n_embd=768,
            block_size=512,
            batch_size=16,
            gradient_accumulation_steps=8,
            max_iters=25000,
        )


@dataclass
class TokenizerConfig:
    """������������ ������������"""
    
    vocab_size: int = 8192
    min_frequency: int = 2
    
    # �������������� ������� ������� ��� �������������
    @classmethod
    def small(cls):
        """��������� ������� - �������, �� ���� ��������"""
        return cls(vocab_size=4096)
    
    @classmethod
    def large(cls):
        """������� ������� - ���������, �� ����� ��������"""
        return cls(vocab_size=16384)


# ������ ������������� � ����:
# from config import TrainingConfig
# config = TrainingConfig.medium()  # ��� .tiny() / .large()
# model = GPT(config)


