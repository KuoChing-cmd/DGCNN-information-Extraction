"""Configuration module for DGCNN-RE."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any, Dict
import yaml
import os


@dataclass
class ModelConfig:
    vocab_size: int = 5000
    num_classes: int = 50
    char_dim: int = 128
    word_dim: int = 0  # 0 indicates not using external dense word vectors
    hidden_dim: int = 128
    max_len: int = 512
    dilation_rates: List[int] = field(
        default_factory=lambda: [1, 2, 5, 1, 2, 5, 1, 2, 5, 1, 1, 1]
    )
    kernel_size: int = 3
    num_heads: int = 8
    size_per_head: int = 16
    dropout: float = 0.25
    conv_dropout: float = 0.1
    num_inter_points: int = 6


@dataclass
class TrainingConfig:
    batch_size: int = 32
    lr: float = 1e-3
    min_lr: float = 1e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 1e-4
    epochs: int = 20
    ema_decay: float = 0.999
    grad_clip: float = 5.0
    device: str = "auto"  # auto, cuda, mps, cpu
    save_dir: str = "checkpoints"
    log_interval: int = 10
    eval_every: int = 1
    patience: int = 6
    seed: int = 42


@dataclass
class DataConfig:
    train_path: str = "data/sample_train.json"
    val_path: Optional[str] = "data/sample_train.json"
    schema_path: str = "data/sample_schemas.json"
    vocab_path: str = "data/vocab.json"
    word2vec_path: Optional[str] = None
    min_char_count: int = 2
    subject_threshold: float = 0.5
    object_threshold: float = 0.5


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from a YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

        model_cfg = ModelConfig(**raw_data.get("model", {}))
        training_cfg = TrainingConfig(**raw_data.get("training", {}))
        data_cfg = DataConfig(**raw_data.get("data", {}))
        return cls(model=model_cfg, training=training_cfg, data=data_cfg)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to a YAML file."""
        os.makedirs(os.path.dirname(os.path.abspath(yaml_path)), exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, sort_keys=False, default_flow_style=False)
