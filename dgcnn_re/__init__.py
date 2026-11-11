"""DGCNN Information Extraction Package (PyTorch Modern Implementation)."""

from dgcnn_re.config import Config, ModelConfig, TrainingConfig, DataConfig
from dgcnn_re.dataset import Vocab, Schema, RelationExtractionDataset, collate_fn
from dgcnn_re.layers import (
    PositionalEncoding,
    DilatedGatedConv1d,
    MultiHeadSelfAttention,
    MaskedGlobalMaxPool1d,
)
from dgcnn_re.model import (
    DGCNNEncoder,
    SubjectPointerDecoder,
    ObjectPointerDecoder,
    DGCNNForRelationExtraction,
)
from dgcnn_re.loss import MaskedRelationExtractionLoss
from dgcnn_re.metrics import ExponentialMovingAverage, compute_spo_metrics
from dgcnn_re.trainer import Trainer
from dgcnn_re.predictor import RelationExtractor

__version__ = "2.0.0"
__all__ = [
    "Config",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "Vocab",
    "Schema",
    "RelationExtractionDataset",
    "collate_fn",
    "PositionalEncoding",
    "DilatedGatedConv1d",
    "MultiHeadSelfAttention",
    "MaskedGlobalMaxPool1d",
    "DGCNNEncoder",
    "SubjectPointerDecoder",
    "ObjectPointerDecoder",
    "DGCNNForRelationExtraction",
    "MaskedRelationExtractionLoss",
    "ExponentialMovingAverage",
    "compute_spo_metrics",
    "Trainer",
    "RelationExtractor",
]

