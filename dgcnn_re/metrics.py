"""Evaluation metrics and Exponential Moving Average (EMA) utility."""

from typing import List, Set, Tuple, Dict, Any
import torch
import torch.nn as nn


class ExponentialMovingAverage:
    """Maintains Exponential Moving Average (EMA) of model parameters for improved generalization."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self.register()

    def register(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self) -> None:
        """Applies shadow weights to model, backing up current weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restores original model weights from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()


def compute_spo_metrics(
    preds_list: List[Set[Tuple[str, str, str]]],
    gold_list: List[Set[Tuple[str, str, str]]],
) -> Dict[str, float]:
    """Computes Precision, Recall, and F1 score for extracted (Subject, Predicate, Object) triplets."""
    num_correct = 0
    num_pred = 0
    num_gold = 0

    for pred, gold in zip(preds_list, gold_list):
        num_correct += len(pred & gold)
        num_pred += len(pred)
        num_gold += len(gold)

    precision = num_correct / max(num_pred, 1e-10)
    recall = num_correct / max(num_gold, 1e-10)
    f1 = (
        (2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": num_correct,
        "predicted": num_pred,
        "gold": num_gold,
    }
