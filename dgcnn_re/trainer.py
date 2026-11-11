"""Training pipeline with device auto-detection, EMA, and validation evaluation."""

import logging
import os
import time
from typing import Optional, Dict, Any, List, Set, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataclasses import asdict
from dgcnn_re.config import Config, ModelConfig
from dgcnn_re.dataset import RelationExtractionDataset, Vocab, Schema, collate_fn
from dgcnn_re.loss import MaskedRelationExtractionLoss
from dgcnn_re.metrics import ExponentialMovingAverage, compute_spo_metrics
from dgcnn_re.model import DGCNNForRelationExtraction
from dgcnn_re.predictor import RelationExtractor

logger = logging.getLogger("dgcnn_re.trainer")


def get_device(requested: str = "auto") -> torch.device:
    """Selects best available compute device."""
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    """Full-featured trainer for DGCNN Relation Extraction models."""

    def __init__(
        self,
        config: Config,
        model: DGCNNForRelationExtraction,
        vocab: Vocab,
        schema: Schema,
        train_dataset: RelationExtractionDataset,
        val_dataset: Optional[RelationExtractionDataset] = None,
    ):
        self.config = config
        self.model = model
        self.vocab = vocab
        self.schema = schema
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.device = get_device(config.training.device)
        self.model.to(self.device)
        logger.info(f"Initialized model on compute device: {self.device}")

        # Training components
        self.criterion = MaskedRelationExtractionLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
        self.ema = (
            ExponentialMovingAverage(self.model, decay=config.training.ema_decay)
            if config.training.ema_decay > 0
            else None
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        self.best_f1 = 0.0
        self.patience_counter = 0
        os.makedirs(config.training.save_dir, exist_ok=True)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        s1_loss_sum = 0.0
        s2_loss_sum = 0.0
        o1_loss_sum = 0.0
        o2_loss_sum = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.training.epochs}",
        )
        for step, batch in enumerate(pbar):
            char_ids = batch["char_ids"].to(self.device)
            mask = batch["mask"].to(self.device)
            s1 = batch["s1"].to(self.device)
            s2 = batch["s2"].to(self.device)
            k1 = batch["k1"].to(self.device)
            k2 = batch["k2"].to(self.device)
            o1 = batch["o1"].to(self.device)
            o2 = batch["o2"].to(self.device)

            self.optimizer.zero_grad()
            out = self.model(char_ids=char_ids, k1=k1, k2=k2, mask=mask)

            loss, loss_dict = self.criterion(
                ps1=out["ps1"],
                ps2=out["ps2"],
                po1=out["po1"],
                po2=out["po2"],
                s1=s1,
                s2=s2,
                o1=o1,
                o2=o2,
                mask=mask,
            )

            loss.backward()
            if self.config.training.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.grad_clip
                )

            self.optimizer.step()
            if self.ema:
                self.ema.update()

            total_loss += loss_dict["loss_total"]
            s1_loss_sum += loss_dict["loss_s1"]
            s2_loss_sum += loss_dict["loss_s2"]
            o1_loss_sum += loss_dict["loss_o1"]
            o2_loss_sum += loss_dict["loss_o2"]

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "s_loss": f"{(loss_dict['loss_s1'] + loss_dict['loss_s2']):.3f}",
                    "o_loss": f"{(loss_dict['loss_o1'] + loss_dict['loss_o2']):.3f}",
                }
            )

        avg_loss = total_loss / max(num_batches, 1)
        return {
            "avg_loss": avg_loss,
            "avg_s1_loss": s1_loss_sum / max(num_batches, 1),
            "avg_s2_loss": s2_loss_sum / max(num_batches, 1),
            "avg_o1_loss": o1_loss_sum / max(num_batches, 1),
            "avg_o2_loss": o2_loss_sum / max(num_batches, 1),
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if not self.val_dataset:
            return {}

        if self.ema:
            self.ema.apply_shadow()

        extractor = RelationExtractor(
            model=self.model,
            vocab=self.vocab,
            schema=self.schema,
            device=self.device,
            sub_start_thresh=self.config.data.subject_threshold,
            obj_start_thresh=self.config.data.object_threshold,
            max_len=self.config.model.max_len,
        )

        preds: List[Set[Tuple[str, str, str]]] = []
        golds: List[Set[Tuple[str, str, str]]] = []

        for item in self.val_dataset:
            text = item["text"]
            gold_spos = {
                (s, p, o)
                for s, p, o in [
                    (sp[0], sp[1], sp[2])
                    if isinstance(sp, (list, tuple))
                    else (sp["subject"], sp["predicate"], sp["object"])
                    for sp in item.get("spo_list", [])
                ]
            }
            pred_spos = set(extractor.extract_triplets(text))
            preds.append(pred_spos)
            golds.append(gold_spos)

        metrics = compute_spo_metrics(preds, golds)

        if self.ema:
            self.ema.restore()

        return metrics

    def save_checkpoint(self, path: str, is_best: bool = False) -> None:
        model_state = (
            self.ema.shadow
            if (self.ema and is_best)
            else self.model.state_dict()
        )
        # Convert state dict tensors to CPU for safe and portable serialization
        cpu_state_dict = {k: v.cpu() for k, v in model_state.items()}
        checkpoint = {
            "model_state_dict": cpu_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config.model),
            "best_f1": self.best_f1,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def fit(self) -> Dict[str, Any]:
        """Runs the complete training and validation cycle."""
        logger.info("Starting training loop...")
        history = []

        for epoch in range(self.config.training.epochs):
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()
            elapsed = time.time() - t0

            f1 = val_metrics.get("f1", 0.0)
            p = val_metrics.get("precision", 0.0)
            r = val_metrics.get("recall", 0.0)

            logger.info(
                f"[Epoch {epoch + 1:02d}/{self.config.training.epochs:02d} ({elapsed:.1f}s)] "
                f"Train Loss: {train_metrics['avg_loss']:.4f} | "
                f"Val P: {p:.4f} R: {r:.4f} F1: {f1:.4f} (Best: {self.best_f1:.4f})"
            )

            # Checkpoint management
            latest_path = os.path.join(
                self.config.training.save_dir, "latest_model.pt"
            )
            self.save_checkpoint(latest_path, is_best=False)

            if f1 > self.best_f1:
                self.best_f1 = f1
                self.patience_counter = 0
                best_path = os.path.join(
                    self.config.training.save_dir, "best_model.pt"
                )
                self.save_checkpoint(best_path, is_best=True)
                logger.info(f"🏆 New best F1 score achieved: {self.best_f1:.4f}!")
            else:
                self.patience_counter += 1
                if (
                    self.patience_counter >= self.config.training.patience
                    and epoch > 5
                ):
                    logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs."
                    )
                    break

            history.append(
                {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
            )

        return {"best_f1": self.best_f1, "history": history}
