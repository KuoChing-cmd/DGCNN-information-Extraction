"""Loss functions for DGCNN Relation Extraction."""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedRelationExtractionLoss(nn.Module):
    """Computes masked binary cross-entropy loss for subject and object pointer predictions."""

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        ps1: torch.Tensor,
        ps2: torch.Tensor,
        po1: torch.Tensor,
        po2: torch.Tensor,
        s1: torch.Tensor,
        s2: torch.Tensor,
        o1: torch.Tensor,
        o2: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Args:

        ps1: Predicted subject start (B, L, 1) ps2: Predicted subject end (B, L,
        1) po1: Predicted object start for all relations (B, L, num_classes) po2:
        Predicted object end for all relations (B, L, num_classes) s1: Ground
        truth subject start (B, L) or (B, L, 1) s2: Ground truth subject end (B,
        L) or (B, L, 1) o1: Ground truth object start (B, L, num_classes) o2:
        Ground truth object end (B, L, num_classes) mask: Sequence mask (B, L)

        Returns:
            total_loss: Scalar loss tensor.
            loss_dict: Dictionary with scalar values for logging.
        """
        if s1.dim() == 2:
            s1 = s1.unsqueeze(-1)
        if s2.dim() == 2:
            s2 = s2.unsqueeze(-1)

        mask_3d = mask.unsqueeze(-1).float()  # (B, L, 1)
        mask_sum = torch.clamp(mask_3d.sum(), min=1.0)

        # Subject losses (BCE clamped for numerical stability)
        ps1_clamped = torch.clamp(ps1, self.eps, 1.0 - self.eps)
        ps2_clamped = torch.clamp(ps2, self.eps, 1.0 - self.eps)

        loss_s1 = F.binary_cross_entropy(ps1_clamped, s1.float(), reduction="none")
        loss_s1 = (loss_s1 * mask_3d).sum() / mask_sum

        loss_s2 = F.binary_cross_entropy(ps2_clamped, s2.float(), reduction="none")
        loss_s2 = (loss_s2 * mask_3d).sum() / mask_sum

        # Object losses
        po1_clamped = torch.clamp(po1, self.eps, 1.0 - self.eps)
        po2_clamped = torch.clamp(po2, self.eps, 1.0 - self.eps)

        loss_o1 = F.binary_cross_entropy(po1_clamped, o1.float(), reduction="none")
        # Sum across relation classes, then average over valid sequence tokens
        loss_o1 = loss_o1.sum(dim=-1, keepdim=True)
        loss_o1 = (loss_o1 * mask_3d).sum() / mask_sum

        loss_o2 = F.binary_cross_entropy(po2_clamped, o2.float(), reduction="none")
        loss_o2 = loss_o2.sum(dim=-1, keepdim=True)
        loss_o2 = (loss_o2 * mask_3d).sum() / mask_sum

        total_loss = (loss_s1 + loss_s2) + (loss_o1 + loss_o2)

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_s1": loss_s1.item(),
            "loss_s2": loss_s2.item(),
            "loss_o1": loss_o1.item(),
            "loss_o2": loss_o2.item(),
        }

        return total_loss, loss_dict
