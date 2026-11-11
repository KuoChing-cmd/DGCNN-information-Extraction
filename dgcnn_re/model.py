"""DGCNN-based joint Entity and Relation Extraction Model."""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgcnn_re.config import ModelConfig
from dgcnn_re.layers import (
    PositionalEncoding,
    DilatedGatedConv1d,
    MultiHeadSelfAttention,
    MaskedGlobalMaxPool1d,
)


class DGCNNEncoder(nn.Module):
    """Encodes character sequences using Character Embedding, Positional Encoding,

    and a stack of Dilated Gated Convolutions (DGCNN) with residual connections.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.char_embedding = nn.Embedding(
            config.vocab_size, config.char_dim, padding_idx=0
        )
        self.word_proj = (
            nn.Linear(config.word_dim, config.char_dim, bias=False)
            if config.word_dim > 0
            else None
        )
        self.pos_encoding = PositionalEncoding(
            d_model=config.char_dim, max_len=config.max_len, learnable=True
        )
        self.dropout = nn.Dropout(config.dropout)

        # DGCNN Conv Stack
        self.conv_layers = nn.ModuleList(
            [
                DilatedGatedConv1d(
                    channels=config.hidden_dim,
                    kernel_size=config.kernel_size,
                    dilation=rate,
                    dropout=config.conv_dropout,
                )
                for rate in config.dilation_rates
            ]
        )

    def forward(
        self,
        char_ids: torch.Tensor,
        word_embeds: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:

        char_ids: Tensor of shape (B, L) with token/character indices.
        word_embeds: Optional Tensor of shape (B, L, word_dim). mask: Optional
        Tensor of shape (B, L) with 1 for valid tokens, 0 for padding.

        Returns:
            t: Tensor of shape (B, L, hidden_dim) containing contextual representations.
            mask: Tensor of shape (B, L) boolean/float mask.
        """
        if mask is None:
            mask = (char_ids > 0).float()
        elif mask.dtype == torch.bool:
            mask = mask.float()

        batch_size, seq_len = char_ids.shape
        device = char_ids.device

        # Character embedding
        t = self.char_embedding(char_ids)

        # Optional word representation projection
        if self.word_proj is not None and word_embeds is not None:
            w = self.word_proj(word_embeds)
            t = t + w

        # Positional encoding
        positions = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )
        pv = self.pos_encoding(positions)
        t = t + pv

        # Dropout and initial masking
        t = self.dropout(t)
        t = t * mask.unsqueeze(-1)

        # Pass through DGCNN stack
        for conv in self.conv_layers:
            t = conv(t, mask=mask)

        return t, mask


class SubjectPointerDecoder(nn.Module):
    """Decodes subject start and end pointer probabilities using prior networks and self-attention."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_dim

        # Prior dense networks
        self.pn1_net = nn.Sequential(
            nn.Linear(dim, config.char_dim), nn.ReLU(), nn.Linear(config.char_dim, 1)
        )
        self.pn2_net = nn.Sequential(
            nn.Linear(dim, config.char_dim), nn.ReLU(), nn.Linear(config.char_dim, 1)
        )

        # Multi-Head Attention
        self.attn = MultiHeadSelfAttention(
            d_model=dim,
            num_heads=config.num_heads,
            size_per_head=config.size_per_head,
            dropout=config.dropout,
        )
        attn_out_dim = config.num_heads * config.size_per_head

        # Conv1D feature projection
        self.conv = nn.Conv1d(
            in_channels=dim + attn_out_dim,
            out_channels=config.char_dim,
            kernel_size=config.kernel_size,
            padding=(config.kernel_size - 1) // 2,
        )

        # Final pointer prediction heads
        self.ps1_head = nn.Linear(config.char_dim, 1)
        self.ps2_head = nn.Linear(config.char_dim, 1)

    def forward(
        self, t: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:

        t: Encoded representations of shape (B, L, D). mask: Tensor of shape
        (B, L).

        Returns:
            ps1: Subject start probabilities (B, L, 1).
            ps2: Subject end probabilities (B, L, 1).
            pn1: Prior start logits/probs (B, L, 1).
            pn2: Prior end logits/probs (B, L, 1).
        """
        # Prior probabilities
        pn1 = torch.sigmoid(self.pn1_net(t))
        pn2 = torch.sigmoid(self.pn2_net(t))

        # Context Attention
        attn_out = self.attn(q=t, key_mask=mask, query_mask=mask)
        h = torch.cat([t, attn_out], dim=-1)

        # Conv1D over sequence: (B, L, C) -> (B, C, L) -> Conv1d -> (B, L, C)
        h = h.transpose(1, 2)
        h = F.relu(self.conv(h))
        h = h.transpose(1, 2)

        # Subject head and tail pointers gated by prior
        ps1 = torch.sigmoid(self.ps1_head(h)) * pn1
        ps2 = torch.sigmoid(self.ps2_head(h)) * pn2

        return ps1, ps2, pn1, pn2


class ObjectPointerDecoder(nn.Module):
    """Decodes object start and end pointers for all candidate relations conditioned on a given subject."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        self.num_inter_points = config.num_inter_points

        # Predicate classification prior
        self.pool = MaskedGlobalMaxPool1d()
        self.pc_net = nn.Sequential(
            nn.Linear(dim, config.char_dim),
            nn.ReLU(),
            nn.Linear(config.char_dim, config.num_classes),
        )

        # Subject interpolation representation via BiGRU
        # Input: 6 points of dimension `dim`
        self.subject_gru = nn.GRU(
            input_size=dim,
            hidden_size=dim,
            batch_first=True,
            bidirectional=True,
        )
        self.pos_encoding = PositionalEncoding(
            d_model=config.char_dim, max_len=config.max_len, learnable=True
        )

        # Context attention
        self.attn = MultiHeadSelfAttention(
            d_model=dim,
            num_heads=config.num_heads,
            size_per_head=config.size_per_head,
            dropout=config.dropout,
        )
        attn_out_dim = config.num_heads * config.size_per_head

        # Object convolution layer
        # Input channels: text_rep (dim) + attn (attn_out_dim) + subject_cond (2 * dim)
        in_channels = dim + attn_out_dim + (2 * dim)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=config.char_dim,
            kernel_size=config.kernel_size,
            padding=(config.kernel_size - 1) // 2,
        )

        # Prediction heads
        self.po_head = nn.Linear(config.char_dim, 1)
        self.po1_head = nn.Linear(config.char_dim, config.num_classes)
        self.po2_head = nn.Linear(config.char_dim, config.num_classes)

    def _get_subject_features(
        self,
        t: torch.Tensor,
        k1: torch.Tensor,
        k2: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolates n=6 points between subject start k1 and end k2, extracts hidden

        states, passes through BiGRU, and combines with relative position
        embeddings.
        """
        batch_size, seq_len, dim = t.shape
        device = t.device

        # Ensure k1, k2 are (B, 1) float
        k1_f = k1.view(batch_size, 1).float()
        k2_f = k2.view(batch_size, 1).float()

        # Generate interpolation coefficients alpha: 0 to 1 with n points
        alphas = torch.linspace(
            0.0, 1.0, steps=self.num_inter_points, device=device
        )  # (n,)

        # Interp indices: (B, n)
        # k1 * (1 - alpha) + k2 * alpha
        interp_idx = torch.round(
            k1_f * (1.0 - alphas) + k2_f * alphas
        ).long()  # (B, n)
        interp_idx = torch.clamp(interp_idx, 0, seq_len - 1)

        # Gather hidden states for interpolated points: (B, n, dim)
        batch_idx = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .expand(-1, self.num_inter_points)
        )
        k_inter = t[batch_idx, interp_idx]  # (B, n, dim)

        # BiGRU over the 6 points
        _, h_n = self.subject_gru(k_inter)
        # h_n shape: (num_directions * num_layers, B, hidden_size) = (2, B, dim)
        # Concat forward and backward final hidden states -> (B, 2 * dim)
        k_subj = torch.cat([h_n[0], h_n[1]], dim=-1)  # (B, 2 * dim)

        # Relative position embeddings
        positions = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )
        rel_pos1 = torch.abs(positions - k1.view(batch_size, 1).long())
        rel_pos2 = torch.abs(positions - k2.view(batch_size, 1).long())
        k1_v = self.pos_encoding(rel_pos1)  # (B, L, char_dim)
        k2_v = self.pos_encoding(rel_pos2)  # (B, L, char_dim)
        k_v = torch.cat([k1_v, k2_v], dim=-1)  # (B, L, 2 * char_dim)

        # Combine subject summary with relative position embeddings: (B, L, 2 * dim)
        k_total = k_subj.unsqueeze(1) + k_v
        return k_total

    def forward(
        self,
        t: torch.Tensor,
        k1: torch.Tensor,
        k2: torch.Tensor,
        mask: torch.Tensor,
        pn1: torch.Tensor,
        pn2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:

        t: Tensor (B, L, D) k1: Subject start index (B, 1) or (B,) k2: Subject
        end index (B, 1) or (B,) mask: Sequence mask (B, L) pn1: Prior start
        probabilities (B, L, 1) pn2: Prior end probabilities (B, L, 1)

        Returns:
            po1: Object start probabilities (B, L, num_classes)
            po2: Object end probabilities (B, L, num_classes)
            pc: Predicate prior probabilities (B, num_classes)
        """
        # Predicate prior from max-pooled text representation
        t_max = self.pool(t, mask)  # (B, D)
        pc = torch.sigmoid(self.pc_net(t_max))  # (B, num_classes)

        # Subject feature aggregation
        k_total = self._get_subject_features(t, k1, k2)  # (B, L, 2D)

        # Context Attention
        attn_out = self.attn(q=t, key_mask=mask, query_mask=mask)

        # Concatenate text, attention, and subject condition
        h = torch.cat([t, attn_out, k_total], dim=-1)

        # Conv1D over sequence
        h = h.transpose(1, 2)
        h = F.relu(self.conv(h))
        h = h.transpose(1, 2)

        # Object existence and relation pointer heads
        po = torch.sigmoid(self.po_head(h))  # (B, L, 1)
        po1 = torch.sigmoid(self.po1_head(h))  # (B, L, num_classes)
        po2 = torch.sigmoid(self.po2_head(h))  # (B, L, num_classes)

        # Hierarchical gating
        pc_expanded = pc.unsqueeze(1)  # (B, 1, num_classes)
        po1 = po * po1 * pc_expanded * pn1
        po2 = po * po2 * pc_expanded * pn2

        return po1, po2, pc


class DGCNNForRelationExtraction(nn.Module):
    """Complete DGCNN Relation Extraction model unifying encoder and hierarchical decoders."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = DGCNNEncoder(config)
        self.subject_decoder = SubjectPointerDecoder(config)
        self.object_decoder = ObjectPointerDecoder(config)

    def forward(
        self,
        char_ids: torch.Tensor,
        k1: Optional[torch.Tensor] = None,
        k2: Optional[torch.Tensor] = None,
        word_embeds: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training or inference.

        If k1 and k2 are provided, object pointers are computed conditioned on
        them.
        """
        t, mask = self.encoder(char_ids, word_embeds=word_embeds, mask=mask)
        ps1, ps2, pn1, pn2 = self.subject_decoder(t, mask)

        output: Dict[str, torch.Tensor] = {
            "ps1": ps1,
            "ps2": ps2,
            "pn1": pn1,
            "pn2": pn2,
            "mask": mask,
            "t": t,
        }

        if k1 is not None and k2 is not None:
            po1, po2, pc = self.object_decoder(t, k1, k2, mask, pn1, pn2)
            output["po1"] = po1
            output["po2"] = po2
            output["pc"] = pc

        return output
