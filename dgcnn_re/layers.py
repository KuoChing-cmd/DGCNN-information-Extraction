"""Neural network layers and modules for DGCNN-RE."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal or learnable positional encoding supporting absolute and relative positions."""

    def __init__(self, d_model: int, max_len: int = 512, learnable: bool = False):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.learnable = learnable

        if learnable:
            self.embedding = nn.Embedding(max_len, d_model)
            nn.init.zeros_(self.embedding.weight)
        else:
            self.embedding = None
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe)

    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        """Args:

        pos_ids: Tensor of shape (batch_size, seq_len) or (batch_size, seq_len,
        ...) containing integer positions.

        Returns:
            Tensor of shape (*pos_ids.shape, d_model).
        """
        # Clamp to max_len - 1 to prevent indexing overflow
        pos_ids = torch.clamp(pos_ids, 0, self.max_len - 1)
        if self.learnable:
            return self.embedding(pos_ids)
        return self.pe[pos_ids]


class DilatedGatedConv1d(nn.Module):
    """Dilated Gated Convolution with residual connection.

    Computes:
        H = Conv1D(x) -> [gate, feature]
        gate = sigmoid(dropout(gate))
        out = (gate * x + (1 - gate) * feature) * mask
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.dilation = dilation
        # padding = dilation ensures sequence length is preserved for kernel_size=3
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels * 2,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Args:

        x: Tensor of shape (batch_size, seq_len, channels) mask: Tensor of
        shape (batch_size, seq_len, 1) or (batch_size, seq_len)

        Returns:
            Tensor of shape (batch_size, seq_len, channels)
        """
        # (B, L, C) -> (B, C, L)
        h = x.transpose(1, 2)
        h = self.conv(h)
        # (B, 2C, L) -> (B, L, 2C)
        h = h.transpose(1, 2)

        gate, feature = torch.chunk(h, chunks=2, dim=-1)
        gate = self.dropout(gate)
        gate = torch.sigmoid(gate)

        out = gate * x + (1.0 - gate) * feature

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            out = out * mask.float()

        return out


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Attention supporting self-attention and cross-attention with masking."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        size_per_head: int = 16,
        dropout: float = 0.1,
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.size_per_head = size_per_head
        self.head_dim = size_per_head
        self.inner_dim = num_heads * size_per_head
        self.out_dim = out_dim or self.inner_dim

        self.q_proj = nn.Linear(d_model, self.inner_dim)
        self.k_proj = nn.Linear(d_model, self.inner_dim)
        self.v_proj = nn.Linear(d_model, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, self.out_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:

        q: (B, L_q, D) k: (B, L_k, D) - defaults to q if None v: (B, L_v, D) -
        defaults to k if None key_mask: (B, L_k) or (B, 1, 1, L_k) - 1 for
        valid, 0 for padding query_mask: (B, L_q) or (B, L_q, 1) - 1 for valid,
        0 for padding

        Returns:
            (B, L_q, out_dim)
        """
        if k is None:
            k = q
        if v is None:
            v = k

        batch_size, q_len, _ = q.shape
        k_len = k.shape[1]

        # Linear projections
        qw = self.q_proj(q).view(batch_size, q_len, self.num_heads, self.head_dim)
        kw = self.k_proj(k).view(batch_size, k_len, self.num_heads, self.head_dim)
        vw = self.v_proj(v).view(batch_size, k_len, self.num_heads, self.head_dim)

        # Transpose to (B, H, L, D)
        qw = qw.transpose(1, 2)
        kw = kw.transpose(1, 2)
        vw = vw.transpose(1, 2)

        # Scaled dot-product: (B, H, L_q, L_k)
        scores = torch.matmul(qw, kw.transpose(-1, -2)) * self.scale

        if key_mask is not None:
            if key_mask.dim() == 2:
                # (B, L_k) -> (B, 1, 1, L_k)
                attn_mask = key_mask.unsqueeze(1).unsqueeze(2)
            elif key_mask.dim() == 3:
                attn_mask = key_mask.unsqueeze(1)
            else:
                attn_mask = key_mask
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Context: (B, H, L_q, D)
        context = torch.matmul(attn_weights, vw)
        # Transpose to (B, L_q, H * D)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, q_len, self.inner_dim)
        )
        out = self.out_proj(context)

        if query_mask is not None:
            if query_mask.dim() == 2:
                query_mask = query_mask.unsqueeze(-1)
            out = out * query_mask.float()

        return out


class MaskedGlobalMaxPool1d(nn.Module):
    """Performs max pooling across sequence dimension while ignoring masked positions."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Args:

        x: Tensor of shape (B, L, D) mask: Tensor of shape (B, L, 1) or (B, L)

        Returns:
            Tensor of shape (B, D)
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        # Replace masked positions with large negative number
        masked_x = x - (1.0 - mask.float()) * 1e9
        return torch.max(masked_x, dim=1).values

