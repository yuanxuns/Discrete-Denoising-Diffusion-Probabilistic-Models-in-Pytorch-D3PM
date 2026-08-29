"""Compact conditional DiT denoiser for discrete diffusion experiments."""

import math

import torch
from torch import nn


class DiTBlock(nn.Module):
    """Transformer block modulated by a per-example time/class conditioning vector."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size))

    def forward(self, x, cond):
        """Apply the block.

        Args:
            x: Token tensor of shape ``(B, N, D)``.
            cond: Conditioning tensor of shape ``(B, D)``.

        Returns:
            Tensor of shape ``(B, N, D)``.
        """
        shift1, scale1, shift2, scale2 = self.modulation(cond).chunk(4, dim=-1)
        h = self.norm1(x) * (1 + scale1[:, None]) + shift1[:, None]
        x = x + self.attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x) * (1 + scale2[:, None]) + shift2[:, None]
        return x + self.mlp(h)


class DiT(nn.Module):
    """A small class-conditional DiT that predicts categorical clean-data logits.

    The model is intentionally shape-agnostic: images use ``input_shape=(H, W)``;
    a two-dimensional point can use ``input_shape=(2,)``.  Each scalar discrete
    state becomes one token, so its output has one categorical logit vector per
    input state.

    Args:
        input_shape: Non-batch data dimensions, e.g. ``(28, 28)`` or ``(2,)``.
        num_classes: Number of D3PM states ``K``.
        num_timesteps: Number of diffusion timesteps ``T``.
        hidden_size: Transformer token width ``D``.
        depth: Number of DiT blocks.
        num_heads: Number of attention heads; must divide ``hidden_size``.
        condition_classes: Number of labels. ``None`` makes the model unconditional.
        class_dropout_prob: Probability of replacing a label with the learned null
            label during training, enabling classifier-free guidance.

    Input:
        ``x`` has shape ``(B, *input_shape)`` with integer values in ``[0, K-1]``;
        ``t`` has shape ``(B,)``; optional ``y`` has shape ``(B,)``.

    Output:
        Tensor of shape ``(B, *input_shape, K)`` containing clean-state logits.
    """

    def __init__(
        self, input_shape, num_classes, num_timesteps, hidden_size=192,
        depth=6, num_heads=6, condition_classes=None, class_dropout_prob=0.1,
    ):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.input_shape = tuple(input_shape)
        self.num_tokens = math.prod(self.input_shape)
        self.num_classes = num_classes
        self.condition_classes = condition_classes
        self.class_dropout_prob = class_dropout_prob
        self.token_embed = nn.Embedding(num_classes, hidden_size)
        self.position_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_size))
        self.time_embed = nn.Embedding(num_timesteps, hidden_size)
        self.class_embed = (
            nn.Embedding(condition_classes + 1, hidden_size)
            if condition_classes is not None else None
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)
        nn.init.normal_(self.position_embed, std=0.02)

    def forward(self, x, t, y=None):
        """Predict clean categorical logits.

        Args:
            x: Integer noisy-state tensor of shape ``(B, *input_shape)``.
            t: Integer timestep tensor of shape ``(B,)`` in ``[0, T-1]``.
            y: Optional class-label tensor of shape ``(B,)``.

        Returns:
            Tensor of shape ``(B, *input_shape, K)``.
        """
        if tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(f"Expected x.shape[1:]={self.input_shape}, got {tuple(x.shape[1:])}.")
        if t.shape != (x.shape[0],):
            raise ValueError("t must have shape (batch_size,).")
        tokens = self.token_embed(x.long().reshape(x.shape[0], self.num_tokens))
        cond = self.time_embed(t.long())
        if self.class_embed is not None:
            if y is None:
                y = torch.full_like(t, self.condition_classes)
            if y.shape != t.shape:
                raise ValueError("y must have shape (batch_size,).")
            y = y.long()
            if self.training and self.class_dropout_prob:
                dropped = torch.rand_like(y, dtype=torch.float) < self.class_dropout_prob
                y = torch.where(dropped, torch.full_like(y, self.condition_classes), y)
            cond = cond + self.class_embed(y)
        h = tokens + self.position_embed
        for block in self.blocks:
            h = block(h, cond)
        logits = self.head(self.norm(h))
        return logits.reshape(x.shape[0], *self.input_shape, self.num_classes)
