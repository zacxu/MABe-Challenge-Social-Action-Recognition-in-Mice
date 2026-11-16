"""Configurable multi-scale TCN/CNN-BiGRU model for Phase 9."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import cast

import torch
from torch import Tensor, nn

from mabe.phase9_config import TemporalArchitectureConfig


class MultiScaleStem(nn.Module):
    def __init__(self, input_features: int, hidden: int, kernels: tuple[int, ...]) -> None:
        super().__init__()
        branch_channels = max(1, hidden // len(kernels))
        self.branches = nn.ModuleList(
            nn.Conv1d(
                input_features,
                branch_channels,
                kernel_size=kernel,
                padding=kernel // 2,
            )
            for kernel in kernels
        )
        self.project = nn.Conv1d(branch_channels * len(kernels), hidden, kernel_size=1)
        self.normalization = nn.GroupNorm(1, hidden)

    def forward(self, value: Tensor) -> Tensor:
        branches = torch.cat([branch(value) for branch in self.branches], dim=1)
        return torch.nn.functional.gelu(self.normalization(self.project(branches)))


class ResidualTCNBlock(nn.Module):
    def __init__(
        self, hidden: int, *, kernel_size: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hidden,
        )
        self.pointwise = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.normalization = nn.GroupNorm(1, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: Tensor) -> Tensor:
        residual = value
        value = self.depthwise(value)
        value = self.pointwise(value)
        value = torch.nn.functional.gelu(self.normalization(value))
        output: Tensor = residual + self.dropout(value)
        return output


class ConvAttentionBlock(nn.Module):
    """Small pre-normalized attention/FFN block after convolutional encoding."""

    def __init__(
        self, hidden: int, *, heads: int, ff_multiplier: int, dropout: float
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.feedforward_norm = nn.LayerNorm(hidden)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden, hidden * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * ff_multiplier, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, value: Tensor) -> Tensor:
        normalized = self.attention_norm(value)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        value = value + attended
        return cast(Tensor, value + self.feedforward(self.feedforward_norm(value)))


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        input_features: int,
        lab_count: int,
        metadata_features: int,
        config: TemporalArchitectureConfig,
    ) -> None:
        super().__init__()
        hidden = config.hidden_channels
        self.stem = MultiScaleStem(input_features, hidden, config.stem_kernel_sizes)
        self.lab_embedding = nn.Embedding(lab_count, config.lab_embedding_dim)
        self.condition = nn.Sequential(
            nn.Linear(config.lab_embedding_dim + metadata_features, config.metadata_hidden_dim),
            nn.GELU(),
            nn.Linear(config.metadata_hidden_dim, hidden),
        )
        self.blocks = nn.ModuleList(
            ResidualTCNBlock(
                hidden,
                kernel_size=config.tcn_kernel_size,
                dilation=dilation,
                dropout=config.dropout,
            )
            for dilation in config.tcn_dilations
        )
        self.attention_blocks = nn.ModuleList()
        if config.family == "conv_attention":
            self.attention_blocks = nn.ModuleList(
                ConvAttentionBlock(
                    hidden,
                    heads=config.attention_heads,
                    ff_multiplier=config.attention_ff_multiplier,
                    dropout=config.dropout,
                )
                for _ in range(config.attention_layers)
            )
        self.adapter_embedding: nn.Embedding | None = None
        self.adapter_project: nn.Linear | None = None
        if config.use_lab_adapter:
            self.adapter_embedding = nn.Embedding(lab_count, config.lab_adapter_dim)
            self.adapter_project = nn.Linear(config.lab_adapter_dim, hidden, bias=False)
        self.gru: nn.GRU | None = None
        self.gru_project: nn.Linear | None = None
        if config.use_bigru:
            self.gru = nn.GRU(
                hidden,
                config.gru_hidden_size,
                batch_first=True,
                bidirectional=True,
            )
            self.gru_project = nn.Linear(config.gru_hidden_size * 2, hidden)

    def forward(self, features: Tensor, lab_index: Tensor, metadata: Tensor) -> Tensor:
        value = self.stem(features.transpose(1, 2))
        condition = self.condition(
            torch.cat([self.lab_embedding(lab_index), metadata], dim=1)
        )
        value = value + condition[:, :, None]
        if self.adapter_embedding is not None and self.adapter_project is not None:
            adapter = self.adapter_project(self.adapter_embedding(lab_index))
            value = value + adapter[:, :, None]
        for block in self.blocks:
            value = block(value)
        value = value.transpose(1, 2)
        if self.gru is not None and self.gru_project is not None:
            value, _ = self.gru(value)
            value = self.gru_project(value)
        for block in self.attention_blocks:
            value = block(value)
        return cast(Tensor, value)


class TemporalBehaviorModel(nn.Module):
    """Separate solo/pair encoders and output heads with shared configuration."""

    def __init__(
        self,
        *,
        solo_input_features: int,
        pair_input_features: int,
        solo_actions: int,
        pair_actions: int,
        lab_count: int,
        metadata_features: int,
        config: TemporalArchitectureConfig,
    ) -> None:
        super().__init__()
        self.solo_encoder = TemporalEncoder(
            solo_input_features, lab_count, metadata_features, config
        )
        self.pair_encoder = TemporalEncoder(
            pair_input_features, lab_count, metadata_features, config
        )
        self.solo_head = nn.Linear(config.hidden_channels, solo_actions)
        self.pair_head = nn.Linear(config.hidden_channels, pair_actions)

    def forward(
        self,
        route: str,
        features: Tensor,
        lab_index: Tensor,
        metadata: Tensor,
        action_validity: Tensor | None = None,
    ) -> Tensor:
        if route == "solo":
            encoded = self.solo_encoder(features, lab_index, metadata)
            logits = self.solo_head(encoded)
        elif route == "pair":
            encoded = self.pair_encoder(features, lab_index, metadata)
            logits = self.pair_head(encoded)
        else:
            raise ValueError(f"unknown temporal route: {route}")
        if action_validity is not None:
            logits = logits.masked_fill(~action_validity[:, None, :], -20.0)
        return cast(Tensor, logits)


def masked_bce_with_logits(
    logits: Tensor,
    targets: Tensor,
    loss_mask: Tensor,
    *,
    positive_weight: float,
    focal_gamma: float = 0.0,
    sample_weights: Tensor | None = None,
) -> Tensor:
    """Masked, sample-weighted BCE with optional focal modulation.

    The positive term alone receives ``positive_weight``. With p=sigmoid(logit)
    and a possibly fractional target y, the focal factor is
    (1 - [p*y + (1-p)*(1-y)])**focal_gamma. The effective supervision weight is
    loss_mask times the optional per-example laboratory weight. Reduction uses
    max(sum(effective_weights), 1), not the count of positives. This adapts the
    focal principle; positive_weight is not the original paper's alpha.
    """
    weights = torch.tensor(positive_weight, dtype=logits.dtype, device=logits.device)
    values = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=weights
    )
    if focal_gamma > 0:
        probability = torch.sigmoid(logits)
        probability_true = probability * targets + (1 - probability) * (1 - targets)
        values = values * (1 - probability_true).pow(focal_gamma)
    weighted_mask = loss_mask
    if sample_weights is not None:
        if sample_weights.ndim != 1 or sample_weights.shape[0] != logits.shape[0]:
            raise ValueError("sample weights must have shape [batch]")
        weighted_mask = weighted_mask * sample_weights[:, None, None]
    denominator = weighted_mask.sum().clamp_min(1.0)
    loss: Tensor = (values * weighted_mask).sum() / denominator
    return loss


def masked_macro_soft_f1_loss(
    logits: Tensor,
    targets: Tensor,
    loss_mask: Tensor,
    *,
    sample_weights: Tensor | None = None,
    epsilon: float = 1e-6,
) -> Tensor:
    """One minus the mean differentiable F1 of batch-supervised actions.

    Weighted soft TP, FP, and FN are sums of w*p*y, w*p*(1-y), and w*(1-p)*y
    over batch and time, where w=loss_mask*sample_weight. Positive BCE weighting
    is deliberately absent here. Epsilon is added to both ratio terms; actions
    with no weighted supervision are excluded, including from the mean's
    denominator. An action with valid negative-only targets is still active.
    Empty supervision returns a differentiable zero. This is a minibatch
    training surrogate, not the decoded interval-based competition scorer.
    """
    weighted_mask = loss_mask
    if sample_weights is not None:
        if sample_weights.ndim != 1 or sample_weights.shape[0] != logits.shape[0]:
            raise ValueError("sample weights must have shape [batch]")
        weighted_mask = weighted_mask * sample_weights[:, None, None]
    probability = torch.sigmoid(logits)
    true_positive = (probability * targets * weighted_mask).sum(dim=(0, 1))
    false_positive = (probability * (1 - targets) * weighted_mask).sum(dim=(0, 1))
    false_negative = ((1 - probability) * targets * weighted_mask).sum(dim=(0, 1))
    soft_f1 = (2 * true_positive + epsilon) / (
        2 * true_positive + false_positive + false_negative + epsilon
    )
    active_actions = weighted_mask.sum(dim=(0, 1)) > 0
    if not bool(active_actions.any()):
        return logits.sum() * 0
    loss: Tensor = 1 - soft_f1[active_actions].mean()
    return loss


def combined_temporal_loss(
    logits: Tensor,
    targets: Tensor,
    loss_mask: Tensor,
    *,
    positive_weight: float,
    soft_f1_weight: float,
    focal_gamma: float = 0.0,
    sample_weights: Tensor | None = None,
) -> Tensor:
    """Project-specific additive focal-BCE plus macro-soft-F1 objective.

    Return focal_BCE + soft_f1_weight*(1 - mean_active_action_soft_F1).
    The coefficient is configured externally, not derived from the F1 identity,
    and does not describe a convex or percentage mixture. Setting it to zero
    disables the auxiliary term without changing focal-BCE behavior. See
    report/temporal_loss_explained.md for derivation, sources, and a worked case.
    """
    bce = masked_bce_with_logits(
        logits,
        targets,
        loss_mask,
        positive_weight=positive_weight,
        focal_gamma=focal_gamma,
        sample_weights=sample_weights,
    )
    if soft_f1_weight == 0:
        return bce
    soft_f1 = masked_macro_soft_f1_loss(
        logits, targets, loss_mask, sample_weights=sample_weights
    )
    return bce + soft_f1_weight * soft_f1


class ExponentialMovingAverage:
    def __init__(self, model: TemporalBehaviorModel, decay: float) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: TemporalBehaviorModel) -> None:
        for shadow, current in zip(
            self.shadow.parameters(), model.parameters(), strict=True
        ):
            shadow.mul_(self.decay).add_(current, alpha=1 - self.decay)
        for shadow_buffer, current_buffer in zip(
            self.shadow.buffers(), model.buffers(), strict=True
        ):
            shadow_buffer.copy_(current_buffer)


def trainable_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def overlap_weights(length: int, edge: int, *, left_boundary: bool, right_boundary: bool) -> Tensor:
    """Edge-downweight overlapping windows while retaining video boundaries."""
    weights = torch.ones(length, dtype=torch.float32)
    if edge <= 0:
        return weights
    ramp = torch.linspace(1 / (edge + 1), edge / (edge + 1), edge)
    if not left_boundary:
        weights[:edge] = ramp
    if not right_boundary:
        weights[-edge:] = torch.flip(ramp, dims=(0,))
    return weights
