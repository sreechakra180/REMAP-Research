"""Gradient history encoding for Meta-Plasticity (F1).

Encodes the gradient history tensor h_t = [g_{t-1}; ...; g_{t-K}] into a
fixed-dimensional representation using a coordinate-wise LSTM, following the
*Learning to learn by gradient descent by gradient descent* (L2L) paradigm.

Also provides gradient pre-processing utilities (log-scale, clipping,
normalization) that stabilise meta-learning optimisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Gradient Preprocessor (functional, no learnable params)
# ---------------------------------------------------------------------------


@dataclass
class GradientPreprocessor:
    """Log-scale gradient preprocessing following Andrychowicz et al. (2016).

    Transforms raw gradients into a two-dimensional feature per coordinate:

        x = [sign(g) · log(|g| + 1),  clip(g, -threshold, threshold) / threshold]

    This representation is more amenable to recurrent processing than raw
    gradient magnitudes, which can span many orders of magnitude.

    Attributes:
        clip_threshold: Absolute value beyond which gradients are clipped in the
            second feature channel.  Defaults to 10.0.
        eps: Small constant added inside log to avoid ``log(0)``.
        normalize: If ``True``, z-score normalize the output across the
            coordinate dimension.
    """

    clip_threshold: float = 10.0
    eps: float = 1e-8
    normalize: bool = True
    _running_mean: Optional[Tensor] = field(default=None, init=False, repr=False)
    _running_var: Optional[Tensor] = field(default=None, init=False, repr=False)

    def __call__(self, gradient: Tensor) -> Tensor:
        """Preprocess a single gradient vector.

        Args:
            gradient: Raw gradient tensor of arbitrary shape.

        Returns:
            Preprocessed tensor with an extra trailing dimension of size 2
            appended: ``(*gradient.shape, 2)``.
        """
        return self.preprocess(gradient)

    def preprocess(self, gradient: Tensor) -> Tensor:
        """Apply log-scale + clipped representation.

        Args:
            gradient: Raw gradient tensor of shape ``(*)``.

        Returns:
            Tensor of shape ``(*, 2)`` with log-magnitude and clipped channels.
        """
        # Channel 1: sign(g) · log(|g| + 1)
        log_feature = torch.sign(gradient) * torch.log(
            torch.abs(gradient) + self.eps
        )

        # Channel 2: clipped / threshold  (bounded in [-1, 1])
        clipped_feature = torch.clamp(
            gradient, -self.clip_threshold, self.clip_threshold
        ) / self.clip_threshold

        # Stack along a new last dimension → (*, 2)
        features = torch.stack([log_feature, clipped_feature], dim=-1)

        if self.normalize:
            features = self._z_normalize(features)

        return features

    # -- internals -----------------------------------------------------------

    def _z_normalize(self, x: Tensor) -> Tensor:
        """Zero-mean, unit-variance normalization across all but the last dim."""
        # Flatten everything except the feature dim for statistics.
        flat = x.reshape(-1, x.shape[-1])
        mean = flat.mean(dim=0)
        std = flat.std(dim=0).clamp(min=self.eps)
        return (x - mean) / std


# ---------------------------------------------------------------------------
# Coordinate-wise Gradient Encoder (LSTM)
# ---------------------------------------------------------------------------


class GradientEncoder(nn.Module):
    r"""Coordinate-wise LSTM encoder for gradient history.

    For each parameter coordinate, an LSTM processes the sequence of
    pre-processed gradient observations ``h_t = [g_{t-K}; ...; g_{t-1}]``
    and produces a fixed-size hidden representation.

    This is the *per-coordinate* design from the L2L paper: all coordinates
    share the same LSTM weights, enabling generalisation across parameter
    dimensions of different sizes.

    The input at each time-step is the 2-D preprocessed gradient feature
    (log-magnitude + clipped value).

    Args:
        input_dim: Number of features per time-step per coordinate (typically
            2 from :class:`GradientPreprocessor`).
        hidden_dim: LSTM hidden-state size per coordinate.
        n_layers: Number of stacked LSTM layers.

    Shape:
        - Input ``gradient_sequence``: ``(batch, seq_len, n_coords, input_dim)``
        - Output: ``(batch, n_coords, hidden_dim)``
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.0 if n_layers == 1 else 0.1,
        )

        # Layer-norm on the final hidden state for training stability.
        self.output_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    # -- weight init ---------------------------------------------------------

    def _init_weights(self) -> None:
        """Orthogonal init for recurrent weights, Xavier for input weights."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget-gate bias to 1 (standard trick).
                hidden = self.hidden_dim
                param.data[hidden : 2 * hidden].fill_(1.0)

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        gradient_sequence: Tensor,
        hidden: Optional[tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """Encode a gradient history sequence.

        Args:
            gradient_sequence: Preprocessed gradient history of shape
                ``(batch, seq_len, n_coords, input_dim)``.
            hidden: Optional initial LSTM state ``(h_0, c_0)`` each of shape
                ``(n_layers, batch * n_coords, hidden_dim)``.

        Returns:
            Encoded history of shape ``(batch, n_coords, hidden_dim)``.
        """
        B, T, N, D = gradient_sequence.shape

        # Merge batch and coordinate dims so the LSTM processes all
        # coordinates in parallel with shared weights.
        # (B, T, N, D) → (B*N, T, D)
        x = gradient_sequence.permute(0, 2, 1, 3).reshape(B * N, T, D)

        # Run the LSTM over the time axis.
        lstm_out, _ = self.lstm(x, hidden)  # (B*N, T, hidden_dim)

        # Take the last time-step's output as the history encoding.
        last_hidden = lstm_out[:, -1, :]  # (B*N, hidden_dim)

        # Un-merge → (B, N, hidden_dim)
        encoded = last_hidden.view(B, N, self.hidden_dim)

        return self.output_norm(encoded)

    def get_initial_state(
        self, batch_size: int, n_coords: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Create zero-initialised LSTM hidden state.

        Args:
            batch_size: Batch size ``B``.
            n_coords: Number of parameter coordinates ``N``.
            device: Target device.

        Returns:
            Tuple ``(h_0, c_0)`` each of shape
            ``(n_layers, B * N, hidden_dim)``.
        """
        total = batch_size * n_coords
        h_0 = torch.zeros(
            self.n_layers, total, self.hidden_dim, device=device
        )
        c_0 = torch.zeros(
            self.n_layers, total, self.hidden_dim, device=device
        )
        return h_0, c_0
