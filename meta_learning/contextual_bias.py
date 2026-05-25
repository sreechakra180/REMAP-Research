"""Context-dependent bias for the Meta-Plasticity update rule.

Implements the bias term from Eq. 3:

    b_ϕ(h, E)

which is a learnable function of the encoded gradient history ``h`` and the
episodic context ``E``.  The bias allows the meta-learner to shift the
parameter update independently of the current task gradient, injecting prior
knowledge about promising search directions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class ContextualBias(nn.Module):
    r"""MLP that produces a context-dependent bias vector.

    Maps the concatenated gradient history encoding ``h`` and episodic
    memory context ``E`` through a small feed-forward network to produce a
    bias vector of the same dimension as the parameter being updated:

        b_ϕ(h, E) = MLP([h; E])   ∈ ℝ^{output_dim}

    The network is initialised so that the initial bias is near zero, leaving
    the meta-update dominated by the preconditioning term ``-M · g`` at the
    start of training.

    Args:
        history_dim: Dimension of the gradient history encoding ``h``.
        context_dim: Dimension of the episodic context ``E``.
        output_dim: Dimension of the produced bias (= parameter dimension).
        hidden_dim: Width of hidden layers.  Defaults to
            ``max(128, output_dim // 4)``.
        n_layers: Number of hidden layers (minimum 1).

    Shape:
        - h: ``(batch, history_dim)``
        - E: ``(batch, context_dim)``
        - Output: ``(batch, output_dim)``
    """

    def __init__(
        self,
        history_dim: int,
        context_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.history_dim = history_dim
        self.context_dim = context_dim
        self.output_dim = output_dim

        in_dim = history_dim + context_dim
        hid = hidden_dim or max(128, output_dim // 4)

        layers: list[nn.Module] = []

        # First hidden layer.
        layers.append(nn.Linear(in_dim, hid))
        layers.append(nn.LayerNorm(hid))
        layers.append(nn.SiLU())

        # Additional hidden layers.
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid, hid))
            layers.append(nn.LayerNorm(hid))
            layers.append(nn.SiLU())

        # Output projection.
        layers.append(nn.Linear(hid, output_dim))

        self.net = nn.Sequential(*layers)

        self._init_weights()

    # -- init ----------------------------------------------------------------

    def _init_weights(self) -> None:
        """Near-zero output at initialisation."""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Shrink the last linear layer so initial bias ≈ 0.
        last_linear = self.net[-1]
        assert isinstance(last_linear, nn.Linear)
        last_linear.weight.data.mul_(0.01)
        if last_linear.bias is not None:
            last_linear.bias.data.zero_()

    # -- forward -------------------------------------------------------------

    def forward(self, h: Tensor, E: Tensor) -> Tensor:
        """Compute the contextual bias vector.

        Args:
            h: Encoded gradient history, ``(batch, history_dim)``.
            E: Episodic context vector, ``(batch, context_dim)``.

        Returns:
            Bias vector ``b_ϕ(h, E)`` of shape ``(batch, output_dim)``.
        """
        ctx = torch.cat([h, E], dim=-1)  # (B, history_dim + context_dim)
        return self.net(ctx)
