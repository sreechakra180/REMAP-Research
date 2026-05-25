"""Baseline optimisers conforming to the meta-learner interface.

These classes provide standard gradient-based optimisers (SGD, Adam) and
established meta-learning algorithms (MAML, L2L) wrapped in an interface
compatible with the Meta-Plasticity module, enabling direct drop-in
comparisons in experiments.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
from torch import Tensor


class BaselineOptimizer:
    """Base interface for baseline optimisers."""

    def step(
        self,
        params: Sequence[Tensor],
        gradients: Sequence[Tensor],
        **kwargs: Any,
    ) -> list[Tensor]:
        """Apply a single optimisation step.

        Args:
            params: Current network parameters.
            gradients: Corresponding parameter gradients.
            **kwargs: Additional contextual info (ignored by most baselines).

        Returns:
            List of updated parameter tensors.
        """
        raise NotImplementedError


class SGDBaseline(BaselineOptimizer):
    """Wraps standard Stochastic Gradient Descent as a meta-learner."""

    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(
        self,
        params: Sequence[Tensor],
        gradients: Sequence[Tensor],
        **kwargs: Any,
    ) -> list[Tensor]:
        """Compute SGD update: θ' = θ - α · ∇θ."""
        new_params = []
        for p, g in zip(params, gradients):
            new_params.append(p - self.lr * g)
        return new_params


class AdamBaseline(BaselineOptimizer):
    """Wraps Adam optimiser as a meta-learner."""

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        self.m: list[Tensor] | None = None
        self.v: list[Tensor] | None = None
        self.t: int = 0

    def reset_state(self) -> None:
        """Clear momentum buffers."""
        self.m = None
        self.v = None
        self.t = 0

    def step(
        self,
        params: Sequence[Tensor],
        gradients: Sequence[Tensor],
        **kwargs: Any,
    ) -> list[Tensor]:
        """Compute Adam update."""
        if self.m is None or self.v is None:
            self.m = [torch.zeros_like(p) for p in params]
            self.v = [torch.zeros_like(p) for p in params]

        self.t += 1
        new_params = []

        for i, (p, g) in enumerate(zip(params, gradients)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            update = self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
            new_params.append(p - update)

        return new_params


class MAMLBaseline(BaselineOptimizer):
    """Model-Agnostic Meta-Learning (MAML) inner-loop optimiser.

    Standard MAML uses SGD in the inner loop with a learnable or fixed
    learning rate, while the outer loop updates the initialisation via
    Adam. This class provides the inner loop step.
    """

    def __init__(self, inner_lr: float = 0.01) -> None:
        self.inner_lr = inner_lr

    def step(
        self,
        params: Sequence[Tensor],
        gradients: Sequence[Tensor],
        **kwargs: Any,
    ) -> list[Tensor]:
        """Compute inner-loop SGD update."""
        new_params = []
        for p, g in zip(params, gradients):
            new_params.append(p - self.inner_lr * g)
        return new_params


class L2LBaseline(BaselineOptimizer, nn.Module):
    r"""Learning to Learn by Gradient Descent by Gradient Descent (L2L).

    Uses a coordinate-wise LSTM to predict parameter updates, representing
    an earlier paradigm of neural optimisers.
    
    The optimiser state is maintained internally across steps within an episode.
    """

    def __init__(self, hidden_dim: int = 20, n_layers: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Input dimension is 2: sign(g)*log(|g|+1) and clipped g
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_dim, 1)
        
        # L2L uses a very specific initialisation to ensure stable early training
        nn.init.constant_(self.linear.weight, 0.1)
        nn.init.constant_(self.linear.bias, 0.0)

        self.state: tuple[Tensor, Tensor] | None = None

    def reset_state(self) -> None:
        """Clear LSTM hidden state."""
        self.state = None

    def _preprocess(self, flat_grads: Tensor, p: float = 10.0) -> Tensor:
        """L2L log-scale preprocessing.
        
        Args:
            flat_grads: Tensor of shape (num_params,)
        Returns:
            Tensor of shape (num_params, 2)
        """
        # Channel 1: log scale
        log_g = torch.sign(flat_grads) * torch.log(torch.abs(flat_grads) + 1.0)
        # Channel 2: clipped and scaled
        clip_g = torch.clamp(flat_grads, -p, p) / p
        return torch.stack([log_g, clip_g], dim=-1)

    def step(
        self,
        params: Sequence[Tensor],
        gradients: Sequence[Tensor],
        **kwargs: Any,
    ) -> list[Tensor]:
        """Compute coordinate-wise LSTM update."""
        # Flatten all gradients into a single vector (num_params,)
        flat_grads = torch.cat([g.reshape(-1) for g in gradients])
        
        # Preprocess -> (num_params, 2)
        prep_grads = self._preprocess(flat_grads)
        
        # LSTM expects (batch, seq, features). 
        # Here batch = num_params, seq = 1.
        lstm_in = prep_grads.unsqueeze(1)  # (num_params, 1, 2)

        if self.state is None:
            lstm_out, self.state = self.lstm(lstm_in)
        else:
            lstm_out, self.state = self.lstm(lstm_in, self.state)

        # Predict update -> (num_params, 1)
        update_flat = self.linear(lstm_out.squeeze(1))
        update_flat = update_flat.squeeze(-1)  # (num_params,)

        # Unflatten updates back to parameter shapes
        new_params = []
        idx = 0
        for p in params:
            numel = p.numel()
            p_update = update_flat[idx : idx + numel].view_as(p)
            new_params.append(p + p_update)
            idx += numel

        return new_params
