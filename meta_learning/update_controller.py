"""Recurrent update controller for gating meta-updates.

Provides a GRU-based controller that modulates the magnitude of the
proposed parameter update at each meta-step.  The controller learns from
the optimisation trajectory *when* to take large exploratory steps and
*when* to take small refinement steps, acting as a learned learning-rate
schedule.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class UpdateController(nn.Module):
    r"""GRU-based controller that gates the meta-parameter update.

    At each step the controller observes the proposed update Δθ (from the
    preconditioning + bias) together with its own recurrent state, and
    produces a *gated* update:

        Δθ_gated = σ(gate) ⊙ Δθ

    where ``σ`` is a sigmoid and ``gate`` is a learned linear projection of
    the GRU output.  This allows the meta-learner to smoothly interpolate
    between applying the full update and suppressing it entirely.

    Args:
        input_dim: Dimension of the proposed update vector.
        hidden_dim: GRU hidden-state dimension.

    Shape:
        - proposed_update: ``(batch, input_dim)``
        - state (optional): ``(1, batch, hidden_dim)`` — GRU hidden state.
        - Returns ``(gated_update, new_state)`` with shapes
          ``(batch, input_dim)`` and ``(1, batch, hidden_dim)``.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # The GRU processes the proposed update as input.
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=False,
        )

        # Projects GRU output → gate vector of same dim as the update.
        self.gate_proj = nn.Linear(hidden_dim, input_dim)

        # Optional magnitude scaling (learned log-scale per coordinate).
        self.log_magnitude = nn.Parameter(torch.zeros(input_dim))

        self._init_weights()

    # -- weight init ---------------------------------------------------------

    def _init_weights(self) -> None:
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Initialise gate bias to a positive value so the gate starts open
        # (sigmoid(+2) ≈ 0.88 → almost identity at init).
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.0)

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        proposed_update: Tensor,
        state: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Gate a proposed parameter update.

        Args:
            proposed_update: The raw update vector Δθ from the
                preconditioner + bias, shape ``(batch, input_dim)``.
            state: GRU hidden state from the previous step, shape
                ``(1, batch, hidden_dim)``.  Pass ``None`` on the first step
                to use a zero-initialised state.

        Returns:
            Tuple of:
                - ``gated_update``: shape ``(batch, input_dim)``.
                - ``new_state``: shape ``(1, batch, hidden_dim)``.
        """
        B = proposed_update.shape[0]

        # Initialise hidden state on the first call.
        if state is None:
            state = torch.zeros(
                1, B, self.hidden_dim,
                device=proposed_update.device,
                dtype=proposed_update.dtype,
            )

        # GRU expects (seq=1, batch, features).
        gru_input = proposed_update.unsqueeze(0)  # (1, B, input_dim)
        gru_out, new_state = self.gru(gru_input, state)  # (1, B, hid), (1, B, hid)

        # Compute gate ∈ (0, 1)^{input_dim}.
        gate = torch.sigmoid(self.gate_proj(gru_out.squeeze(0)))  # (B, input_dim)

        # Apply magnitude scaling and gate.
        magnitude = self.log_magnitude.exp()  # (input_dim,)
        gated_update = gate * magnitude * proposed_update

        return gated_update, new_state

    def get_initial_state(
        self, batch_size: int, device: torch.device
    ) -> Tensor:
        """Create a zero-initialised GRU hidden state.

        Args:
            batch_size: Number of parallel optimisation trajectories.
            device: Target device.

        Returns:
            Tensor of shape ``(1, batch_size, hidden_dim)``.
        """
        return torch.zeros(
            1, batch_size, self.hidden_dim, device=device
        )
