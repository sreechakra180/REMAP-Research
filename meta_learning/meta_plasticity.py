"""Meta-Plasticity Module (F1) for REMAP-Net.

Implements Eq. 2-3 from the paper:
    θ(t+1) = θ(t) + G_ϕ(g_t, h_t, E_t)
    G_ϕ(g, h, E) = -M_ϕ(h, E) · g + b_ϕ(h, E)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .gradient_encoder import GradientEncoder, GradientPreprocessor
from .low_rank import LowRankPreconditioner
from .contextual_bias import ContextualBias
from .update_controller import UpdateController


class MetaPlasticityModule(nn.Module):
    r"""F1 Meta-Plasticity Module implementing the task-adaptive update rule.

    Given the current gradient ``g_t``, gradient history ``h_t``, and episodic
    context ``E_t`` from the memory module, this computes the parameter update:

        G_ϕ(g_t, h_t, E_t) = -M_ϕ(h_t, E_t) · g_t + b_ϕ(h_t, E_t)

    where M_ϕ is a low-rank preconditioning matrix and b_ϕ is a contextual bias.
    The update is further refined by a recurrent update controller.

    Args:
        param_dim: Total number of parameters to update.
        history_depth: Length of the gradient history sequence K.
        context_dim: Dimension of the episodic context E_t.
        rank: Rank for the low-rank preconditioner M_ϕ.
    """

    def __init__(
        self,
        param_dim: int,
        history_depth: int = 10,
        context_dim: int = 128,
        rank: int = 32,
    ) -> None:
        super().__init__()
        self.param_dim = param_dim
        self.history_depth = history_depth
        self.context_dim = context_dim
        self.rank = rank

        self.history_dim = 64  # Hidden dimension from GradientEncoder

        self.preprocessor = GradientPreprocessor()
        # GradientEncoder processes (batch, seq, param_dim, input_dim) -> (batch, param_dim, history_dim)
        # input_dim is 2 from GradientPreprocessor
        self.gradient_encoder = GradientEncoder(
            input_dim=2, hidden_dim=self.history_dim, n_layers=2
        )

        self.preconditioning = LowRankPreconditioner(
            param_dim=param_dim,
            rank=rank,
            context_dim=self.history_dim + context_dim,
        )

        self.contextual_bias = ContextualBias(
            history_dim=self.history_dim,
            context_dim=context_dim,
            output_dim=param_dim,
        )

        self.update_controller = UpdateController(
            input_dim=param_dim, hidden_dim=128
        )

    def forward(
        self,
        gradient: Tensor,
        history: Tensor,
        context: Tensor,
        controller_state: Optional[Tensor] = None,
        encoder_state: Optional[tuple[Tensor, Tensor]] = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        """Compute the meta-plasticity parameter update.

        Args:
            gradient: Current task gradient, shape ``(batch, param_dim)``.
            history: Gradient history sequence, shape ``(batch, seq_len, param_dim)``.
            context: Episodic context from memory, shape ``(batch, context_dim)``.
            controller_state: Optional GRU state for the update controller.
            encoder_state: Optional LSTM state for the gradient encoder.

        Returns:
            Tuple of:
                - `gated_update`: The final parameter update to apply.
                - `new_controller_state`: Updated GRU state.
                - `new_encoder_state`: Updated LSTM state (though usually not returned 
                  from the file's GradientEncoder directly in this structure, we just dummy it).
        """
        B, seq_len, D = history.shape
        assert D == self.param_dim, f"History dim {D} != param_dim {self.param_dim}"

        # 1. Preprocess history: (B, seq, D) -> (B, seq, D, 2)
        h_prep = self.preprocessor(history)

        # 2. Encode history: (B, D, history_dim)
        # Wait, the encoder returns (B, D, hidden_dim) and NO state in the return tuple?
        # Let's check GradientEncoder: it just returns encoded tensor if we look closely...
        # Wait, looking at the code for GradientEncoder, it doesn't return state! It only returns `self.output_norm(encoded)`.
        # So we just get `h_enc`.
        h_enc = self.gradient_encoder(h_prep, encoder_state)

        # Aggregate across parameter dimensions to get a global history representation:
        # (B, history_dim)
        h_global = h_enc.mean(dim=1)

        # 3. M_ϕ * g:
        # Note: LowRankPreconditioner takes (h_global, context) and context_dim in its init is the *combined* length!
        # But wait, looking at low_rank.py, it says: `ctx = torch.cat([h, E], dim=-1)` inside its forward!
        # So it actually expects h and E as separate args. Wait, in low_rank.py `__init__` it asks for `context_dim: Combined dimension of gradient-history encoding h and episodic context E`.
        # So if we passed context_dim=history_dim + context_dim to LowRankPreconditioner, it matches.
        
        M_phi = self.preconditioning(h_global, context)  # (B, D, D)
        
        # M_ϕ * g -> (B, D, D) bmm (B, D, 1) -> (B, D)
        m_g = torch.bmm(M_phi, gradient.unsqueeze(-1)).squeeze(-1)

        # 4. Contextual bias
        # ContextualBias expects (h, E) as separate args.
        b_phi = self.contextual_bias(h_global, context)  # (B, D)

        # 5. Proposed update
        proposed_update = -m_g + b_phi  # (B, D)

        # 6. Gate the update via controller
        gated_update, new_controller_state = self.update_controller(
            proposed_update, controller_state
        )

        return gated_update, new_controller_state, None

    def compute_update(
        self,
        params: Tensor,
        gradient: Tensor,
        history: Tensor,
        context: Tensor,
        controller_state: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute the new parameters using Eq. 2-3.
        
        θ(t+1) = θ(t) + G_ϕ(g_t, h_t, E_t)
        """
        update, new_state, _ = self.forward(
            gradient, history, context, controller_state
        )
        new_params = params + update
        return new_params, new_state

    def get_preconditioning_matrix(
        self, history: Tensor, context: Tensor
    ) -> Tensor:
        """Return the preconditioning matrix M_ϕ for analysis.

        Args:
            history: Gradient history sequence, shape ``(batch, seq_len, param_dim)``.
            context: Episodic context, shape ``(batch, context_dim)``.

        Returns:
            The M_ϕ matrix, shape ``(batch, param_dim, param_dim)``.
        """
        h_prep = self.preprocessor(history)
        h_enc = self.gradient_encoder(h_prep)
        h_global = h_enc.mean(dim=1)
        _, _, M = self.preconditioning.get_effective_matrix(h_global, context)
        return M
