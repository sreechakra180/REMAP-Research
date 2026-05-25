"""Low-rank preconditioning for the Meta-Plasticity update rule.

Implements the preconditioning matrix from Eq. 3:

    M_ϕ(h, E) = U(h, E) · V(h, E)^T

where U, V ∈ ℝ^{d × r} with rank r ≪ d.  Both factors are *generated* by
small hyper-networks conditioned on the encoded gradient history ``h`` and
episodic context ``E``, enabling task-adaptive curvature approximation.

Also provides an :class:`AdaptivePreconditioner` that augments the low-rank
core with per-coordinate diagonal scaling and learned temperature.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LowRankPreconditioner(nn.Module):
    r"""Context-conditioned low-rank preconditioning matrix.

    Generates ``M_ϕ = U · V^T`` where the factors ``U, V ∈ ℝ^{d × r}`` are
    produced by lightweight MLPs conditioned on the concatenated context
    ``[h; E]``.

    Keeping the rank ``r ≪ d`` reduces the parameter count from O(d²) to
    O(d · r) and implicitly regularises the curvature approximation.

    Args:
        param_dim: Dimension ``d`` of the parameter vector being updated.
        rank: Rank ``r`` of the low-rank factorisation (default 32).
        context_dim: Combined dimension of gradient-history encoding ``h``
            and episodic context ``E`` that conditions the factors.

    Shape:
        - h: ``(batch, history_dim)``
        - E: ``(batch, episodic_dim)``  where ``history_dim + episodic_dim = context_dim``
        - Output ``M_ϕ``: ``(batch, param_dim, param_dim)``
    """

    def __init__(
        self,
        param_dim: int,
        rank: int = 32,
        context_dim: int = 128,
    ) -> None:
        super().__init__()
        self.param_dim = param_dim
        self.rank = rank
        self.context_dim = context_dim

        # Hyper-networks that map context → factors.
        self.u_net = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, param_dim * rank),
        )
        self.v_net = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, param_dim * rank),
        )

        # Learnable scaling to control initial magnitude of M.
        self.log_scale = nn.Parameter(torch.tensor(math.log(0.01)))

        self._init_weights()

    # -- init ----------------------------------------------------------------

    def _init_weights(self) -> None:
        """Small-magnitude initialisation so M starts near zero."""
        for net in (self.u_net, self.v_net):
            for module in net.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(
                        module.weight, a=math.sqrt(5)
                    )
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            # Shrink the last layer so that M ≈ 0 at init.
            last_layer = net[-1]
            assert isinstance(last_layer, nn.Linear)
            last_layer.weight.data.mul_(0.01)
            if last_layer.bias is not None:
                last_layer.bias.data.zero_()

    # -- forward -------------------------------------------------------------

    def forward(self, h: Tensor, E: Tensor) -> Tensor:
        """Compute the context-conditioned preconditioning matrix.

        Args:
            h: Encoded gradient history, ``(batch, history_dim)``.
            E: Episodic context vector, ``(batch, episodic_dim)``.

        Returns:
            ``M_ϕ`` of shape ``(batch, param_dim, param_dim)``.
        """
        ctx = torch.cat([h, E], dim=-1)  # (B, context_dim)

        # Generate factors.
        U = self.u_net(ctx).view(-1, self.param_dim, self.rank)  # (B, d, r)
        V = self.v_net(ctx).view(-1, self.param_dim, self.rank)  # (B, d, r)

        # M = scale · U · V^T  → (B, d, d)
        scale = self.log_scale.exp()
        M = scale * torch.bmm(U, V.transpose(1, 2))

        return M

    # -- analysis helpers ----------------------------------------------------

    def get_effective_matrix(
        self, h: Tensor, E: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return the factors and the full matrix for analysis.

        Args:
            h: Gradient history encoding.
            E: Episodic context.

        Returns:
            Tuple ``(U, V, M)`` with shapes ``(B, d, r)``, ``(B, d, r)``,
            ``(B, d, d)``.
        """
        ctx = torch.cat([h, E], dim=-1)
        U = self.u_net(ctx).view(-1, self.param_dim, self.rank)
        V = self.v_net(ctx).view(-1, self.param_dim, self.rank)
        scale = self.log_scale.exp()
        M = scale * torch.bmm(U, V.transpose(1, 2))
        return U, V, M


# ---------------------------------------------------------------------------
# Adaptive Preconditioner (extends low-rank with task-adaptive scaling)
# ---------------------------------------------------------------------------


class AdaptivePreconditioner(nn.Module):
    r"""Task-adaptive preconditioner with low-rank core and diagonal modulation.

    Extends :class:`LowRankPreconditioner` with a per-coordinate diagonal
    scaling predicted from the context, plus a learned temperature parameter:

        M_adaptive = diag(s(ctx)) · M_lowrank · diag(s(ctx)) + τ · I

    where ``s(ctx)`` is a context-dependent positive scaling vector and ``τ``
    is a learnable temperature that ensures positive-definiteness.

    Args:
        param_dim: Parameter vector dimension.
        rank: Low-rank factorisation rank.
        context_dim: Conditioning dimension ``|[h; E]|``.
        temperature_init: Initial value for the diagonal temperature ``τ``.
    """

    def __init__(
        self,
        param_dim: int,
        rank: int = 32,
        context_dim: int = 128,
        temperature_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.param_dim = param_dim

        # Core low-rank component.
        self.low_rank = LowRankPreconditioner(
            param_dim=param_dim, rank=rank, context_dim=context_dim
        )

        # Context → positive per-coordinate scaling.
        self.scale_net = nn.Sequential(
            nn.Linear(context_dim, context_dim // 2),
            nn.SiLU(),
            nn.Linear(context_dim // 2, param_dim),
            nn.Softplus(),
        )

        # Learnable temperature for the identity term.
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(temperature_init))
        )

    def forward(self, h: Tensor, E: Tensor) -> Tensor:
        """Compute the adaptive preconditioning matrix.

        Args:
            h: Gradient history encoding, ``(batch, history_dim)``.
            E: Episodic context, ``(batch, episodic_dim)``.

        Returns:
            ``M_adaptive`` of shape ``(batch, param_dim, param_dim)``.
        """
        ctx = torch.cat([h, E], dim=-1)
        B = ctx.shape[0]

        # Low-rank core.
        M_lr = self.low_rank(h, E)  # (B, d, d)

        # Per-coordinate scaling (diagonal).
        s = self.scale_net(ctx)  # (B, d)
        S = torch.diag_embed(s)  # (B, d, d)

        # M_adaptive = S · M_lr · S  + τ·I
        M_scaled = torch.bmm(S, torch.bmm(M_lr, S))
        tau = self.log_temperature.exp()
        identity = torch.eye(
            self.param_dim, device=ctx.device, dtype=ctx.dtype
        ).unsqueeze(0)
        M_adaptive = M_scaled + tau * identity

        return M_adaptive
