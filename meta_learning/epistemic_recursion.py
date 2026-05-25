"""Epistemic Recursion Module (F2) for REMAP-Net.

Implements Eq. 4-5 from the paper:
    M(ϕ, θ_{1:T}, D_val) = E_{(x,y)~D_val}[ℓ(F0(x; θ_T), y)] + α·R(ϕ)
    ϕ(t+1) = ϕ(t) + H_ψ(∇_ϕ M(ϕ_t, θ^(t)_{1:T}, D^(t)_val))

Where H_ψ has the same structural form as G_ϕ (recursive self-similarity).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .meta_plasticity import MetaPlasticityModule


class EpistemicRecursionModule(nn.Module):
    r"""F2 Epistemic Recursion Module.

    Updates the meta-parameters ϕ using higher-order gradients derived from
    a meta-objective evaluated on validation data. Employs recursive 
    self-similarity by using a structurally identical update module (Meta-Plasticity) 
    to optimise the meta-parameters themselves.

    Args:
        meta_param_dim: Total number of meta-parameters in ϕ.
        rank: Rank for the higher-level preconditioner.
        context_dim: Dimension of the meta-level episodic context.
        history_depth: Depth of the meta-gradient history.
    """

    def __init__(
        self,
        meta_param_dim: int,
        rank: int = 16,
        context_dim: int = 64,
        history_depth: int = 10,
    ) -> None:
        super().__init__()
        self.meta_param_dim = meta_param_dim

        # Structural self-similarity: H_ψ has the exact same form as G_ϕ
        self.meta_updater = MetaPlasticityModule(
            param_dim=meta_param_dim,
            history_depth=history_depth,
            context_dim=context_dim,
            rank=rank,
        )

    def forward(
        self,
        meta_gradient: Tensor,
        meta_history: Tensor,
        meta_context: Tensor,
        controller_state: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute the update for the meta-parameters ϕ.

        Args:
            meta_gradient: Higher-order gradient ∇_ϕ M, shape ``(batch, meta_param_dim)``.
            meta_history: Meta-gradient history, shape ``(batch, seq, meta_param_dim)``.
            meta_context: Meta-level episodic context, shape ``(batch, context_dim)``.
            controller_state: Optional state for the meta-updater's GRU.

        Returns:
            Tuple of:
                - `phi_update`: The update to apply to ϕ.
                - `new_state`: Updated GRU state.
        """
        phi_update, new_state, _ = self.meta_updater(
            meta_gradient, meta_history, meta_context, controller_state
        )
        return phi_update, new_state

    def compute_meta_objective(
        self,
        f0: nn.Module,
        val_data: tuple[Tensor, Tensor],
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        alpha: float = 0.01,
        R_phi: Optional[Tensor] = None,
    ) -> Tensor:
        """Evaluate the meta-objective M(ϕ, θ_{1:T}, D_val).

        Args:
            f0: Base network with adapted parameters θ_T.
            val_data: Validation data tuple `(x_val, y_val)`.
            loss_fn: Task loss function.
            alpha: Regularisation strength for the meta-parameters.
            R_phi: Regularisation penalty on ϕ (e.g., L2 norm or complexity).

        Returns:
            Scalar meta-objective tensor.
        """
        x_val, y_val = val_data
        preds = f0(x_val)
        task_loss = loss_fn(preds, y_val)

        meta_obj = task_loss
        if R_phi is not None:
            meta_obj = meta_obj + alpha * R_phi

        return meta_obj

    def compute_higher_order_gradients(
        self,
        meta_objective: Tensor,
        phi_params: Sequence[Tensor],
    ) -> list[Tensor]:
        """Compute ∇_ϕ M(ϕ_t, ...) using unrolled differentiation.

        Requires that the forward passes through the inner loop were computed
        with `create_graph=True` so that the computational graph connects the
        validation loss back to the meta-parameters ϕ.

        Args:
            meta_objective: The computed meta-objective scalar.
            phi_params: The sequence of meta-parameters ϕ to differentiate w.r.t.

        Returns:
            List of gradient tensors matching `phi_params`.
        """
        grads = torch.autograd.grad(
            meta_objective,
            phi_params,
            allow_unused=True,
            retain_graph=False,
        )

        # Handle any parameters that weren't part of the computational graph
        # by returning zero gradients.
        clean_grads = []
        for g, p in zip(grads, phi_params):
            if g is None:
                clean_grads.append(torch.zeros_like(p))
            else:
                clean_grads.append(g)

        return clean_grads

    def approximate_inverse_hessian_hutchinson(
        self,
        loss: Tensor,
        params: Sequence[Tensor],
        vec: Sequence[Tensor],
        max_iter: int = 5,
    ) -> list[Tensor]:
        r"""Approximate Inverse Hessian-vector product using Neumann series.

        Uses Hutchinson's method / Neumann series to efficiently approximate
        H^{-1} @ v without materialising the full Hessian:
            H^{-1} @ v ≈ Σ_{i=0}^K (I - H)^i @ v

        Args:
            loss: The scalar loss to differentiate.
            params: Parameters to differentiate w.r.t (θ).
            vec: The vector to multiply with the inverse Hessian (v).
            max_iter: Number of Neumann series terms K.

        Returns:
            List of tensors representing the approximated inverse HVP.
        """
        # First-order gradients ∇_θ L
        grads = torch.autograd.grad(loss, params, create_graph=True)

        # Initialize Neumann series components
        # res = v_0 = v
        # v_i = v + (I - H) v_{i-1} = v + v_{i-1} - H v_{i-1}
        res = [v.clone() for v in vec]
        v_i = [v.clone() for v in vec]

        for _ in range(max_iter):
            # Compute Hessian-vector product: H @ v_i
            # H @ v_i = ∇_θ (∇_θ L @ v_i)
            grad_dot_v = sum(torch.sum(g * vi) for g, vi in zip(grads, v_i))
            
            # The retaining of the graph is necessary to do multiple HvPs
            hvp = torch.autograd.grad(
                grad_dot_v, params, retain_graph=True, allow_unused=True
            )

            # Clean hvp in case of detached subgraphs
            hvp_clean = [
                h if h is not None else torch.zeros_like(p) 
                for h, p in zip(hvp, params)
            ]

            # v_{i+1} = v + v_i - H @ v_i
            v_i_next = []
            res_next = []
            for v_orig, vi_curr, h_curr, r_curr in zip(vec, v_i, hvp_clean, res):
                vi_next_val = v_orig + vi_curr - h_curr
                v_i_next.append(vi_next_val)
                res_next.append(r_curr + vi_next_val)

            v_i = v_i_next
            res = res_next

        return res
