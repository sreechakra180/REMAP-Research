"""Meta-learning components for REMAP-Net.

Provides the implementations for the F1 (Meta-Plasticity) and F2
(Epistemic Recursion) modules, as well as several baselines.
"""

from __future__ import annotations

from .baselines import AdamBaseline, L2LBaseline, MAMLBaseline, SGDBaseline
from .contextual_bias import ContextualBias
from .epistemic_recursion import EpistemicRecursionModule
from .gradient_encoder import GradientEncoder, GradientPreprocessor
from .low_rank import AdaptivePreconditioner, LowRankPreconditioner
from .meta_plasticity import MetaPlasticityModule
from .preconditioning import DiagonalPreconditioner, KroneckerPreconditioner
from .update_controller import UpdateController

__all__ = [
    "AdamBaseline",
    "L2LBaseline",
    "MAMLBaseline",
    "SGDBaseline",
    "ContextualBias",
    "EpistemicRecursionModule",
    "GradientEncoder",
    "GradientPreprocessor",
    "AdaptivePreconditioner",
    "LowRankPreconditioner",
    "MetaPlasticityModule",
    "DiagonalPreconditioner",
    "KroneckerPreconditioner",
    "UpdateController",
]
