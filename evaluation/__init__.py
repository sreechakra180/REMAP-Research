"""Evaluation module for computing metrics and statistics."""

from .metrics import accuracy, top_k_accuracy, f1_score, few_shot_accuracy, backward_transfer, forward_transfer
from .statistical import wilcoxon_signed_rank, paired_t_test, compute_confidence_interval, bootstrap_ci, effect_size_cohens_d, multi_seed_summary
from .aggregation import ResultAggregator
from .tables import generate_main_results_table, generate_ablation_table, generate_few_shot_table, save_table
from .plots import plot_training_curves, plot_few_shot_comparison, plot_ablation_bars, plot_stability_trajectory, plot_forgetting_curves

__all__ = [
    'accuracy', 'top_k_accuracy', 'f1_score', 'few_shot_accuracy', 'backward_transfer', 'forward_transfer',
    'wilcoxon_signed_rank', 'paired_t_test', 'compute_confidence_interval', 'bootstrap_ci', 'effect_size_cohens_d', 'multi_seed_summary',
    'ResultAggregator',
    'generate_main_results_table', 'generate_ablation_table', 'generate_few_shot_table', 'save_table',
    'plot_training_curves', 'plot_few_shot_comparison', 'plot_ablation_bars', 'plot_stability_trajectory', 'plot_forgetting_curves'
]
