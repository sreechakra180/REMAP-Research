from .style import set_ieee_style, get_color, IEEE_RCPARAMS, COLOR_PALETTE
from .training import plot_loss_curves, plot_learning_rate_schedule, plot_gradient_norms, plot_phase_transitions, plot_meta_update_magnitudes
from .stability import plot_lyapunov_energy, plot_energy_components, plot_stability_landscape_2d, plot_projection_events, plot_convergence_rate
from .abstraction import plot_tsne_latent_space, plot_umap_latent_space, plot_abstraction_hierarchy, plot_information_plane, plot_compression_vs_quality
from .memory import plot_memory_utilization, plot_retrieval_similarity, plot_memory_evolution, plot_forgetting_heatmap
from .recursion import plot_recursive_update_flow, plot_update_correlation, plot_meta_gradient_flow
from .publication import create_figure_grid, add_significance_brackets, save_publication_figure, create_comparison_bar_chart, create_radar_chart

__all__ = [
    'set_ieee_style', 'get_color', 'IEEE_RCPARAMS', 'COLOR_PALETTE',
    'plot_loss_curves', 'plot_learning_rate_schedule', 'plot_gradient_norms', 'plot_phase_transitions', 'plot_meta_update_magnitudes',
    'plot_lyapunov_energy', 'plot_energy_components', 'plot_stability_landscape_2d', 'plot_projection_events', 'plot_convergence_rate',
    'plot_tsne_latent_space', 'plot_umap_latent_space', 'plot_abstraction_hierarchy', 'plot_information_plane', 'plot_compression_vs_quality',
    'plot_memory_utilization', 'plot_retrieval_similarity', 'plot_memory_evolution', 'plot_forgetting_heatmap',
    'plot_recursive_update_flow', 'plot_update_correlation', 'plot_meta_gradient_flow',
    'create_figure_grid', 'add_significance_brackets', 'save_publication_figure', 'create_comparison_bar_chart', 'create_radar_chart'
]
