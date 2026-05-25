import json
import os
from typing import Dict, Any
import numpy as np
from ..evaluation.statistical import wilcoxon_signed_rank

class AblationAnalyzer:
    """Analyzer for ablation study results."""
    
    @staticmethod
    def load_results(results_dir: str) -> Dict[str, Any]:
        """Load ablation results from a directory."""
        results = {}
        if not os.path.exists(results_dir):
            return results
            
        for file in os.listdir(results_dir):
            if file.endswith('.json'):
                name = file.split('.')[0]
                with open(os.path.join(results_dir, file), 'r') as f:
                    results[name] = json.load(f)
        return results
        
    @staticmethod
    def compute_relative_performance(results: Dict[str, Dict], baseline: str = 'full') -> Dict[str, float]:
        """Compute performance relative to a baseline configuration."""
        if baseline not in results:
            raise ValueError(f"Baseline '{baseline}' not found in results.")
            
        base_score = results[baseline].get('score', 0)
        relative_scores = {}
        
        for name, res in results.items():
            if name != baseline:
                relative_scores[name] = res.get('score', 0) - base_score
                
        return relative_scores
        
    @staticmethod
    def statistical_comparison(results: Dict[str, Dict], baseline: str = 'full') -> Dict[str, Dict]:
        """Perform statistical significance tests against baseline."""
        significance = {}
        if baseline not in results or 'seed_runs' not in results[baseline]:
            return significance
            
        base_runs = [r['score'] for r in results[baseline]['seed_runs']]
        
        for name, res in results.items():
            if name != baseline and 'seed_runs' in res:
                comp_runs = [r['score'] for r in res['seed_runs']]
                
                # Check if we have enough samples for Wilcoxon
                if len(base_runs) >= 5 and len(comp_runs) >= 5:
                    p_value, stat = wilcoxon_signed_rank(base_runs, comp_runs)
                    is_significant = p_value < 0.05
                else:
                    p_value, stat, is_significant = None, None, None
                    
                significance[name] = {
                    'p_value': p_value,
                    'statistic': stat,
                    'significant': is_significant,
                    'delta': float(np.mean(comp_runs) - np.mean(base_runs))
                }
                
        return significance
        
    @staticmethod
    def generate_report(results: Dict[str, Dict], baseline: str = 'full') -> str:
        """Generate a markdown report summarizing ablation findings."""
        report = "# REMAP-Net Ablation Study Report\n\n"
        
        report += "## Overall Performance\n"
        for name, res in results.items():
            score = res.get('score', 0)
            std = res.get('std', 0)
            report += f"- **{name}**: {score:.4f} ± {std:.4f}\n"
            
        report += "\n## Relative Impacts\n"
        try:
            rel_perf = AblationAnalyzer.compute_relative_performance(results, baseline)
            for name, delta in sorted(rel_perf.items(), key=lambda x: x[1]):
                report += f"- **{name}**: {delta:+.4f}\n"
        except Exception as e:
            report += f"Could not compute relative performance: {e}\n"
            
        report += "\n## Statistical Significance (vs Baseline)\n"
        try:
            sig = AblationAnalyzer.statistical_comparison(results, baseline)
            for name, res in sig.items():
                sig_str = "Yes" if res['significant'] else "No"
                p_val = f"{res['p_value']:.4f}" if res['p_value'] is not None else "N/A"
                report += f"- **{name}**: Significant: {sig_str} (p={p_val}), $\\Delta$: {res['delta']:+.4f}\n"
        except Exception as e:
            report += f"Could not compute statistical significance: {e}\n"
            
        return report
