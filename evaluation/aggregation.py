import pandas as pd
import numpy as np
from typing import Dict, Any, List
from .statistical import compute_confidence_interval

class ResultAggregator:
    """Aggregates benchmark results over multiple seeds."""
    
    def __init__(self):
        self.runs = []
        
    def add_run(self, seed: int, results: Dict[str, Any]):
        """Add results from a single run/seed."""
        flat_results = self._flatten_dict(results)
        flat_results['seed'] = seed
        self.runs.append(flat_results)
        
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
        
    def aggregate(self) -> Dict[str, Dict[str, float]]:
        """Compute aggregate statistics over all runs."""
        df = pd.DataFrame(self.runs)
        metrics = [col for col in df.columns if col != 'seed']
        
        aggregated = {}
        for metric in metrics:
            values = df[metric].dropna().tolist()
            if not values:
                continue
            mean, ci_lower, ci_upper = compute_confidence_interval(values)
            aggregated[metric] = {
                'mean': mean,
                'std': float(np.std(values)),
                'ci_95': float((ci_upper - ci_lower) / 2)
            }
        return aggregated
        
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all runs to a Pandas DataFrame."""
        return pd.DataFrame(self.runs)
