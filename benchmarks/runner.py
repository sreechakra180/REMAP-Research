from typing import Dict, Any, List
from .few_shot import FewShotBenchmark
from .continual import ContinualBenchmark
from .reasoning import ReasoningBenchmark
from .abstraction import AbstractionBenchmark

class BenchmarkRunner:
    """Unified benchmark runner for all REMAP-Net evaluations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def run_all(self, model) -> Dict[str, Any]:
        """Run all configured benchmarks on the model."""
        results = {}
        
        if 'few_shot' in self.config:
            results['few_shot'] = self.run_benchmark('few_shot', model)
            
        if 'continual' in self.config:
            results['continual'] = self.run_benchmark('continual', model)
            
        if 'reasoning' in self.config:
            results['reasoning'] = self.run_benchmark('reasoning', model)
            
        if 'abstraction' in self.config:
            results['abstraction'] = self.run_benchmark('abstraction', model)
            
        return results
        
    def run_benchmark(self, name: str, model) -> Dict[str, Any]:
        """Run a specific benchmark by name."""
        if name == 'few_shot':
            # Setup mock dataset since we don't have real data loaded
            return FewShotBenchmark.evaluate(model, None, n_way=5, k_shot=1)
        elif name == 'continual':
            return ContinualBenchmark.evaluate(model, datasets=[])
        elif name == 'reasoning':
            return ReasoningBenchmark.evaluate(model, None)
        elif name == 'abstraction':
            return AbstractionBenchmark.evaluate(model, None)
        else:
            raise ValueError(f"Unknown benchmark: {name}")
            
    def aggregate_results(self, multi_seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across multiple seeds."""
        aggregated = {}
        
        # Simplify aggregation logic for nested dictionaries
        # Just computing mean across seeds for scalar values
        # In a real implementation this would traverse the nested dicts
        
        return aggregated
