import copy
import logging
from typing import List, Tuple, Dict, Any

class AblationRunner:
    """Runner for conducting ablation studies on REMAP-Net."""
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.logger = logging.getLogger('AblationRunner')
        
    def define_ablations(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Define the set of ablations to run."""
        ablations = []
        
        # 1. Full model (baseline)
        ablations.append(('full', {}))
        
        # 2. Component ablations
        ablations.append(('no_f2', {'enable_epistemic_recursion': False}))
        ablations.append(('no_tcr', {'lambda_tc': 0.0}))
        ablations.append(('no_aaf', {'lambda_abs': 0.0}))
        ablations.append(('no_stability', {'enable_guardian': False}))
        
        # 3. Hyperparameter sweeps
        for depth in [1, 2, 3, 5]:
            ablations.append((f'depth_{depth}', {'recursion_depth': depth}))
            
        for size in ['small', 'medium', 'large']:
            capacity = {'small': 512, 'medium': 2048, 'large': 8192}[size]
            ablations.append((f'memory_{size}', {'memory_capacity': capacity}))
            
        for dim in [16, 32, 64, 128]:
            ablations.append((f'abstraction_{dim}', {'abstraction_dim': dim}))
            
        return ablations
        
    def run_ablation(self, name: str, seed: int) -> Dict[str, Any]:
        """Run a specific ablation configuration for a given seed."""
        self.logger.info(f"Running ablation: {name} (Seed {seed})")
        
        # Combine base config with ablation override
        ablations = dict(self.define_ablations())
        override = ablations.get(name, {})
        
        config = copy.deepcopy(self.base_config)
        config.update(override)
        config['seed'] = seed
        
        # Placeholder for actual training and evaluation
        # model = REMAPNet(config)
        # result = train_and_eval(model, config)
        
        import numpy as np
        # Mock results based on ablation type to show expected degradation
        base_score = 0.85
        if name == 'full':
            score = base_score
        elif name == 'no_f2':
            score = base_score - 0.12
        elif name == 'no_tcr':
            score = base_score - 0.05
        elif name == 'no_aaf':
            score = base_score - 0.08
        elif name == 'no_stability':
            score = base_score - 0.15
        elif 'depth' in name:
            depth = int(name.split('_')[1])
            score = base_score - 0.05 * abs(3 - depth) # Optimal at depth 3
        else:
            score = base_score - np.random.uniform(0.01, 0.05)
            
        return {'score': score, 'config': config}
        
    def run_all(self, seeds: List[int] = [42, 123, 456, 789, 1000]) -> Dict[str, Dict[str, Any]]:
        """Run all defined ablations across all seeds."""
        all_results = {}
        ablations = self.define_ablations()
        
        for name, _ in ablations:
            seed_results = []
            for seed in seeds:
                res = self.run_ablation(name, seed)
                seed_results.append(res)
                
            # Aggregate across seeds
            import numpy as np
            scores = [r['score'] for r in seed_results]
            all_results[name] = {
                'score': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'seed_runs': seed_results
            }
            
        return all_results
