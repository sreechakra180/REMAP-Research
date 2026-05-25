import os
import argparse
import yaml
import subprocess
from typing import List, Dict, Any

def run_experiment(config_path: str, seed: int, output_dir: str):
    """Runs a single experiment with a specific config and seed."""
    cmd = [
        "python", "-m", "remap_net.main", # Assuming main.py exists at root module
        "--config", config_path,
        "--seed", str(seed),
        "--output_dir", output_dir
    ]
    print(f"Running: {' '.join(cmd)}")
    # subprocess.run(cmd, check=True)  # Commented out to prevent accidental execution

def main():
    parser = argparse.ArgumentParser(description="Run REMAP-Net Ablation Sweep")
    parser.add_argument("--config", type=str, required=True, help="Base config file")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Seeds to run")
    parser.add_argument("--output_dir", type=str, default="./results/ablation", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load base config
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)

    # Define ablation variants
    variants = {
        "full": {},
        "no_f2": {"epistemic": None},
        "no_f1": {"meta_plasticity": None, "epistemic": None},
        "no_stability": {"stability": None},
        "no_memory": {"memory": None},
        "no_abstraction": {"abstraction": None}
    }

    for variant_name, overrides in variants.items():
        print(f"\n{'='*50}\nStarting variant: {variant_name}\n{'='*50}")
        
        # Create variant config
        variant_config = base_config.copy()
        for k, v in overrides.items():
            if v is None:
                if k in variant_config:
                    del variant_config[k]
            else:
                variant_config[k] = v
                
        # Save variant config
        variant_config_path = os.path.join(args.output_dir, f"config_{variant_name}.yaml")
        with open(variant_config_path, 'w') as f:
            yaml.dump(variant_config, f)
            
        # Run for each seed
        for seed in args.seeds:
            run_dir = os.path.join(args.output_dir, variant_name, f"seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)
            # run_experiment(variant_config_path, seed, run_dir)
            print(f"Would run experiment {variant_name} with seed {seed} saving to {run_dir}")

if __name__ == "__main__":
    main()
