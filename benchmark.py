import os
import yaml

def run_benchmark(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    seeds = cfg.get('benchmark', {}).get('seeds', [42, 43, 44])
    tasks = cfg.get('benchmark', {}).get('tasks', ['few_shot', 'continual'])
    
    results = {}
    for task in tasks:
        results[task] = []
        for seed in seeds:
            print(f"Running benchmark for task={task}, seed={seed}")
            results[task].append({'seed': seed, 'score': 0.0}) # Placeholder
            
    print("Benchmark complete. Results:")
    print(results)
    
    with open("benchmark_results.txt", "w") as f:
        f.write("REMAP-Net Benchmark Results\n")
        f.write("===========================\n")
        for task, data in results.items():
            f.write(f"\nTask: {task}\n")
            for d in data:
                f.write(f"  Seed {d['seed']}: {d['score']}\n")

if __name__ == "__main__":
    if os.path.exists("configs/config.yaml"):
        run_benchmark("configs/config.yaml")
    else:
        print("Config not found. Please create configs/config.yaml.")
