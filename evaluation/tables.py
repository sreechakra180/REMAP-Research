import os
from typing import Dict, List, Any

def generate_main_results_table(results: Dict[str, Dict], baselines: Dict[str, Dict]) -> str:
    """Generate LaTeX table for main comparison results."""
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{l|cccc}\n"
    latex += "\\hline\n"
    latex += "Model & Few-Shot Acc & BWT & FWT & Reason Rate \\\\\n"
    latex += "\\hline\n"
    
    # Baselines
    for name, metrics in baselines.items():
        latex += f"{name} & {metrics.get('few_shot', '-')} & {metrics.get('bwt', '-')} & {metrics.get('fwt', '-')} & {metrics.get('reason', '-')} \\\\\n"
        
    latex += "\\hline\n"
    
    # REMAP-Net
    latex += f"\\textbf{{REMAP-Net}} & {results.get('few_shot', '-')} & {results.get('bwt', '-')} & {results.get('fwt', '-')} & {results.get('reason', '-')} \\\\\n"
    
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\caption{Main evaluation results across datasets.}\n"
    latex += "\\end{table}"
    
    return latex

def generate_ablation_table(ablation_results: Dict[str, Dict]) -> str:
    """Generate LaTeX table for ablation studies."""
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{l|c}\n"
    latex += "\\hline\n"
    latex += "Configuration & Performance ($\\Delta$) \\\\\n"
    latex += "\\hline\n"
    
    full_perf = ablation_results.get('full', {}).get('score', 0)
    for name, result in ablation_results.items():
        score = result.get('score', 0)
        delta = score - full_perf
        sign = "+" if delta >= 0 else ""
        latex += f"{name} & {score:.2f} ({sign}{delta:.2f}) \\\\\n"
        
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\caption{Ablation study of REMAP-Net components.}\n"
    latex += "\\end{table}"
    
    return latex

def generate_few_shot_table(results: Dict[str, Dict]) -> str:
    """Generate LaTeX table for few-shot specific results."""
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{l|cc|cc}\n"
    latex += "\\hline\n"
    latex += " & \\multicolumn{2}{c|}{Omniglot} & \\multicolumn{2}{c}{miniImageNet} \\\\\n"
    latex += "Model & 5-way 1-shot & 5-way 5-shot & 5-way 1-shot & 5-way 5-shot \\\\\n"
    latex += "\\hline\n"
    # Placeholder rows
    latex += "\\end{tabular}\n"
    latex += "\\end{table}"
    
    return latex

def save_table(latex_str: str, filepath: str):
    """Save generated LaTeX table to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(latex_str)
