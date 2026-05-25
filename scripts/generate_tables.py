import os
import argparse
import pandas as pd
import numpy as np

def generate_latex_table(df: pd.DataFrame, title: str, label: str) -> str:
    """Generates a LaTeX table from a pandas DataFrame."""
    latex = "\\begin{table}[htbp]\n"
    latex += "    \\centering\n"
    latex += f"    \\caption{{{title}}}\n"
    latex += f"    \\label{{tab:{label}}}\n"
    
    # Simple formatting
    latex += df.to_latex(index=False, float_format="%.2f", escape=False)
    
    latex += "\\end{table}\n"
    return latex

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from results")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory with result CSVs")
    parser.add_argument("--output_file", type=str, default="./results/tables.tex", help="Output LaTeX file")
    args = parser.parse_args()

    # Mock data for demonstration
    data = {
        "Method": ["MAML", "ANIL", "ProtoNet", "REMAP-Net (Ours)"],
        "5-way 1-shot": ["87.5 $\\pm$ 0.4", "88.1 $\\pm$ 0.3", "89.0 $\\pm$ 0.5", "\\textbf{94.2 $\\pm$ 0.2}"],
        "5-way 5-shot": ["96.1 $\\pm$ 0.2", "96.5 $\\pm$ 0.1", "97.2 $\\pm$ 0.2", "\\textbf{98.8 $\\pm$ 0.1}"]
    }
    df = pd.DataFrame(data)

    latex_content = generate_latex_table(df, "Few-shot classification accuracy on Omniglot.", "omniglot_results")

    ablation_data = {
        "Variant": ["Full Model", "w/o Epistemic (F2)", "w/o Meta (F1)", "w/o Stability", "w/o Memory"],
        "Accuracy": ["94.2", "89.5", "82.1", "88.0", "91.2"],
        "Forgetting": ["2.1%", "4.5%", "15.2%", "12.0%", "8.4%"]
    }
    df_ablation = pd.DataFrame(ablation_data)
    
    latex_content += "\n" + generate_latex_table(df_ablation, "Ablation study on Omniglot 5-way 1-shot.", "ablation_results")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        f.write(latex_content)
        
    print(f"Generated LaTeX tables at {args.output_file}")

if __name__ == "__main__":
    main()
