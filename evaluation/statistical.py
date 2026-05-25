import numpy as np
import scipy.stats as stats
from typing import Tuple, List, Dict

def wilcoxon_signed_rank(scores_a: List[float], scores_b: List[float]) -> Tuple[float, float]:
    """Perform Wilcoxon signed-rank test to compare two models."""
    statistic, p_value = stats.wilcoxon(scores_a, scores_b)
    return float(p_value), float(statistic)

def paired_t_test(scores_a: List[float], scores_b: List[float]) -> float:
    """Perform paired t-test."""
    statistic, p_value = stats.ttest_rel(scores_a, scores_b)
    return float(p_value)

def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute mean and confidence interval for given data."""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return float(m), float(m - h), float(m + h)

def bootstrap_ci(data: List[float], n_bootstrap: int = 10000, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval."""
    a = np.array(data)
    bootstrapped_means = np.random.choice(a, size=(n_bootstrap, len(a)), replace=True).mean(axis=1)
    m = np.mean(data)
    lower = np.percentile(bootstrapped_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrapped_means, (1 + confidence) / 2 * 100)
    return float(m), float(lower), float(upper)

def effect_size_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    d = (np.mean(group1) - np.mean(group2)) / np.sqrt(pooled_var)
    return float(d)

def multi_seed_summary(results_per_seed: List[Dict]) -> Dict:
    """Summarize results across multiple seeds."""
    summary = {}
    if not results_per_seed:
        return summary
        
    keys = results_per_seed[0].keys()
    for key in keys:
        try:
            values = [res[key] for res in results_per_seed if key in res]
            mean, lower, upper = compute_confidence_interval(values)
            summary[key] = {
                'mean': mean,
                'ci_lower': lower,
                'ci_upper': upper,
                'std': float(np.std(values))
            }
        except:
            pass
    return summary
