#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention vs SHAP Comparison Analysis for Thesis.

This script provides comprehensive statistical analysis comparing attention
weights with SHAP values across multiple samples. Designed for thesis-ready
outputs including tables, figures, and statistical tests.

Features:
- Aggregate analysis across all samples in a batch
- Statistical significance testing (correlation, paired tests)
- Publication-ready figures and tables
- LaTeX-formatted output for thesis inclusion

Usage:
    # Run comparison on existing SHAP analysis results
    python compare_attention_shap.py --dataset BPIC2012-O --results_dir outputs/BPIC2012-O/shap_batch_analysis

    # Run full pipeline (attention + SHAP + comparison)
    python compare_attention_shap.py --dataset BPIC2012-O --full_pipeline --batch_size 32
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SampleComparison:
    """Comparison data for a single sample."""
    batch_index: int
    case_id: str
    prefix_length: int
    predicted_label: str
    predicted_prob: float

    shap_values: np.ndarray
    attention_weights: Optional[np.ndarray]

    # Correlation metrics
    pearson_attn_phi: float = np.nan
    spearman_attn_phi: float = np.nan
    pearson_attn_absphi: float = np.nan
    spearman_attn_absphi: float = np.nan

    # Top-k overlap
    top3_overlap: float = np.nan
    top5_overlap: float = np.nan

    # Feature rankings
    top_attn_indices: List[int] = field(default_factory=list)
    top_shap_indices: List[int] = field(default_factory=list)


@dataclass
class ComparisonSummary:
    """Aggregated comparison summary."""
    dataset: str
    n_samples: int
    n_with_attention: int

    # Correlation statistics
    pearson_attn_absphi_mean: float
    pearson_attn_absphi_std: float
    pearson_attn_absphi_ci95: Tuple[float, float]

    spearman_attn_absphi_mean: float
    spearman_attn_absphi_std: float
    spearman_attn_absphi_ci95: Tuple[float, float]

    pearson_attn_phi_mean: float
    pearson_attn_phi_std: float

    # Top-k overlap statistics
    top3_overlap_mean: float
    top3_overlap_std: float
    top5_overlap_mean: float
    top5_overlap_std: float

    # Statistical tests
    correlation_pvalue: float  # Test if mean correlation > 0
    paired_test_pvalue: float  # Test if attention predicts SHAP rankings


# =============================================================================
# Analysis Functions
# =============================================================================

def load_batch_results(results_path: str) -> Tuple[dict, List[dict]]:
    """Load batch analysis results from JSON."""
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    individual = data.get("individual_results", [])

    return summary, individual


def extract_comparisons(individual_results: List[dict]) -> List[SampleComparison]:
    """Extract comparison data from individual results."""
    comparisons = []

    for r in individual_results:
        ac = r.get("attention_comparison", {})
        has_attn = ac.get("attention_available", False)

        comp = SampleComparison(
            batch_index=r.get("batch_index", -1),
            case_id=str(r.get("case_id", "")),
            prefix_length=r.get("prefix_length", 0),
            predicted_label=r.get("model_predicted_label", ""),
            predicted_prob=r.get("model_predicted_prob", 0.0),
            shap_values=np.array(r.get("shap_values", [])),
            attention_weights=np.array(ac.get("attention_weights", [])) if has_attn else None,
        )

        if has_attn:
            comp.pearson_attn_phi = ac.get("pearson_attn_vs_phi", np.nan)
            comp.spearman_attn_phi = ac.get("spearman_attn_vs_phi", np.nan)
            comp.pearson_attn_absphi = ac.get("pearson_attn_vs_absphi", np.nan)
            comp.spearman_attn_absphi = ac.get("spearman_attn_vs_absphi", np.nan)
            comp.top3_overlap = ac.get("top3_overlap_ratio_attn_absphi", np.nan)
            comp.top5_overlap = ac.get("top5_overlap_ratio_attn_absphi", np.nan)

        comparisons.append(comp)

    return comparisons


def compute_ci95(values: np.ndarray) -> Tuple[float, float]:
    """Compute 95% confidence interval using bootstrap."""
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return (np.nan, np.nan)

    n_bootstrap = 1000
    means = []
    rng = np.random.default_rng(42)

    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))

    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def one_sample_ttest(values: np.ndarray, null_value: float = 0) -> float:
    """One-sample t-test p-value (two-tailed)."""
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return np.nan

    try:
        from scipy import stats
        _, pval = stats.ttest_1samp(values, null_value)
        return float(pval)
    except ImportError:
        # Manual calculation
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        se = std / np.sqrt(n)
        t = (mean - null_value) / se

        # Approximate p-value using normal distribution
        return float(2 * (1 - 0.5 * (1 + np.tanh(t / np.sqrt(2)))))


def compute_summary(
    comparisons: List[SampleComparison],
    dataset: str
) -> ComparisonSummary:
    """Compute aggregate summary statistics."""
    n_samples = len(comparisons)
    with_attn = [c for c in comparisons if c.attention_weights is not None]
    n_with_attn = len(with_attn)

    if n_with_attn == 0:
        return ComparisonSummary(
            dataset=dataset,
            n_samples=n_samples,
            n_with_attention=0,
            pearson_attn_absphi_mean=np.nan,
            pearson_attn_absphi_std=np.nan,
            pearson_attn_absphi_ci95=(np.nan, np.nan),
            spearman_attn_absphi_mean=np.nan,
            spearman_attn_absphi_std=np.nan,
            spearman_attn_absphi_ci95=(np.nan, np.nan),
            pearson_attn_phi_mean=np.nan,
            pearson_attn_phi_std=np.nan,
            top3_overlap_mean=np.nan,
            top3_overlap_std=np.nan,
            top5_overlap_mean=np.nan,
            top5_overlap_std=np.nan,
            correlation_pvalue=np.nan,
            paired_test_pvalue=np.nan,
        )

    # Extract metrics
    pearson_absphi = np.array([c.pearson_attn_absphi for c in with_attn])
    spearman_absphi = np.array([c.spearman_attn_absphi for c in with_attn])
    pearson_phi = np.array([c.pearson_attn_phi for c in with_attn])
    top3 = np.array([c.top3_overlap for c in with_attn])
    top5 = np.array([c.top5_overlap for c in with_attn])

    # Filter NaN
    valid_pearson = pearson_absphi[~np.isnan(pearson_absphi)]
    valid_spearman = spearman_absphi[~np.isnan(spearman_absphi)]
    valid_phi = pearson_phi[~np.isnan(pearson_phi)]
    valid_top3 = top3[~np.isnan(top3)]
    valid_top5 = top5[~np.isnan(top5)]

    return ComparisonSummary(
        dataset=dataset,
        n_samples=n_samples,
        n_with_attention=n_with_attn,
        pearson_attn_absphi_mean=float(np.mean(valid_pearson)) if len(valid_pearson) > 0 else np.nan,
        pearson_attn_absphi_std=float(np.std(valid_pearson)) if len(valid_pearson) > 0 else np.nan,
        pearson_attn_absphi_ci95=compute_ci95(valid_pearson),
        spearman_attn_absphi_mean=float(np.mean(valid_spearman)) if len(valid_spearman) > 0 else np.nan,
        spearman_attn_absphi_std=float(np.std(valid_spearman)) if len(valid_spearman) > 0 else np.nan,
        spearman_attn_absphi_ci95=compute_ci95(valid_spearman),
        pearson_attn_phi_mean=float(np.mean(valid_phi)) if len(valid_phi) > 0 else np.nan,
        pearson_attn_phi_std=float(np.std(valid_phi)) if len(valid_phi) > 0 else np.nan,
        top3_overlap_mean=float(np.mean(valid_top3)) if len(valid_top3) > 0 else np.nan,
        top3_overlap_std=float(np.std(valid_top3)) if len(valid_top3) > 0 else np.nan,
        top5_overlap_mean=float(np.mean(valid_top5)) if len(valid_top5) > 0 else np.nan,
        top5_overlap_std=float(np.std(valid_top5)) if len(valid_top5) > 0 else np.nan,
        correlation_pvalue=one_sample_ttest(valid_pearson, 0),
        paired_test_pvalue=one_sample_ttest(valid_spearman, 0),
    )


# =============================================================================
# Visualization
# =============================================================================

def plot_correlation_distribution(
    comparisons: List[SampleComparison],
    out_dir: str,
    dataset: str
):
    """Plot distribution of correlation coefficients."""
    with_attn = [c for c in comparisons if c.attention_weights is not None]

    pearson_absphi = [c.pearson_attn_absphi for c in with_attn if not np.isnan(c.pearson_attn_absphi)]
    spearman_absphi = [c.spearman_attn_absphi for c in with_attn if not np.isnan(c.spearman_attn_absphi)]
    pearson_phi = [c.pearson_attn_phi for c in with_attn if not np.isnan(c.pearson_attn_phi)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Pearson (attn vs |SHAP|)
    axes[0].hist(pearson_absphi, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(np.mean(pearson_absphi), color='red', linestyle='--',
                    label=f'Mean={np.mean(pearson_absphi):.3f}')
    axes[0].axvline(0, color='gray', linestyle=':')
    axes[0].set_xlabel('Pearson Correlation')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Attention vs |SHAP| (Pearson)')
    axes[0].legend()

    # Spearman (attn vs |SHAP|)
    axes[1].hist(spearman_absphi, bins=20, edgecolor='black', alpha=0.7, color='darkorange')
    axes[1].axvline(np.mean(spearman_absphi), color='red', linestyle='--',
                    label=f'Mean={np.mean(spearman_absphi):.3f}')
    axes[1].axvline(0, color='gray', linestyle=':')
    axes[1].set_xlabel('Spearman Correlation')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Attention vs |SHAP| (Spearman)')
    axes[1].legend()

    # Pearson (attn vs signed SHAP)
    axes[2].hist(pearson_phi, bins=20, edgecolor='black', alpha=0.7, color='seagreen')
    axes[2].axvline(np.mean(pearson_phi), color='red', linestyle='--',
                    label=f'Mean={np.mean(pearson_phi):.3f}')
    axes[2].axvline(0, color='gray', linestyle=':')
    axes[2].set_xlabel('Pearson Correlation')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Attention vs Signed SHAP (Pearson)')
    axes[2].legend()

    plt.suptitle(f'Correlation Distributions - {dataset}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'correlation_distributions.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, 'correlation_distributions.pdf'), bbox_inches='tight')
    plt.close()


def plot_overlap_analysis(
    comparisons: List[SampleComparison],
    out_dir: str,
    dataset: str
):
    """Plot top-k overlap analysis."""
    with_attn = [c for c in comparisons if c.attention_weights is not None]

    top3 = [c.top3_overlap for c in with_attn if not np.isnan(c.top3_overlap)]
    top5 = [c.top5_overlap for c in with_attn if not np.isnan(c.top5_overlap)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Top-3 overlap
    axes[0].hist(top3, bins=10, edgecolor='black', alpha=0.7, color='purple')
    axes[0].axvline(np.mean(top3), color='red', linestyle='--',
                    label=f'Mean={np.mean(top3):.3f}')
    axes[0].axvline(1/3, color='gray', linestyle=':', label='Random baseline')
    axes[0].set_xlabel('Overlap Ratio')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Top-3 Feature Overlap (Attention vs |SHAP|)')
    axes[0].legend()

    # Top-5 overlap
    axes[1].hist(top5, bins=10, edgecolor='black', alpha=0.7, color='teal')
    axes[1].axvline(np.mean(top5), color='red', linestyle='--',
                    label=f'Mean={np.mean(top5):.3f}')
    axes[1].axvline(1/5, color='gray', linestyle=':', label='Random baseline')
    axes[1].set_xlabel('Overlap Ratio')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Top-5 Feature Overlap (Attention vs |SHAP|)')
    axes[1].legend()

    plt.suptitle(f'Top-k Overlap Analysis - {dataset}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'overlap_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, 'overlap_analysis.pdf'), bbox_inches='tight')
    plt.close()


def plot_correlation_vs_length(
    comparisons: List[SampleComparison],
    out_dir: str,
    dataset: str
):
    """Plot correlation as a function of prefix length."""
    with_attn = [c for c in comparisons if c.attention_weights is not None]

    lengths = [c.prefix_length for c in with_attn]
    pearson = [c.pearson_attn_absphi for c in with_attn]
    spearman = [c.spearman_attn_absphi for c in with_attn]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(lengths, pearson, alpha=0.6, label='Pearson', color='steelblue')
    ax.scatter(lengths, spearman, alpha=0.6, label='Spearman', color='darkorange', marker='^')

    # Add trend lines
    valid_p = [(l, p) for l, p in zip(lengths, pearson) if not np.isnan(p)]
    valid_s = [(l, s) for l, s in zip(lengths, spearman) if not np.isnan(s)]

    if len(valid_p) > 2:
        z = np.polyfit([x[0] for x in valid_p], [x[1] for x in valid_p], 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(lengths), max(lengths), 100)
        ax.plot(x_line, p(x_line), '--', color='steelblue', alpha=0.5)

    if len(valid_s) > 2:
        z = np.polyfit([x[0] for x in valid_s], [x[1] for x in valid_s], 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(lengths), max(lengths), 100)
        ax.plot(x_line, p(x_line), '--', color='darkorange', alpha=0.5)

    ax.axhline(0, color='gray', linestyle=':')
    ax.set_xlabel('Prefix Length')
    ax.set_ylabel('Correlation (Attention vs |SHAP|)')
    ax.set_title(f'Correlation vs Prefix Length - {dataset}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'correlation_vs_length.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, 'correlation_vs_length.pdf'), bbox_inches='tight')
    plt.close()


def plot_scatter_examples(
    comparisons: List[SampleComparison],
    out_dir: str,
    n_examples: int = 4
):
    """Plot scatter examples of attention vs SHAP for individual samples."""
    with_attn = [c for c in comparisons if c.attention_weights is not None]

    # Select diverse examples
    if len(with_attn) <= n_examples:
        selected = with_attn
    else:
        # Select samples with different correlation values
        sorted_by_corr = sorted(with_attn, key=lambda c: c.pearson_attn_absphi if not np.isnan(c.pearson_attn_absphi) else 0)
        indices = np.linspace(0, len(sorted_by_corr)-1, n_examples, dtype=int)
        selected = [sorted_by_corr[i] for i in indices]

    n_cols = 2
    n_rows = (len(selected) + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if len(selected) == 1 else axes

    for idx, comp in enumerate(selected):
        ax = axes[idx]
        attn = comp.attention_weights
        absphi = np.abs(comp.shap_values)

        ax.scatter(attn, absphi, alpha=0.7, s=60)

        # Add labels for top events
        for i in range(len(attn)):
            ax.annotate(f'E{i+1}', (attn[i], absphi[i]), fontsize=8, alpha=0.7)

        ax.set_xlabel('Attention Weight')
        ax.set_ylabel('|SHAP Value|')
        ax.set_title(
            f'Sample {comp.batch_index} (L={comp.prefix_length})\n'
            f'r={comp.pearson_attn_absphi:.3f}, ρ={comp.spearman_attn_absphi:.3f}'
        )

    # Hide unused axes
    for idx in range(len(selected), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'scatter_examples.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, 'scatter_examples.pdf'), bbox_inches='tight')
    plt.close()


def plot_summary_figure(
    summary: ComparisonSummary,
    out_dir: str
):
    """Create a summary figure with key metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Correlation comparison
    metrics = ['Pearson\n(attn vs |φ|)', 'Spearman\n(attn vs |φ|)', 'Pearson\n(attn vs φ)']
    means = [summary.pearson_attn_absphi_mean, summary.spearman_attn_absphi_mean, summary.pearson_attn_phi_mean]
    stds = [summary.pearson_attn_absphi_std, summary.spearman_attn_absphi_std, summary.pearson_attn_phi_std]

    colors = ['steelblue', 'darkorange', 'seagreen']
    bars = axes[0].bar(metrics, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    axes[0].axhline(0, color='gray', linestyle=':')
    axes[0].set_ylabel('Correlation')
    axes[0].set_title('Mean Correlation Coefficients')
    axes[0].set_ylim(-0.5, 1.0)

    # Top-k overlap
    overlap_metrics = ['Top-3', 'Top-5']
    overlap_means = [summary.top3_overlap_mean, summary.top5_overlap_mean]
    overlap_stds = [summary.top3_overlap_std, summary.top5_overlap_std]
    random_baselines = [1/3, 1/5]

    x = np.arange(len(overlap_metrics))
    width = 0.35

    axes[1].bar(x - width/2, overlap_means, width, yerr=overlap_stds, capsize=5,
                label='Observed', color='purple', alpha=0.7, edgecolor='black')
    axes[1].bar(x + width/2, random_baselines, width,
                label='Random', color='gray', alpha=0.5, edgecolor='black')
    axes[1].set_ylabel('Overlap Ratio')
    axes[1].set_title('Top-k Feature Overlap')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(overlap_metrics)
    axes[1].legend()
    axes[1].set_ylim(0, 1)

    # Sample counts and p-values
    text_content = (
        f"Dataset: {summary.dataset}\n\n"
        f"Samples analyzed: {summary.n_samples}\n"
        f"With attention: {summary.n_with_attention}\n\n"
        f"Statistical Tests:\n"
        f"  Pearson ≠ 0: p = {summary.correlation_pvalue:.4f}\n"
        f"  Spearman ≠ 0: p = {summary.paired_test_pvalue:.4f}\n\n"
        f"95% CI (Pearson):\n"
        f"  [{summary.pearson_attn_absphi_ci95[0]:.3f}, {summary.pearson_attn_absphi_ci95[1]:.3f}]"
    )

    axes[2].text(0.1, 0.5, text_content, transform=axes[2].transAxes,
                 fontsize=11, verticalalignment='center', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[2].axis('off')
    axes[2].set_title('Summary Statistics')

    plt.suptitle(f'Attention vs SHAP Comparison Summary - {summary.dataset}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'summary_figure.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, 'summary_figure.pdf'), bbox_inches='tight')
    plt.close()


# =============================================================================
# Output Generation
# =============================================================================

def generate_latex_table(summary: ComparisonSummary, out_dir: str):
    """Generate LaTeX table for thesis inclusion."""
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Attention vs SHAP Value Comparison for %(dataset)s}
\label{tab:attn_shap_comparison}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Mean $\pm$ Std} & \textbf{95\%% CI} \\
\midrule
Pearson (Attn vs $|\phi|$) & $%(pearson_mean).3f \pm %(pearson_std).3f$ & $[%(pearson_ci_lo).3f, %(pearson_ci_hi).3f]$ \\
Spearman (Attn vs $|\phi|$) & $%(spearman_mean).3f \pm %(spearman_std).3f$ & $[%(spearman_ci_lo).3f, %(spearman_ci_hi).3f]$ \\
Pearson (Attn vs $\phi$) & $%(phi_mean).3f \pm %(phi_std).3f$ & -- \\
\midrule
Top-3 Overlap & $%(top3_mean).3f \pm %(top3_std).3f$ & -- \\
Top-5 Overlap & $%(top5_mean).3f \pm %(top5_std).3f$ & -- \\
\midrule
\multicolumn{3}{l}{\textit{Samples: %(n_samples)d total, %(n_attn)d with attention}} \\
\multicolumn{3}{l}{\textit{$H_0: \rho = 0$, p-value = %(pvalue).4f}} \\
\bottomrule
\end{tabular}
\end{table}
""" % {
        'dataset': summary.dataset,
        'pearson_mean': summary.pearson_attn_absphi_mean,
        'pearson_std': summary.pearson_attn_absphi_std,
        'pearson_ci_lo': summary.pearson_attn_absphi_ci95[0],
        'pearson_ci_hi': summary.pearson_attn_absphi_ci95[1],
        'spearman_mean': summary.spearman_attn_absphi_mean,
        'spearman_std': summary.spearman_attn_absphi_std,
        'spearman_ci_lo': summary.spearman_attn_absphi_ci95[0],
        'spearman_ci_hi': summary.spearman_attn_absphi_ci95[1],
        'phi_mean': summary.pearson_attn_phi_mean,
        'phi_std': summary.pearson_attn_phi_std,
        'top3_mean': summary.top3_overlap_mean,
        'top3_std': summary.top3_overlap_std,
        'top5_mean': summary.top5_overlap_mean,
        'top5_std': summary.top5_overlap_std,
        'n_samples': summary.n_samples,
        'n_attn': summary.n_with_attention,
        'pvalue': summary.correlation_pvalue,
    }

    with open(os.path.join(out_dir, 'comparison_table.tex'), 'w') as f:
        f.write(latex)


def generate_report(
    summary: ComparisonSummary,
    comparisons: List[SampleComparison],
    out_dir: str
):
    """Generate comprehensive report."""
    report = {
        "summary": {
            "dataset": summary.dataset,
            "n_samples": summary.n_samples,
            "n_with_attention": summary.n_with_attention,
            "correlation_metrics": {
                "pearson_attn_vs_absphi": {
                    "mean": summary.pearson_attn_absphi_mean,
                    "std": summary.pearson_attn_absphi_std,
                    "ci95_lower": summary.pearson_attn_absphi_ci95[0],
                    "ci95_upper": summary.pearson_attn_absphi_ci95[1],
                },
                "spearman_attn_vs_absphi": {
                    "mean": summary.spearman_attn_absphi_mean,
                    "std": summary.spearman_attn_absphi_std,
                    "ci95_lower": summary.spearman_attn_absphi_ci95[0],
                    "ci95_upper": summary.spearman_attn_absphi_ci95[1],
                },
                "pearson_attn_vs_phi": {
                    "mean": summary.pearson_attn_phi_mean,
                    "std": summary.pearson_attn_phi_std,
                },
            },
            "overlap_metrics": {
                "top3": {"mean": summary.top3_overlap_mean, "std": summary.top3_overlap_std},
                "top5": {"mean": summary.top5_overlap_mean, "std": summary.top5_overlap_std},
            },
            "statistical_tests": {
                "correlation_pvalue": summary.correlation_pvalue,
                "paired_test_pvalue": summary.paired_test_pvalue,
            },
        },
        "interpretation": generate_interpretation(summary),
    }

    with open(os.path.join(out_dir, 'comparison_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=lambda x: None if np.isnan(x) else x)


def generate_interpretation(summary: ComparisonSummary) -> str:
    """Generate natural language interpretation of results."""
    parts = []

    # Correlation interpretation
    if not np.isnan(summary.pearson_attn_absphi_mean):
        if summary.pearson_attn_absphi_mean > 0.5:
            strength = "strong positive"
        elif summary.pearson_attn_absphi_mean > 0.3:
            strength = "moderate positive"
        elif summary.pearson_attn_absphi_mean > 0.1:
            strength = "weak positive"
        elif summary.pearson_attn_absphi_mean > -0.1:
            strength = "negligible"
        else:
            strength = "negative"

        parts.append(
            f"The analysis reveals a {strength} correlation (r={summary.pearson_attn_absphi_mean:.3f}) "
            f"between attention weights and absolute SHAP values across {summary.n_with_attention} samples."
        )

    # Statistical significance
    if not np.isnan(summary.correlation_pvalue):
        if summary.correlation_pvalue < 0.001:
            parts.append("This correlation is highly statistically significant (p < 0.001).")
        elif summary.correlation_pvalue < 0.05:
            parts.append(f"This correlation is statistically significant (p = {summary.correlation_pvalue:.4f}).")
        else:
            parts.append(f"This correlation is not statistically significant (p = {summary.correlation_pvalue:.4f}).")

    # Overlap interpretation
    if not np.isnan(summary.top5_overlap_mean):
        random_baseline = 0.2  # For top-5
        if summary.top5_overlap_mean > 2 * random_baseline:
            parts.append(
                f"The top-5 feature overlap ({summary.top5_overlap_mean:.1%}) is substantially "
                f"above random chance ({random_baseline:.1%}), indicating attention identifies "
                f"similar important features as SHAP."
            )
        elif summary.top5_overlap_mean > random_baseline:
            parts.append(
                f"The top-5 feature overlap ({summary.top5_overlap_mean:.1%}) is moderately "
                f"above random chance ({random_baseline:.1%})."
            )
        else:
            parts.append(
                f"The top-5 feature overlap ({summary.top5_overlap_mean:.1%}) is near or below "
                f"random chance ({random_baseline:.1%}), suggesting attention and SHAP identify "
                f"different important features."
            )

    # Signed vs unsigned
    if not np.isnan(summary.pearson_attn_phi_mean) and not np.isnan(summary.pearson_attn_absphi_mean):
        diff = summary.pearson_attn_absphi_mean - summary.pearson_attn_phi_mean
        if diff > 0.1:
            parts.append(
                "Attention correlates more strongly with absolute SHAP values than signed values, "
                "suggesting attention captures importance magnitude but not direction."
            )

    return " ".join(parts)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare Attention and SHAP values for thesis analysis"
    )
    parser.add_argument("--dataset", default="BPIC2012-O", help="Dataset name")
    parser.add_argument("--results_dir", default=None,
                        help="Directory containing SHAP batch analysis results")
    parser.add_argument("--out_dir", default=None, help="Output directory for comparison results")
    parser.add_argument("--full_pipeline", action="store_true",
                        help="Run full pipeline (attention extraction + SHAP + comparison)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for full pipeline")
    parser.add_argument("--repo_root", default=".", help="Repository root path")

    args = parser.parse_args()

    # Set default paths
    if args.results_dir is None:
        args.results_dir = os.path.join(args.repo_root, "outputs", args.dataset, "shap_batch_analysis")

    if args.out_dir is None:
        args.out_dir = os.path.join(args.repo_root, "outputs", args.dataset, "attention_shap_comparison")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Attention vs SHAP Comparison Analysis")
    print(f"Dataset: {args.dataset}")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.out_dir}")
    print()

    # Check if we need to run SHAP analysis first
    results_file = os.path.join(args.results_dir, "shap_analysis_batch.json")

    if args.full_pipeline or not os.path.exists(results_file):
        print("Running SHAP batch analysis first...")
        import subprocess
        # Use the deletion-only SHAP script (preferred for proper Shapley semantics)
        cmd = [
            "python", "shapley_ppm_deletion_only_shap_pkg.py",
            "--dataset", args.dataset,
            "--batch_mode",
            "--start_idx", "0",
            "--end_idx", str(args.batch_size),
            "--repo_root", args.repo_root,
        ]
        subprocess.run(cmd, check=True)
        # Update results_dir to point to the deletion script's output
        args.results_dir = os.path.join(args.repo_root, "outputs", args.dataset, "shap_deletion_batch_analysis")
        results_file = os.path.join(args.results_dir, "shap_analysis_batch.json")
        print()

    # Load results
    print("Loading SHAP analysis results...")
    summary_data, individual_results = load_batch_results(results_file)

    # Extract comparisons
    comparisons = extract_comparisons(individual_results)
    print(f"Loaded {len(comparisons)} samples")

    # Compute summary
    summary = compute_summary(comparisons, args.dataset)

    # Generate outputs
    print("\nGenerating visualizations...")
    plot_correlation_distribution(comparisons, args.out_dir, args.dataset)
    plot_overlap_analysis(comparisons, args.out_dir, args.dataset)
    plot_correlation_vs_length(comparisons, args.out_dir, args.dataset)
    plot_scatter_examples(comparisons, args.out_dir)
    plot_summary_figure(summary, args.out_dir)

    print("Generating reports...")
    generate_latex_table(summary, args.out_dir)
    generate_report(summary, comparisons, args.out_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("ATTENTION vs SHAP COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Dataset: {summary.dataset}")
    print(f"Samples: {summary.n_samples} total, {summary.n_with_attention} with attention")
    print()
    print("CORRELATION METRICS:")
    print(f"  Pearson (attn vs |φ|):  {summary.pearson_attn_absphi_mean:.3f} ± {summary.pearson_attn_absphi_std:.3f}")
    print(f"  95% CI: [{summary.pearson_attn_absphi_ci95[0]:.3f}, {summary.pearson_attn_absphi_ci95[1]:.3f}]")
    print(f"  Spearman (attn vs |φ|): {summary.spearman_attn_absphi_mean:.3f} ± {summary.spearman_attn_absphi_std:.3f}")
    print(f"  Pearson (attn vs φ):    {summary.pearson_attn_phi_mean:.3f} ± {summary.pearson_attn_phi_std:.3f}")
    print()
    print("TOP-K OVERLAP:")
    print(f"  Top-3: {summary.top3_overlap_mean:.3f} ± {summary.top3_overlap_std:.3f} (random: 0.333)")
    print(f"  Top-5: {summary.top5_overlap_mean:.3f} ± {summary.top5_overlap_std:.3f} (random: 0.200)")
    print()
    print("STATISTICAL TESTS:")
    print(f"  H0: correlation = 0, p-value = {summary.correlation_pvalue:.4f}")
    print()
    print("INTERPRETATION:")
    print(generate_interpretation(summary))
    print()
    print(f"Outputs saved to: {args.out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
