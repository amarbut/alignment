"""
Analyze grid search results and create visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

def analyze_grid_search(csv_file):
    """Analyze grid search results from CSV file."""

    # Load results
    df = pd.read_csv(csv_file)

    print("="*80)
    print("GRID SEARCH ANALYSIS")
    print("="*80)
    print(f"\nLoaded {len(df)} results from {csv_file}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for intervention_type in ['refusal', 'response']:
        subset = df[df['intervention_type'] == intervention_type]
        dataset = subset['dataset'].iloc[0]

        print(f"\n{intervention_type.upper()} INDUCTION (on {dataset}):")
        print("-"*60)

        # Substring-matching
        print(f"\nSubstring-Matching Refusal Rates:")
        print(f"  Min:  {subset['substring_refusal_rate'].min():.4f}")
        print(f"  Max:  {subset['substring_refusal_rate'].max():.4f}")
        print(f"  Mean: {subset['substring_refusal_rate'].mean():.4f}")

        best_idx = subset['substring_refusal_rate'].idxmax()
        print(f"  Best: {subset.loc[best_idx, 'substring_refusal_rate']:.4f} " +
              f"(eps={subset.loc[best_idx, 'epsilon']}, thr={subset.loc[best_idx, 'threshold']})")

        # LlamaGuard2
        print(f"\nLlamaGuard2 Refusal Rates:")
        print(f"  Min:  {subset['llamaguard2_refusal_rate'].min():.4f}")
        print(f"  Max:  {subset['llamaguard2_refusal_rate'].max():.4f}")
        print(f"  Mean: {subset['llamaguard2_refusal_rate'].mean():.4f}")

        best_idx = subset['llamaguard2_refusal_rate'].idxmax()
        print(f"  Best: {subset.loc[best_idx, 'llamaguard2_refusal_rate']:.4f} " +
              f"(eps={subset.loc[best_idx, 'epsilon']}, thr={subset.loc[best_idx, 'threshold']})")

    # Pivot tables for heatmaps
    print("\n" + "="*80)
    print("PIVOT TABLES")
    print("="*80)

    for intervention_type in ['refusal', 'response']:
        subset = df[df['intervention_type'] == intervention_type]

        print(f"\n{intervention_type.upper()} - Substring-Matching Refusal Rate:")
        pivot = subset.pivot(index='epsilon', columns='threshold', values='substring_refusal_rate')
        print(pivot.to_string())

        print(f"\n{intervention_type.upper()} - LlamaGuard2 Refusal Rate:")
        pivot = subset.pivot(index='epsilon', columns='threshold', values='llamaguard2_refusal_rate')
        print(pivot.to_string())

    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    output_dir = Path(csv_file).parent / "grid_search_plots"
    output_dir.mkdir(exist_ok=True)

    # Heatmaps
    for intervention_type in ['refusal', 'response']:
        subset = df[df['intervention_type'] == intervention_type]

        # Substring-matching heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot = subset.pivot(index='epsilon', columns='threshold', values='substring_refusal_rate')
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax)
        ax.set_title(f'{intervention_type.title()} Induction - Substring-Matching Refusal Rate')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Epsilon')
        plt.tight_layout()
        plt.savefig(output_dir / f'{intervention_type}_substring_heatmap.png', dpi=150)
        plt.close()

        # LlamaGuard2 heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot = subset.pivot(index='epsilon', columns='threshold', values='llamaguard2_refusal_rate')
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax)
        ax.set_title(f'{intervention_type.title()} Induction - LlamaGuard2 Refusal Rate')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Epsilon')
        plt.tight_layout()
        plt.savefig(output_dir / f'{intervention_type}_llamaguard2_heatmap.png', dpi=150)
        plt.close()

    # Line plots by epsilon
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, intervention_type in enumerate(['refusal', 'response']):
        subset = df[df['intervention_type'] == intervention_type]

        # Substring
        ax = axes[i, 0]
        for threshold in sorted(subset['threshold'].unique()):
            data = subset[subset['threshold'] == threshold]
            ax.plot(data['epsilon'], data['substring_refusal_rate'],
                   marker='o', label=f'thr={threshold}')
        ax.set_xlabel('Epsilon')
        ax.set_ylabel('Refusal Rate')
        ax.set_title(f'{intervention_type.title()} - Substring-Matching')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # LlamaGuard2
        ax = axes[i, 1]
        for threshold in sorted(subset['threshold'].unique()):
            data = subset[subset['threshold'] == threshold]
            ax.plot(data['epsilon'], data['llamaguard2_refusal_rate'],
                   marker='o', label=f'thr={threshold}')
        ax.set_xlabel('Epsilon')
        ax.set_ylabel('Refusal Rate')
        ax.set_title(f'{intervention_type.title()} - LlamaGuard2')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'epsilon_trends.png', dpi=150)
    plt.close()

    print(f"\nSaved visualizations to: {output_dir}")
    print(f"  - {intervention_type}_substring_heatmap.png")
    print(f"  - {intervention_type}_llamaguard2_heatmap.png")
    print(f"  - epsilon_trends.png")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_grid_search.py <results.csv>")
        sys.exit(1)

    csv_file = sys.argv[1]
    analyze_grid_search(csv_file)
