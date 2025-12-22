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

    # Compute coherence metrics for response induction
    # When substring >> llamaguard2, likely producing gibberish (responds but not actually harmful)
    response_mask = df['intervention_type'] == 'response'
    df.loc[response_mask, 'jailbreak_divergence'] = (
        df.loc[response_mask, 'substring_jailbreak_rate'] -
        df.loc[response_mask, 'llamaguard2_jailbreak_rate']
    )
    df.loc[response_mask, 'likely_nonsense'] = df.loc[response_mask, 'jailbreak_divergence'] > 0.5
    df.loc[response_mask, 'coherent_jailbreak_rate'] = df.loc[response_mask, 'llamaguard2_jailbreak_rate']

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

    # Coherence analysis for response induction
    print("\n" + "="*80)
    print("COHERENCE ANALYSIS (Response Induction)")
    print("="*80)
    print("\nDetecting nonsense outputs using divergence heuristic:")
    print("  Heuristic: substring_jailbreak >> llamaguard2_jailbreak indicates gibberish")
    print("  (Model responds without refusal strings, but content is safe = nonsense)")

    response_subset = df[df['intervention_type'] == 'response'].copy()
    nonsense_count = response_subset['likely_nonsense'].sum()
    total_count = len(response_subset)

    print(f"\nResults flagged as likely nonsense: {nonsense_count}/{total_count} ({100*nonsense_count/total_count:.1f}%)")
    print(f"\nDivergence statistics (substring_jailbreak - llamaguard2_jailbreak):")
    print(f"  Min:  {response_subset['jailbreak_divergence'].min():.4f}")
    print(f"  Max:  {response_subset['jailbreak_divergence'].max():.4f}")
    print(f"  Mean: {response_subset['jailbreak_divergence'].mean():.4f}")
    print(f"  Cases with divergence > 0.5: {nonsense_count}")

    print(f"\n{'Epsilon':<10} {'Threshold':<12} {'Substring JB':<15} {'LlamaG2 JB':<15} {'Divergence':<12} {'Nonsense?'}")
    print("-"*80)
    for _, row in response_subset.iterrows():
        nonsense_flag = "⚠ YES" if row['likely_nonsense'] else "  no"
        print(f"{row['epsilon']:<10} {row['threshold']:<12} "
              f"{row['substring_jailbreak_rate']:<15.4f} {row['llamaguard2_jailbreak_rate']:<15.4f} "
              f"{row['jailbreak_divergence']:<12.4f} {nonsense_flag}")

    print(f"\nCoherent Jailbreak Rate (using LlamaGuard2 as ground truth):")
    print(f"  Max coherent jailbreak rate: {response_subset['coherent_jailbreak_rate'].max():.4f}")
    if response_subset['coherent_jailbreak_rate'].max() > 0:
        best_idx = response_subset['coherent_jailbreak_rate'].idxmax()
        print(f"    (eps={response_subset.loc[best_idx, 'epsilon']}, thr={response_subset.loc[best_idx, 'threshold']})")
    else:
        print(f"  → No coherent jailbreaks found (all successful jailbreaks appear to be nonsense)")

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

    # Coherent jailbreak heatmap for response induction
    response_subset = df[df['intervention_type'] == 'response']
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot = response_subset.pivot(index='epsilon', columns='threshold', values='coherent_jailbreak_rate')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, vmin=0, vmax=1)
    ax.set_title('Response Induction - Coherent Jailbreak Rate (LlamaGuard2)\nFilters out nonsense/gibberish outputs')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Epsilon')
    plt.tight_layout()
    plt.savefig(output_dir / 'response_coherent_jailbreak_heatmap.png', dpi=150)
    plt.close()

    # Divergence heatmap (to visualize nonsense distribution)
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot = response_subset.pivot(index='epsilon', columns='threshold', values='jailbreak_divergence')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='coolwarm', ax=ax, center=0)
    ax.set_title('Response Induction - Jailbreak Divergence\n(Substring - LlamaGuard2; >0.5 indicates likely nonsense)')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Epsilon')
    plt.tight_layout()
    plt.savefig(output_dir / 'response_divergence_heatmap.png', dpi=150)
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
    print(f"  - refusal_substring_heatmap.png")
    print(f"  - refusal_llamaguard2_heatmap.png")
    print(f"  - response_substring_heatmap.png")
    print(f"  - response_llamaguard2_heatmap.png")
    print(f"  - response_coherent_jailbreak_heatmap.png (NEW: filters out nonsense)")
    print(f"  - response_divergence_heatmap.png (NEW: shows where nonsense occurs)")
    print(f"  - epsilon_trends.png")

    # Save enhanced dataframe with coherence metrics
    enhanced_csv = Path(csv_file).parent / (Path(csv_file).stem + "_with_coherence.csv")
    df.to_csv(enhanced_csv, index=False)
    print(f"\nSaved enhanced CSV with coherence metrics to: {enhanced_csv}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_grid_search.py <results.csv>")
        sys.exit(1)

    csv_file = sys.argv[1]
    analyze_grid_search(csv_file)
