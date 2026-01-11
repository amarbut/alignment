#!/usr/bin/env python3
"""
Compare the performance of the selected expert vs all evaluated experts.

This script analyzes the results from run_all_experts_evaluation.py and compares
them with your selected expert to validate the selection process.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare selected expert with all evaluated experts"
    )
    parser.add_argument(
        'all_experts_dir',
        type=str,
        help='Directory containing all experts evaluation results'
    )
    parser.add_argument(
        'selected_expert_dir',
        type=str,
        help='Directory containing selected expert results'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='substring_matching_asr',
        choices=['substring_matching_asr', 'llamaguard2_asr', 'openai_asr'],
        help='Metric to compare'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save plots and analysis (defaults to all_experts_dir)'
    )

    return parser.parse_args()


def load_all_experts_results(all_experts_dir: Path) -> List[Dict]:
    """Load results from all experts evaluation."""
    summary_file = all_experts_dir / "summary_all_experts.json"

    if summary_file.exists():
        with open(summary_file, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Summary file not found: {summary_file}")


def load_selected_expert_result(selected_expert_dir: Path) -> Dict:
    """Load result from the selected expert."""
    # Load metadata to get layer, expert_id, and position
    metadata_file = Path(selected_expert_dir) / "direction_metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Load evaluation
    eval_file = Path(selected_expert_dir) / "completions" / "actadd" / "jailbreakbench_evaluations.json"
    with open(eval_file, 'r') as f:
        evaluation = json.load(f)

    return {
        "layer": metadata["layer"],
        "expert_id": metadata["expert_id"],
        "position": metadata.get("pos", metadata.get("position", None)),  # Support both field names
        "substring_matching_asr": evaluation.get("substring_matching_success_rate", None),
        "llamaguard2_asr": evaluation.get("llamaguard2_success_rate", None),
        "openai_asr": evaluation.get("openai_success_rate", None),
    }


def calculate_statistics(all_results: List[Dict], metric: str) -> Dict:
    """Calculate statistics for the given metric."""
    values = [r[metric] for r in all_results if r[metric] is not None]

    if not values:
        return {}

    return {
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "q25": np.percentile(values, 25),
        "q75": np.percentile(values, 75),
    }


def calculate_percentile_rank(all_results: List[Dict], selected_value: float, metric: str) -> float:
    """Calculate what percentile the selected expert is at."""
    values = [r[metric] for r in all_results if r[metric] is not None]

    if not values:
        return None

    # Percentile rank = percentage of values below this value
    percentile = (sum(1 for v in values if v < selected_value) / len(values)) * 100
    return percentile


def print_comparison(all_results: List[Dict], selected_result: Dict, metric: str):
    """Print detailed comparison."""
    stats = calculate_statistics(all_results, metric)
    selected_value = selected_result[metric]

    if selected_value is None:
        print(f"ERROR: Selected expert has no {metric} value")
        return

    percentile = calculate_percentile_rank(all_results, selected_value, metric)

    print("\n" + "="*80)
    print("EXPERT SELECTION VALIDATION")
    print("="*80)

    print(f"\nSelected Expert:")
    print(f"  Layer: {selected_result['layer']}")
    print(f"  Expert ID: {selected_result['expert_id']}")
    if selected_result.get('position') is not None:
        print(f"  Position: {selected_result['position']}")
    print(f"  {metric}: {selected_value:.2%}")

    print(f"\nAll Experts Statistics ({metric}):")
    print(f"  Mean:     {stats['mean']:.2%}")
    print(f"  Median:   {stats['median']:.2%}")
    print(f"  Std Dev:  {stats['std']:.2%}")
    print(f"  Min:      {stats['min']:.2%}")
    print(f"  Max:      {stats['max']:.2%}")
    print(f"  25th %ile: {stats['q25']:.2%}")
    print(f"  75th %ile: {stats['q75']:.2%}")

    print(f"\nPerformance Comparison:")
    print(f"  Selected Expert Percentile: {percentile:.1f}th")
    print(f"  Above Mean: {'+' if selected_value > stats['mean'] else ''}{(selected_value - stats['mean'])*100:.2f} percentage points")
    print(f"  Above Median: {'+' if selected_value > stats['median'] else ''}{(selected_value - stats['median'])*100:.2f} percentage points")

    # Determine how good the selection is
    if percentile >= 95:
        quality = "EXCELLENT (top 5%)"
    elif percentile >= 90:
        quality = "VERY GOOD (top 10%)"
    elif percentile >= 75:
        quality = "GOOD (top 25%)"
    elif percentile >= 50:
        quality = "ABOVE AVERAGE"
    else:
        quality = "BELOW AVERAGE"

    print(f"\nSelection Quality: {quality}")

    # Calculate how much better than random
    improvement_over_mean = ((selected_value - stats['mean']) / stats['mean']) * 100
    print(f"Improvement over mean: {improvement_over_mean:+.1f}%")


def create_distribution_plot(all_results: List[Dict], selected_result: Dict, metric: str, output_path: Path):
    """Create histogram showing distribution with selected expert marked."""
    values = [r[metric] for r in all_results if r[metric] is not None]
    selected_value = selected_result[metric]

    plt.figure(figsize=(12, 6))

    # Histogram
    plt.hist(values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')

    # Mark selected expert
    plt.axvline(selected_value, color='red', linestyle='--', linewidth=2, label='Selected Expert')

    # Mark mean and median
    mean_val = np.mean(values)
    median_val = np.median(values)
    plt.axvline(mean_val, color='green', linestyle=':', linewidth=2, label='Mean')
    plt.axvline(median_val, color='orange', linestyle=':', linewidth=2, label='Median')

    plt.xlabel(metric.replace('_', ' ').title())
    plt.ylabel('Number of Experts')
    plt.title(f'Distribution of Expert Performance\nSelected Expert: Layer {selected_result["layer"]}, Expert {selected_result["expert_id"]}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def create_layer_heatmap(all_results: List[Dict], selected_result: Dict, metric: str, output_path: Path):
    """Create heatmap showing performance by layer and expert."""
    # Create matrix: layers x experts
    n_layers = 21
    n_experts = 32

    matrix = np.full((n_layers, n_experts), np.nan)

    for result in all_results:
        if result[metric] is not None:
            matrix[result['layer'], result['expert_id']] = result[metric]

    plt.figure(figsize=(16, 8))

    # Create heatmap
    sns.heatmap(matrix, cmap='RdYlGn', vmin=0, vmax=1, cbar_kws={'label': metric})

    # Mark selected expert
    plt.scatter(
        selected_result['expert_id'] + 0.5,
        selected_result['layer'] + 0.5,
        s=200, c='blue', marker='*', edgecolors='white', linewidths=2,
        label='Selected Expert'
    )

    plt.xlabel('Expert ID')
    plt.ylabel('Layer')
    plt.title(f'Expert Performance Heatmap ({metric})\nSelected: Layer {selected_result["layer"]}, Expert {selected_result["expert_id"]}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {output_path}")
    plt.close()


def main():
    args = parse_arguments()

    all_experts_dir = Path(args.all_experts_dir)
    selected_expert_dir = Path(args.selected_expert_dir)

    output_dir = Path(args.output_dir) if args.output_dir else all_experts_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print("Loading results...")
    all_results = load_all_experts_results(all_experts_dir)
    selected_result = load_selected_expert_result(selected_expert_dir)

    # Print comparison
    print_comparison(all_results, selected_result, args.metric)

    # Create plots
    print("\nGenerating visualizations...")

    try:
        distribution_plot = output_dir / f"distribution_{args.metric}.png"
        create_distribution_plot(all_results, selected_result, args.metric, distribution_plot)
    except Exception as e:
        print(f"Could not create distribution plot: {e}")

    try:
        heatmap_plot = output_dir / f"heatmap_{args.metric}.png"
        create_layer_heatmap(all_results, selected_result, args.metric, heatmap_plot)
    except Exception as e:
        print(f"Could not create heatmap: {e}")

    # Save detailed comparison to JSON
    comparison = {
        "metric": args.metric,
        "selected_expert": selected_result,
        "statistics": calculate_statistics(all_results, args.metric),
        "percentile_rank": calculate_percentile_rank(all_results, selected_result[args.metric], args.metric),
        "n_experts_evaluated": len(all_results),
    }

    comparison_path = output_dir / f"comparison_{args.metric}.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\nDetailed comparison saved to: {comparison_path}")


if __name__ == "__main__":
    main()
