"""
Exploratory analysis script for expert routing patterns.

This script can be run locally after downloading expert_routing_data.json
to perform initial exploratory data analysis.
"""

import json
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import argparse

def load_routing_data(filepath):
    """Load the expert routing data."""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_expert_distribution(results, label):
    """Analyze the distribution of expert selections."""
    print(f"\n{'='*60}")
    print(f"Analysis for {label.upper()} prompts")
    print(f"{'='*60}")

    # Get layer info
    if not results:
        print("No results found!")
        return

    num_layers = len(results[0]["layer_routing"])
    print(f"Number of prompts: {len(results)}")
    print(f"Number of MoE layers: {num_layers}")

    # Analyze each layer
    for layer_idx in range(num_layers):
        layer_key = f"layer_{layer_idx}"
        print(f"\n--- Layer {layer_idx} ---")

        # Collect all expert selections across all prompts for this layer
        all_experts = []
        for result in results:
            top_experts = result["layer_routing"][layer_key]["top_expert"]
            all_experts.extend(top_experts)

        # Count expert usage
        expert_counts = Counter(all_experts)
        num_experts = results[0]["layer_routing"][layer_key]["num_experts"]

        print(f"Number of experts: {num_experts}")
        print(f"Most common experts (top 10):")
        for expert_id, count in expert_counts.most_common(10):
            percentage = (count / len(all_experts)) * 100
            print(f"  Expert {expert_id}: {count} times ({percentage:.2f}%)")

        # Calculate entropy (diversity of expert usage)
        probs = np.array([expert_counts.get(i, 0) for i in range(num_experts)])
        probs = probs / probs.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        print(f"Expert selection entropy: {entropy:.3f} (max: {np.log(num_experts):.3f})")

        # Calculate how many experts are actually used
        active_experts = len([c for c in expert_counts.values() if c > 0])
        print(f"Active experts: {active_experts}/{num_experts} ({100*active_experts/num_experts:.1f}%)")

    return expert_counts

def compare_distributions(harmful_results, harmless_results):
    """Compare expert distributions between harmful and harmless prompts."""
    print(f"\n{'='*60}")
    print("COMPARATIVE ANALYSIS: Harmful vs Harmless")
    print(f"{'='*60}")

    if not harmful_results or not harmless_results:
        print("Need both harmful and harmless results!")
        return

    num_layers = len(harmful_results[0]["layer_routing"])

    for layer_idx in range(num_layers):
        layer_key = f"layer_{layer_idx}"
        print(f"\n--- Layer {layer_idx} ---")

        # Collect expert selections
        harmful_experts = []
        harmless_experts = []

        for result in harmful_results:
            harmful_experts.extend(result["layer_routing"][layer_key]["top_expert"])

        for result in harmless_results:
            harmless_experts.extend(result["layer_routing"][layer_key]["top_expert"])

        harmful_counts = Counter(harmful_experts)
        harmless_counts = Counter(harmless_experts)

        # Normalize to get frequencies
        harmful_total = sum(harmful_counts.values())
        harmless_total = sum(harmless_counts.values())

        # Find experts with biggest differences
        num_experts = harmful_results[0]["layer_routing"][layer_key]["num_experts"]
        differences = {}
        for expert_id in range(num_experts):
            harmful_freq = harmful_counts.get(expert_id, 0) / harmful_total
            harmless_freq = harmless_counts.get(expert_id, 0) / harmless_total
            differences[expert_id] = harmful_freq - harmless_freq

        # Sort by absolute difference
        sorted_diffs = sorted(differences.items(), key=lambda x: abs(x[1]), reverse=True)

        print("\nExperts with largest frequency differences (harmful - harmless):")
        print("(Positive = more common in harmful, Negative = more common in harmless)")
        for expert_id, diff in sorted_diffs[:15]:
            harmful_pct = (harmful_counts.get(expert_id, 0) / harmful_total) * 100
            harmless_pct = (harmless_counts.get(expert_id, 0) / harmless_total) * 100
            print(f"  Expert {expert_id:2d}: {diff:+.4f} (harmful: {harmful_pct:.2f}%, harmless: {harmless_pct:.2f}%)")

        # Calculate KL divergence
        harmful_probs = np.array([harmful_counts.get(i, 0) for i in range(num_experts)])
        harmless_probs = np.array([harmless_counts.get(i, 0) for i in range(num_experts)])
        harmful_probs = (harmful_probs + 1) / (harmful_probs.sum() + num_experts)  # Add-one smoothing
        harmless_probs = (harmless_probs + 1) / (harmless_probs.sum() + num_experts)

        kl_div = np.sum(harmful_probs * np.log(harmful_probs / harmless_probs))
        print(f"\nKL Divergence (Harmful || Harmless): {kl_div:.4f}")

def analyze_token_position_patterns(results, label):
    """Analyze if expert selection varies by token position."""
    print(f"\n{'='*60}")
    print(f"TOKEN POSITION ANALYSIS for {label.upper()}")
    print(f"{'='*60}")

    if not results:
        return

    num_layers = len(results[0]["layer_routing"])

    for layer_idx in range(num_layers):
        layer_key = f"layer_{layer_idx}"
        print(f"\n--- Layer {layer_idx} ---")

        # Collect experts by position (0-4 for last 5 tokens)
        position_experts = defaultdict(list)

        for result in results:
            top_experts = result["layer_routing"][layer_key]["top_expert"]
            for pos_idx, expert_id in enumerate(top_experts):
                position_experts[pos_idx].append(expert_id)

        # Analyze each position
        for pos_idx in sorted(position_experts.keys()):
            experts = position_experts[pos_idx]
            expert_counts = Counter(experts)
            total = len(experts)

            print(f"\n  Position {pos_idx} (token index from end: -{5-pos_idx}):")
            print(f"    Top 5 experts:")
            for expert_id, count in expert_counts.most_common(5):
                print(f"      Expert {expert_id}: {count}/{total} ({100*count/total:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Analyze expert routing patterns")
    parser.add_argument("data_file", type=str, help="Path to expert_routing_data.json")
    parser.add_argument("--detailed", action="store_true", help="Include detailed per-position analysis")

    args = parser.parse_args()

    # Load data
    print("Loading expert routing data...")
    data = load_routing_data(args.data_file)

    harmful_results = data["harmful_results"]
    harmless_results = data["harmless_results"]

    print(f"Model: {data['model']}")
    print(f"Harmful prompts: {data['num_harmful']}")
    print(f"Harmless prompts: {data['num_harmless']}")

    # Analyze each distribution
    analyze_expert_distribution(harmful_results, "harmful")
    analyze_expert_distribution(harmless_results, "harmless")

    # Compare distributions
    compare_distributions(harmful_results, harmless_results)

    # Optional detailed analysis
    if args.detailed:
        analyze_token_position_patterns(harmful_results, "harmful")
        analyze_token_position_patterns(harmless_results, "harmless")

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == "__main__":
    main()
