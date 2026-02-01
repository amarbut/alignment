"""
Expert selection based on activation frequency differences.

Loads expert routing analysis and selects candidate experts based on
differential activation between harmful and harmless prompts.
"""

import json
from typing import List, Tuple, Dict
from pathlib import Path


def load_expert_diffs(filepath: str = "expert_explore/oss_expert_diffs.json") -> Dict:
    """
    Load expert frequency differences from JSON file.

    Handles both formats:
    - Direct format: {"layer_0": [...], "layer_1": [...], ...}
    - Metadata format: {"model_path": ..., "expert_diffs": {"layer_0": [...], ...}}
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Check if this is the new format with metadata
    if 'expert_diffs' in data:
        return data['expert_diffs']

    return data


def select_experts_by_threshold(
    expert_diffs: Dict,
    threshold: float = 15.0,
    return_type: str = "harmful_preferred"
) -> List[Tuple[int, int, float]]:
    """
    Select experts with activation frequency difference above threshold.

    Args:
        expert_diffs: Dictionary mapping layer -> list of [expert_id, diff]
        threshold: Minimum absolute difference (percentage points)
        return_type: "harmful_preferred", "harmless_preferred", or "both"

    Returns:
        List of (layer, expert_id, diff) tuples
    """
    selected = []

    for layer_str, experts_data in expert_diffs.items():
        # Handle both "layer_0" format and plain "0" format
        if layer_str.startswith('layer_'):
            layer_idx = int(layer_str.replace('layer_', ''))
        else:
            layer_idx = int(layer_str)

        for expert_id, diff in experts_data:
            abs_diff = abs(diff)

            if abs_diff >= threshold:
                # Check if this expert matches the requested type
                if return_type == "both":
                    selected.append((layer_idx, expert_id, diff))
                elif return_type == "harmful_preferred" and diff > 0:
                    selected.append((layer_idx, expert_id, diff))
                elif return_type == "harmless_preferred" and diff < 0:
                    selected.append((layer_idx, expert_id, diff))

    # Sort by absolute difference (largest first)
    selected.sort(key=lambda x: abs(x[2]), reverse=True)

    return selected


def print_expert_selection_summary(selected_experts: List[Tuple[int, int, float]]):
    """Print summary of selected experts."""
    print("="*80)
    print("SELECTED CANDIDATE EXPERTS")
    print("="*80)
    print(f"Total experts selected: {len(selected_experts)}")
    print()

    harmful_preferred = [e for e in selected_experts if e[2] > 0]
    harmless_preferred = [e for e in selected_experts if e[2] < 0]

    print(f"Harmful-preferred experts: {len(harmful_preferred)}")
    print(f"Harmless-preferred experts: {len(harmless_preferred)}")
    print()

    print("Top 10 by absolute difference:")
    print(f"{'Layer':<8} {'Expert':<8} {'Diff (%)':<12} {'Type':<20}")
    print("-" * 80)

    for layer, expert_id, diff in selected_experts[:10]:
        expert_type = "Harmful-preferred" if diff > 0 else "Harmless-preferred"
        print(f"{layer:<8} {expert_id:<8} {diff:>10.2f}   {expert_type:<20}")

    if len(selected_experts) > 10:
        print(f"\n... and {len(selected_experts) - 10} more experts")

    print("="*80)


def get_candidate_experts(
    threshold: float = 15.0,
    expert_type: str = "harmful_preferred",
    expert_diffs_path: str = "expert_explore/oss_expert_diffs.json"
) -> List[Tuple[int, int, float]]:
    """
    Main function to load and select candidate experts.

    Args:
        threshold: Minimum absolute difference (percentage points)
        expert_type: "harmful_preferred", "harmless_preferred", or "both"
        expert_diffs_path: Path to expert diffs JSON file

    Returns:
        List of (layer, expert_id, diff) tuples
    """
    print(f"Loading expert diffs from {expert_diffs_path}...")
    expert_diffs = load_expert_diffs(expert_diffs_path)

    print(f"Selecting experts with threshold={threshold}%, type={expert_type}...")
    selected = select_experts_by_threshold(
        expert_diffs,
        threshold=threshold,
        return_type=expert_type
    )

    print_expert_selection_summary(selected)

    return selected


if __name__ == "__main__":
    # Test with different thresholds
    print("\nTesting with 15% threshold:")
    experts_15 = get_candidate_experts(threshold=15.0, expert_type="harmful_preferred")

    print("\n" + "="*80)
    print("\nTesting with 10% threshold:")
    experts_10 = get_candidate_experts(threshold=10.0, expert_type="harmful_preferred")

    print("\n" + "="*80)
    print("\nTesting with both types at 15%:")
    experts_both = get_candidate_experts(threshold=15.0, expert_type="both")
