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


def _parse_entry(entry):
    """
    Parse an expert diff entry, handling old and new formats.

    Old format: [expert_id, diff]
    New format: [expert_id, diff, harmful_pct, harmless_pct]

    Returns: (expert_id, diff, harmful_pct, harmless_pct)
    """
    if len(entry) == 4:
        return entry[0], entry[1], entry[2], entry[3]
    else:
        return entry[0], entry[1], None, None


def _layer_idx(layer_str):
    return int(layer_str.replace('layer_', '')) if layer_str.startswith('layer_') else int(layer_str)


def _matches_type(diff, return_type):
    if return_type == "both":
        return True
    elif return_type == "harmful_preferred":
        return diff > 0
    elif return_type == "harmless_preferred":
        return diff < 0
    return False


def select_experts_by_threshold(
    expert_diffs: Dict,
    threshold: float = 15.0,
    return_type: str = "harmful_preferred"
) -> List[Tuple[int, int, float]]:
    """
    Select experts with activation frequency difference above threshold.

    Args:
        expert_diffs: Dictionary mapping layer -> list of entries
        threshold: Minimum absolute difference (percentage points)
        return_type: "harmful_preferred", "harmless_preferred", or "both"

    Returns:
        List of (layer, expert_id, diff) tuples, sorted by abs(diff) descending
    """
    selected = []

    for layer_str, experts_data in expert_diffs.items():
        layer_idx = _layer_idx(layer_str)
        for entry in experts_data:
            expert_id, diff, _, _ = _parse_entry(entry)
            if abs(diff) >= threshold and _matches_type(diff, return_type):
                selected.append((layer_idx, expert_id, diff))

    selected.sort(key=lambda x: abs(x[2]), reverse=True)
    return selected


def select_experts_by_top_pct(
    expert_diffs: Dict,
    top_pct: float = 5.0,
    return_type: str = "harmful_preferred"
) -> List[Tuple[int, int, float]]:
    """
    Select the top N% of experts by absolute activation frequency difference.

    Args:
        expert_diffs: Dictionary mapping layer -> list of entries
        top_pct: Percentage of experts to keep (e.g. 5.0 = top 5%)
        return_type: "harmful_preferred", "harmless_preferred", or "both"

    Returns:
        List of (layer, expert_id, diff) tuples, sorted by abs(diff) descending
    """
    import math

    all_matching = []
    for layer_str, experts_data in expert_diffs.items():
        layer_idx = _layer_idx(layer_str)
        for entry in experts_data:
            expert_id, diff, _, _ = _parse_entry(entry)
            if _matches_type(diff, return_type):
                all_matching.append((layer_idx, expert_id, diff))

    all_matching.sort(key=lambda x: abs(x[2]), reverse=True)
    n = max(1, math.ceil(len(all_matching) * top_pct / 100.0))
    return all_matching[:n]


def select_experts_by_score(
    expert_diffs: Dict,
    top_pct: float = 5.0,
    return_type: str = "harmful_preferred"
) -> List[Tuple[int, int, float]]:
    """
    Select the top N% of experts by balanced score: abs(diff) * harmful_pct.

    This rewards experts that are both strongly differential AND frequently
    activated on harmful prompts, ensuring the intervention actually fires
    at inference time.

    Falls back to abs(diff) if raw frequencies are unavailable (old format).

    Args:
        expert_diffs: Dictionary mapping layer -> list of entries
        top_pct: Percentage of experts to keep (e.g. 5.0 = top 5%)
        return_type: "harmful_preferred", "harmless_preferred", or "both"

    Returns:
        List of (layer, expert_id, diff) tuples, sorted by score descending
    """
    import math

    all_matching = []
    for layer_str, experts_data in expert_diffs.items():
        layer_idx = _layer_idx(layer_str)
        for entry in experts_data:
            expert_id, diff, harmful_pct, harmless_pct = _parse_entry(entry)
            if _matches_type(diff, return_type):
                score = abs(diff) * harmful_pct if harmful_pct is not None else abs(diff)
                all_matching.append((layer_idx, expert_id, diff, score))

    all_matching.sort(key=lambda x: x[3], reverse=True)
    n = max(1, math.ceil(len(all_matching) * top_pct / 100.0))
    return [(layer, eid, diff) for layer, eid, diff, _ in all_matching[:n]]


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
    expert_diffs_path: str = "expert_explore/oss_expert_diffs.json",
    selection_mode: str = "threshold",
    top_pct: float = 5.0,
) -> List[Tuple[int, int, float]]:
    """
    Load and select candidate experts.

    Args:
        threshold: Minimum absolute difference (percentage points). Used by 'threshold' mode.
        expert_type: "harmful_preferred", "harmless_preferred", or "both"
        expert_diffs_path: Path to expert diffs JSON file
        selection_mode: One of:
            "threshold" - keep all experts with abs(diff) >= threshold (default)
            "top_pct"   - keep top N% by abs(diff)
            "score"     - keep top N% by abs(diff) * harmful_pct
        top_pct: Percentage for top_pct and score modes (e.g. 5.0 = top 5%)

    Returns:
        List of (layer, expert_id, diff) tuples
    """
    print(f"Loading expert diffs from {expert_diffs_path}...")
    expert_diffs = load_expert_diffs(expert_diffs_path)

    if selection_mode == "top_pct":
        print(f"Selecting top {top_pct}% of experts by abs(diff), type={expert_type}...")
        selected = select_experts_by_top_pct(expert_diffs, top_pct=top_pct, return_type=expert_type)
    elif selection_mode == "score":
        print(f"Selecting top {top_pct}% of experts by score (abs(diff)*harmful_pct), type={expert_type}...")
        selected = select_experts_by_score(expert_diffs, top_pct=top_pct, return_type=expert_type)
    else:
        print(f"Selecting experts with threshold={threshold}%, type={expert_type}...")
        selected = select_experts_by_threshold(expert_diffs, threshold=threshold, return_type=expert_type)

    print_expert_selection_summary(selected)
    return selected
