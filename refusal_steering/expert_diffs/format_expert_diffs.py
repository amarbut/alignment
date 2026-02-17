"""
Format expert activation frequency diffs into a sorted, human-readable table.

Usage:
    python format_expert_diffs.py                          # formats all files in this directory
    python format_expert_diffs.py oss_expert_diffs.json    # format a specific file
    python format_expert_diffs.py --top 20 *.json          # show only top 20 by |diff|
"""

import json
import argparse
import glob
import os


def format_expert_diffs(filepath, top_n=None):
    with open(filepath) as f:
        data = json.load(f)

    # Collect metadata
    model = data.get("model_path", "unknown")
    num_layers = data.get("num_layers", "?")
    num_experts = data.get("num_experts", "?")
    routing = data.get("routing_mode", "?")
    n_harmful = data.get("num_harmful", "?")
    n_harmless = data.get("num_harmless", "?")

    # Flatten all (layer, expert, diff) entries
    entries = []
    for layer_key, expert_list in data["expert_diffs"].items():
        layer_idx = int(layer_key.split("_")[1])
        for expert_id, diff_pct in expert_list:
            entries.append((layer_idx, int(expert_id), diff_pct))

    # Sort by absolute diff, descending
    entries.sort(key=lambda x: abs(x[2]), reverse=True)

    if top_n is not None:
        entries = entries[:top_n]

    # Print
    print(f"\n{'=' * 70}")
    print(f"Expert Activation Frequency Diffs: {os.path.basename(filepath)}")
    print(f"{'=' * 70}")
    print(f"  Model:    {model}")
    print(f"  Layers:   {num_layers}  |  Experts/layer: {num_experts}  |  Routing: {routing}")
    print(f"  Samples:  {n_harmful} harmful, {n_harmless} harmless")
    if top_n is not None:
        print(f"  Showing:  top {top_n} by |diff|")
    print(f"{'=' * 70}")
    print(f"  {'Rank':>4}  {'Expert':>8}  {'Diff (%%)':>10}  {'Direction':>18}")
    print(f"  {'-' * 4}  {'-' * 8}  {'-' * 10}  {'-' * 18}")

    for rank, (layer, expert, diff) in enumerate(entries, 1):
        direction = "harmful-preferred" if diff > 0 else "harmless-preferred"
        print(f"  {rank:>4}  L{layer:>2}E{expert:<3}  {diff:>+10.2f}  {direction:>18}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Format expert diffs into sorted tables")
    parser.add_argument("files", nargs="*", help="JSON files to format (default: all *_expert_diffs.json in this dir)")
    parser.add_argument("--top", type=int, default=None, help="Show only top N experts by |diff|")
    args = parser.parse_args()

    if args.files:
        files = args.files
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        files = sorted(glob.glob(os.path.join(script_dir, "*_expert_diffs.json")))

    if not files:
        print("No expert diff files found.")
        return

    for filepath in files:
        format_expert_diffs(filepath, top_n=args.top)


if __name__ == "__main__":
    main()
