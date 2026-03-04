"""
Ablation: random below-threshold expert steering.

Validates that the threshold-based expert selection is doing meaningful work
by running the same full 2-stage judge grid on N experts that did NOT pass the
expert-diff threshold — i.e. experts with no strong harmful/harmless routing
preference.

The experiment is identical to Mode 1 (threshold) in run_expert_steering.py,
except the candidate experts are sampled randomly from those BELOW the
threshold rather than selected from those above it.  This gives the control
group the same fair opportunity (full grid search) as the treatment group.

Sampling is stratified by layer to avoid clustering in a single part of the
network.

Usage:
  python run_ablation_random_experts.py \\
      --model_path allenai/OLMoE-1B-7B-0924-Instruct \\
      --system_prompt lightweight \\
      --ablation_n 10 \\
      --ablation_seed 42 \\
      --skip_harmless --skip_baseline --max_new_tokens 50
"""

# =============================================================================
# CRITICAL: Set HF cache BEFORE importing any HuggingFace libraries
# =============================================================================
import sys
import argparse as _argparse_early

_early_parser = _argparse_early.ArgumentParser(add_help=False)
_early_parser.add_argument('--model_path', type=str, default='')
_early_args, _ = _early_parser.parse_known_args()

if _early_args.model_path:
    from model_utils.hf_cache_config import set_hf_cache_from_path
    set_hf_cache_from_path(_early_args.model_path)
# =============================================================================

try:
    from unsloth import FastLanguageModel
    from peft import PeftModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

import random
import os
import argparse
import json

from config import Config
from model_utils.model_factory_moe import construct_model_base
from submodules.expert_steering.expert_selection import load_expert_diffs

# Reuse all shared pipeline logic from run_expert_steering
from run_expert_steering import (
    load_and_sample_datasets,
    filter_data,
    _get_shared_cache_dir,
    _load_or_compute_layer_directions,
    run_grid_search,
    run_single_experiment,
)


# =============================================================================
# Expert sampling
# =============================================================================

def get_random_below_threshold_experts(n, threshold, expert_diffs_path, seed=42):
    """
    Sample N experts with |diff| < threshold, stratified by layer.

    Returns a list of (layer, expert_id, diff) tuples — the same format used
    by get_candidate_experts() in the main pipeline.
    """
    expert_diffs = load_expert_diffs(expert_diffs_path)

    # Collect below-threshold experts grouped by layer
    by_layer = {}
    for layer_str, entries in expert_diffs.items():
        layer_idx = int(layer_str.replace('layer_', ''))
        below = [
            (layer_idx, int(e[0]), float(e[1]))
            for e in entries
            if abs(e[1]) < threshold
        ]
        if below:
            by_layer[layer_idx] = below

    total_below = sum(len(v) for v in by_layer.values())
    print(f"Found {total_below} experts with |diff| < {threshold}% "
          f"across {len(by_layer)} layers")

    if n > total_below:
        raise ValueError(
            f"Requested {n} below-threshold experts but only {total_below} exist"
        )

    # Stratified sampling: floor(n / n_layers) per layer, remainder from pool
    rng = random.Random(seed)
    n_layers = len(by_layer)
    per_layer = n // n_layers
    remainder = n % n_layers

    sampled, leftovers = [], []
    for layer_idx in sorted(by_layer):
        pool = list(by_layer[layer_idx])
        rng.shuffle(pool)
        sampled.extend(pool[:per_layer])
        leftovers.extend(pool[per_layer:])

    rng.shuffle(leftovers)
    sampled.extend(leftovers[:remainder])

    return sampled


# =============================================================================
# Argument parsing
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Ablation: run the expert steering judge grid on N randomly "
                    "sampled below-threshold experts"
    )

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model')
    parser.add_argument('--system_prompt', type=str, default=None,
                        choices=['none', 'llama_2', 'lightweight'],
                        help='System prompt (default: Config default)')

    # Ablation-specific
    parser.add_argument('--ablation_n', type=int, default=10,
                        help='Number of below-threshold experts to sample (default: 10)')
    parser.add_argument('--ablation_seed', type=int, default=42,
                        help='Random seed for expert sampling (default: 42)')

    # Expert selection
    parser.add_argument('--threshold', type=float, default=None,
                        help='Threshold used to define "below threshold" (default: model card value)')

    # Grid search
    parser.add_argument('--grid_coeffs', type=float, nargs='+',
                        default=[25, 50, 75, 100, 150, 200, 250, 300],
                        help='Coefficients to sweep in grid search')
    parser.add_argument('--judge_grid_tokens', type=int, default=25,
                        help='Max new tokens for judge grid generations (default: 25)')
    parser.add_argument('--judge_grid_n_samples', type=int, default=25,
                        help='Harmful val samples for judge grid (default: 25)')

    # Generation / evaluation
    parser.add_argument('--max_new_tokens', type=int, default=100,
                        help='Max new tokens for full evaluation (default: 100)')
    parser.add_argument('--skip_baseline', action='store_true',
                        help='Skip baseline evaluation')
    parser.add_argument('--skip_harmless', action='store_true',
                        help='Skip harmless (refusal induction) evaluations')
    parser.add_argument('--eval_datasets', type=str, nargs='+', default=None,
                        help='Evaluation datasets (default: Config default)')
    parser.add_argument('--force_generate', action='store_true',
                        help='Recompute cached directions even if they exist')

    # Dataset / sampling
    parser.add_argument('--n_train', type=int, default=None)
    parser.add_argument('--n_val',   type=int, default=None)
    parser.add_argument('--n_test',  type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_arguments()

    print("=" * 80)
    print("ABLATION: RANDOM BELOW-THRESHOLD EXPERT STEERING")
    print("=" * 80)
    print(f"Model:        {args.model_path}")
    print(f"Ablation N:   {args.ablation_n}")
    print(f"Seed:         {args.ablation_seed}")
    print(f"Grid coeffs:  {args.grid_coeffs}")
    print("=" * 80)

    base_model_name = os.path.basename(args.model_path)

    # ------------------------------------------------------------------
    # Config — results go to expert_steering_ablation/
    # ------------------------------------------------------------------
    config_kwargs = {
        "model_alias": (f"{base_model_name}/expert_steering_ablation"
                        f"/sys_prompt_{args.system_prompt}"),
        "model_path": args.model_path,
    }
    if args.system_prompt is not None:
        config_kwargs["system_prompt"] = args.system_prompt
    if args.n_train is not None:
        config_kwargs["n_train"] = args.n_train
    if args.n_val is not None:
        config_kwargs["n_val"] = args.n_val
    if args.n_test is not None:
        config_kwargs["n_test"] = args.n_test
    cfg = Config(**config_kwargs)
    print(f"System prompt: {cfg.system_prompt}")

    if args.eval_datasets is not None:
        cfg.evaluation_datasets = tuple(args.eval_datasets)

    base_output_dir = os.path.join(cfg.artifact_path(), "grid_search")
    os.makedirs(base_output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    model_base = construct_model_base(args.model_path, system_prompt=cfg.system_prompt)

    from model_utils.model_card_factory import create_model_card
    model_card = create_model_card(model_base)

    threshold = (args.threshold if args.threshold is not None
                 else model_card.get_expert_steering_thresholds().get('expert_diff_threshold', 15.0))
    print(f"Expert diff threshold: {threshold}%  (ablation selects experts BELOW this)")

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)
    (harmful_train, harmless_train, harmful_val, harmless_val,
     harmful_test, harmless_test) = load_and_sample_datasets(cfg)
    print("\nFiltering training data based on refusal scores...")
    harmful_train, harmless_train = filter_data(cfg, model_base, harmful_train, harmless_train)

    # ------------------------------------------------------------------
    # Select random below-threshold experts
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SELECTING RANDOM BELOW-THRESHOLD EXPERTS")
    print("=" * 80)

    expert_diffs_filename = model_card.get_expert_diffs_filename()
    expert_diffs_dir = f"expert_diffs/sys_prompt_{cfg.system_prompt}"
    expert_diffs_path = os.path.join(expert_diffs_dir, expert_diffs_filename)

    if not os.path.exists(expert_diffs_path):
        print(f"Expert diffs not found at {expert_diffs_path}, generating...")
        os.makedirs(expert_diffs_dir, exist_ok=True)
        model_card.generate_expert_diffs(
            harmful_dataset_path="dataset/splits/harmful_train.json",
            harmless_dataset_path="dataset/splits/harmless_train.json",
            output_path=expert_diffs_path,
            batch_size=args.batch_size
        )

    candidate_experts = get_random_below_threshold_experts(
        n=args.ablation_n,
        threshold=threshold,
        expert_diffs_path=expert_diffs_path,
        seed=args.ablation_seed,
    )
    expert_ranks = list(range(1, len(candidate_experts) + 1))

    # Save the sampled expert list alongside results for reproducibility
    ablation_manifest = {
        "ablation_n": args.ablation_n,
        "ablation_seed": args.ablation_seed,
        "threshold": threshold,
        "sampled_experts": [
            {"rank": r, "layer": l, "expert_id": e, "diff_pct": d}
            for r, (l, e, d) in enumerate(candidate_experts, 1)
        ],
    }
    manifest_path = os.path.join(base_output_dir, "ablation_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(ablation_manifest, f, indent=2)
    print(f"\nAblation manifest saved to: {manifest_path}")

    # ------------------------------------------------------------------
    # Load / compute directions (hits the shared cache)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LOADING / COMPUTING DIRECTIONS")
    print("=" * 80)

    shared_cache_dir = _get_shared_cache_dir(base_model_name, cfg.system_prompt)
    print(f"  Cache: {shared_cache_dir}/layer_{{L}}.pt")

    from collections import defaultdict
    layer_to_ranks = defaultdict(list)
    for rank in expert_ranks:
        layer, expert_id, _ = candidate_experts[rank - 1]
        layer_to_ranks[layer].append((rank, expert_id))

    expert_data = {}
    for layer_idx in sorted(layer_to_ranks):
        rank_expert_pairs = layer_to_ranks[layer_idx]
        expert_ids_needed = [eid for _, eid in rank_expert_pairs]
        layer_cache = _load_or_compute_layer_directions(
            layer_idx=layer_idx,
            expert_ids_needed=expert_ids_needed,
            model_base=model_base,
            harmful_train=harmful_train,
            harmless_train=harmless_train,
            batch_size=args.batch_size,
            cache_dir=shared_cache_dir,
            force_generate=args.force_generate,
        )
        for rank, expert_id in rank_expert_pairs:
            expert_data[rank] = layer_cache[expert_id]

    # ------------------------------------------------------------------
    # Stage 1+2: judge grid over all N sampled experts
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("JUDGE GRID SEARCH (over below-threshold experts)")
    print("=" * 80)

    best_rank, best_position, best_coeff, all_results = run_grid_search(
        model_base=model_base,
        model_card=model_card,
        candidate_experts=candidate_experts,
        expert_ranks=expert_ranks,
        expert_data=expert_data,
        harmful_val=harmful_val,
        harmless_val=harmless_val,
        grid_coeffs=args.grid_coeffs,
        max_new_tokens=args.judge_grid_tokens,
        n_samples=args.judge_grid_n_samples,
        batch_size=args.batch_size,
        output_dir=os.path.join(base_output_dir, "judge_grid_search"),
    )

    grid_summary = {
        "best_rank": best_rank,
        "best_layer": int(candidate_experts[best_rank - 1][0]),
        "best_expert": int(candidate_experts[best_rank - 1][1]),
        "best_diff_pct": float(candidate_experts[best_rank - 1][2]),
        "best_position": int(best_position),
        "best_coeff": float(best_coeff),
        "all_results": all_results,
    }
    with open(os.path.join(base_output_dir, "judge_grid_search", "ablation_grid_results.json"), "w") as f:
        json.dump(grid_summary, f, indent=2)

    best_layer = candidate_experts[best_rank - 1][0]
    best_expert_id = candidate_experts[best_rank - 1][1]
    print(f"\nBest candidate: rank={best_rank} "
          f"(L{best_layer} E{best_expert_id}), "
          f"position={best_position}, coeff={best_coeff}")

    # ------------------------------------------------------------------
    # Full evaluation on best (expert, position, coeff)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FULL EVALUATION (best below-threshold expert)")
    print("=" * 80)

    run_single_experiment(
        args=args,
        cfg=cfg,
        model_base=model_base,
        expert_entry=candidate_experts[best_rank - 1],
        rank=best_rank,
        mean_diff=expert_data[best_rank],
        position=best_position,
        harmful_test=harmful_test,
        harmless_test=harmless_test,
        base_output_dir=base_output_dir,
        coeff_override=best_coeff,
    )

    print("\n" + "=" * 80)
    print("ABLATION COMPLETE")
    print("=" * 80)
    print(f"Results saved under: {base_output_dir}")


if __name__ == "__main__":
    main()
