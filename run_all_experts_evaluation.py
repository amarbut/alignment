#!/usr/bin/env python3
"""
Evaluate steering interventions for all experts (or a specified subset).

This script generates steering vectors for multiple experts and evaluates each one,
allowing you to validate that your selection process identifies better-than-random experts.

Usage:
    # Evaluate all experts in specific layers
    python run_all_experts_evaluation.py --layers 10 11 12

    # Evaluate all experts in all layers
    python run_all_experts_evaluation.py --all_layers

    # Evaluate a random sample
    python run_all_experts_evaluation.py --random_sample 50

    # Resume from previous run
    python run_all_experts_evaluation.py --layers 10 11 12 --resume
"""

import torch
import random
import json
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

from dataset.load_dataset import load_dataset_split, load_dataset
from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak

from expert_specific_activations import get_expert_mean_diff
from expert_intervention import get_expert_weighted_intervention_hooks


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate steering for all/multiple experts"
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='unsloth/gpt-oss-20b-unsloth-bnb-4bit',
        help='Path to the model'
    )

    # Expert selection arguments (mutually exclusive)
    expert_group = parser.add_mutually_exclusive_group(required=True)
    expert_group.add_argument(
        '--layers',
        type=int,
        nargs='+',
        help='Specific layers to evaluate all experts in (e.g., --layers 10 11 12)'
    )
    expert_group.add_argument(
        '--all_layers',
        action='store_true',
        help='Evaluate all experts in all layers (672 total)'
    )
    expert_group.add_argument(
        '--random_sample',
        type=int,
        help='Evaluate N randomly sampled experts'
    )
    expert_group.add_argument(
        '--expert_list',
        type=str,
        help='JSON file with list of (layer, expert_id) tuples to evaluate'
    )

    parser.add_argument(
        '--n_train',
        type=int,
        default=500,
        help='Number of training samples for direction generation'
    )

    parser.add_argument(
        '--n_test',
        type=int,
        default=100,
        help='Number of test samples for evaluation'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing'
    )

    parser.add_argument(
        '--coeff',
        type=float,
        default=1.0,
        help='Coefficient for activation addition'
    )

    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=256,
        help='Maximum new tokens for generation'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='pipeline/runs/all_experts_evaluation',
        help='Output directory for results'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous run (skip already completed experts)'
    )

    parser.add_argument(
        '--eval_datasets',
        type=str,
        nargs='+',
        default=['jailbreakbench'],
        help='Datasets to evaluate on'
    )

    return parser.parse_args()


def get_expert_list(args) -> List[Tuple[int, int]]:
    """
    Generate list of (layer, expert_id) pairs to evaluate.

    Returns:
        List of (layer_idx, expert_id) tuples
    """
    if args.layers:
        # All experts in specified layers
        experts = []
        for layer in args.layers:
            for expert_id in range(32):  # OSS-20B has 32 experts per layer
                experts.append((layer, expert_id))
        print(f"Evaluating all experts in layers {args.layers}: {len(experts)} total")

    elif args.all_layers:
        # All experts in all layers
        experts = []
        for layer in range(21):  # OSS-20B has 21 layers
            for expert_id in range(32):
                experts.append((layer, expert_id))
        print(f"Evaluating ALL experts: {len(experts)} total")

    elif args.random_sample:
        # Random sample
        all_experts = [(layer, expert_id) for layer in range(21) for expert_id in range(32)]
        random.seed(42)
        experts = random.sample(all_experts, args.random_sample)
        print(f"Evaluating random sample of {args.random_sample} experts")

    elif args.expert_list:
        # Load from JSON file
        with open(args.expert_list, 'r') as f:
            experts = json.load(f)
        print(f"Loaded {len(experts)} experts from {args.expert_list}")

    return experts


def load_and_sample_datasets(cfg):
    """Load and sample datasets."""
    random.seed(42)

    harmful_train = random.sample(
        load_dataset_split(harmtype='harmful', split='train', instructions_only=True),
        cfg.n_train
    )
    harmless_train = random.sample(
        load_dataset_split(harmtype='harmless', split='train', instructions_only=True),
        cfg.n_train
    )
    harmful_test = random.sample(
        load_dataset_split(harmtype='harmful', split='test', instructions_only=True),
        cfg.n_test
    )
    harmless_test = random.sample(
        load_dataset_split(harmtype='harmless', split='test', instructions_only=False),
        cfg.n_test
    )

    return harmful_train, harmless_train, harmful_test, harmless_test


def is_expert_completed(output_dir: Path, layer: int, expert_id: int) -> bool:
    """Check if an expert has already been evaluated."""
    expert_dir = output_dir / f"layer_{layer}" / f"expert_{expert_id}"

    # Check if any position has been evaluated (position-specific structure)
    # We check for pos_0 through pos_4 (5 positions)
    for pos in range(5):
        pos_dir = expert_dir / f"pos_{pos}"
        eval_file = pos_dir / "completions" / "actadd" / "jailbreakbench_evaluations.json"
        if eval_file.exists():
            # If any position is completed, consider it done (for resume purposes)
            # Full check would verify all 5 positions, but this is simpler
            return True

    # Also check non-position-specific structure for backward compatibility
    eval_file = expert_dir / "completions" / "actadd" / "jailbreakbench_evaluations.json"
    return eval_file.exists()


def generate_expert_direction(
    model_base,
    harmful_train: List[str],
    harmless_train: List[str],
    layer: int,
    expert_id: int,
    batch_size: int = 32
) -> torch.Tensor:
    """Generate steering direction for a single expert."""

    print(f"Generating direction for Layer {layer}, Expert {expert_id}...")

    mean_diff = get_expert_mean_diff(
        model_base,
        harmful_train,
        harmless_train,
        layer,
        expert_id,
        batch_size=batch_size
    )

    return mean_diff


def evaluate_expert(
    model_base,
    layer: int,
    expert_id: int,
    direction: torch.Tensor,
    harmful_test: List,
    harmless_test: List,
    output_dir: Path,
    coeff: float,
    max_new_tokens: int,
    eval_datasets: List[str],
    eval_methodologies: List[str]
):
    """
    Run intervention and evaluation for a single expert.

    If direction is position-specific [n_pos, d_model], evaluates each position separately.
    """

    # Check if direction is position-specific
    if direction.dim() == 2:
        n_positions = direction.shape[0]
        print(f"\nEvaluating Layer {layer}, Expert {expert_id} across {n_positions} positions...")

        # Evaluate each position separately
        for pos_idx in range(n_positions):
            pos_direction = direction[pos_idx, :]  # Extract single position [d_model]

            print(f"\n  Position {pos_idx}:")
            evaluate_expert_single_direction(
                model_base=model_base,
                layer=layer,
                expert_id=expert_id,
                position=pos_idx,
                direction=pos_direction,
                harmful_test=harmful_test,
                harmless_test=harmless_test,
                output_dir=output_dir,
                coeff=coeff,
                max_new_tokens=max_new_tokens,
                eval_datasets=eval_datasets,
                eval_methodologies=eval_methodologies
            )
    else:
        # Single direction [d_model]
        print(f"\nEvaluating Layer {layer}, Expert {expert_id}...")
        evaluate_expert_single_direction(
            model_base=model_base,
            layer=layer,
            expert_id=expert_id,
            position=None,
            direction=direction,
            harmful_test=harmful_test,
            harmless_test=harmless_test,
            output_dir=output_dir,
            coeff=coeff,
            max_new_tokens=max_new_tokens,
            eval_datasets=eval_datasets,
            eval_methodologies=eval_methodologies
        )


def evaluate_expert_single_direction(
    model_base,
    layer: int,
    expert_id: int,
    position: int,
    direction: torch.Tensor,
    harmful_test: List,
    harmless_test: List,
    output_dir: Path,
    coeff: float,
    max_new_tokens: int,
    eval_datasets: List[str],
    eval_methodologies: List[str]
):
    """Evaluate a single direction for an expert (at a specific position)."""

    # Create output directory
    if position is not None:
        expert_dir = output_dir / f"layer_{layer}" / f"expert_{expert_id}" / f"pos_{position}"
    else:
        expert_dir = output_dir / f"layer_{layer}" / f"expert_{expert_id}"

    expert_dir.mkdir(parents=True, exist_ok=True)

    # Save direction
    torch.save(direction, expert_dir / "direction.pt")

    # Save metadata
    metadata = {
        "layer": layer,
        "expert_id": expert_id,
        "position": position,
        "coeff": coeff
    }
    with open(expert_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Move direction to correct device/dtype
    direction = direction.to(model_base.model.device, dtype=model_base.model.dtype)

    # Run interventions and evaluations
    for dataset_name in eval_datasets:
        # ActAdd intervention (suppress refusal)
        fwd_pre_hooks, fwd_hooks = get_expert_weighted_intervention_hooks(
            model_base,
            layer_idx=layer,
            expert_id=expert_id,
            direction=direction,
            coeff=-coeff  # Negative to suppress refusal
        )

        # Generate completions
        dataset = load_dataset(dataset_name)
        completions = model_base.generate_completions(
            dataset,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            max_new_tokens=max_new_tokens
        )

        # Save and evaluate
        completions_dir = expert_dir / "completions" / "actadd"
        completions_dir.mkdir(parents=True, exist_ok=True)

        completions_path = completions_dir / f"{dataset_name}_completions.json"
        with open(completions_path, "w", encoding="utf-8") as f:
            json.dump(completions, f, indent=4, ensure_ascii=False)

        eval_path = completions_dir / f"{dataset_name}_evaluations.json"
        evaluation = evaluate_jailbreak(
            completions=completions,
            methodologies=eval_methodologies,
            evaluation_path=str(eval_path)
        )

        asr = evaluation.get('substring_matching_success_rate', 'N/A')
        if position is not None:
            print(f"    {dataset_name} - ASR: {asr}")
        else:
            print(f"{dataset_name} - ASR: {asr}")


def generate_summary_report(output_dir: Path, expert_list: List[Tuple[int, int]]):
    """Generate summary comparing all evaluated experts (across all positions)."""

    results = []

    for layer, expert_id in expert_list:
        expert_dir = output_dir / f"layer_{layer}" / f"expert_{expert_id}"

        # Check for position-specific results
        for pos in range(5):  # 5 positions
            pos_dir = expert_dir / f"pos_{pos}"
            eval_file = pos_dir / "completions" / "actadd" / "jailbreakbench_evaluations.json"

            if eval_file.exists():
                with open(eval_file, 'r') as f:
                    evaluation = json.load(f)

                results.append({
                    "layer": layer,
                    "expert_id": expert_id,
                    "position": pos,
                    "substring_matching_asr": evaluation.get("substring_matching_success_rate", None),
                    "llamaguard2_asr": evaluation.get("llamaguard2_success_rate", None),
                    "openai_asr": evaluation.get("openai_success_rate", None),
                })

        # Also check non-position-specific (backward compatibility)
        eval_file = expert_dir / "completions" / "actadd" / "jailbreakbench_evaluations.json"
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                evaluation = json.load(f)

            results.append({
                "layer": layer,
                "expert_id": expert_id,
                "position": None,
                "substring_matching_asr": evaluation.get("substring_matching_success_rate", None),
                "llamaguard2_asr": evaluation.get("llamaguard2_success_rate", None),
                "openai_asr": evaluation.get("openai_success_rate", None),
            })

    # Sort by substring matching ASR (descending)
    results.sort(key=lambda x: x["substring_matching_asr"] if x["substring_matching_asr"] is not None else -1, reverse=True)

    # Save full results
    summary_path = output_dir / "summary_all_experts.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    print(f"\nTotal candidates evaluated: {len(results)}")
    print(f"\nTop 10 candidates by substring matching ASR:")
    for i, result in enumerate(results[:10], 1):
        pos_str = f", Pos {result['position']}" if result['position'] is not None else ""
        print(f"{i:2d}. Layer {result['layer']:2d}, Expert {result['expert_id']:2d}{pos_str}: "
              f"ASR = {result['substring_matching_asr']:.2%}")

    print(f"\nBottom 10 candidates by substring matching ASR:")
    for i, result in enumerate(results[-10:], 1):
        pos_str = f", Pos {result['position']}" if result['position'] is not None else ""
        print(f"{i:2d}. Layer {result['layer']:2d}, Expert {result['expert_id']:2d}{pos_str}: "
              f"ASR = {result['substring_matching_asr']:.2%}")

    print(f"\nFull results saved to: {summary_path}")

    return results


def main():
    args = parse_arguments()

    print("="*80)
    print("ALL EXPERTS EVALUATION")
    print("="*80)

    # Get list of experts to evaluate
    expert_list = get_expert_list(args)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save expert list
    with open(output_dir / "expert_list.json", 'w') as f:
        json.dump(expert_list, f, indent=2)

    # Filter out completed experts if resuming
    if args.resume:
        original_count = len(expert_list)
        expert_list = [
            (layer, expert_id) for layer, expert_id in expert_list
            if not is_expert_completed(output_dir, layer, expert_id)
        ]
        skipped = original_count - len(expert_list)
        if skipped > 0:
            print(f"\nResuming: Skipping {skipped} already completed experts")
            print(f"Remaining: {len(expert_list)} experts to evaluate")

    if len(expert_list) == 0:
        print("\nAll experts already evaluated!")
        generate_summary_report(output_dir, get_expert_list(args))
        return

    # Load model
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    model_base = construct_model_base(args.model_path)
    
    # Setup evaluation config
    cfg = Config(model_alias="all_experts_eval", model_path=args.model_path)
    # Load datasets
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)
    (harmful_train, harmless_train,
     harmful_test, harmless_test) = load_and_sample_datasets(cfg)

    print(f"Training: {len(harmful_train)} harmful, {len(harmless_train)} harmless")
    print(f"Test: {len(harmful_test)} harmful, {len(harmless_test)} harmless")

    # Evaluate each expert
    print("\n" + "="*80)
    print(f"EVALUATING {len(expert_list)} EXPERTS")
    print("="*80)

    for idx, (layer, expert_id) in enumerate(expert_list, 1):
        print(f"\n[{idx}/{len(expert_list)}] Layer {layer}, Expert {expert_id}")
        print("-"*80)

        try:
            # Generate direction
            direction = generate_expert_direction(
                model_base,
                harmful_train,
                harmless_train,
                layer,
                expert_id,
                batch_size=args.batch_size
            )

            # Evaluate
            evaluate_expert(
                model_base,
                layer,
                expert_id,
                direction,
                harmful_test,
                harmless_test,
                output_dir,
                coeff=args.coeff,
                max_new_tokens=args.max_new_tokens,
                eval_datasets=args.eval_datasets,
                eval_methodologies=cfg.jailbreak_eval_methodologies
            )

        except Exception as e:
            print(f"ERROR evaluating Layer {layer}, Expert {expert_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate summary
    print("\n" + "="*80)
    print("GENERATING SUMMARY")
    print("="*80)
    generate_summary_report(output_dir, get_expert_list(args))

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
