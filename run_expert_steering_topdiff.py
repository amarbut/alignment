"""
Expert-Specific Refusal Vector Pipeline (Top-Diff Selection)

Simplified pipeline that skips Arditi-style direction selection entirely.
Instead, it picks experts by activation frequency difference rank and uses
their mean-diff vectors directly for steering.

Modes:
    Single expert, single position (default):
        python run_expert_steering_topdiff.py --coeff 100
    Pick the 2nd-ranked expert:
        python run_expert_steering_topdiff.py --expert_rank 2 --coeff 100
    Sweep top 5 experts:
        python run_expert_steering_topdiff.py --sweep_experts 5 --coeff 100
    Sweep all 5 token positions for one expert:
        python run_expert_steering_topdiff.py --position all --coeff 100
    Sweep top 3 experts x all positions:
        python run_expert_steering_topdiff.py --sweep_experts 3 --position all --coeff 100
"""

# =============================================================================
# CRITICAL: Set HF cache BEFORE importing any HuggingFace libraries
# HF libraries cache environment variables at import time, so this must happen first
# =============================================================================
import sys
import argparse as _argparse_early

# Minimal early arg parsing just to get model_path for cache config
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

import torch
import random
import json
import os
import argparse
from pathlib import Path

from dataset.load_dataset import load_dataset_split, load_dataset
from config import Config
from model_utils.model_factory_moe import construct_model_base
from submodules.arditi.select_direction import get_refusal_scores
from submodules.evaluate_jailbreak import evaluate_jailbreak

from submodules.expert_steering.expert_selection import get_candidate_experts
from submodules.expert_steering.expert_specific_activations import get_expert_mean_diff
from submodules.expert_steering.expert_intervention import get_expert_weighted_intervention_hooks


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Expert-specific refusal vector pipeline (top-diff selection)"
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='unsloth/gpt-oss-20b-unsloth-bnb-4bit',
        help='Path to the base model'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=15.0,
        help='Expert selection threshold (percentage points)'
    )

    parser.add_argument(
        '--expert_type',
        type=str,
        default='both',
        choices=['harmful_preferred', 'harmless_preferred', 'both'],
        help='Which type of experts to select'
    )

    parser.add_argument(
        '--expert_rank',
        type=int,
        default=1,
        help='Which expert to use by activation diff rank, 1-indexed (default: 1 = largest diff)'
    )

    parser.add_argument(
        '--sweep_experts',
        type=int,
        default=None,
        help='Sweep the top N experts (overrides --expert_rank, evaluates each independently)'
    )

    parser.add_argument(
        '--skip_generate',
        action='store_true',
        help='Skip generation and load from cache'
    )

    parser.add_argument(
        '--skip_eval',
        action='store_true',
        help='Skip evaluation (only generate)'
    )

    parser.add_argument(
        '--skip_baseline',
        action='store_true',
        help='Skip eval on baseline model'
    )

    parser.add_argument(
        '--n_train',
        type=int,
        default=None,
        help='Number of training samples (default: use Config default)'
    )

    parser.add_argument(
        '--n_val',
        type=int,
        default=None,
        help='Number of validation samples (default: use Config default)'
    )

    parser.add_argument(
        '--n_test',
        type=int,
        default=None,
        help='Number of test samples (default: use Config default)'
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
        default=100,
        help='Maximum new tokens for generation'
    )

    parser.add_argument(
        '--position',
        type=str,
        default='-1',
        help='Token position to use: an integer (default: -1, i.e. last token) or "all" to sweep positions -5 to -1'
    )

    parser.add_argument(
        '--eval_datasets',
        type=str,
        nargs='+',
        default=None,
        help='Evaluation datasets to use (default: uses config). Options: jailbreakbench, advbench, tdc2023, maliciousinstruct, strongreject, harmbench_test'
    )

    parser.add_argument(
        '--system_prompt',
        type=str,
        default=None,
        choices=['none', 'llama_2', 'lightweight'],
        help='System prompt to use (default: use Config default)'
    )

    parser.add_argument(
        '--normalize',
        type=str,
        default='none',
        choices=['none', 'unit', 'expert_scale'],
        help='Direction normalization mode: none=raw vectors, unit=unit norm, '
             'expert_scale=unit norm scaled by expert activation RMS (default: none)'
    )

    return parser.parse_args()


def load_and_sample_datasets(cfg):
    """Load and sample datasets with size safety checks."""
    random.seed(42)

    # Load full datasets
    harmful_train_full = load_dataset_split(harmtype='harmful', split='train', instructions_only=True)
    harmless_train_full = load_dataset_split(harmtype='harmless', split='train', instructions_only=True)
    harmful_test_full = load_dataset_split(harmtype='harmful', split='test', instructions_only=True)
    harmless_test_full = load_dataset_split(harmtype='harmless', split='test', instructions_only=False)

    # Sample with size checks
    n_train_actual = min(cfg.n_train, len(harmful_train_full), len(harmless_train_full))
    n_test_actual = min(cfg.n_test, len(harmful_test_full), len(harmless_test_full))

    if n_train_actual < cfg.n_train:
        print(f"Warning: Requested {cfg.n_train} train samples but only {n_train_actual} available")
    if n_test_actual < cfg.n_test:
        print(f"Warning: Requested {cfg.n_test} test samples but only {n_test_actual} available")

    harmful_train = random.sample(harmful_train_full, n_train_actual)
    harmless_train = random.sample(harmless_train_full, n_train_actual)
    harmful_test = random.sample(harmful_test_full, n_test_actual)
    harmless_test = random.sample(harmless_test_full, n_test_actual)

    return harmful_train, harmless_train, harmful_test, harmless_test


def filter_data(cfg, model_base, harmful_train, harmless_train):
    """
    Filter training datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train)
    """
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        print("\n  Filtering training data...")
        harmful_train_scores = get_refusal_scores(
            model_base.model, harmful_train, model_base.tokenize_instructions_fn,
            model_base.refusal_toks, tokenizer=model_base.tokenizer,
            refusal_score_suffix_toks=model_base.refusal_score_suffix_toks
        )
        harmless_train_scores = get_refusal_scores(
            model_base.model, harmless_train, model_base.tokenize_instructions_fn,
            model_base.refusal_toks, tokenizer=model_base.tokenizer,
            refusal_score_suffix_toks=model_base.refusal_score_suffix_toks
        )

        harmful_train_filtered = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train_filtered = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

        print(f"    Harmful train: {len(harmful_train)} -> {len(harmful_train_filtered)}")
        print(f"    Harmless train: {len(harmless_train)} -> {len(harmless_train_filtered)}")

        harmful_train = harmful_train_filtered
        harmless_train = harmless_train_filtered

    return harmful_train, harmless_train


def generate_and_evaluate_completions(
    model_base,
    dataset_name,
    expert_info,  # (layer, expert_id, direction)
    coeff,
    output_dir,
    intervention_label,
    eval_methodologies,
    max_new_tokens=256,
    dataset=None
):
    """Generate completions with expert-specific intervention and evaluate."""

    # Create output directory
    completions_dir = os.path.join(output_dir, 'completions', intervention_label)
    os.makedirs(completions_dir, exist_ok=True)

    # Load dataset if not provided
    if dataset is None:
        dataset = load_dataset(dataset_name)
        dataset = random.sample(dataset, min(100, len(dataset)))

    # Get intervention hooks
    layer, expert_id, direction = expert_info
    fwd_pre_hooks, fwd_hooks = get_expert_weighted_intervention_hooks(
        model_base,
        layer_idx=layer,
        expert_id=expert_id,
        direction=direction,
        coeff=coeff
    )

    # Generate completions
    print(f"\nGenerating completions for {dataset_name} with {intervention_label}...")
    completions = model_base.generate_completions(
        dataset,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        max_new_tokens=max_new_tokens
    )

    # Save completions
    completions_path = os.path.join(
        completions_dir,
        f"{dataset_name}_completions.json"
    )
    with open(completions_path, "w", encoding="utf-8") as f:
        json.dump(completions, f, indent=4, ensure_ascii=False)

    print(f"Saved completions to: {completions_path}")

    # Evaluate
    print(f"Evaluating completions...")
    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(completions_dir, f"{dataset_name}_evaluations.json"),
    )

    # Save evaluations
    eval_path = os.path.join(completions_dir, f"{dataset_name}_evaluations.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=4, ensure_ascii=False)

    print(f"Saved evaluations to: {eval_path}")

    return completions, evaluation


def normalize_direction(direction, activation_rms, position, mode):
    """Apply normalization to a direction vector. Returns (normalized_direction, log_string)."""
    orig_norm = direction.norm().item()
    if mode == 'unit':
        direction = direction / direction.norm()
        log = f"norm {orig_norm:.4f} -> 1.0 (unit)"
    elif mode == 'expert_scale':
        scale = activation_rms[position].to(direction.device)
        direction = direction / direction.norm() * scale
        log = f"norm {orig_norm:.4f} -> {direction.norm().item():.4f} (expert_scale={scale.item():.4f})"
    else:
        log = f"norm {orig_norm:.4f} (no normalization)"
    return direction, log


def run_single_experiment(
    args, cfg, model_base, expert_entry, rank,
    mean_diff, activation_rms, position,
    harmful_test, harmless_test, base_output_dir
):
    """
    Run generate+evaluate for one (expert, position) combination.

    Args:
        expert_entry: (layer, expert_id, diff_pct) tuple
        rank: 1-indexed rank of this expert by diff
        mean_diff: [n_positions, d_model] tensor for this expert
        activation_rms: [n_positions] tensor (or None)
        position: int, the token position index to use
        base_output_dir: root output dir for this run
    """
    layer, expert_id, diff_pct = expert_entry

    # Build output subdirectory based on what we're sweeping
    sweep_parts = []
    if args.sweep_experts is not None:
        sweep_parts.append(f"rank_{rank}_L{layer}_E{expert_id}")
    if args.position == 'all':
        sweep_parts.append(f"pos_{position}")

    if sweep_parts:
        output_dir = os.path.join(base_output_dir, *sweep_parts)
    else:
        output_dir = base_output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n" + "#"*80)
    print(f"# EXPERIMENT: rank={rank} (L{layer} E{expert_id}, diff={diff_pct:.2f}%), position={position}")
    print(f"# Output: {output_dir}")
    print(f"#" + "#"*79)

    # Extract direction at chosen position
    direction = mean_diff[position].to(model_base.model.device, dtype=model_base.model.dtype)

    # Normalize
    normalize_mode = args.normalize
    if normalize_mode == 'expert_scale' and activation_rms is None:
        normalize_mode = 'none'
    direction, norm_log = normalize_direction(direction, activation_rms, position, normalize_mode)
    print(f"  Direction: {norm_log}")

    # Save experiment metadata
    metadata = {
        "layer": int(layer),
        "expert": int(expert_id),
        "diff_pct": float(diff_pct),
        "rank": rank,
        "position": position,
        "coeff": args.coeff,
        "normalize": normalize_mode,
        "direction_norm": direction.norm().item(),
        "selection_method": "top_diff"
    }
    with open(os.path.join(output_dir, "experiment_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    expert_info = (layer, expert_id, direction)

    if args.skip_eval:
        print("  Skipping evaluation.")
        return

    # Baseline (shared across experiments if in base_output_dir, but we re-run
    # per-experiment when sweeping to keep output self-contained)
    if not args.skip_baseline:
        print("\n  " + "-"*60)
        print("  BASELINE (No Intervention)")
        print("  " + "-"*60)

        for dataset_name in cfg.evaluation_datasets:
            generate_and_evaluate_completions(
                model_base,
                dataset_name,
                expert_info,
                coeff=0.0,
                output_dir=output_dir,
                intervention_label='baseline',
                eval_methodologies=cfg.jailbreak_eval_methodologies,
                max_new_tokens=args.max_new_tokens
            )

        generate_and_evaluate_completions(
            model_base,
            'harmless',
            expert_info,
            coeff=0.0,
            output_dir=output_dir,
            intervention_label='baseline',
            eval_methodologies=cfg.refusal_eval_methodologies,
            max_new_tokens=args.max_new_tokens,
            dataset=harmless_test
        )

    # ActAdd intervention
    print("\n  " + "-"*60)
    print(f"  ACTADD INTERVENTION (coeff={-args.coeff})")
    print("  " + "-"*60)

    for dataset_name in cfg.evaluation_datasets:
        generate_and_evaluate_completions(
            model_base,
            dataset_name,
            expert_info,
            coeff=-args.coeff,
            output_dir=output_dir,
            intervention_label='actadd',
            eval_methodologies=cfg.jailbreak_eval_methodologies,
            max_new_tokens=args.max_new_tokens
        )

    generate_and_evaluate_completions(
        model_base,
        'harmless',
        expert_info,
        coeff=args.coeff,
        output_dir=output_dir,
        intervention_label='actadd',
        eval_methodologies=cfg.refusal_eval_methodologies,
        max_new_tokens=args.max_new_tokens,
        dataset=harmless_test
    )


def run_topdiff_pipeline(args):
    """Run the top-diff expert steering pipeline."""

    # Parse position arg
    if args.position == 'all':
        positions = list(range(-5, 0))  # [-5, -4, -3, -2, -1]
    else:
        positions = [int(args.position)]

    # Determine which expert ranks to run
    if args.sweep_experts is not None:
        expert_ranks = list(range(1, args.sweep_experts + 1))
    else:
        expert_ranks = [args.expert_rank]

    print("="*80)
    print("EXPERT STEERING PIPELINE (TOP-DIFF SELECTION)")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Expert threshold: {args.threshold}%")
    print(f"Coefficient: {args.coeff}")
    print(f"Expert rank(s): {expert_ranks}")
    print(f"Position(s): {positions}")
    print(f"Normalize mode: {args.normalize}")
    print(f"Expert type: {args.expert_type}")
    print("="*80)

    model_alias = f"{os.path.basename(args.model_path)}/expert_steering_topdiff_t{args.threshold}/sys_prompt_{args.system_prompt}"

    # Create config with CLI args (only override if explicitly specified)
    config_kwargs = {
        "model_alias": model_alias,
        "model_path": args.model_path,
        "coeff": args.coeff,
        "threshold": args.threshold,
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

    # Override evaluation datasets if specified via CLI
    if args.eval_datasets is not None:
        cfg.evaluation_datasets = tuple(args.eval_datasets)
        print(f"Using custom evaluation datasets: {cfg.evaluation_datasets}")

    # Create output directory
    base_output_dir = os.path.join(cfg.artifact_path(), f"coeff_{args.coeff}")
    os.makedirs(base_output_dir, exist_ok=True)

    # Load model
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    model_base = construct_model_base(args.model_path, system_prompt=cfg.system_prompt)

    # Create model card for MoE-specific operations
    from model_utils.model_card_factory import create_model_card
    model_card = create_model_card(model_base)
    print(f"Model card: {type(model_card).__name__}")

    # Load datasets
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)
    harmful_train, harmless_train, harmful_test, harmless_test = load_and_sample_datasets(cfg)

    print(f"Raw data:")
    print(f"  Training: {len(harmful_train)} harmful, {len(harmless_train)} harmless")
    print(f"  Test: {len(harmful_test)} harmful, {len(harmless_test)} harmless")

    # Filter training datasets based on refusal scores
    print("\nFiltering datasets based on refusal scores...")
    harmful_train, harmless_train = filter_data(cfg, model_base, harmful_train, harmless_train)

    print(f"\nFiltered data:")
    print(f"  Training: {len(harmful_train)} harmful, {len(harmless_train)} harmless")

    # Select candidate experts
    print("\n" + "="*80)
    print("SELECTING CANDIDATE EXPERTS")
    print("="*80)

    # Get model-specific expert diffs path
    expert_diffs_filename = model_card.get_expert_diffs_filename()
    expert_diffs_path = f"expert_diffs/{expert_diffs_filename}"

    # Generate expert diffs if they don't exist
    if not os.path.exists(expert_diffs_path):
        print(f"Expert diffs not found at {expert_diffs_path}, generating...")
        os.makedirs("expert_diffs", exist_ok=True)
        model_card.generate_expert_diffs(
            harmful_dataset_path="dataset/splits/harmful_train.json",
            harmless_dataset_path="dataset/splits/harmless_train.json",
            output_path=expert_diffs_path,
            batch_size=args.batch_size
        )
        print(f"Expert diffs saved to {expert_diffs_path}")

    candidate_experts = get_candidate_experts(
        threshold=args.threshold,
        expert_type=args.expert_type,
        expert_diffs_path=expert_diffs_path
    )

    max_rank = max(expert_ranks)
    if max_rank > len(candidate_experts):
        print(f"ERROR: Requested rank {max_rank} but only {len(candidate_experts)} experts above threshold")
        sys.exit(1)

    # Print which experts we'll be using
    print(f"\nExperts to evaluate:")
    for rank in expert_ranks:
        layer, expert_id, diff_pct = candidate_experts[rank - 1]
        print(f"  Rank {rank}: Layer {layer}, Expert {expert_id}, Diff {diff_pct:.2f}%")

    # Direction generation / loading
    directions_dir = os.path.join(base_output_dir, "expert_directions")

    # Generate or load directions for all needed experts
    expert_data = {}  # rank -> (mean_diff, activation_rms)

    print("\n" + "="*80)
    print("GENERATING / LOADING DIRECTIONS")
    print("="*80)

    for rank in expert_ranks:
        layer, expert_id, diff_pct = candidate_experts[rank - 1]
        save_path = os.path.join(directions_dir, f"expert_L{layer}_E{expert_id}_mean_diff.pt")
        scale_path = os.path.join(directions_dir, f"expert_L{layer}_E{expert_id}_scale.pt")

        if not args.skip_generate:
            os.makedirs(directions_dir, exist_ok=True)

            mean_diff, activation_rms = get_expert_mean_diff(
                model_base,
                harmful_train,
                harmless_train,
                layer,
                expert_id,
                batch_size=args.batch_size
            )

            torch.save(mean_diff, save_path)
            torch.save(activation_rms, scale_path)
            print(f"  Saved: {save_path}")
        else:
            print(f"\n  Loading rank {rank} (L{layer} E{expert_id}) from cache...")
            mean_diff = torch.load(save_path)
            print(f"    Direction: {save_path}")

            activation_rms = None
            if os.path.exists(scale_path):
                activation_rms = torch.load(scale_path)
                print(f"    Scale: {scale_path}")
            elif args.normalize == 'expert_scale':
                print(f"    WARNING: No scale file, falling back to normalize='none'")

        expert_data[rank] = (mean_diff, activation_rms)

    # Run experiments
    if not args.skip_eval:
        print("\n" + "="*80)
        print("EVALUATION")
        print("="*80)

        n_experiments = len(expert_ranks) * len(positions)
        exp_num = 0

        for rank in expert_ranks:
            expert_entry = candidate_experts[rank - 1]
            mean_diff, activation_rms = expert_data[rank]

            for position in positions:
                exp_num += 1
                print(f"\n{'='*80}")
                print(f"EXPERIMENT {exp_num}/{n_experiments}")
                print(f"{'='*80}")

                run_single_experiment(
                    args=args,
                    cfg=cfg,
                    model_base=model_base,
                    expert_entry=expert_entry,
                    rank=rank,
                    mean_diff=mean_diff,
                    activation_rms=activation_rms,
                    position=position,
                    harmful_test=harmful_test,
                    harmless_test=harmless_test,
                    base_output_dir=base_output_dir
                )

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Results saved under: {base_output_dir}")


if __name__ == "__main__":
    args = parse_arguments()
    run_topdiff_pipeline(args)
