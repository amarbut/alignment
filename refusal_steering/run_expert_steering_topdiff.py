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
    Grid search for best (position, coeff) then full eval:
        python run_expert_steering_topdiff.py --grid_search --expert_rank 2
    Unified grid search over ALL experts x positions x coeffs:
        python run_expert_steering_topdiff.py --unified_grid
    Judge-based grid search (generate + OpenAI judge):
        python run_expert_steering_topdiff.py --judge_grid
        python run_expert_steering_topdiff.py --judge_grid --judge_grid_tokens 50 --judge_grid_n_samples 25
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
from submodules.arditi.select_direction import get_refusal_scores, get_last_position_logits, kl_div_fn
from submodules.evaluate_jailbreak import evaluate_jailbreak

from submodules.expert_steering.expert_selection import get_candidate_experts
from submodules.expert_steering.expert_specific_activations import get_expert_mean_diff
from submodules.expert_steering.expert_intervention import (
    get_expert_weighted_intervention_hooks,
    get_expert_weighted_activation_addition_hook,
)


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
        help='Expert selection threshold in percentage points. Used with --selection_mode=threshold.'
    )

    parser.add_argument(
        '--expert_type',
        type=str,
        default='both',
        choices=['harmful_preferred', 'harmless_preferred', 'both'],
        help='Which type of experts to select'
    )

    parser.add_argument(
        '--selection_mode',
        type=str,
        default='threshold',
        choices=['threshold', 'top_pct', 'score'],
        help=(
            'How to select candidate experts: '
            '"threshold" = all experts with abs(diff) >= --threshold (default); '
            '"top_pct" = top --top_pct%% by abs(diff); '
            '"score" = top --top_pct%% by abs(diff)*harmful_pct (balances '
            'differential activation with raw harmful-prompt frequency)'
        )
    )

    parser.add_argument(
        '--top_pct',
        type=float,
        default=5.0,
        help='Percentage of experts to keep in top_pct and score selection modes (default: 5.0)'
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
        '--skip_harmless',
        action='store_true',
        help='Skip harmless (refusal induction) evaluations'
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
        '--expert_diff_system_prompt',
        type=str,
        default=None,
        choices=['none', 'llama_2', 'lightweight'],
        help='System prompt whose expert diffs to load, independent of --system_prompt. '
             'If not set, uses the evaluation system prompt (current behavior). '
             'Useful for reusing diffs computed with system_prompt=none across eval configs.'
    )

    parser.add_argument(
        '--normalize',
        type=str,
        default='none',
        choices=['none', 'unit', 'expert_scale'],
        help='Direction normalization mode: none=raw vectors, unit=unit norm, '
             'expert_scale=unit norm scaled by expert activation RMS (default: none)'
    )

    # Grid search options
    parser.add_argument(
        '--grid_search',
        action='store_true',
        help='Run grid search over (position, coeff) using refusal scores on val set, '
             'then full eval with best combo'
    )

    parser.add_argument(
        '--grid_coeffs',
        type=float,
        nargs='+',
        default=[25, 50, 75, 100, 150, 200, 250, 300],
        help='Coeff values to search over in grid search (default: 25 50 75 100 150 200 250 300)'
    )

    parser.add_argument(
        '--unified_grid',
        action='store_true',
        help='Run a single unified grid search over ALL experts above threshold '
             '(plus position and coeff). Picks the single best (expert, position, coeff) combo. '
             'Implies --grid_search.'
    )

    parser.add_argument(
        '--judge_grid',
        action='store_true',
        help='Run judge-based grid search: generate short completions and use OpenAI judge '
             'to pick the best (expert, position, coeff) combo. Searches all experts above '
             'threshold. Implies --grid_search.'
    )

    parser.add_argument(
        '--judge_grid_tokens',
        type=int,
        default=25,
        help='Max new tokens for judge grid search generations (default: 25)'
    )

    parser.add_argument(
        '--judge_grid_n_samples',
        type=int,
        default=25,
        help='Number of harmful val samples to use for judge grid search (default: 25)'
    )

    return parser.parse_args()


def load_and_sample_datasets(cfg):
    """Load and sample datasets with size safety checks."""
    random.seed(42)

    # Load full datasets
    harmful_train_full = load_dataset_split(harmtype='harmful', split='train', instructions_only=True)
    harmless_train_full = load_dataset_split(harmtype='harmless', split='train', instructions_only=True)
    harmful_val_full = load_dataset_split(harmtype='harmful', split='val', instructions_only=False)
    harmless_val_full = load_dataset_split(harmtype='harmless', split='val', instructions_only=True)
    harmful_test_full = load_dataset_split(harmtype='harmful', split='test', instructions_only=True)
    harmless_test_full = load_dataset_split(harmtype='harmless', split='test', instructions_only=False)

    # Sample with size checks
    n_train_actual = min(cfg.n_train, len(harmful_train_full), len(harmless_train_full))
    n_val_actual = min(cfg.n_val, len(harmful_val_full), len(harmless_val_full))
    n_test_actual = min(cfg.n_test, len(harmful_test_full), len(harmless_test_full))

    if n_train_actual < cfg.n_train:
        print(f"Warning: Requested {cfg.n_train} train samples but only {n_train_actual} available")
    if n_val_actual < cfg.n_val:
        print(f"Warning: Requested {cfg.n_val} val samples but only {n_val_actual} available")
    if n_test_actual < cfg.n_test:
        print(f"Warning: Requested {cfg.n_test} test samples but only {n_test_actual} available")

    harmful_train = random.sample(harmful_train_full, n_train_actual)
    harmless_train = random.sample(harmless_train_full, n_train_actual)
    harmful_val = random.sample(harmful_val_full, n_val_actual)
    harmless_val = random.sample(harmless_val_full, n_val_actual)
    harmful_test = random.sample(harmful_test_full, n_test_actual)
    harmless_test = random.sample(harmless_test_full, n_test_actual)

    return (harmful_train, harmless_train, harmful_val, harmless_val,
            harmful_test, harmless_test)


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


def run_grid_search(
    model_base, model_card, mean_diff, activation_rms,
    layer, expert_id,
    harmful_val, harmless_val, normalize_mode,
    grid_coeffs, batch_size=32, kl_threshold=None
):
    """
    Grid search over (position, coeff) using Arditi refusal scores on val set,
    filtered by KL divergence on harmless data to discard combos that break coherence.

    For each (position, coeff) combo:
    1. Compute mean refusal score on harmful_val (lower = more suppression)
    2. Compute KL divergence on harmless_val vs baseline (measures coherence damage)
    3. Discard combos where KL > threshold

    Among passing combos, pick the one with lowest refusal score.

    Returns:
        (best_position, best_coeff, all_results, filtered_results) where
        all_results contains every combo and filtered_results contains only
        combos that passed the KL filter, both sorted by refusal score.
    """
    from tqdm import tqdm

    # Extract instruction strings if harmful_val contains dicts
    harmful_val_instructions = [x['instruction'] if isinstance(x, dict) else x for x in harmful_val]
    harmless_val_instructions = [x['instruction'] if isinstance(x, dict) else x for x in harmless_val]

    # Get KL threshold from model card or use default
    if kl_threshold is None:
        if hasattr(model_card, 'get_expert_steering_thresholds'):
            thresholds = model_card.get_expert_steering_thresholds()
            kl_threshold = thresholds.get('kl_threshold', 1.0)
        else:
            kl_threshold = 1.0

    positions = list(range(-5, 0))  # [-5, -4, -3, -2, -1]

    print(f"\n  Grid search: {len(positions)} positions x {len(grid_coeffs)} coeffs = {len(positions) * len(grid_coeffs)} combos")
    print(f"  Positions: {positions}")
    print(f"  Coeffs: {grid_coeffs}")
    print(f"  KL threshold: {kl_threshold}")

    # Get baseline refusal score (no intervention)
    baseline_scores = get_refusal_scores(
        model_base.model, harmful_val_instructions,
        model_base.tokenize_instructions_fn, model_base.refusal_toks,
        fwd_hooks=[], batch_size=batch_size,
        tokenizer=model_base.tokenizer,
        refusal_score_suffix_toks=model_base.refusal_score_suffix_toks
    )
    baseline_mean = baseline_scores.mean().item()
    print(f"  Baseline harmful refusal score: {baseline_mean:.4f}")

    # Get baseline harmless logits for KL computation
    print(f"  Collecting baseline harmless logits...")
    baseline_harmless_logits = get_last_position_logits(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=harmless_val_instructions,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        fwd_pre_hooks=[],
        fwd_hooks=[],
        batch_size=batch_size
    )

    all_results = []
    best_score = float('inf')
    best_position = None
    best_coeff = None

    mlp_module = model_card.get_mlp_module(layer)

    total = len(positions) * len(grid_coeffs)
    with tqdm(total=total, desc="  Grid search") as pbar:
        for position in positions:
            direction = mean_diff[position].to(model_base.model.device, dtype=model_base.model.dtype)

            if normalize_mode == 'unit':
                direction = direction / direction.norm()
            elif normalize_mode == 'expert_scale' and activation_rms is not None:
                scale = activation_rms[position].to(direction.device)
                direction = direction / direction.norm() * scale

            for coeff in grid_coeffs:
                # Create hook with negative coeff (suppress refusal)
                hook_fn = get_expert_weighted_activation_addition_hook(
                    direction=direction,
                    expert_id=expert_id,
                    coeff=-coeff,
                    model_card=model_card
                )
                fwd_hooks = [(mlp_module, hook_fn)]

                # Compute refusal score on harmful val
                scores = get_refusal_scores(
                    model_base.model, harmful_val_instructions,
                    model_base.tokenize_instructions_fn, model_base.refusal_toks,
                    fwd_hooks=fwd_hooks, batch_size=batch_size,
                    tokenizer=model_base.tokenizer,
                    refusal_score_suffix_toks=model_base.refusal_score_suffix_toks
                )
                mean_score = scores.mean().item()

                # Compute KL divergence on harmless val
                intervention_logits = get_last_position_logits(
                    model=model_base.model,
                    tokenizer=model_base.tokenizer,
                    instructions=harmless_val_instructions,
                    tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                    fwd_pre_hooks=[],
                    fwd_hooks=fwd_hooks,
                    batch_size=batch_size
                )
                kl_div = kl_div_fn(
                    baseline_harmless_logits, intervention_logits, mask=None
                ).mean(dim=0).item()

                passed_kl = kl_div <= kl_threshold

                all_results.append({
                    "position": position,
                    "coeff": coeff,
                    "refusal_score": mean_score,
                    "refusal_reduction": baseline_mean - mean_score,
                    "kl_div": kl_div,
                    "passed_kl": passed_kl,
                })

                if passed_kl and mean_score < best_score:
                    best_score = mean_score
                    best_position = position
                    best_coeff = coeff

                pbar.update(1)
                status = f"pos={best_position} coeff={best_coeff} score={best_score:.4f}" if best_position is not None else "no passing combo yet"
                pbar.set_postfix_str(f"best: {status}")

    # Sort all results by refusal score
    all_results.sort(key=lambda x: x["refusal_score"])

    # Create filtered list (passing KL only), sorted by refusal score
    filtered_results = [r for r in all_results if r["passed_kl"]]

    n_passed = len(filtered_results)
    n_failed = len(all_results) - n_passed

    print(f"\n  Grid search: {n_passed} passed KL filter (threshold={kl_threshold}), {n_failed} filtered out")
    print(f"\n  Top 10 results (passing KL filter):")
    print(f"  {'Pos':>5} {'Coeff':>8} {'Refusal':>10} {'Reduction':>10} {'KL':>10}")
    print(f"  {'-'*5} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for r in filtered_results[:10]:
        print(f"  {r['position']:>5} {r['coeff']:>8.1f} {r['refusal_score']:>10.4f} {r['refusal_reduction']:>10.4f} {r['kl_div']:>10.4f}")

    if best_position is not None:
        print(f"\n  Best: position={best_position}, coeff={best_coeff}, refusal_score={best_score:.4f}")
    else:
        print(f"\n  WARNING: No combo passed KL filter! Using lowest-KL combo as fallback.")
        all_results_by_kl = sorted(all_results, key=lambda x: x["kl_div"])
        best_position = all_results_by_kl[0]["position"]
        best_coeff = all_results_by_kl[0]["coeff"]
        best_score = all_results_by_kl[0]["refusal_score"]
        print(f"  Fallback: position={best_position}, coeff={best_coeff}, refusal_score={best_score:.4f}, kl_div={all_results_by_kl[0]['kl_div']:.4f}")

    return best_position, best_coeff, all_results, filtered_results


def run_unified_grid_search(
    model_base, model_card, expert_data, candidate_experts, expert_ranks,
    harmful_val, harmless_val, normalize_mode,
    grid_coeffs, batch_size=32, kl_threshold=None
):
    """
    Unified grid search over (expert, position, coeff) in a single pass.

    Searches all experts over threshold simultaneously, picking the single best
    (expert, position, coeff) combo based on refusal score with KL-div filtering.

    Args:
        expert_data: dict of rank -> (mean_diff, activation_rms)
        candidate_experts: list of (layer, expert_id, diff_pct) tuples
        expert_ranks: list of 1-indexed ranks to search over

    Returns:
        (best_rank, best_position, best_coeff, all_results, filtered_results)
    """
    from tqdm import tqdm

    # Extract instruction strings if val sets contain dicts
    harmful_val_instructions = [x['instruction'] if isinstance(x, dict) else x for x in harmful_val]
    harmless_val_instructions = [x['instruction'] if isinstance(x, dict) else x for x in harmless_val]

    # Get KL threshold from model card or use default
    if kl_threshold is None:
        if hasattr(model_card, 'get_expert_steering_thresholds'):
            thresholds = model_card.get_expert_steering_thresholds()
            kl_threshold = thresholds.get('kl_threshold', 1.0)
        else:
            kl_threshold = 1.0

    positions = list(range(-5, 0))  # [-5, -4, -3, -2, -1]

    n_experts = len(expert_ranks)
    total = n_experts * len(positions) * len(grid_coeffs)
    print(f"\n  Unified grid search: {n_experts} experts x {len(positions)} positions x {len(grid_coeffs)} coeffs = {total} combos")
    print(f"  Experts: {[(candidate_experts[r-1][0], candidate_experts[r-1][1]) for r in expert_ranks]}")
    print(f"  Positions: {positions}")
    print(f"  Coeffs: {grid_coeffs}")
    print(f"  KL threshold: {kl_threshold}")

    # Get baseline refusal score (no intervention)
    baseline_scores = get_refusal_scores(
        model_base.model, harmful_val_instructions,
        model_base.tokenize_instructions_fn, model_base.refusal_toks,
        fwd_hooks=[], batch_size=batch_size,
        tokenizer=model_base.tokenizer,
        refusal_score_suffix_toks=model_base.refusal_score_suffix_toks
    )
    baseline_mean = baseline_scores.mean().item()
    print(f"  Baseline harmful refusal score: {baseline_mean:.4f}")

    # Get baseline harmless logits for KL computation
    print(f"  Collecting baseline harmless logits...")
    baseline_harmless_logits = get_last_position_logits(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=harmless_val_instructions,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        fwd_pre_hooks=[],
        fwd_hooks=[],
        batch_size=batch_size
    )

    all_results = []
    best_score = float('inf')
    best_rank = None
    best_position = None
    best_coeff = None

    with tqdm(total=total, desc="  Unified grid search") as pbar:
        for rank in expert_ranks:
            layer, expert_id, diff_pct = candidate_experts[rank - 1]
            mean_diff, activation_rms = expert_data[rank]

            # Resolve normalize mode for this expert
            norm_mode = normalize_mode
            if norm_mode == 'expert_scale' and activation_rms is None:
                norm_mode = 'none'

            mlp_module = model_card.get_mlp_module(layer)

            for position in positions:
                direction = mean_diff[position].to(model_base.model.device, dtype=model_base.model.dtype)

                if norm_mode == 'unit':
                    direction = direction / direction.norm()
                elif norm_mode == 'expert_scale' and activation_rms is not None:
                    scale = activation_rms[position].to(direction.device)
                    direction = direction / direction.norm() * scale

                for coeff in grid_coeffs:
                    hook_fn = get_expert_weighted_activation_addition_hook(
                        direction=direction,
                        expert_id=expert_id,
                        coeff=-coeff,
                        model_card=model_card
                    )
                    fwd_hooks = [(mlp_module, hook_fn)]

                    # Compute refusal score on harmful val
                    scores = get_refusal_scores(
                        model_base.model, harmful_val_instructions,
                        model_base.tokenize_instructions_fn, model_base.refusal_toks,
                        fwd_hooks=fwd_hooks, batch_size=batch_size,
                        tokenizer=model_base.tokenizer,
                        refusal_score_suffix_toks=model_base.refusal_score_suffix_toks
                    )
                    mean_score = scores.mean().item()

                    # Compute KL divergence on harmless val
                    intervention_logits = get_last_position_logits(
                        model=model_base.model,
                        tokenizer=model_base.tokenizer,
                        instructions=harmless_val_instructions,
                        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                        fwd_pre_hooks=[],
                        fwd_hooks=fwd_hooks,
                        batch_size=batch_size
                    )
                    kl_div = kl_div_fn(
                        baseline_harmless_logits, intervention_logits, mask=None
                    ).mean(dim=0).item()

                    passed_kl = kl_div <= kl_threshold

                    all_results.append({
                        "rank": rank,
                        "layer": layer,
                        "expert_id": expert_id,
                        "diff_pct": diff_pct,
                        "position": position,
                        "coeff": coeff,
                        "refusal_score": mean_score,
                        "refusal_reduction": baseline_mean - mean_score,
                        "kl_div": kl_div,
                        "passed_kl": passed_kl,
                    })

                    if passed_kl and mean_score < best_score:
                        best_score = mean_score
                        best_rank = rank
                        best_position = position
                        best_coeff = coeff

                    pbar.update(1)
                    if best_rank is not None:
                        bl, be = candidate_experts[best_rank - 1][0], candidate_experts[best_rank - 1][1]
                        status = f"L{bl}E{be} pos={best_position} coeff={best_coeff} score={best_score:.4f}"
                    else:
                        status = "no passing combo yet"
                    pbar.set_postfix_str(f"best: {status}")

    # Sort all results by refusal score
    all_results.sort(key=lambda x: x["refusal_score"])

    # Create filtered list (passing KL only)
    filtered_results = [r for r in all_results if r["passed_kl"]]

    n_passed = len(filtered_results)
    n_failed = len(all_results) - n_passed

    print(f"\n  Unified grid search: {n_passed} passed KL filter (threshold={kl_threshold}), {n_failed} filtered out")
    print(f"\n  Top 10 results (passing KL filter):")
    print(f"  {'Rank':>5} {'Layer':>6} {'Expert':>7} {'Pos':>5} {'Coeff':>8} {'Refusal':>10} {'Reduction':>10} {'KL':>10}")
    print(f"  {'-'*5} {'-'*6} {'-'*7} {'-'*5} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for r in filtered_results[:10]:
        print(f"  {r['rank']:>5} {r['layer']:>6} {r['expert_id']:>7} {r['position']:>5} {r['coeff']:>8.1f} {r['refusal_score']:>10.4f} {r['refusal_reduction']:>10.4f} {r['kl_div']:>10.4f}")

    if best_rank is not None:
        bl, be = candidate_experts[best_rank - 1][0], candidate_experts[best_rank - 1][1]
        print(f"\n  Best: rank={best_rank} (L{bl} E{be}), position={best_position}, coeff={best_coeff}, refusal_score={best_score:.4f}")
    else:
        print(f"\n  WARNING: No combo passed KL filter! Using lowest-KL combo as fallback.")
        all_results_by_kl = sorted(all_results, key=lambda x: x["kl_div"])
        fallback = all_results_by_kl[0]
        best_rank = fallback["rank"]
        best_position = fallback["position"]
        best_coeff = fallback["coeff"]
        best_score = fallback["refusal_score"]
        print(f"  Fallback: rank={best_rank}, position={best_position}, coeff={best_coeff}, refusal_score={best_score:.4f}, kl_div={fallback['kl_div']:.4f}")

    return best_rank, best_position, best_coeff, all_results, filtered_results


def run_judge_grid_search(
    model_base, model_card, expert_data, candidate_experts, expert_ranks,
    harmful_val, harmless_val, normalize_mode, grid_coeffs,
    max_new_tokens=25, n_samples=25, batch_size=32,
    output_dir=None, kl_threshold=None
):
    """
    Two-stage judge-based grid search over (expert, position, coeff).

    Stage 1: Fast forward-pass refusal scores + KL-div over all combos. Combos
    that fail the KL filter are excluded. From the passing combos, keeps the top
    max(ceil(total * 0.05), 15) candidates by lowest refusal score.

    Stage 2: Generate short completions and run the OpenAI judge only on the
    surviving candidates.

    Per-combo completions and evaluations are saved to output_dir/combos/ for
    debugging and inspection.

    Returns:
        (best_rank, best_position, best_coeff, all_results)
        all_results contains all combos with refusal_score from stage 1, and
        asr/full_response/refusal/non_response for stage-2 candidates.
    """
    from tqdm import tqdm
    import math

    positions = list(range(-5, 0))  # [-5, -4, -3, -2, -1]

    # harmful_val contains dicts; extract instructions for refusal score stage
    harmful_val_instructions = [x['instruction'] if isinstance(x, dict) else x for x in harmful_val]
    # Subsample for judge stage (dicts needed for generate_completions)
    sample = harmful_val[:n_samples]

    n_experts = len(expert_ranks)
    total = n_experts * len(positions) * len(grid_coeffs)
    n_judge = max(math.ceil(total * 0.025), 15)

    print(f"\n  Judge grid search (2-stage): {n_experts} experts x {len(positions)} positions x {len(grid_coeffs)} coeffs = {total} combos")
    print(f"  Stage 1: refusal score sweep (all {total} combos)")
    print(f"  Stage 2: OpenAI judge on top {n_judge} ({max(2.55, round(100*n_judge/total))}%) candidates")
    print(f"  Positions: {positions}, Coeffs: {grid_coeffs}")
    print(f"  Judge samples: {n_samples}, max_new_tokens: {max_new_tokens}")

    # -------------------------------------------------------------------------
    # Stage 1: fast refusal score + KL-div sweep
    # -------------------------------------------------------------------------
    print(f"\n  --- Stage 1: Refusal score + KL-div sweep ---")

    # KL threshold
    if kl_threshold is None:
        if hasattr(model_card, 'get_expert_steering_thresholds'):
            kl_threshold = model_card.get_expert_steering_thresholds().get('kl_threshold', 1.0)
        else:
            kl_threshold = 1.0
    print(f"  KL threshold: {kl_threshold}")

    # Extract harmless instructions for KL computation
    harmless_val_instructions = [x['instruction'] if isinstance(x, dict) else x for x in harmless_val]

    # Baseline refusal score and harmless logits
    baseline_scores = get_refusal_scores(
        model_base.model, harmful_val_instructions,
        model_base.tokenize_instructions_fn, model_base.refusal_toks,
        fwd_hooks=[], batch_size=batch_size,
        tokenizer=model_base.tokenizer,
        refusal_score_suffix_toks=model_base.refusal_score_suffix_toks
    )
    baseline_mean = baseline_scores.mean().item()
    print(f"  Baseline refusal score: {baseline_mean:.4f}")

    print(f"  Collecting baseline harmless logits...")
    baseline_harmless_logits = get_last_position_logits(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=harmless_val_instructions,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        fwd_pre_hooks=[],
        fwd_hooks=[],
        batch_size=batch_size
    )

    stage1_results = []
    with tqdm(total=total, desc="  Stage 1 sweep") as pbar:
        for rank in expert_ranks:
            layer, expert_id, diff_pct = candidate_experts[rank - 1]
            mean_diff, activation_rms = expert_data[rank]

            norm_mode = normalize_mode
            if norm_mode == 'expert_scale' and activation_rms is None:
                norm_mode = 'none'

            mlp_module = model_card.get_mlp_module(layer)

            for position in positions:
                direction = mean_diff[position].to(model_base.model.device, dtype=model_base.model.dtype)

                if norm_mode == 'unit':
                    direction = direction / direction.norm()
                elif norm_mode == 'expert_scale' and activation_rms is not None:
                    scale = activation_rms[position].to(direction.device)
                    direction = direction / direction.norm() * scale

                for coeff in grid_coeffs:
                    hook_fn = get_expert_weighted_activation_addition_hook(
                        direction=direction,
                        expert_id=expert_id,
                        coeff=-coeff,
                        model_card=model_card
                    )
                    fwd_hooks = [(mlp_module, hook_fn)]

                    scores = get_refusal_scores(
                        model_base.model, harmful_val_instructions,
                        model_base.tokenize_instructions_fn, model_base.refusal_toks,
                        fwd_hooks=fwd_hooks, batch_size=batch_size,
                        tokenizer=model_base.tokenizer,
                        refusal_score_suffix_toks=model_base.refusal_score_suffix_toks
                    )
                    mean_score = scores.mean().item()

                    intervention_logits = get_last_position_logits(
                        model=model_base.model,
                        tokenizer=model_base.tokenizer,
                        instructions=harmless_val_instructions,
                        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                        fwd_pre_hooks=[],
                        fwd_hooks=fwd_hooks,
                        batch_size=batch_size
                    )
                    kl_div = kl_div_fn(
                        baseline_harmless_logits, intervention_logits, mask=None
                    ).mean(dim=0).item()

                    passed_kl = kl_div <= kl_threshold

                    stage1_results.append({
                        "rank": rank,
                        "layer": layer,
                        "expert_id": expert_id,
                        "diff_pct": diff_pct,
                        "position": position,
                        "coeff": coeff,
                        "refusal_score": mean_score,
                        "refusal_reduction": baseline_mean - mean_score,
                        "kl_div": kl_div,
                        "passed_kl": passed_kl,
                    })
                    pbar.update(1)
                    pbar.set_postfix_str(f"score={mean_score:.3f} kl={kl_div:.3f}{'✓' if passed_kl else '✗'}")

    # Filter by KL, then sort survivors by refusal score
    passing = [r for r in stage1_results if r["passed_kl"]]
    failing = [r for r in stage1_results if not r["passed_kl"]]
    passing.sort(key=lambda x: x["refusal_score"])

    print(f"\n  Stage 1: {len(passing)} passed KL filter, {len(failing)} excluded")

    if not passing:
        print(f"  WARNING: No combos passed KL filter (threshold={kl_threshold}). "
              f"Falling back to lowest-KL combos.")
        stage1_results.sort(key=lambda x: x["kl_div"])
        passing = stage1_results[:n_judge]

    candidates = passing[:n_judge]

    print(f"\n  Stage 1 top {len(candidates)} candidates (passed KL, sorted by refusal score):")
    print(f"  {'Rank':>5} {'Layer':>6} {'Expert':>7} {'Pos':>5} {'Coeff':>8} {'Refusal':>10} {'Reduction':>10} {'KL':>8}")
    print(f"  {'-'*5} {'-'*6} {'-'*7} {'-'*5} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for r in candidates:
        print(f"  {r['rank']:>5} {r['layer']:>6} {r['expert_id']:>7} {r['position']:>5} {r['coeff']:>8.1f} {r['refusal_score']:>10.4f} {r['refusal_reduction']:>10.4f} {r['kl_div']:>8.4f}")

    # -------------------------------------------------------------------------
    # Stage 2: OpenAI judge on surviving candidates
    # -------------------------------------------------------------------------
    print(f"\n  --- Stage 2: OpenAI judge on {len(candidates)} candidates ---")

    # Create combos output directory
    combos_dir = None
    if output_dir is not None:
        combos_dir = os.path.join(output_dir, "combos")
        os.makedirs(combos_dir, exist_ok=True)

    best_asr = -1.0
    best_rank = None
    best_position = None
    best_coeff = None

    for i, candidate in enumerate(candidates):
        rank = candidate["rank"]
        layer = candidate["layer"]
        expert_id = candidate["expert_id"]
        position = candidate["position"]
        coeff = candidate["coeff"]

        print(f"\n  [{i+1}/{len(candidates)}] Rank {rank} L{layer} E{expert_id} pos={position} coeff={coeff} (refusal={candidate['refusal_score']:.4f})")

        mean_diff, activation_rms = expert_data[rank]
        norm_mode = normalize_mode
        if norm_mode == 'expert_scale' and activation_rms is None:
            norm_mode = 'none'

        direction = mean_diff[position].to(model_base.model.device, dtype=model_base.model.dtype)
        if norm_mode == 'unit':
            direction = direction / direction.norm()
        elif norm_mode == 'expert_scale' and activation_rms is not None:
            scale = activation_rms[position].to(direction.device)
            direction = direction / direction.norm() * scale

        fwd_pre_hooks, fwd_hooks = get_expert_weighted_intervention_hooks(
            model_base,
            layer_idx=layer,
            expert_id=expert_id,
            direction=direction,
            coeff=-coeff
        )

        completions = model_base.generate_completions(
            sample,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size
        )

        combo_label = f"R{rank}_L{layer}_E{expert_id}_pos{position}_c{coeff}"
        eval_path = os.path.join(combos_dir, f"{combo_label}_eval.json") if combos_dir else os.devnull

        evaluation = evaluate_jailbreak(
            completions=completions,
            methodologies=["openai"],
            evaluation_path=eval_path,
            openai_delay=0.1
        )

        asr = evaluation.get("openai_success_rate", 0.0)
        counts = evaluation.get("openai_overall_counts", {})
        n_full = counts.get("full_response", 0)
        n_refusal = counts.get("refusal", 0)
        n_non = counts.get("non_response", 0)

        print(f"    ASR={asr:.2%} (full={n_full}, refusal={n_refusal}, non_response={n_non})")

        candidate.update({
            "asr": asr,
            "full_response": n_full,
            "refusal_count": n_refusal,
            "non_response": n_non,
        })

        if asr > best_asr:
            best_asr = asr
            best_rank = rank
            best_position = position
            best_coeff = coeff

    # Build all_results: stage-2 candidates sorted by ASR, then rest sorted by refusal score
    judged = sorted(candidates, key=lambda x: (-x.get("asr", -1), x.get("non_response", 0)))
    not_judged = stage1_results[n_judge:]  # already sorted by refusal score
    all_results = judged + not_judged

    print(f"\n  Top stage-2 results (by ASR):")
    print(f"  {'Rank':>5} {'Layer':>6} {'Expert':>7} {'Pos':>5} {'Coeff':>8} {'ASR':>8} {'Full':>6} {'Ref':>6} {'Non':>6} {'RefScore':>10}")
    print(f"  {'-'*5} {'-'*6} {'-'*7} {'-'*5} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*10}")
    for r in judged[:10]:
        print(f"  {r['rank']:>5} {r['layer']:>6} {r['expert_id']:>7} {r['position']:>5} {r['coeff']:>8.1f} {r.get('asr', 0):>8.2%} {r.get('full_response', '-'):>6} {r.get('refusal_count', '-'):>6} {r.get('non_response', '-'):>6} {r['refusal_score']:>10.4f}")

    if best_rank is not None:
        bl, be = candidate_experts[best_rank - 1][0], candidate_experts[best_rank - 1][1]
        print(f"\n  Best: rank={best_rank} (L{bl} E{be}), position={best_position}, coeff={best_coeff}, ASR={best_asr:.2%}")

    return best_rank, best_position, best_coeff, all_results


def run_single_experiment(
    args, cfg, model_base, expert_entry, rank,
    mean_diff, activation_rms, position,
    harmful_test, harmless_test, base_output_dir,
    coeff_override=None
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
        coeff_override: if set, use this coeff instead of args.coeff
    """
    layer, expert_id, diff_pct = expert_entry
    coeff = coeff_override if coeff_override is not None else args.coeff

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
    print(f"# EXPERIMENT: rank={rank} (L{layer} E{expert_id}, diff={diff_pct:.2f}%), position={position}, coeff={coeff}")
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
        "coeff": coeff,
        "normalize": normalize_mode,
        "direction_norm": direction.norm().item(),
        "selection_method": "grid_search" if args.grid_search else "top_diff"
    }
    with open(os.path.join(output_dir, "experiment_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    expert_info = (layer, expert_id, direction)

    if args.skip_eval:
        print("  Skipping evaluation.")
        return

    # Baseline
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

        if not args.skip_harmless:
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
    print(f"  ACTADD INTERVENTION (coeff={-coeff})")
    print("  " + "-"*60)

    for dataset_name in cfg.evaluation_datasets:
        generate_and_evaluate_completions(
            model_base,
            dataset_name,
            expert_info,
            coeff=-coeff,
            output_dir=output_dir,
            intervention_label='actadd',
            eval_methodologies=cfg.jailbreak_eval_methodologies,
            max_new_tokens=args.max_new_tokens
        )

    if not args.skip_harmless:
        generate_and_evaluate_completions(
            model_base,
            'harmless',
            expert_info,
            coeff=coeff,
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

    # --unified_grid and --judge_grid imply --grid_search
    if args.unified_grid or args.judge_grid:
        args.grid_search = True

    # Determine which expert ranks to run
    if args.sweep_experts is not None:
        expert_ranks = list(range(1, args.sweep_experts + 1))
    else:
        expert_ranks = [args.expert_rank]

    print("="*80)
    print("EXPERT STEERING PIPELINE (TOP-DIFF SELECTION)")
    print("="*80)
    print(f"Model: {args.model_path}")
    if args.selection_mode == 'threshold':
        print(f"Expert selection: threshold={args.threshold}%")
    else:
        print(f"Expert selection: {args.selection_mode}, top {args.top_pct}%")
    if args.judge_grid:
        print(f"Mode: JUDGE GRID SEARCH over experts x positions x coeffs {args.grid_coeffs} ({args.judge_grid_n_samples} samples, {args.judge_grid_tokens} tokens)")
    elif args.unified_grid:
        print(f"Mode: UNIFIED GRID SEARCH over experts x positions x coeffs {args.grid_coeffs}")
    elif args.grid_search:
        print(f"Mode: GRID SEARCH over positions x coeffs {args.grid_coeffs}")
    else:
        print(f"Coefficient: {args.coeff}")
    print(f"Expert rank(s): {expert_ranks}")
    print(f"Position(s): {'grid_search' if args.grid_search else positions}")
    print(f"Normalize mode: {args.normalize}")
    print(f"Expert type: {args.expert_type}")
    print(f"Skip harmless: {args.skip_harmless}")
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

    # Create base output directory (coeff is part of path only when not grid searching)
    if args.grid_search:
        base_output_dir = os.path.join(cfg.artifact_path(), "grid_search")
    else:
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
    (harmful_train, harmless_train, harmful_val, harmless_val,
     harmful_test, harmless_test) = load_and_sample_datasets(cfg)

    print(f"Raw data:")
    print(f"  Training: {len(harmful_train)} harmful, {len(harmless_train)} harmless")
    print(f"  Validation: {len(harmful_val)} harmful, {len(harmless_val)} harmless")
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

    # Get model-specific expert diffs path (system-prompt-specific)
    expert_diffs_filename = model_card.get_expert_diffs_filename()
    diff_sys_prompt = args.expert_diff_system_prompt if args.expert_diff_system_prompt is not None else cfg.system_prompt
    expert_diffs_dir = f"expert_diffs/sys_prompt_{diff_sys_prompt}"
    expert_diffs_path = os.path.join(expert_diffs_dir, expert_diffs_filename)
    if args.expert_diff_system_prompt is not None:
        print(f"  Using expert diffs from system_prompt='{diff_sys_prompt}' (overrides eval system_prompt='{cfg.system_prompt}')")

    def _needs_raw_freqs(selection_mode):
        """Return True if the selection mode requires raw harmful/harmless_pct values."""
        return selection_mode == 'score'

    def _diffs_have_raw_freqs(path):
        """Check whether the saved diffs file contains raw frequency columns."""
        import json as _json
        with open(path) as f:
            data = _json.load(f)
        diffs = data.get('expert_diffs', data)
        first_layer = next(iter(diffs.values()))
        return len(first_layer[0]) >= 4

    def _generate_diffs():
        os.makedirs(expert_diffs_dir, exist_ok=True)
        if args.expert_diff_system_prompt is not None and args.expert_diff_system_prompt != cfg.system_prompt:
            print(f"  Constructing temporary model with system_prompt='{diff_sys_prompt}' for diff generation...")
            diff_model_base = construct_model_base(args.model_path, system_prompt=diff_sys_prompt)
            diff_model_card = create_model_card(diff_model_base)
        else:
            diff_model_base = model_base
            diff_model_card = model_card
        diff_model_card.generate_expert_diffs(
            harmful_dataset_path="dataset/splits/harmful_train.json",
            harmless_dataset_path="dataset/splits/harmless_train.json",
            output_path=expert_diffs_path,
            batch_size=args.batch_size
        )
        print(f"Expert diffs saved to {expert_diffs_path}")

    # Generate expert diffs if missing or in wrong format for the selection mode
    if not os.path.exists(expert_diffs_path):
        print(f"Expert diffs not found at {expert_diffs_path}, generating...")
        _generate_diffs()
    elif _needs_raw_freqs(args.selection_mode) and not _diffs_have_raw_freqs(expert_diffs_path):
        print(f"Expert diffs at {expert_diffs_path} are in old format (no raw frequencies).")
        print(f"Selection mode '{args.selection_mode}' requires raw harmful/harmless frequencies — regenerating...")
        _generate_diffs()

    candidate_experts = get_candidate_experts(
        threshold=args.threshold,
        expert_type=args.expert_type,
        expert_diffs_path=expert_diffs_path,
        selection_mode=args.selection_mode,
        top_pct=args.top_pct,
    )

    # For unified_grid or judge_grid, or any non-threshold selection mode,
    # use ALL selected candidates (the selection already determined the set)
    if args.unified_grid or args.judge_grid or args.selection_mode != 'threshold':
        expert_ranks = list(range(1, len(candidate_experts) + 1))
        print(f"\nUsing all {len(candidate_experts)} selected experts (selection_mode={args.selection_mode})")

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

    # Grid search mode
    if args.grid_search:
        print("\n" + "="*80)
        print("GRID SEARCH")
        print("="*80)

        if args.judge_grid:
            # Judge-based grid search: generate short completions + OpenAI judge
            grid_output_dir = os.path.join(base_output_dir, "judge_grid_search")
            os.makedirs(grid_output_dir, exist_ok=True)

            best_rank, best_position, best_coeff, all_results = run_judge_grid_search(
                model_base=model_base,
                model_card=model_card,
                expert_data=expert_data,
                candidate_experts=candidate_experts,
                expert_ranks=expert_ranks,
                harmful_val=harmful_val,
                harmless_val=harmless_val,
                normalize_mode=args.normalize,
                grid_coeffs=args.grid_coeffs,
                max_new_tokens=args.judge_grid_tokens,
                n_samples=args.judge_grid_n_samples,
                batch_size=args.batch_size,
                output_dir=grid_output_dir
            )

            best_layer, best_expert_id, best_diff_pct = candidate_experts[best_rank - 1]

            with open(os.path.join(grid_output_dir, "judge_grid_results.json"), 'w') as f:
                json.dump({
                    "best_rank": best_rank,
                    "best_layer": best_layer,
                    "best_expert_id": best_expert_id,
                    "best_position": best_position,
                    "best_coeff": best_coeff,
                    "n_experts": len(expert_ranks),
                    "max_new_tokens": args.judge_grid_tokens,
                    "n_samples": args.judge_grid_n_samples,
                    "results": all_results
                }, f, indent=2)

            # Run full eval with best combo
            if not args.skip_eval:
                best_mean_diff, best_activation_rms = expert_data[best_rank]
                print(f"\n  Running full evaluation with best combo: L{best_layer} E{best_expert_id}, position={best_position}, coeff={best_coeff}")
                run_single_experiment(
                    args=args,
                    cfg=cfg,
                    model_base=model_base,
                    expert_entry=(best_layer, best_expert_id, best_diff_pct),
                    rank=best_rank,
                    mean_diff=best_mean_diff,
                    activation_rms=best_activation_rms,
                    position=best_position,
                    harmful_test=harmful_test,
                    harmless_test=harmless_test,
                    base_output_dir=base_output_dir,
                    coeff_override=best_coeff
                )

        elif args.unified_grid:
            # Single unified search over all experts x positions x coeffs (refusal score + KL)
            best_rank, best_position, best_coeff, all_results, filtered_results = run_unified_grid_search(
                model_base=model_base,
                model_card=model_card,
                expert_data=expert_data,
                candidate_experts=candidate_experts,
                expert_ranks=expert_ranks,
                harmful_val=harmful_val,
                harmless_val=harmless_val,
                normalize_mode=args.normalize,
                grid_coeffs=args.grid_coeffs,
                batch_size=args.batch_size
            )

            # Save results
            grid_output_dir = os.path.join(base_output_dir, "unified_grid_search")
            os.makedirs(grid_output_dir, exist_ok=True)

            best_layer, best_expert_id, best_diff_pct = candidate_experts[best_rank - 1]

            with open(os.path.join(grid_output_dir, "grid_search_all_results.json"), 'w') as f:
                json.dump({
                    "best_rank": best_rank,
                    "best_layer": best_layer,
                    "best_expert_id": best_expert_id,
                    "best_position": best_position,
                    "best_coeff": best_coeff,
                    "n_experts": len(expert_ranks),
                    "results": all_results
                }, f, indent=2)

            with open(os.path.join(grid_output_dir, "grid_search_filtered_results.json"), 'w') as f:
                json.dump({
                    "best_rank": best_rank,
                    "best_layer": best_layer,
                    "best_expert_id": best_expert_id,
                    "best_position": best_position,
                    "best_coeff": best_coeff,
                    "n_experts": len(expert_ranks),
                    "results": filtered_results
                }, f, indent=2)

            # Run full eval with best combo
            if not args.skip_eval:
                best_mean_diff, best_activation_rms = expert_data[best_rank]
                print(f"\n  Running full evaluation with best combo: L{best_layer} E{best_expert_id}, position={best_position}, coeff={best_coeff}")
                run_single_experiment(
                    args=args,
                    cfg=cfg,
                    model_base=model_base,
                    expert_entry=(best_layer, best_expert_id, best_diff_pct),
                    rank=best_rank,
                    mean_diff=best_mean_diff,
                    activation_rms=best_activation_rms,
                    position=best_position,
                    harmful_test=harmful_test,
                    harmless_test=harmless_test,
                    base_output_dir=base_output_dir,
                    coeff_override=best_coeff
                )

        else:
            # Per-expert grid search (original behavior)
            for rank in expert_ranks:
                layer, expert_id, diff_pct = candidate_experts[rank - 1]
                mean_diff, activation_rms = expert_data[rank]

                print(f"\n  Grid search for rank {rank}: L{layer} E{expert_id} (diff={diff_pct:.2f}%)")

                normalize_mode = args.normalize
                if normalize_mode == 'expert_scale' and activation_rms is None:
                    normalize_mode = 'none'

                best_position, best_coeff, all_results, filtered_results = run_grid_search(
                    model_base=model_base,
                    model_card=model_card,
                    mean_diff=mean_diff,
                    activation_rms=activation_rms,
                    layer=layer,
                    expert_id=expert_id,
                    harmful_val=harmful_val,
                    harmless_val=harmless_val,
                    normalize_mode=normalize_mode,
                    grid_coeffs=args.grid_coeffs,
                    batch_size=args.batch_size
                )

                # Save grid search results
                grid_output_dir = os.path.join(base_output_dir, f"rank_{rank}_L{layer}_E{expert_id}")
                os.makedirs(grid_output_dir, exist_ok=True)

                with open(os.path.join(grid_output_dir, "grid_search_all_results.json"), 'w') as f:
                    json.dump({
                        "best_position": best_position,
                        "best_coeff": best_coeff,
                        "layer": layer,
                        "expert_id": expert_id,
                        "rank": rank,
                        "results": all_results
                    }, f, indent=2)

                with open(os.path.join(grid_output_dir, "grid_search_filtered_results.json"), 'w') as f:
                    json.dump({
                        "best_position": best_position,
                        "best_coeff": best_coeff,
                        "layer": layer,
                        "expert_id": expert_id,
                        "rank": rank,
                        "results": filtered_results
                    }, f, indent=2)

                # Run full eval with best combo
                if not args.skip_eval:
                    print(f"\n  Running full evaluation with best combo: position={best_position}, coeff={best_coeff}")
                    run_single_experiment(
                        args=args,
                        cfg=cfg,
                        model_base=model_base,
                        expert_entry=(layer, expert_id, diff_pct),
                        rank=rank,
                        mean_diff=mean_diff,
                        activation_rms=activation_rms,
                        position=best_position,
                        harmful_test=harmful_test,
                        harmless_test=harmless_test,
                        base_output_dir=base_output_dir,
                        coeff_override=best_coeff
                    )

    # Standard (non-grid-search) mode
    else:
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
