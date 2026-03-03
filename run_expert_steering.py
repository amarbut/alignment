"""
Expert-Specific Refusal Vector Pipeline (Consolidated)

Five modes, selected by CLI flags:

  Mode 1 — THRESHOLD GRID (default):
    judge grid over experts >= model_card threshold,
    sweeping 5 positions x grid_coeffs.

  Mode 2 — ALLEX GRID (--allex):
    judge grid over ALL experts at ALL MoE layers,
    sweeping 5 positions x grid_coeffs.

  Mode 3 — RANK GRID (--expert_rank N):
    judge grid over the top-N experts by diff rank,
    sweeping 5 positions x grid_coeffs.

  Mode 4 — BY NAME GRID (--layer L --expert E):
    judge grid for one named expert,
    sweeping 5 positions x grid_coeffs.

  Mode 5 — BY NAME+POS GRID (--layer L --expert E --position P):
    judge grid for one named expert at a fixed position,
    sweeping only grid_coeffs.

Usage examples:
  python run_expert_steering.py --model_path allenai/OLMoE-1B-7B-0924-Instruct
  python run_expert_steering.py --model_path ... --allex
  python run_expert_steering.py --model_path ... --expert_rank 5
  python run_expert_steering.py --model_path ... --layer 9 --expert 39
  python run_expert_steering.py --model_path ... --layer 9 --expert 39 --position -1
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
    get_expert_weighted_activation_addition_hook,
    get_expert_weighted_intervention_hooks,
    get_all_expert_weighted_activation_addition_hook,
    get_all_expert_weighted_intervention_hooks,
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Expert-specific refusal vector pipeline (consolidated, 5 modes)"
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the base model'
    )

    # -------------------------------------------------------------------------
    # Mode selection flags
    # -------------------------------------------------------------------------
    parser.add_argument(
        '--allex',
        action='store_true',
        help='[Mode 2] Run judge grid over ALL experts at ALL MoE layers x positions x coeffs'
    )

    parser.add_argument(
        '--expert_rank',
        type=int,
        default=None,
        help='[Mode 3] Run judge grid for top-N experts by diff rank (1-indexed). '
             'E.g. --expert_rank 5 searches experts ranked 1-5.'
    )

    parser.add_argument(
        '--layer',
        type=int,
        default=None,
        help='[Modes 4 & 5] Layer index of a specific expert to steer'
    )

    parser.add_argument(
        '--expert',
        type=int,
        default=None,
        help='[Modes 4 & 5] Expert index within the layer to steer'
    )

    parser.add_argument(
        '--position',
        type=int,
        default=None,
        help='[Mode 5 only] Fix token position (e.g. -1) instead of grid-searching positions. '
             'Only used when --layer and --expert are also set.'
    )

    # -------------------------------------------------------------------------
    # Expert selection
    # -------------------------------------------------------------------------
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Expert selection threshold in percentage points (Mode 1 / 3). '
             'If not set, uses model card expert_diff_threshold.'
    )

    parser.add_argument(
        '--expert_type',
        type=str,
        default='both',
        choices=['harmful_preferred', 'harmless_preferred', 'both'],
        help='Which type of experts to select (default: both)'
    )

    # -------------------------------------------------------------------------
    # Grid search options
    # -------------------------------------------------------------------------
    parser.add_argument(
        '--grid_coeffs',
        type=float,
        nargs='+',
        default=[25, 50, 75, 100, 150, 200, 250, 300],
        help='Coeff values to search over in grid (default: 25 50 75 100 150 200 250 300)'
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

    # -------------------------------------------------------------------------
    # Generation / evaluation
    # -------------------------------------------------------------------------
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=100,
        help='Maximum new tokens for full evaluation generation (default: 100)'
    )

    parser.add_argument(
        '--force_generate',
        action='store_true',
        help='Recompute and overwrite cached directions even if they already exist'
    )

    parser.add_argument(
        '--skip_baseline',
        action='store_true',
        help='Skip baseline model evaluation'
    )

    parser.add_argument(
        '--skip_harmless',
        action='store_true',
        help='Skip harmless (refusal induction) evaluations'
    )

    # -------------------------------------------------------------------------
    # Dataset / sampling
    # -------------------------------------------------------------------------
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
        help='Batch size for processing (default: 32)'
    )

    parser.add_argument(
        '--system_prompt',
        type=str,
        default=None,
        choices=['none', 'llama_2', 'lightweight'],
        help='System prompt to use (default: use Config default)'
    )

    parser.add_argument(
        '--eval_datasets',
        type=str,
        nargs='+',
        default=None,
        help='Evaluation datasets (default: uses config). Options: jailbreakbench, advbench, '
             'tdc2023, maliciousinstruct, strongreject, harmbench_test'
    )

    return parser.parse_args()


# =============================================================================
# Data helpers
# =============================================================================

def load_and_sample_datasets(cfg):
    """Load and sample datasets with size safety checks."""
    random.seed(42)

    harmful_train_full = load_dataset_split(harmtype='harmful', split='train', instructions_only=True)
    harmless_train_full = load_dataset_split(harmtype='harmless', split='train', instructions_only=True)
    # Load harmful_val as dicts (instructions_only=False) so judge stage can generate completions
    harmful_val_full = load_dataset_split(harmtype='harmful', split='val', instructions_only=False)
    harmless_val_full = load_dataset_split(harmtype='harmless', split='val', instructions_only=True)
    harmful_test_full = load_dataset_split(harmtype='harmful', split='test', instructions_only=True)
    harmless_test_full = load_dataset_split(harmtype='harmless', split='test', instructions_only=False)

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
    """Filter training datasets based on refusal scores."""
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


# =============================================================================
# Direction cache — shared across all modes
# =============================================================================
#
# Cache layout:
#   runs/{model}/expert_directions/sys_prompt_{sp}/layer_{L}.pt
#     → dict {expert_id (int): tensor[n_pos, d_model]}
#
# Any mode that needs directions for a given (layer, expert) pair will hit the
# same file. If the expert is already present it is loaded; if not it is
# computed and the file is updated in-place. This means running threshold mode
# first and then allex will reuse the directions already cached.
# =============================================================================

def _get_shared_cache_dir(base_model_name, system_prompt):
    """Return the mode-agnostic direction cache directory."""
    return os.path.join("runs", base_model_name, "expert_directions",
                        f"sys_prompt_{system_prompt}")


def _load_or_compute_layer_directions(
    layer_idx, expert_ids_needed,
    model_base, harmful_train, harmless_train,
    batch_size, cache_dir, force_generate=False
):
    """
    Load cached expert directions for a layer, computing any that are missing.

    Cache file: {cache_dir}/layer_{layer_idx}.pt
      → dict {expert_id (int): tensor[n_pos, d_model]}

    The file is updated (not overwritten) whenever new experts are computed,
    so multiple runs accumulate into a single growing cache.

    Args:
        expert_ids_needed: list of expert_id ints required by the caller
        force_generate: if True, ignore the cache and recompute all requested
                        experts, overwriting any previously cached values

    Returns:
        dict {expert_id: tensor[n_pos, d_model]}
        (contains at minimum all expert_ids_needed)
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"layer_{layer_idx}.pt")

    # Load existing cache (skipped when force_generate)
    cached = {}
    if not force_generate and os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location='cpu')

    missing = [eid for eid in expert_ids_needed if eid not in cached]

    if not missing:
        if expert_ids_needed:
            print(f"  [Layer {layer_idx}] {len(expert_ids_needed)} direction(s) loaded from cache")
        return cached

    # Compute missing experts and update the cache file
    from tqdm import tqdm
    n_existing = len(cached)
    desc = f"  Layer {layer_idx:>3} ({n_existing} cached + {len(missing)} new)"
    with tqdm(missing, desc=desc, unit="expert", leave=True) as pbar:
        for expert_id in pbar:
            mean_diff, _ = get_expert_mean_diff(
                model_base, harmful_train, harmless_train,
                layer_idx, expert_id, batch_size=batch_size
            )
            cached[expert_id] = mean_diff.cpu()
            torch.cuda.empty_cache()

    torch.save(cached, cache_path)
    print(f"  [Layer {layer_idx}] Cache updated: {len(cached)} total experts → {cache_path}")

    return cached


# =============================================================================
# Grid search functions
# =============================================================================

def run_grid_search(
    model_base, model_card, expert_data, candidate_experts, expert_ranks,
    harmful_val, harmless_val,
    grid_coeffs, max_new_tokens=25, n_samples=25, batch_size=32,
    output_dir=None
):
    """
    Two-stage judge-based grid search over (expert, position, coeff).

    Used by modes 1, 3, and 4.

    Stage 1: Fast forward-pass refusal scores + KL-div over all combos.
             Keeps the top max(ceil(total * 0.025), 15) candidates by
             lowest refusal score (excluding KL failures).

    Stage 2: Generate short completions and run the OpenAI judge only on
             the surviving candidates.

    Args:
        expert_data: dict of rank -> mean_diff tensor [n_pos, d_model]
        candidate_experts: list of (layer, expert_id, diff_pct) tuples
        expert_ranks: list of 1-indexed ranks to search

    Returns:
        (best_rank, best_position, best_coeff, all_results)
    """
    from tqdm import tqdm
    import math

    positions = list(range(-5, 0))  # [-5, -4, -3, -2, -1]

    # harmful_val contains dicts; extract instructions for refusal score stage
    harmful_val_instructions = [x['instruction'] if isinstance(x, dict) else x for x in harmful_val]
    harmless_val_instructions = [x['instruction'] if isinstance(x, dict) else x for x in harmless_val]

    # Subsample for judge stage (dicts needed for generate_completions)
    sample = harmful_val[:n_samples]

    # KL threshold from model card
    if hasattr(model_card, 'get_expert_steering_thresholds'):
        kl_threshold = model_card.get_expert_steering_thresholds().get('kl_threshold', 1.0)
    else:
        kl_threshold = 1.0

    n_experts = len(expert_ranks)
    total = n_experts * len(positions) * len(grid_coeffs)
    n_judge = max(math.ceil(total * 0.025), 15)

    print(f"\n  Grid search (2-stage): {n_experts} experts x {len(positions)} positions x "
          f"{len(grid_coeffs)} coeffs = {total} combos")
    print(f"  Stage 1: refusal score + KL-div sweep (all {total} combos)")
    print(f"  Stage 2: OpenAI judge on top {n_judge} ({max(2.5, round(100*n_judge/total, 1))}%) candidates")
    print(f"  Positions: {positions}, Coeffs: {grid_coeffs}")
    print(f"  KL threshold: {kl_threshold}")
    print(f"  Judge samples: {n_samples}, max_new_tokens: {max_new_tokens}")

    # -------------------------------------------------------------------------
    # Stage 1: fast refusal score + KL-div sweep
    # -------------------------------------------------------------------------
    print(f"\n  --- Stage 1: Refusal score + KL-div sweep ---")

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
            mean_diff = expert_data[rank]

            mlp_module = model_card.get_mlp_module(layer)

            for position in positions:
                direction = mean_diff[position].to(model_base.model.device, dtype=model_base.model.dtype)

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

    print(f"\n  Stage 1 top {len(candidates)} candidates (sorted by refusal score):")
    print(f"  {'Rank':>5} {'Layer':>6} {'Expert':>7} {'Pos':>5} {'Coeff':>8} {'Refusal':>10} {'Reduction':>10} {'KL':>8}")
    print(f"  {'-'*5} {'-'*6} {'-'*7} {'-'*5} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for r in candidates:
        print(f"  {r['rank']:>5} {r['layer']:>6} {r['expert_id']:>7} {r['position']:>5} "
              f"{r['coeff']:>8.1f} {r['refusal_score']:>10.4f} {r['refusal_reduction']:>10.4f} "
              f"{r['kl_div']:>8.4f}")

    # -------------------------------------------------------------------------
    # Stage 2: OpenAI judge on surviving candidates
    # -------------------------------------------------------------------------
    print(f"\n  --- Stage 2: OpenAI judge on {len(candidates)} candidates ---")

    combos_dir = None
    if output_dir is not None:
        combos_dir = os.path.join(output_dir, "combos")
        os.makedirs(combos_dir, exist_ok=True)

    best_asr = -1.0
    best_rank = None
    best_position = None
    best_coeff = None

    with tqdm(candidates, desc="  Stage 2 judge", unit="combo") as pbar:
        for candidate in pbar:
            rank = candidate["rank"]
            layer = candidate["layer"]
            expert_id = candidate["expert_id"]
            position = candidate["position"]
            coeff = candidate["coeff"]

            mean_diff = expert_data[rank]
            direction = mean_diff[position].to(model_base.model.device, dtype=model_base.model.dtype)

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
                openai_delay=0.1,
                verbose=False,
            )

            asr = evaluation.get("openai_success_rate", 0.0)
            counts = evaluation.get("openai_overall_counts", {})
            n_full = counts.get("full_response", 0)
            n_refusal = counts.get("refusal", 0)
            n_non = counts.get("non_response", 0)

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

            pbar.set_postfix_str(f"best ASR={best_asr:.0%}")

    judged = sorted(candidates, key=lambda x: (-x.get("asr", -1), x.get("non_response", 0)))
    not_judged = stage1_results[n_judge:]
    all_results = judged + not_judged

    print(f"\n  Top stage-2 results (by ASR):")
    print(f"  {'Rank':>5} {'Layer':>6} {'Expert':>7} {'Pos':>5} {'Coeff':>8} "
          f"{'ASR':>8} {'Full':>6} {'Ref':>6} {'Non':>6} {'RefScore':>10}")
    print(f"  {'-'*5} {'-'*6} {'-'*7} {'-'*5} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*10}")
    for r in judged[:10]:
        print(f"  {r['rank']:>5} {r['layer']:>6} {r['expert_id']:>7} {r['position']:>5} "
              f"{r['coeff']:>8.1f} {r.get('asr', 0):>8.2%} "
              f"{r.get('full_response', '-'):>6} {r.get('refusal_count', '-'):>6} "
              f"{r.get('non_response', '-'):>6} {r['refusal_score']:>10.4f}")

    if best_rank is not None:
        bl, be = candidate_experts[best_rank - 1][0], candidate_experts[best_rank - 1][1]
        print(f"\n  Best: rank={best_rank} (L{bl} E{be}), position={best_position}, "
              f"coeff={best_coeff}, ASR={best_asr:.2%}")

    return best_rank, best_position, best_coeff, all_results


def run_allex_grid_search(
    model_base, model_card, all_layer_directions,
    harmful_val,   # list of dicts
    harmless_val,  # list of strings
    grid_coeffs=None,
    max_new_tokens=25, n_samples=25, batch_size=32,
    output_dir=None
):
    """
    Two-stage judge-based grid search over (layer, position, coeff) for allex.

    Used by mode 2 (--allex).

    Stage 1: Fast forward-pass refusal scores + KL-div over all combos.
             Keeps top max(ceil(total * 0.025), 15) candidates.

    Stage 2: Generate short completions and run OpenAI judge on candidates.

    Returns:
        (best_layer, best_position, best_coeff, all_results)
    """
    from tqdm import tqdm
    import math

    if grid_coeffs is None:
        grid_coeffs = [25, 50, 75, 100, 150, 200, 250, 300]

    positions = list(range(-5, 0))
    layer_indices = sorted(all_layer_directions.keys())

    harmful_val_instructions = [x['instruction'] if isinstance(x, dict) else x for x in harmful_val]
    harmless_val_instructions = [x['instruction'] if isinstance(x, dict) else x for x in harmless_val]

    sample = random.sample(harmful_val, min(n_samples, len(harmful_val)))

    if hasattr(model_card, 'get_expert_steering_thresholds'):
        kl_threshold = model_card.get_expert_steering_thresholds().get('kl_threshold', 1.0)
    else:
        kl_threshold = 1.0

    total = len(layer_indices) * len(positions) * len(grid_coeffs)
    n_judge = max(math.ceil(total * 0.025), 15)

    print(f"\n  Allex grid (2-stage): {len(layer_indices)} layers x {len(positions)} pos x "
          f"{len(grid_coeffs)} coeffs = {total} combos")
    print(f"  Stage 1: refusal score + KL-div sweep (all {total} combos)")
    print(f"  Stage 2: OpenAI judge on top {n_judge} ({max(2.5, round(100*n_judge/total, 1))}%) candidates")
    print(f"  Positions: {positions}, Coeffs: {grid_coeffs}")
    print(f"  KL threshold: {kl_threshold}")
    print(f"  Judge samples: {n_samples}, max_new_tokens: {max_new_tokens}")

    # -------------------------------------------------------------------------
    # Stage 1
    # -------------------------------------------------------------------------
    print(f"\n  --- Stage 1: Refusal score + KL-div sweep ---")

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
        for layer_idx in layer_indices:
            dirs = all_layer_directions[layer_idx]  # [n_experts, n_pos, d_model]
            mlp_module = model_card.get_mlp_module(layer_idx)

            for position in positions:
                dirs_at_pos = dirs[:, position, :]  # [n_experts, d_model]

                for coeff in grid_coeffs:
                    hook = get_all_expert_weighted_activation_addition_hook(
                        directions_for_layer=dirs_at_pos,
                        coeff=-coeff,
                        model_card=model_card
                    )
                    fwd_hooks = [(mlp_module, hook)]

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
                        "layer": layer_idx,
                        "position": position,
                        "coeff": coeff,
                        "refusal_score": mean_score,
                        "refusal_reduction": baseline_mean - mean_score,
                        "kl_div": kl_div,
                        "passed_kl": passed_kl,
                    })
                    pbar.update(1)
                    pbar.set_postfix_str(f"score={mean_score:.3f} kl={kl_div:.3f}{'✓' if passed_kl else '✗'}")

    passing = [r for r in stage1_results if r["passed_kl"]]
    passing.sort(key=lambda x: x["refusal_score"])
    print(f"\n  Stage 1: {len(passing)} passed KL filter (threshold={kl_threshold}), "
          f"{len(stage1_results) - len(passing)} excluded")

    if not passing:
        print(f"  WARNING: No combos passed KL filter. Falling back to lowest-KL combos.")
        stage1_results.sort(key=lambda x: x["kl_div"])
        passing = stage1_results[:n_judge]

    candidates = passing[:n_judge]

    print(f"\n  Stage 1 top {len(candidates)} candidates (sorted by refusal score):")
    print(f"  {'Layer':>7} {'Pos':>5} {'Coeff':>8} {'Refusal':>10} {'Reduction':>10} {'KL':>8}")
    print(f"  {'-'*7} {'-'*5} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for r in candidates:
        print(f"  {r['layer']:>7} {r['position']:>5} {r['coeff']:>8.1f} "
              f"{r['refusal_score']:>10.4f} {r['refusal_reduction']:>10.4f} {r['kl_div']:>8.4f}")

    # -------------------------------------------------------------------------
    # Stage 2
    # -------------------------------------------------------------------------
    print(f"\n  --- Stage 2: OpenAI judge on {len(candidates)} candidates ---")

    combos_dir = None
    if output_dir is not None:
        combos_dir = os.path.join(output_dir, "combos")
        os.makedirs(combos_dir, exist_ok=True)

    best_asr = -1.0
    best_layer = None
    best_position = None
    best_coeff = None

    with tqdm(candidates, desc="  Stage 2 judge", unit="combo") as pbar:
        for candidate in pbar:
            layer_idx = candidate["layer"]
            position = candidate["position"]
            coeff = candidate["coeff"]

            dirs = all_layer_directions[layer_idx]
            dirs_at_pos = dirs[:, position, :].to(model_base.model.device, dtype=model_base.model.dtype)

            fwd_pre_hooks, fwd_hooks = get_all_expert_weighted_intervention_hooks(
                model_base, layer_idx=layer_idx,
                directions_for_layer=dirs_at_pos,
                coeff=-coeff, model_card=model_card
            )

            completions = model_base.generate_completions(
                sample,
                fwd_pre_hooks=fwd_pre_hooks,
                fwd_hooks=fwd_hooks,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size
            )

            combo_label = f"L{layer_idx}_pos{position}_c{coeff}"
            eval_path = os.path.join(combos_dir, f"{combo_label}_eval.json") if combos_dir else os.devnull

            evaluation = evaluate_jailbreak(
                completions=completions,
                methodologies=["openai"],
                evaluation_path=eval_path,
                openai_delay=0.1,
                verbose=False,
            )

            asr = evaluation.get("openai_success_rate", 0.0)
            counts = evaluation.get("openai_overall_counts", {})
            n_full = counts.get("full_response", 0)
            n_refusal = counts.get("refusal", 0)
            n_non = counts.get("non_response", 0)

            candidate.update({
                "asr": asr,
                "full_response": n_full,
                "refusal_count": n_refusal,
                "non_response": n_non,
            })

            if asr > best_asr:
                best_asr = asr
                best_layer = layer_idx
                best_position = position
                best_coeff = coeff

            pbar.set_postfix_str(f"best ASR={best_asr:.0%}")

    judged = sorted(candidates, key=lambda x: (-x.get("asr", -1), x.get("non_response", 0)))
    not_judged = stage1_results[n_judge:]
    all_results = judged + not_judged

    print(f"\n  Top stage-2 results (by ASR):")
    print(f"  {'Layer':>7} {'Pos':>5} {'Coeff':>8} {'ASR':>8} "
          f"{'Full':>6} {'Ref':>6} {'Non':>6} {'RefScore':>10}")
    print(f"  {'-'*7} {'-'*5} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*10}")
    for r in judged[:10]:
        print(f"  {r['layer']:>7} {r['position']:>5} {r['coeff']:>8.1f} "
              f"{r.get('asr', 0):>8.2%} {r.get('full_response', '-'):>6} "
              f"{r.get('refusal_count', '-'):>6} {r.get('non_response', '-'):>6} "
              f"{r['refusal_score']:>10.4f}")

    if best_layer is not None:
        print(f"\n  Best: L{best_layer}, pos={best_position}, coeff={best_coeff}, ASR={best_asr:.2%}")

    return best_layer, best_position, best_coeff, all_results


# =============================================================================
# Full evaluation (post-grid-search)
# =============================================================================

def run_single_experiment(
    args, cfg, model_base, expert_entry, rank,
    mean_diff, position,
    harmful_test, harmless_test, base_output_dir,
    coeff_override
):
    """
    Run generate+evaluate for one (expert, position, coeff) combination.

    Args:
        expert_entry: (layer, expert_id, diff_pct) tuple
        rank: 1-indexed rank of this expert by diff (for subdir naming)
        mean_diff: [n_positions, d_model] tensor for this expert
        position: int, the token position index to use
        base_output_dir: root output dir for this run
        coeff_override: coeff to use for the actadd intervention
    """
    layer, expert_id, diff_pct = expert_entry
    coeff = coeff_override

    output_dir = base_output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n" + "#"*80)
    print(f"# EXPERIMENT: rank={rank} (L{layer} E{expert_id}, diff={diff_pct:.2f}%), "
          f"position={position}, coeff={coeff}")
    print(f"# Output: {output_dir}")
    print("#" + "#"*79)

    direction = mean_diff[position].to(model_base.model.device, dtype=model_base.model.dtype)

    metadata = {
        "layer": int(layer),
        "expert": int(expert_id),
        "diff_pct": float(diff_pct),
        "rank": rank,
        "position": position,
        "coeff": coeff,
        "direction_norm": direction.norm().item(),
        "selection_method": "judge_grid"
    }
    with open(os.path.join(output_dir, "experiment_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    expert_info = (layer, expert_id, direction)

    def _generate_eval(dataset_name, coeff_val, intervention_label, eval_methodologies, dataset=None):
        completions_dir = os.path.join(output_dir, 'completions', intervention_label)
        os.makedirs(completions_dir, exist_ok=True)

        if dataset is None:
            dataset = load_dataset(dataset_name)
            dataset = random.sample(dataset, min(100, len(dataset)))

        fwd_pre_hooks, fwd_hooks = get_expert_weighted_intervention_hooks(
            model_base,
            layer_idx=layer,
            expert_id=expert_id,
            direction=direction,
            coeff=coeff_val
        )

        print(f"\nGenerating completions for {dataset_name} with {intervention_label}...")
        completions = model_base.generate_completions(
            dataset,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            max_new_tokens=args.max_new_tokens
        )

        completions_path = os.path.join(completions_dir, f"{dataset_name}_completions.json")
        with open(completions_path, "w", encoding="utf-8") as f:
            json.dump(completions, f, indent=4, ensure_ascii=False)
        print(f"Saved completions to: {completions_path}")

        evaluation = evaluate_jailbreak(
            completions=completions,
            methodologies=eval_methodologies,
            evaluation_path=os.path.join(completions_dir, f"{dataset_name}_evaluations.json"),
        )
        eval_path = os.path.join(completions_dir, f"{dataset_name}_evaluations.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=4, ensure_ascii=False)
        print(f"Saved evaluations to: {eval_path}")

    # Baseline
    if not args.skip_baseline:
        print("\n  " + "-"*60)
        print("  BASELINE (No Intervention)")
        print("  " + "-"*60)

        for dataset_name in cfg.evaluation_datasets:
            _generate_eval(dataset_name, 0.0, 'baseline', cfg.jailbreak_eval_methodologies)

        if not args.skip_harmless:
            _generate_eval('harmless', 0.0, 'baseline', cfg.refusal_eval_methodologies, dataset=harmless_test)

    # ActAdd intervention
    print("\n  " + "-"*60)
    print(f"  ACTADD INTERVENTION (coeff={-coeff})")
    print("  " + "-"*60)

    for dataset_name in cfg.evaluation_datasets:
        _generate_eval(dataset_name, -coeff, 'actadd', cfg.jailbreak_eval_methodologies)

    if not args.skip_harmless:
        _generate_eval('harmless', coeff, 'actadd', cfg.refusal_eval_methodologies, dataset=harmless_test)


def run_single_allex_experiment(
    args, cfg, model_base, model_card,
    all_layer_directions, best_layer, best_position, best_coeff,
    harmful_test, harmless_test, output_dir
):
    """Run full evaluation for the best allex (layer, position, coeff) combo."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n" + "#"*80)
    print(f"# ALLEX EXPERIMENT: L{best_layer} pos={best_position} coeff={best_coeff}")
    print(f"# Output: {output_dir}")
    print("#" + "#"*79)

    best_dirs = all_layer_directions[best_layer][:, best_position, :]
    dirs_dev = best_dirs.to(model_base.model.device, dtype=model_base.model.dtype)

    def _generate_eval(dataset_name, coeff_val, intervention_label, eval_methodologies, dataset=None):
        completions_dir = os.path.join(output_dir, 'completions', intervention_label)
        os.makedirs(completions_dir, exist_ok=True)

        if dataset is None:
            dataset = load_dataset(dataset_name)
            dataset = random.sample(dataset, min(100, len(dataset)))

        fwd_pre_hooks, fwd_hooks = get_all_expert_weighted_intervention_hooks(
            model_base, layer_idx=best_layer,
            directions_for_layer=dirs_dev,
            coeff=coeff_val, model_card=model_card
        )

        print(f"\nGenerating completions for {dataset_name} with {intervention_label}...")
        completions = model_base.generate_completions(
            dataset,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            max_new_tokens=args.max_new_tokens
        )

        completions_path = os.path.join(completions_dir, f"{dataset_name}_completions.json")
        with open(completions_path, "w", encoding="utf-8") as f:
            json.dump(completions, f, indent=4, ensure_ascii=False)
        print(f"Saved completions to: {completions_path}")

        evaluation = evaluate_jailbreak(
            completions=completions,
            methodologies=eval_methodologies,
            evaluation_path=os.path.join(completions_dir, f"{dataset_name}_evaluations.json"),
        )
        eval_path = os.path.join(completions_dir, f"{dataset_name}_evaluations.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=4, ensure_ascii=False)
        print(f"Saved evaluations to: {eval_path}")

    if not args.skip_baseline:
        print("\n  " + "-"*60)
        print("  BASELINE (No Intervention)")
        print("  " + "-"*60)

        for dataset_name in cfg.evaluation_datasets:
            _generate_eval(dataset_name, 0.0, 'baseline', cfg.jailbreak_eval_methodologies)

        if not args.skip_harmless:
            _generate_eval('harmless', 0.0, 'baseline', cfg.refusal_eval_methodologies, dataset=harmless_test)

    print("\n  " + "-"*60)
    print(f"  ACTADD INTERVENTION (coeff={-best_coeff})")
    print("  " + "-"*60)

    for dataset_name in cfg.evaluation_datasets:
        _generate_eval(dataset_name, -best_coeff, 'actadd', cfg.jailbreak_eval_methodologies)

    if not args.skip_harmless:
        _generate_eval('harmless', best_coeff, 'actadd', cfg.refusal_eval_methodologies, dataset=harmless_test)


# =============================================================================
# Main pipeline
# =============================================================================

def run_pipeline(args):
    """Run the consolidated expert steering pipeline."""

    # ------------------------------------------------------------------
    # Determine mode
    # ------------------------------------------------------------------
    if args.allex:
        mode = "allex"
    elif args.layer is not None and args.position is not None:
        mode = "by_name_pos"
    elif args.layer is not None:
        mode = "by_name"
    elif args.expert_rank is not None:
        mode = "rank"
    else:
        mode = "threshold"

    # ------------------------------------------------------------------
    # Print startup header
    # ------------------------------------------------------------------
    print("="*80)
    print("EXPERT STEERING PIPELINE")
    print("="*80)
    print(f"Model: {args.model_path}")

    if mode == "threshold":
        threshold_display = args.threshold if args.threshold is not None else "(from model card)"
        print(f"Mode: THRESHOLD GRID      (threshold={threshold_display})")
    elif mode == "allex":
        print(f"Mode: ALLEX GRID          (all experts, all layers)")
    elif mode == "rank":
        print(f"Mode: RANK GRID           (top {args.expert_rank} experts by diff rank)")
    elif mode == "by_name":
        print(f"Mode: BY NAME GRID        (L{args.layer} E{args.expert}, grid over positions x coeffs)")
    elif mode == "by_name_pos":
        print(f"Mode: BY NAME+POS GRID    (L{args.layer} E{args.expert} position={args.position}, grid over coeffs)")

    print(f"Grid coeffs: {args.grid_coeffs}")
    print(f"Expert type: {args.expert_type}")
    print(f"Skip harmless: {args.skip_harmless}")
    print("="*80)

    base_model_name = os.path.basename(args.model_path)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    config_kwargs = {
        "model_alias": f"{base_model_name}/expert_steering_{mode}/sys_prompt_{args.system_prompt}",
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
        print(f"Using custom evaluation datasets: {cfg.evaluation_datasets}")

    base_output_dir = os.path.join(cfg.artifact_path(), "grid_search")
    os.makedirs(base_output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    model_base = construct_model_base(args.model_path, system_prompt=cfg.system_prompt)

    from model_utils.model_card_factory import create_model_card
    model_card = create_model_card(model_base)
    print(f"Model card: {type(model_card).__name__}")

    # Resolve threshold (modes 1, 3, and implicitly 4/5 for diff generation)
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = model_card.get_expert_steering_thresholds().get('expert_diff_threshold', 15.0)
    print(f"Expert diff threshold: {threshold}%")

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)
    (harmful_train, harmless_train, harmful_val, harmless_val,
     harmful_test, harmless_test) = load_and_sample_datasets(cfg)

    print(f"Raw data:")
    print(f"  Training: {len(harmful_train)} harmful, {len(harmless_train)} harmless")
    print(f"  Validation: {len(harmful_val)} harmful, {len(harmless_val)} harmless")
    print(f"  Test: {len(harmful_test)} harmful, {len(harmless_test)} harmless")

    print("\nFiltering training data based on refusal scores...")
    harmful_train, harmless_train = filter_data(cfg, model_base, harmful_train, harmless_train)

    print(f"\nFiltered training data:")
    print(f"  Training: {len(harmful_train)} harmful, {len(harmless_train)} harmless")

    # ------------------------------------------------------------------
    # Mode 2: ALLEX
    # ------------------------------------------------------------------
    if mode == "allex":
        shared_cache_dir = _get_shared_cache_dir(base_model_name, cfg.system_prompt)

        print("\n" + "="*80)
        print("LOADING / COMPUTING ALL-EXPERT DIRECTIONS")
        print("="*80)
        print(f"  Cache: {shared_cache_dir}/layer_{{L}}.pt")

        all_layer_directions = {}
        num_layers = model_card.get_num_layers()
        for layer_idx in range(num_layers):
            if not model_card.is_moe_layer(layer_idx):
                continue
            n_experts = model_card.get_num_experts(layer_idx)
            layer_cache = _load_or_compute_layer_directions(
                layer_idx=layer_idx,
                expert_ids_needed=list(range(n_experts)),
                model_base=model_base,
                harmful_train=harmful_train,
                harmless_train=harmless_train,
                batch_size=args.batch_size,
                cache_dir=shared_cache_dir,
                force_generate=args.force_generate,
            )
            # Stack in expert-id order: [n_experts, n_pos, d_model]
            all_layer_directions[layer_idx] = torch.stack(
                [layer_cache[eid] for eid in range(n_experts)], dim=0
            )

        print(f"\n  Loaded/computed directions for {len(all_layer_directions)} MoE layers")

        print("\n" + "="*80)
        print("ALLEX GRID SEARCH (LAYER x POSITION x COEFF)")
        print("="*80)

        grid_output_dir = os.path.join(base_output_dir, "allex_grid_search")
        os.makedirs(grid_output_dir, exist_ok=True)

        best_layer, best_position, best_coeff, all_results = run_allex_grid_search(
            model_base=model_base,
            model_card=model_card,
            all_layer_directions=all_layer_directions,
            harmful_val=harmful_val,
            harmless_val=harmless_val,
            grid_coeffs=args.grid_coeffs,
            max_new_tokens=args.judge_grid_tokens,
            n_samples=args.judge_grid_n_samples,
            batch_size=args.batch_size,
            output_dir=grid_output_dir
        )

        with open(os.path.join(grid_output_dir, "allex_grid_results.json"), 'w') as f:
            json.dump({
                "best_layer": int(best_layer),
                "best_position": int(best_position),
                "best_coeff": float(best_coeff),
                "n_layers": len(all_layer_directions),
                "max_new_tokens": args.judge_grid_tokens,
                "n_samples": args.judge_grid_n_samples,
                "grid_coeffs": args.grid_coeffs,
                "results": all_results
            }, f, indent=2)

        print(f"\nSaved allex grid results to: {grid_output_dir}/allex_grid_results.json")

        # Full evaluation
        eval_output_dir = os.path.join(base_output_dir, f"allex_L{best_layer}_pos{best_position}_c{best_coeff}")
        run_single_allex_experiment(
            args=args, cfg=cfg,
            model_base=model_base, model_card=model_card,
            all_layer_directions=all_layer_directions,
            best_layer=best_layer, best_position=best_position, best_coeff=best_coeff,
            harmful_test=harmful_test, harmless_test=harmless_test,
            output_dir=eval_output_dir
        )

        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print(f"Results saved under: {base_output_dir}")
        return

    # ------------------------------------------------------------------
    # Modes 1, 3, 4, 5 — per-expert directions
    # ------------------------------------------------------------------

    # Select candidate experts
    print("\n" + "="*80)
    print("SELECTING CANDIDATE EXPERTS")
    print("="*80)

    expert_diffs_filename = model_card.get_expert_diffs_filename()
    expert_diffs_dir = f"expert_diffs/sys_prompt_{cfg.system_prompt}"
    expert_diffs_path = os.path.join(expert_diffs_dir, expert_diffs_filename)

    if mode in ("by_name", "by_name_pos"):
        # Modes 4 & 5: single named expert — no threshold needed
        candidate_experts = [(args.layer, args.expert, 0.0)]
        expert_ranks = [1]
        print(f"  Named expert: L{args.layer} E{args.expert} (diff_pct unknown; 0.0 placeholder)")
    else:
        # Modes 1 & 3: load or generate expert diffs
        if not os.path.exists(expert_diffs_path):
            print(f"Expert diffs not found at {expert_diffs_path}, generating...")
            os.makedirs(expert_diffs_dir, exist_ok=True)
            model_card.generate_expert_diffs(
                harmful_dataset_path="dataset/splits/harmful_train.json",
                harmless_dataset_path="dataset/splits/harmless_train.json",
                output_path=expert_diffs_path,
                batch_size=args.batch_size
            )
            print(f"Expert diffs saved to {expert_diffs_path}")

        candidate_experts = get_candidate_experts(
            threshold=threshold,
            expert_type=args.expert_type,
            expert_diffs_path=expert_diffs_path,
        )

        if mode == "rank":
            # Mode 3: top N experts
            n = args.expert_rank
            if n > len(candidate_experts):
                print(f"ERROR: Requested top {n} experts but only {len(candidate_experts)} above threshold")
                sys.exit(1)
            expert_ranks = list(range(1, n + 1))
            print(f"\nMode 3: using top {n} experts (ranks 1-{n})")
        else:
            # Mode 1: all experts above threshold
            expert_ranks = list(range(1, len(candidate_experts) + 1))
            n_selected = len(candidate_experts)
            print(f"\nMode 1: threshold={threshold}%, {n_selected} experts selected")
            # Update mode header now that we know the count
            print(f"Mode: THRESHOLD GRID      (threshold={threshold}%, {n_selected} experts selected)")

    print(f"\nExperts to evaluate:")
    for rank in expert_ranks:
        layer, expert_id, diff_pct = candidate_experts[rank - 1]
        print(f"  Rank {rank}: Layer {layer}, Expert {expert_id}, Diff {diff_pct:.2f}%")

    print("\n" + "="*80)
    print("LOADING / COMPUTING DIRECTIONS")
    print("="*80)

    shared_cache_dir = _get_shared_cache_dir(base_model_name, cfg.system_prompt)
    print(f"  Cache: {shared_cache_dir}/layer_{{L}}.pt")

    # Group expert_ids by layer so each layer's cache file is touched once
    from collections import defaultdict
    layer_to_ranks = defaultdict(list)  # layer_idx -> [(rank, expert_id)]
    for rank in expert_ranks:
        layer, expert_id, _ = candidate_experts[rank - 1]
        layer_to_ranks[layer].append((rank, expert_id))

    expert_data = {}  # rank -> mean_diff tensor
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
    # Mode 5: BY NAME+POS — grid over coeffs only (position fixed)
    # ------------------------------------------------------------------
    if mode == "by_name_pos":
        layer, expert_id, diff_pct = candidate_experts[0]
        mean_diff = expert_data[1]
        position = args.position

        print("\n" + "="*80)
        print(f"MODE 5: COEFF GRID (L{layer} E{expert_id}, position={position})")
        print("="*80)

        # Use run_grid_search with a single (rank=1, position=fixed) entry
        # We build a mini candidate_experts list with position baked in.
        # We'll call a simplified single-position grid.

        print(f"  Grid over {len(args.grid_coeffs)} coeffs (position fixed at {position})")
        print(f"  Coeffs: {args.grid_coeffs}")

        harmful_val_instructions = [x['instruction'] if isinstance(x, dict) else x for x in harmful_val]
        harmless_val_instructions = [x['instruction'] if isinstance(x, dict) else x for x in harmless_val]

        if hasattr(model_card, 'get_expert_steering_thresholds'):
            kl_threshold = model_card.get_expert_steering_thresholds().get('kl_threshold', 1.0)
        else:
            kl_threshold = 1.0

        from tqdm import tqdm
        import math

        sample = harmful_val[:args.judge_grid_n_samples]
        direction = mean_diff[position].to(model_base.model.device, dtype=model_base.model.dtype)
        mlp_module = model_card.get_mlp_module(layer)

        baseline_scores = get_refusal_scores(
            model_base.model, harmful_val_instructions,
            model_base.tokenize_instructions_fn, model_base.refusal_toks,
            fwd_hooks=[], batch_size=args.batch_size,
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
            batch_size=args.batch_size
        )

        stage1_results = []
        with tqdm(total=len(args.grid_coeffs), desc="  Coeff sweep") as pbar:
            for coeff in args.grid_coeffs:
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
                    fwd_hooks=fwd_hooks, batch_size=args.batch_size,
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
                    batch_size=args.batch_size
                )
                kl_div = kl_div_fn(
                    baseline_harmless_logits, intervention_logits, mask=None
                ).mean(dim=0).item()
                passed_kl = kl_div <= kl_threshold

                stage1_results.append({
                    "rank": 1,
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

        passing = [r for r in stage1_results if r["passed_kl"]]
        passing.sort(key=lambda x: x["refusal_score"])
        if not passing:
            print(f"  WARNING: No combos passed KL filter. Using lowest-KL coeff.")
            stage1_results.sort(key=lambda x: x["kl_div"])
            passing = stage1_results[:1]

        n_judge = max(math.ceil(len(args.grid_coeffs) * 0.025), min(5, len(passing)))
        candidates = passing[:n_judge]

        print(f"\n  Stage 2: judging top {len(candidates)} coeff candidates...")

        grid_output_dir = os.path.join(base_output_dir, f"mode5_L{layer}_E{expert_id}_pos{position}")
        os.makedirs(grid_output_dir, exist_ok=True)
        combos_dir = os.path.join(grid_output_dir, "combos")
        os.makedirs(combos_dir, exist_ok=True)

        best_asr = -1.0
        best_coeff = None

        for i, candidate in enumerate(candidates):
            coeff = candidate["coeff"]
            print(f"\n  [{i+1}/{len(candidates)}] coeff={coeff} (refusal={candidate['refusal_score']:.4f})")

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
                max_new_tokens=args.judge_grid_tokens,
                batch_size=args.batch_size
            )

            eval_path = os.path.join(combos_dir, f"L{layer}_E{expert_id}_pos{position}_c{coeff}_eval.json")
            evaluation = evaluate_jailbreak(
                completions=completions,
                methodologies=["openai"],
                evaluation_path=eval_path,
                openai_delay=0.1
            )

            asr = evaluation.get("openai_success_rate", 0.0)
            counts = evaluation.get("openai_overall_counts", {})
            print(f"    ASR={asr:.2%} (full={counts.get('full_response',0)}, "
                  f"refusal={counts.get('refusal',0)}, non_response={counts.get('non_response',0)})")

            candidate.update({"asr": asr, **{k: counts.get(k, 0)
                                              for k in ["full_response", "refusal", "non_response"]}})

            if asr > best_asr:
                best_asr = asr
                best_coeff = coeff

        all_results = sorted(candidates, key=lambda x: -x.get("asr", -1)) + \
                      [r for r in stage1_results if r not in candidates]

        with open(os.path.join(grid_output_dir, "coeff_grid_results.json"), 'w') as f:
            json.dump({
                "best_layer": layer, "best_expert_id": expert_id,
                "best_position": position, "best_coeff": best_coeff,
                "results": all_results
            }, f, indent=2)

        if best_coeff is not None:
            print(f"\n  Best: coeff={best_coeff}, ASR={best_asr:.2%}")
            run_single_experiment(
                args=args, cfg=cfg, model_base=model_base,
                expert_entry=(layer, expert_id, diff_pct), rank=1,
                mean_diff=mean_diff, position=position,
                harmful_test=harmful_test, harmless_test=harmless_test,
                base_output_dir=os.path.join(base_output_dir, f"mode5_L{layer}_E{expert_id}_pos{position}_c{best_coeff}"),
                coeff_override=best_coeff
            )

        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print(f"Results saved under: {base_output_dir}")
        return

    # ------------------------------------------------------------------
    # Modes 1, 3, 4: run_grid_search (judge grid over experts x positions x coeffs)
    # ------------------------------------------------------------------
    print("\n" + "="*80)
    print("JUDGE GRID SEARCH")
    print("="*80)

    grid_output_dir = os.path.join(base_output_dir, "judge_grid_search")
    os.makedirs(grid_output_dir, exist_ok=True)

    best_rank, best_position, best_coeff, all_results = run_grid_search(
        model_base=model_base,
        model_card=model_card,
        expert_data=expert_data,
        candidate_experts=candidate_experts,
        expert_ranks=expert_ranks,
        harmful_val=harmful_val,
        harmless_val=harmless_val,
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

    # Full evaluation with best combo
    best_mean_diff = expert_data[best_rank]
    print(f"\n  Running full evaluation: L{best_layer} E{best_expert_id} "
          f"pos={best_position} coeff={best_coeff}")
    run_single_experiment(
        args=args, cfg=cfg, model_base=model_base,
        expert_entry=(best_layer, best_expert_id, best_diff_pct),
        rank=best_rank,
        mean_diff=best_mean_diff,
        position=best_position,
        harmful_test=harmful_test, harmless_test=harmless_test,
        base_output_dir=base_output_dir,
        coeff_override=best_coeff
    )

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Results saved under: {base_output_dir}")


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(args)
