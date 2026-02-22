"""
Alternative Expert Steering Interventions

Two methods:
  --method unweighted: Same expert selection as original, but applies steering
      vector without multiplying by expert's routing probability.
  --method allex: Generates mean-diff vectors for ALL experts at ALL MoE layers,
      then at inference applies the router-weighted sum of all expert vectors.

Direction caching is shared across methods/coefficients under:
  runs/{model}/expert_directions/sys_prompt_{sp}/t{threshold}/   (unweighted)
  runs/{model}/expert_directions/sys_prompt_{sp}/all_experts/     (allex)

Modes (allex):
    Single coeff, Arditi selection:
        python run_expert_steering_alt.py --method allex --coeff 100
    Judge-based grid search over (layer, position, coeff):
        python run_expert_steering_alt.py --method allex --judge_grid
        python run_expert_steering_alt.py --method allex --judge_grid --judge_grid_tokens 50 --judge_grid_n_samples 25
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
    get_expert_unweighted_intervention_hooks,
    get_all_expert_weighted_intervention_hooks,
    get_all_expert_weighted_activation_addition_hook,
)
from submodules.expert_steering.select_direction_moe import (
    select_expert_direction_unweighted,
    select_layer_direction,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Alternative expert steering methods (unweighted / all-expert)"
    )

    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['unweighted', 'allex'],
        help='Steering method: unweighted (no router weighting) or allex (all-expert layer steering)'
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
        help='Expert selection threshold (percentage points). Only used by unweighted method.'
    )

    parser.add_argument(
        '--expert_type',
        type=str,
        default='both',
        choices=['harmful_preferred', 'harmless_preferred', 'both'],
        help='Which type of experts to select (unweighted only)'
    )

    parser.add_argument(
        '--skip_generate',
        action='store_true',
        help='Skip generation and load directions from cache'
    )

    parser.add_argument(
        '--skip_select',
        action='store_true',
        help='Skip selection and load from cache'
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
        help='Number of training samples'
    )

    parser.add_argument(
        '--n_val',
        type=int,
        default=None,
        help='Number of validation samples'
    )

    parser.add_argument(
        '--n_test',
        type=int,
        default=None,
        help='Number of test samples'
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
        help='Coefficient for activation addition (ignored when --judge_grid is set)'
    )

    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=100,
        help='Maximum new tokens for generation'
    )

    parser.add_argument(
        '--top_n',
        type=int,
        default=1,
        help='Number of top vectors to select (unweighted only, allex always picks 1 layer)'
    )

    parser.add_argument(
        '--eval_datasets',
        type=str,
        nargs='+',
        default=None,
        help='Evaluation datasets to use'
    )

    parser.add_argument(
        '--system_prompt',
        type=str,
        default=None,
        choices=['none', 'llama_2', 'lightweight'],
        help='System prompt to use'
    )

    # --- Judge grid search options (allex only) ---
    parser.add_argument(
        '--judge_grid',
        action='store_true',
        help='(allex only) Run judge-based grid search: generate short completions and use '
             'OpenAI judge to pick the best (layer, position, coeff) combo.'
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

    parser.add_argument(
        '--grid_coeffs',
        type=float,
        nargs='+',
        default=[25, 50, 75, 100, 150, 200, 250, 300],
        help='Coeff values to search over in judge grid (default: 25 50 75 100 150 200 250 300)'
    )

    return parser.parse_args()


# ============================================================================
# Shared helpers
# ============================================================================

def load_and_sample_datasets(cfg):
    random.seed(42)

    harmful_train_full = load_dataset_split(harmtype='harmful', split='train', instructions_only=True)
    harmless_train_full = load_dataset_split(harmtype='harmless', split='train', instructions_only=True)
    # Load harmful_val as dicts (instructions_only=False) so the judge stage can generate completions
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
    harmful_val = random.sample(harmful_val_full, n_val_actual)   # list of dicts
    harmless_val = random.sample(harmless_val_full, n_val_actual)
    harmful_test = random.sample(harmful_test_full, n_test_actual)
    harmless_test = random.sample(harmless_test_full, n_test_actual)

    return (harmful_train, harmless_train, harmful_val,
            harmless_val, harmful_test, harmless_test)


def _to_instructions(dataset):
    """Extract instruction strings from a dataset that may contain dicts or strings."""
    return [x['instruction'] if isinstance(x, dict) else x for x in dataset]


def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """Filter datasets based on refusal scores. harmful_val may be dicts or strings."""
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

    if cfg.filter_val:
        print("\n  Filtering validation data...")
        # harmful_val may be dicts; extract strings for scoring
        harmful_val_str = _to_instructions(harmful_val)
        harmless_val_str = _to_instructions(harmless_val)

        harmful_val_scores = get_refusal_scores(
            model_base.model, harmful_val_str, model_base.tokenize_instructions_fn,
            model_base.refusal_toks, tokenizer=model_base.tokenizer,
            refusal_score_suffix_toks=model_base.refusal_score_suffix_toks
        )
        harmless_val_scores = get_refusal_scores(
            model_base.model, harmless_val_str, model_base.tokenize_instructions_fn,
            model_base.refusal_toks, tokenizer=model_base.tokenizer,
            refusal_score_suffix_toks=model_base.refusal_score_suffix_toks
        )

        harmful_val_filtered = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val_filtered = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)

        print(f"    Harmful val: {len(harmful_val)} -> {len(harmful_val_filtered)}")
        print(f"    Harmless val: {len(harmless_val)} -> {len(harmless_val_filtered)}")

        harmful_val = harmful_val_filtered
        harmless_val = harmless_val_filtered

    return harmful_train, harmless_train, harmful_val, harmless_val


def create_candidate_tensor(expert_directions):
    """Convert expert directions dict to tensor + mapping."""
    first_key = list(expert_directions.keys())[0]
    first_tensor = expert_directions[first_key]
    n_positions, d_model = first_tensor.shape

    n_candidates = len(expert_directions)

    candidates = torch.zeros(n_positions, n_candidates, d_model,
                            dtype=first_tensor.dtype, device=first_tensor.device)

    candidate_mapping = {}

    for idx, ((layer, expert), direction) in enumerate(expert_directions.items()):
        candidates[:, idx, :] = direction
        candidate_mapping[idx] = (layer, expert)

    return candidates, candidate_mapping


# ============================================================================
# Direction generation
# ============================================================================

def generate_expert_specific_directions(
    model_base, harmful_train, harmless_train,
    candidate_experts, artifact_dir, batch_size=32
):
    """Generate mean difference vectors for each candidate expert."""
    os.makedirs(artifact_dir, exist_ok=True)

    expert_directions = {}

    print("\n" + "="*80)
    print("GENERATING EXPERT-SPECIFIC DIRECTIONS")
    print("="*80)

    for layer_idx, expert_id, diff_pct in candidate_experts:
        # Check cache
        save_path = os.path.join(
            artifact_dir,
            f"expert_L{layer_idx}_E{expert_id}_mean_diff.pt"
        )
        if os.path.exists(save_path):
            print(f"\n[Expert: Layer {layer_idx}, Expert {expert_id}] Loading from cache")
            mean_diff = torch.load(save_path)
            expert_directions[(layer_idx, expert_id)] = mean_diff
            continue

        print(f"\n[Expert: Layer {layer_idx}, Expert {expert_id}, Diff: {diff_pct:.2f}%]")

        mean_diff, _ = get_expert_mean_diff(
            model_base, harmful_train, harmless_train,
            layer_idx, expert_id, batch_size=batch_size
        )

        expert_directions[(layer_idx, expert_id)] = mean_diff

        torch.save(mean_diff, save_path)
        print(f"  Saved to: {save_path}")

    # Save combined
    combined_path = os.path.join(artifact_dir, "all_expert_directions.pt")
    torch.save(expert_directions, combined_path)
    print(f"\nSaved all expert directions to: {combined_path}")

    return expert_directions


def generate_all_expert_directions(
    model_base, harmful_train, harmless_train,
    model_card, artifact_dir, batch_size=32
):
    """
    Generate mean difference vectors for ALL experts at ALL MoE layers.

    Saves per-layer to enable incremental progress. Returns dict:
      {layer_idx: tensor[n_experts, n_pos, d_model]}
    """
    os.makedirs(artifact_dir, exist_ok=True)

    num_layers = model_card.get_num_layers()
    all_layer_directions = {}

    print("\n" + "="*80)
    print("GENERATING ALL-EXPERT DIRECTIONS")
    print("="*80)

    for layer_idx in range(num_layers):
        if not model_card.is_moe_layer(layer_idx):
            continue

        layer_path = os.path.join(artifact_dir, f"layer_{layer_idx}_all_experts.pt")

        # Check per-layer cache
        if os.path.exists(layer_path):
            print(f"\n[Layer {layer_idx}] Loading from cache")
            layer_dirs = torch.load(layer_path, map_location='cpu')
            all_layer_directions[layer_idx] = layer_dirs
            continue

        n_experts = model_card.get_num_experts(layer_idx)
        print(f"\n[Layer {layer_idx}] Generating directions for {n_experts} experts")

        expert_diffs = []
        for expert_id in range(n_experts):
            print(f"  Expert {expert_id}/{n_experts-1}")
            mean_diff, _ = get_expert_mean_diff(
                model_base, harmful_train, harmless_train,
                layer_idx, expert_id, batch_size=batch_size
            )
            expert_diffs.append(mean_diff)  # [n_pos, d_model]

        # Stack: [n_experts, n_pos, d_model]
        layer_dirs = torch.stack(expert_diffs, dim=0).cpu()
        all_layer_directions[layer_idx] = layer_dirs

        # Save immediately
        torch.save(layer_dirs, layer_path)
        print(f"  Saved layer {layer_idx} directions to: {layer_path}")

        # Free memory
        del expert_diffs
        torch.cuda.empty_cache()

    # Save combined
    combined_path = os.path.join(artifact_dir, "all_layer_expert_directions.pt")
    torch.save(all_layer_directions, combined_path)
    print(f"\nSaved all layer-expert directions to: {combined_path}")

    return all_layer_directions


# ============================================================================
# Evaluation helpers
# ============================================================================

def generate_and_evaluate_completions(
    model_base, dataset_name, get_hooks_fn,
    coeff, output_dir, intervention_label,
    eval_methodologies, max_new_tokens=256, dataset=None
):
    """
    Generate completions with intervention and evaluate.

    Args:
        get_hooks_fn: callable(coeff) -> (fwd_pre_hooks, fwd_hooks)
    """
    completions_dir = os.path.join(output_dir, 'completions', intervention_label)
    os.makedirs(completions_dir, exist_ok=True)

    if dataset is None:
        dataset = load_dataset(dataset_name)
        dataset = random.sample(dataset, min(100, len(dataset)))

    fwd_pre_hooks, fwd_hooks = get_hooks_fn(coeff)

    print(f"\nGenerating completions for {dataset_name} with {intervention_label}...")
    completions = model_base.generate_completions(
        dataset,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        max_new_tokens=max_new_tokens
    )

    completions_path = os.path.join(
        completions_dir,
        f"{dataset_name}_completions.json"
    )
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

    return completions, evaluation


def _run_evaluation(args, cfg, model_base, output_dir, get_hooks_fn, harmful_test, harmless_test,
                    coeff_override=None):
    """
    Run baseline + intervention evaluation.

    Args:
        coeff_override: If set, use this coeff for actadd instead of args.coeff.
                        Used by judge_grid to evaluate with the grid-selected coeff.
    """
    actual_coeff = coeff_override if coeff_override is not None else args.coeff

    if not args.skip_baseline:
        print("\n" + "-"*80)
        print("BASELINE (No Intervention)")
        print("-"*80)

        def baseline_hooks(c):
            return [], []

        for dataset_name in cfg.evaluation_datasets:
            generate_and_evaluate_completions(
                model_base, dataset_name, baseline_hooks,
                coeff=0.0, output_dir=output_dir,
                intervention_label='baseline',
                eval_methodologies=cfg.jailbreak_eval_methodologies,
                max_new_tokens=args.max_new_tokens
            )

        if not args.skip_harmless:
            generate_and_evaluate_completions(
                model_base, 'harmless', baseline_hooks,
                coeff=0.0, output_dir=output_dir,
                intervention_label='baseline',
                eval_methodologies=cfg.refusal_eval_methodologies,
                max_new_tokens=args.max_new_tokens,
                dataset=harmless_test
            )

    # ActAdd intervention (suppress refusal)
    print("\n" + "-"*80)
    print(f"ACTADD INTERVENTION (coeff={-actual_coeff})")
    print("-"*80)

    for dataset_name in cfg.evaluation_datasets:
        generate_and_evaluate_completions(
            model_base, dataset_name, get_hooks_fn,
            coeff=-actual_coeff,  # Negative to suppress refusal
            output_dir=output_dir,
            intervention_label='actadd',
            eval_methodologies=cfg.jailbreak_eval_methodologies,
            max_new_tokens=args.max_new_tokens
        )

    if not args.skip_harmless:
        generate_and_evaluate_completions(
            model_base, 'harmless', get_hooks_fn,
            coeff=actual_coeff,  # Positive to induce refusal
            output_dir=output_dir,
            intervention_label='actadd',
            eval_methodologies=cfg.refusal_eval_methodologies,
            max_new_tokens=args.max_new_tokens,
            dataset=harmless_test
        )


# ============================================================================
# Method 1: Unweighted expert steering
# ============================================================================

def run_unweighted_pipeline(args, cfg, model_base, model_card,
                            harmful_train, harmless_train,
                            harmful_val, harmless_val,
                            harmful_test, harmless_test):
    """Run the unweighted expert steering pipeline."""

    print("\n" + "="*80)
    print("METHOD: UNWEIGHTED EXPERT STEERING")
    print("="*80)

    # --- Shared direction cache path ---
    base_model_name = os.path.basename(args.model_path)
    direction_cache_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "runs", base_model_name,
        "expert_directions",
        f"sys_prompt_{cfg.system_prompt}",
        f"t{args.threshold}"
    )

    # --- Method+coeff-specific output dir ---
    method_alias = f"{base_model_name}/expert_steering_unweighted_t{args.threshold}/sys_prompt_{cfg.system_prompt}"
    output_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "runs", method_alias, f"coeff_{args.coeff}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # --- Select candidate experts ---
    print("\n" + "="*80)
    print("SELECTING CANDIDATE EXPERTS")
    print("="*80)

    expert_diffs_filename = model_card.get_expert_diffs_filename()
    expert_diffs_path = f"expert_diffs/{expert_diffs_filename}"

    if not os.path.exists(expert_diffs_path):
        print(f"Expert diffs not found at {expert_diffs_path}, generating...")
        os.makedirs("expert_diffs", exist_ok=True)
        model_card.generate_expert_diffs(
            harmful_dataset_path="dataset/splits/harmful_train.json",
            harmless_dataset_path="dataset/splits/harmless_train.json",
            output_path=expert_diffs_path,
            batch_size=args.batch_size
        )

    candidate_experts = get_candidate_experts(
        threshold=args.threshold,
        expert_type=args.expert_type,
        expert_diffs_path=expert_diffs_path
    )

    expert_info_path = os.path.join(output_dir, "candidate_experts.json")
    with open(expert_info_path, 'w') as f:
        json.dump([
            {"layer": int(l), "expert": int(e), "diff_pct": float(d)}
            for l, e, d in candidate_experts
        ], f, indent=2)

    # --- Generate directions (shared cache) ---
    if not args.skip_generate:
        expert_directions = generate_expert_specific_directions(
            model_base, harmful_train, harmless_train,
            candidate_experts,
            artifact_dir=direction_cache_dir,
            batch_size=args.batch_size
        )
    else:
        print("\nSkipping generation, loading from cache...")
        directions_path = os.path.join(direction_cache_dir, "all_expert_directions.pt")
        expert_directions = torch.load(directions_path)

    # --- Select best direction(s) with unweighted hooks ---
    # Extract instruction strings from harmful_val (may be dicts)
    harmful_val_str = _to_instructions(harmful_val)
    harmless_val_str = _to_instructions(harmless_val)

    if not args.skip_select:
        candidates, candidate_mapping = create_candidate_tensor(expert_directions)
        mu_b = torch.zeros(candidates.size(-1), device=candidates.device)

        print("\n" + "="*80)
        if args.top_n == 1:
            print("SELECTING BEST EXPERT DIRECTION (UNWEIGHTED)")
        else:
            print(f"SELECTING TOP {args.top_n} EXPERT DIRECTIONS (UNWEIGHTED)")
        print("="*80)

        result = select_expert_direction_unweighted(
            model_base, harmful_val_str, harmless_val_str,
            candidates, candidate_mapping,
            artifact_dir=os.path.join(output_dir, "selection"),
            coeff=args.coeff,
            mu_b=mu_b,
            tau=1.0,
            top_n=args.top_n,
            model_card=model_card
        )

        if args.top_n == 1:
            pos, candidate_idx, direction = result
            layer, expert_id = candidate_mapping[candidate_idx]

            metadata = {
                "method": "unweighted",
                "top_n": 1,
                "pos": int(pos),
                "layer": int(layer),
                "expert_id": int(expert_id),
                "threshold": args.threshold,
                "expert_type": args.expert_type
            }
            with open(os.path.join(output_dir, "direction_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            torch.save(direction, os.path.join(output_dir, "direction.pt"))

            print(f"\nBest direction: Layer {layer}, Expert {expert_id}, Position {pos}")
        else:
            selected_directions = []
            for pos, candidate_idx, direction in result:
                layer, expert_id = candidate_mapping[candidate_idx]
                selected_directions.append((pos, layer, expert_id, direction))

            metadata = {
                "method": "unweighted",
                "top_n": args.top_n,
                "threshold": args.threshold,
                "expert_type": args.expert_type,
                "directions": [
                    {"pos": int(p), "layer": int(l), "expert_id": int(e)}
                    for p, l, e, _ in selected_directions
                ]
            }
            with open(os.path.join(output_dir, "direction_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            torch.save(
                {f"direction_{i}": d for i, (_, _, _, d) in enumerate(selected_directions)},
                os.path.join(output_dir, "directions.pt")
            )
    else:
        print("\nSkipping selection, loading from cache...")
        evaluations_path = os.path.join(output_dir, "selection", "expert_direction_evaluations_filtered.json")
        with open(evaluations_path, 'r') as f:
            filtered_evals = json.load(f)

        all_directions_path = os.path.join(direction_cache_dir, "all_expert_directions.pt")
        all_expert_directions = torch.load(all_directions_path)

        n_to_load = min(args.top_n, len(filtered_evals))

        if args.top_n == 1:
            entry = filtered_evals[0]
            pos = entry["position"]
            layer = entry["layer"]
            expert_id = entry["expert"]
            full_direction = all_expert_directions[(layer, expert_id)]
            direction = full_direction[pos]

            print(f"\nLoaded: Layer {layer}, Expert {expert_id}, Position {pos}")
        else:
            selected_directions = []
            for i in range(n_to_load):
                entry = filtered_evals[i]
                full_direction = all_expert_directions[(entry["layer"], entry["expert"])]
                direction = full_direction[entry["position"]]
                selected_directions.append((entry["position"], entry["layer"], entry["expert"], direction))

    # --- Evaluation ---
    print("\n" + "="*80)
    print("EVALUATION (UNWEIGHTED)")
    print("="*80)

    # Build hook factory for unweighted intervention
    if args.top_n == 1:
        direction_dev = direction.to(model_base.model.device, dtype=model_base.model.dtype)

        def get_hooks_fn(c):
            return get_expert_unweighted_intervention_hooks(
                model_base, layer_idx=layer, expert_id=expert_id,
                direction=direction_dev, coeff=c, model_card=model_card
            )
    else:
        dirs_on_device = [
            (l, e, d.to(model_base.model.device, dtype=model_base.model.dtype))
            for _, l, e, d in selected_directions
        ]

        def get_hooks_fn(c):
            all_pre = []
            all_fwd = []
            for l, e, d in dirs_on_device:
                pre, fwd = get_expert_unweighted_intervention_hooks(
                    model_base, layer_idx=l, expert_id=e,
                    direction=d, coeff=c, model_card=model_card
                )
                all_pre.extend(pre)
                all_fwd.extend(fwd)
            return all_pre, all_fwd

    _run_evaluation(args, cfg, model_base, output_dir, get_hooks_fn, harmful_test, harmless_test)


# ============================================================================
# Method 2: All-expert layer steering — judge grid search
# ============================================================================

def run_allex_judge_grid(
    model_base, model_card, all_layer_directions,
    harmful_val,  # list of dicts (instructions_only=False)
    harmless_val=None,  # list of dicts or strings for KL filter
    grid_coeffs=None,
    max_new_tokens=25, n_samples=25, batch_size=32,
    output_dir=None, kl_threshold=None
):
    """
    Two-stage judge-based grid search over (layer, position, coeff) for allex.

    Stage 1: Fast forward-pass refusal scores + KL-div over all combos. Combos
             that fail the KL filter are excluded. From the passing combos, keeps
             the top max(ceil(total * 0.025), 15) candidates by lowest refusal score.

    Stage 2: Generate short completions and run the OpenAI judge only on the
             surviving candidates.

    Per-combo completions and evaluations are saved to output_dir/combos/.

    Returns:
        (best_layer, best_position, best_coeff, all_results)
    """
    from tqdm import tqdm
    import math

    if grid_coeffs is None:
        grid_coeffs = [25, 50, 75, 100, 150, 200, 250, 300]

    positions = list(range(-5, 0))
    layer_indices = sorted(all_layer_directions.keys())

    # Extract instruction strings for stage 1 scoring
    harmful_val_instructions = _to_instructions(harmful_val)

    # Subsample for judge stage (dicts needed for generate_completions)
    sample = random.sample(harmful_val, min(n_samples, len(harmful_val)))

    # Resolve KL threshold
    use_kl = harmless_val is not None
    if use_kl:
        if kl_threshold is None:
            if hasattr(model_card, 'get_expert_steering_thresholds'):
                kl_threshold = model_card.get_expert_steering_thresholds().get('kl_threshold', 1.0)
            else:
                kl_threshold = 1.0
        harmless_val_instructions = _to_instructions(harmless_val)

    total = len(layer_indices) * len(positions) * len(grid_coeffs)
    n_judge = max(math.ceil(total * 0.025), 15)

    print(f"\n  Judge grid (2-stage): {len(layer_indices)} layers x {len(positions)} pos x "
          f"{len(grid_coeffs)} coeffs = {total} combos")
    print(f"  Stage 1: refusal score + {'KL-div ' if use_kl else ''}sweep (all {total} combos)")
    print(f"  Stage 2: OpenAI judge on top {n_judge} ({max(2.5, round(100*n_judge/total, 1))}%) candidates")
    print(f"  Positions: {positions}, Coeffs: {grid_coeffs}")
    print(f"  Judge samples: {n_samples}, max_new_tokens: {max_new_tokens}")
    if use_kl:
        print(f"  KL threshold: {kl_threshold}")

    # -------------------------------------------------------------------------
    # Stage 1: fast refusal score + optional KL-div sweep
    # -------------------------------------------------------------------------
    print(f"\n  --- Stage 1: Refusal score{' + KL-div' if use_kl else ''} sweep ---")

    baseline_scores = get_refusal_scores(
        model_base.model, harmful_val_instructions,
        model_base.tokenize_instructions_fn, model_base.refusal_toks,
        fwd_hooks=[], batch_size=batch_size,
        tokenizer=model_base.tokenizer,
        refusal_score_suffix_toks=model_base.refusal_score_suffix_toks
    )
    baseline_mean = baseline_scores.mean().item()
    print(f"  Baseline refusal score: {baseline_mean:.4f}")

    baseline_harmless_logits = None
    if use_kl:
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

                    kl_div = None
                    passed_kl = True
                    if use_kl:
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
                    status = f"score={mean_score:.3f}"
                    if use_kl:
                        status += f" kl={kl_div:.3f}{'✓' if passed_kl else '✗'}"
                    pbar.set_postfix_str(status)

    # Filter by KL, sort survivors by refusal score, fall back if nothing passes
    if use_kl:
        passing = [r for r in stage1_results if r["passed_kl"]]
        passing.sort(key=lambda x: x["refusal_score"])
        print(f"\n  Stage 1: {len(passing)} passed KL filter (threshold={kl_threshold}), "
              f"{len(stage1_results) - len(passing)} excluded")
        if not passing:
            print(f"  WARNING: No combos passed KL filter. Falling back to lowest-KL combos.")
            stage1_results.sort(key=lambda x: x["kl_div"])
            passing = stage1_results[:n_judge]
        candidates = passing[:n_judge]
    else:
        stage1_results.sort(key=lambda x: x["refusal_score"])
        candidates = stage1_results[:n_judge]

    print(f"\n  Stage 1 top {len(candidates)} candidates (passed KL filter, sorted by refusal score):")
    if use_kl:
        print(f"  {'Layer':>7} {'Pos':>5} {'Coeff':>8} {'Refusal':>10} {'Reduction':>10} {'KL':>8}")
        print(f"  {'-'*7} {'-'*5} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
        for r in candidates:
            print(f"  {r['layer']:>7} {r['position']:>5} {r['coeff']:>8.1f} "
                  f"{r['refusal_score']:>10.4f} {r['refusal_reduction']:>10.4f} {r['kl_div']:>8.4f}")
    else:
        print(f"  {'Layer':>7} {'Pos':>5} {'Coeff':>8} {'Refusal':>10} {'Reduction':>10}")
        print(f"  {'-'*7} {'-'*5} {'-'*8} {'-'*10} {'-'*10}")
        for r in candidates:
            print(f"  {r['layer']:>7} {r['position']:>5} {r['coeff']:>8.1f} "
                  f"{r['refusal_score']:>10.4f} {r['refusal_reduction']:>10.4f}")

    # -------------------------------------------------------------------------
    # Stage 2: OpenAI judge on surviving candidates
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

    for i, candidate in enumerate(candidates):
        layer_idx = candidate["layer"]
        position = candidate["position"]
        coeff = candidate["coeff"]

        print(f"\n  [{i+1}/{len(candidates)}] L{layer_idx} pos={position} coeff={coeff} "
              f"(refusal={candidate['refusal_score']:.4f})")

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
            best_layer = layer_idx
            best_position = position
            best_coeff = coeff

    # Build all_results: judged candidates sorted by ASR, then rest by refusal score
    judged = sorted(candidates, key=lambda x: (-x.get("asr", -1), x.get("non_response", 0)))
    not_judged = stage1_results[n_judge:]  # already sorted by refusal score
    all_results = judged + not_judged

    print(f"\n  Top stage-2 results (by ASR):")
    print(f"  {'Layer':>7} {'Pos':>5} {'Coeff':>8} {'ASR':>8} {'Full':>6} {'Ref':>6} {'Non':>6} {'RefScore':>10}")
    print(f"  {'-'*7} {'-'*5} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*10}")
    for r in judged[:10]:
        print(f"  {r['layer']:>7} {r['position']:>5} {r['coeff']:>8.1f} "
              f"{r.get('asr', 0):>8.2%} {r.get('full_response', '-'):>6} "
              f"{r.get('refusal_count', '-'):>6} {r.get('non_response', '-'):>6} "
              f"{r['refusal_score']:>10.4f}")

    if best_layer is not None:
        print(f"\n  Best: L{best_layer}, pos={best_position}, coeff={best_coeff}, ASR={best_asr:.2%}")

    return best_layer, best_position, best_coeff, all_results


# ============================================================================
# Method 2: All-expert layer steering — main pipeline
# ============================================================================

def run_allex_pipeline(args, cfg, model_base, model_card,
                       harmful_train, harmless_train,
                       harmful_val, harmless_val,
                       harmful_test, harmless_test):
    """Run the all-expert layer steering pipeline."""

    print("\n" + "="*80)
    print("METHOD: ALL-EXPERT LAYER STEERING")
    print("="*80)

    base_model_name = os.path.basename(args.model_path)
    method_alias = f"{base_model_name}/expert_steering_allex/sys_prompt_{cfg.system_prompt}"
    base_method_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "runs", method_alias
    )

    # Shared direction cache (independent of coeff)
    direction_cache_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "runs", base_model_name,
        "expert_directions",
        f"sys_prompt_{cfg.system_prompt}",
        "all_experts"
    )

    # Coeff-specific output dir (only relevant when not using judge_grid)
    output_dir = os.path.join(base_method_dir, f"coeff_{args.coeff}")

    # --- Generate directions for ALL experts at ALL MoE layers ---
    if not args.skip_generate:
        all_layer_directions = generate_all_expert_directions(
            model_base, harmful_train, harmless_train,
            model_card,
            artifact_dir=direction_cache_dir,
            batch_size=args.batch_size
        )
    else:
        print("\nSkipping generation, loading from cache...")
        combined_path = os.path.join(direction_cache_dir, "all_layer_expert_directions.pt")
        if os.path.exists(combined_path):
            all_layer_directions = torch.load(combined_path, map_location='cpu')
        else:
            # Load per-layer files
            all_layer_directions = {}
            num_layers = model_card.get_num_layers()
            for layer_idx in range(num_layers):
                if not model_card.is_moe_layer(layer_idx):
                    continue
                layer_path = os.path.join(direction_cache_dir, f"layer_{layer_idx}_all_experts.pt")
                if os.path.exists(layer_path):
                    all_layer_directions[layer_idx] = torch.load(layer_path, map_location='cpu')
            print(f"Loaded directions for {len(all_layer_directions)} MoE layers")

    # -------------------------------------------------------------------------
    # Branch: judge_grid vs Arditi selection
    # -------------------------------------------------------------------------
    if args.judge_grid:
        judge_grid_dir = os.path.join(base_method_dir, "judge_grid")
        os.makedirs(judge_grid_dir, exist_ok=True)

        print("\n" + "="*80)
        print("JUDGE GRID SEARCH (LAYER x POSITION x COEFF)")
        print("="*80)

        best_layer, best_position, best_coeff, all_results = run_allex_judge_grid(
            model_base, model_card, all_layer_directions,
            harmful_val,  # dicts
            harmless_val=harmless_val,
            grid_coeffs=args.grid_coeffs,
            max_new_tokens=args.judge_grid_tokens,
            n_samples=args.judge_grid_n_samples,
            batch_size=args.batch_size,
            output_dir=judge_grid_dir
        )

        # Save summary
        with open(os.path.join(judge_grid_dir, "judge_grid_results.json"), 'w') as f:
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

        print(f"\nSaved judge grid results to: {judge_grid_dir}/judge_grid_results.json")

        # Run full eval with best (layer, position, coeff)
        eval_output_dir = os.path.join(base_method_dir, f"coeff_{best_coeff}")
        os.makedirs(eval_output_dir, exist_ok=True)

        print(f"\n  Running full evaluation: L{best_layer} pos={best_position} coeff={best_coeff}")

        best_dirs = all_layer_directions[best_layer][:, best_position, :]
        dirs_dev = best_dirs.to(model_base.model.device, dtype=model_base.model.dtype)

        def get_hooks_fn(c):
            return get_all_expert_weighted_intervention_hooks(
                model_base, layer_idx=best_layer,
                directions_for_layer=dirs_dev,
                coeff=c, model_card=model_card
            )

        _run_evaluation(
            args, cfg, model_base, eval_output_dir, get_hooks_fn,
            harmful_test, harmless_test,
            coeff_override=best_coeff
        )

    elif not args.skip_select:
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*80)
        print("SELECTING BEST (LAYER, POSITION) FOR ALL-EXPERT STEERING")
        print("="*80)

        # Extract instruction strings from dicts
        harmful_val_str = _to_instructions(harmful_val)
        harmless_val_str = _to_instructions(harmless_val)

        best_layer, best_pos, directions_for_layer = select_layer_direction(
            model_base, harmful_val_str, harmless_val_str,
            all_layer_directions,
            artifact_dir=os.path.join(output_dir, "selection"),
            coeff=args.coeff,
            model_card=model_card,
            batch_size=args.batch_size
        )

        metadata = {
            "method": "allex",
            "layer": int(best_layer),
            "position": int(best_pos),
            "n_experts": int(directions_for_layer.shape[0]),
        }
        with open(os.path.join(output_dir, "direction_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        torch.save(directions_for_layer, os.path.join(output_dir, "directions_for_layer.pt"))

        print(f"\nBest: Layer {best_layer}, Position {best_pos}")

        dirs_dev = directions_for_layer.to(model_base.model.device, dtype=model_base.model.dtype)

        def get_hooks_fn(c):
            return get_all_expert_weighted_intervention_hooks(
                model_base, layer_idx=best_layer,
                directions_for_layer=dirs_dev,
                coeff=c, model_card=model_card
            )

        print("\n" + "="*80)
        print("EVALUATION (ALL-EXPERT)")
        print("="*80)
        _run_evaluation(args, cfg, model_base, output_dir, get_hooks_fn, harmful_test, harmless_test)

    else:
        # skip_select: load from cache
        os.makedirs(output_dir, exist_ok=True)

        print("\nSkipping selection, loading from cache...")
        meta_path = os.path.join(output_dir, "direction_metadata.json")
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        best_layer = metadata["layer"]
        best_pos = metadata["position"]
        directions_for_layer = torch.load(os.path.join(output_dir, "directions_for_layer.pt"))

        print(f"\nLoaded: Layer {best_layer}, Position {best_pos}")

        dirs_dev = directions_for_layer.to(model_base.model.device, dtype=model_base.model.dtype)

        def get_hooks_fn(c):
            return get_all_expert_weighted_intervention_hooks(
                model_base, layer_idx=best_layer,
                directions_for_layer=dirs_dev,
                coeff=c, model_card=model_card
            )

        print("\n" + "="*80)
        print("EVALUATION (ALL-EXPERT)")
        print("="*80)
        _run_evaluation(args, cfg, model_base, output_dir, get_hooks_fn, harmful_test, harmless_test)


# ============================================================================
# Main
# ============================================================================

def run_alt_pipeline(args):
    """Run the alternative expert steering pipeline."""

    print("="*80)
    print(f"ALTERNATIVE EXPERT STEERING PIPELINE: {args.method.upper()}")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Method: {args.method}")
    if args.method == 'allex' and args.judge_grid:
        print(f"Mode: JUDGE GRID SEARCH over layers x positions x coeffs {args.grid_coeffs} "
              f"({args.judge_grid_n_samples} samples, {args.judge_grid_tokens} tokens)")
    else:
        print(f"Coefficient: {args.coeff}")
    if args.method == 'unweighted':
        print(f"Expert threshold: {args.threshold}%")
        print(f"Expert type: {args.expert_type}")
    print(f"Skip harmless: {args.skip_harmless}")
    print("="*80)

    # Build model alias for Config
    base_model_name = os.path.basename(args.model_path)
    if args.method == 'unweighted':
        model_alias = f"{base_model_name}/expert_steering_unweighted_t{args.threshold}/sys_prompt_{args.system_prompt}"
    else:
        model_alias = f"{base_model_name}/expert_steering_allex/sys_prompt_{args.system_prompt}"

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

    if args.eval_datasets is not None:
        cfg.evaluation_datasets = tuple(args.eval_datasets)
        print(f"Using custom evaluation datasets: {cfg.evaluation_datasets}")

    # Load model
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    model_base = construct_model_base(args.model_path, system_prompt=cfg.system_prompt)

    from model_utils.model_card_factory import create_model_card
    model_card = create_model_card(model_base)
    print(f"Model card: {type(model_card).__name__}")

    # Load datasets
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)
    (harmful_train, harmless_train, harmful_val,
     harmless_val, harmful_test, harmless_test) = load_and_sample_datasets(cfg)

    print(f"Raw data:")
    print(f"  Training: {len(harmful_train)} harmful, {len(harmless_train)} harmless")
    print(f"  Validation: {len(harmful_val)} harmful, {len(harmless_val)} harmless")
    print(f"  Test: {len(harmful_test)} harmful, {len(harmless_test)} harmless")

    # Filter datasets
    print("\nFiltering datasets based on refusal scores...")
    (harmful_train, harmless_train,
     harmful_val, harmless_val) = filter_data(
        cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val
    )

    print(f"\nFiltered data:")
    print(f"  Training: {len(harmful_train)} harmful, {len(harmless_train)} harmless")
    print(f"  Validation: {len(harmful_val)} harmful, {len(harmless_val)} harmless")
    print(f"  Test: {len(harmful_test)} harmful, {len(harmless_test)} harmless")

    # Dispatch to method-specific pipeline
    if args.method == 'unweighted':
        run_unweighted_pipeline(
            args, cfg, model_base, model_card,
            harmful_train, harmless_train,
            harmful_val, harmless_val,
            harmful_test, harmless_test
        )
    elif args.method == 'allex':
        run_allex_pipeline(
            args, cfg, model_base, model_card,
            harmful_train, harmless_train,
            harmful_val, harmless_val,
            harmful_test, harmless_test
        )

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    args = parse_arguments()
    run_alt_pipeline(args)
