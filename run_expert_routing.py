"""
Expert Routing Intervention Pipeline
    - Identifies refusal/response experts from activation diffs
    - Forces or suppresses experts at inference to induce refusal/response
"""

# =============================================================================
# CRITICAL: Set HF cache BEFORE importing any HuggingFace libraries
# HF libraries cache environment variables at import time, so this must happen first
# =============================================================================
import argparse as _argparse_early

# Minimal early arg parsing just to get model_path for cache config
_early_parser = _argparse_early.ArgumentParser(add_help=False)
_early_parser.add_argument('--model_path', type=str, default='')
_early_args, _ = _early_parser.parse_known_args()

if _early_args.model_path:
    from model_utils.hf_cache_config import set_hf_cache_from_path
    set_hf_cache_from_path(_early_args.model_path)
# =============================================================================

import os
import json
import random
import argparse
from datetime import datetime
from typing import Dict

from config import Config
from model_utils.model_factory_moe import construct_model_base
from model_utils.model_card_factory import create_model_card
from dataset.load_dataset import load_dataset_split, load_dataset

from submodules.expert_diff_generator import generate_expert_diffs_for_model
from submodules.expert_routing.expert_routing_hooks import (
    ExpertInterventionConfig,
    apply_expert_interventions,
    remove_expert_interventions,
    print_intervention_summary,
    load_expert_diffs,
    get_select_experts_refusal_induction_config,
    get_select_experts_response_induction_config,
    get_top_experts_refusal_config,
    get_top_experts_jailbreak_config,
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run expert routing intervention pipeline"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='unsloth/gpt-oss-20b-unsloth-bnb-4bit',
        help='Path to the model'
    )
    parser.add_argument(
        '--system_prompt',
        type=str,
        default=None,
        choices=['none', 'llama_2', 'lightweight'],
        help='System prompt to use (default: use Config default)'
    )
    parser.add_argument(
        '--intervention',
        type=str,
        default='select_experts',
        choices=['baseline', 'select_experts', 'top_experts'],
        help='Intervention type: select_experts (threshold-based) or top_experts (top-1 per layer). Each runs both refusal and response induction.'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=10.0,
        help='Threshold for select_experts interventions (default: 10.0)'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.01,
        help='Epsilon for calibrated adjustment (default: 0.01)'
    )
    parser.add_argument(
        '--n_test',
        type=int,
        default=None,
        help='Number of test examples (default: use Config default)'
    )
    parser.add_argument(
        '--regenerate_diffs',
        action='store_true',
        help='Force regeneration of expert diffs file'
    )
    parser.add_argument(
        '--skip_baseline',
        action='store_true',
        help='Skip baseline evaluation'
    )
    parser.add_argument(
        '--expert_diffs_path',
        type=str,
        default='expert_diffs',
        help='Directory containing expert diffs files'
    )
    return parser.parse_args()


def generate_and_save_completions(
    cfg,
    model_base,
    intervention_label: str,
    dataset_name: str,
    dataset=None
):
    """Generate and save completions for a dataset."""
    output_dir = os.path.join(cfg.artifact_path(), 'completions', intervention_label)
    os.makedirs(output_dir, exist_ok=True)

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(
        dataset,
        fwd_pre_hooks=[],
        fwd_hooks=[],
        max_new_tokens=cfg.max_new_tokens
    )

    # Print first few completions
    print("\nSample completions:")
    for c in completions[:2]:
        print(f"  Prompt: {c['prompt'][:50]}...")
        print(f"  Response: {c['response'][:100]}...")
        print()

    output_path = os.path.join(output_dir, f'{dataset_name}_completions.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(completions, f, indent=4, ensure_ascii=False)

    print(f"Saved completions to: {output_path}")
    return completions


def evaluate_and_save_results(
    cfg,
    intervention_label: str,
    dataset_name: str,
    eval_methodologies
):
    """Evaluate completions and save results."""
    from submodules.evaluate_jailbreak import evaluate_jailbreak

    completions_dir = os.path.join(cfg.artifact_path(), 'completions', intervention_label)
    completions_path = os.path.join(completions_dir, f'{dataset_name}_completions.json')

    with open(completions_path, 'r', encoding='utf-8') as f:
        completions = json.load(f)

    eval_path = os.path.join(completions_dir, f'{dataset_name}_evaluations.json')

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=eval_path,
    )

    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, indent=4, ensure_ascii=False)

    print(f"Saved evaluations to: {eval_path}")
    return evaluation


def save_metadata(cfg, args, config: ExpertInterventionConfig, intervention_name: str):
    """Save run metadata to JSON."""
    force_count, suppress_count = config.get_intervention_count()

    metadata = {
        "model_path": args.model_path,
        "intervention_type": "expert_routing",
        "intervention_name": intervention_name,
        "system_prompt": args.system_prompt,
        "threshold": args.threshold,
        "epsilon": args.epsilon,
        "n_test": cfg.n_test,
        "num_forced_experts": force_count,
        "num_suppressed_experts": suppress_count,
        "timestamp": datetime.now().isoformat(),
    }

    output_dir = os.path.join(cfg.artifact_path(), 'completions', intervention_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


def run_single_intervention(
    cfg,
    model_base,
    model_card,
    config: ExpertInterventionConfig,
    intervention_name: str,
    args,
    harmful_datasets: Dict,
    harmless_test
):
    """Run a single intervention configuration."""
    print("\n" + "=" * 80)
    print(f"RUNNING INTERVENTION: {intervention_name}")
    print("=" * 80)

    print_intervention_summary(config)

    # Apply intervention
    if intervention_name != 'baseline':
        print("\nApplying expert interventions...")
        original_biases = apply_expert_interventions(model_base, model_card, config)
    else:
        original_biases = {}

    try:
        # Evaluate on harmful datasets
        print("\n--- Evaluating on harmful datasets ---")
        for dataset_name, dataset in harmful_datasets.items():
            print(f"\nProcessing {dataset_name}...")
            generate_and_save_completions(cfg, model_base, intervention_name, dataset_name, dataset)
            evaluate_and_save_results(cfg, intervention_name, dataset_name, cfg.jailbreak_eval_methodologies)

        # Evaluate on harmless dataset
        print("\n--- Evaluating on harmless dataset ---")
        generate_and_save_completions(cfg, model_base, intervention_name, 'harmless', harmless_test)
        evaluate_and_save_results(cfg, intervention_name, 'harmless', cfg.refusal_eval_methodologies)

        # Save metadata
        save_metadata(cfg, args, config, intervention_name)

    finally:
        # Restore original biases
        if original_biases:
            print("\nRestoring original biases...")
            remove_expert_interventions(model_base, model_card, original_biases)

    print(f"\nCompleted intervention: {intervention_name}")


def run_pipeline(args):
    """Run the expert routing intervention pipeline."""
    print("=" * 80)
    print("EXPERT ROUTING INTERVENTION PIPELINE")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Intervention: {args.intervention}")
    print(f"Threshold: {args.threshold}")
    print(f"Epsilon: {args.epsilon}")
    print("=" * 80)

    # Setup configuration (only override if explicitly specified)
    sys = args.system_prompt if args.system_prompt is not None else "lightweight"
    model_alias = os.path.basename(args.model_path) + f"/expert_routing_t{args.threshold}" + f"/sys_prompt_{sys}"
    config_kwargs = {
        "model_alias": model_alias,
        "model_path": args.model_path,
    }
    if args.system_prompt is not None:
        config_kwargs["system_prompt"] = args.system_prompt
    if args.n_test is not None:
        config_kwargs["n_test"] = args.n_test
    cfg = Config(**config_kwargs)
    print(f"System prompt: {cfg.system_prompt}")

    print(f"\nArtifact path: {cfg.artifact_path()}")

    # Load model with system prompt from config
    print("\nLoading model...")
    model_base = construct_model_base(args.model_path, system_prompt=cfg.system_prompt)
    model_base.model.config.pad_token_id = model_base.tokenizer.pad_token_id
    print(f"Model loaded with system prompt: {cfg.system_prompt}")

    # Create model card
    model_card = create_model_card(model_base)
    print(f"Model card: {type(model_card).__name__}")
    print(f"Number of layers: {model_card.get_num_layers()}")

    # Load or generate expert diffs — use sys_prompt subdirectory so each
    # system prompt variant uses its own routing diffs
    expert_diffs_dir = os.path.join(args.expert_diffs_path, f'sys_prompt_{sys}')
    expert_diffs_file = os.path.join(expert_diffs_dir, model_card.get_expert_diffs_filename())

    if args.regenerate_diffs or not os.path.exists(expert_diffs_file):
        print(f"\nExpert diffs not found or regeneration requested...")
        print(f"Generating expert diffs for {args.model_path}...")
        os.makedirs(expert_diffs_dir, exist_ok=True)
        generate_expert_diffs_for_model(
            model_base=model_base,
            model_card=model_card,
            harmful_dataset_path="dataset/splits/harmful_train.json",
            harmless_dataset_path="dataset/splits/harmless_train.json",
            output_path=expert_diffs_file,
            batch_size=4,
            last_n_tokens=5
        )
        print(f"Expert diffs saved to: {expert_diffs_file}")

    print(f"\nLoading expert diffs from: {expert_diffs_file}")
    expert_diffs = load_expert_diffs(model_card, expert_diffs_dir)
    print(f"Loaded diffs for {len(expert_diffs)} layers")

    # Load test datasets
    print("\nLoading test datasets...")
    random.seed(42)

    harmful_datasets = {}
    for dataset_name in cfg.evaluation_datasets:
        dataset = load_dataset(dataset_name)
        dataset = random.sample(dataset, min(cfg.n_test, len(dataset)))
        harmful_datasets[dataset_name] = dataset

    harmless_test = load_dataset_split(harmtype='harmless', split='test')
    harmless_test = random.sample(harmless_test, min(cfg.n_test, len(harmless_test)))

    print(f"Loaded {len(harmful_datasets)} harmful datasets and harmless test set")

    # Build intervention configs
    configs_to_run = {}

    if args.intervention == 'baseline':
        configs_to_run['baseline'] = ExpertInterventionConfig(epsilon=args.epsilon)

    elif args.intervention == 'select_experts':
        # Threshold-based: runs both refusal and response induction
        if not args.skip_baseline:
            configs_to_run['baseline'] = ExpertInterventionConfig(epsilon=args.epsilon)
        configs_to_run['select_refusal'] = get_select_experts_refusal_induction_config(
            expert_diffs, args.threshold, args.epsilon
        )
        configs_to_run['select_response'] = get_select_experts_response_induction_config(
            expert_diffs, args.threshold, args.epsilon
        )

    elif args.intervention == 'top_experts':
        # Top-1 per layer: runs both refusal and response induction
        if not args.skip_baseline:
            configs_to_run['baseline'] = ExpertInterventionConfig(epsilon=args.epsilon)
        configs_to_run['top_refusal'] = get_top_experts_refusal_config(
            expert_diffs, model_card, args.epsilon
        )
        configs_to_run['top_response'] = get_top_experts_jailbreak_config(
            expert_diffs, model_card, args.epsilon
        )

    print(f"\nRunning {len(configs_to_run)} intervention configurations:")
    for name in configs_to_run.keys():
        print(f"  - {name}")

    # Run each intervention
    for intervention_name, config in configs_to_run.items():
        run_single_intervention(
            cfg, model_base, model_card, config, intervention_name,
            args, harmful_datasets, harmless_test
        )

    # Print summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {cfg.artifact_path()}/completions/")


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(args)
