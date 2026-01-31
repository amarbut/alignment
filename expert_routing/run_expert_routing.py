"""
Expert Routing Intervention Pipeline

Clean implementation of expert routing interventions that:
- Uses ModelCard abstraction for model-specific operations
- Dynamically loads expert diffs from model-specific files
- Supports system prompt configuration
- Saves comprehensive metadata for reproducibility

Based on expert_explore/run_paired_interventions_v4.py but cleaner.
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime
from typing import Dict

# Add the alignment directory to path
alignment_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if alignment_dir not in sys.path:
    sys.path.insert(0, alignment_dir)

from pipeline.config import Config, SYSTEM_PROMPTS
from pipeline.model_utils.model_factory_moe import construct_model_base
from pipeline.model_utils.model_card_factory import create_model_card
from dataset.load_dataset import load_dataset_split, load_dataset

from expert_routing.expert_routing_hooks import (
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
        default='llama_2',
        choices=['none', 'llama_2', 'lightweight'],
        help='System prompt to use (default: llama_2)'
    )
    parser.add_argument(
        '--intervention',
        type=str,
        default='select_refusal',
        choices=[
            'baseline',
            'select_refusal',
            'select_response',
            'top_refusal',
            'top_jailbreak',
            'all'
        ],
        help='Intervention type to run'
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
        default=100,
        help='Number of test examples'
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
        default='expert_explore',
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
    from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak

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
        "n_test": args.n_test,
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
    print(f"System prompt: {args.system_prompt}")
    print(f"Intervention: {args.intervention}")
    print(f"Threshold: {args.threshold}")
    print(f"Epsilon: {args.epsilon}")
    print("=" * 80)

    # Setup configuration
    model_alias = os.path.basename(args.model_path) + f"/expert_routing_t{args.threshold}"
    cfg = Config(
        model_alias=model_alias,
        model_path=args.model_path,
        n_test=args.n_test,
        system_prompt=args.system_prompt
    )

    print(f"\nArtifact path: {cfg.artifact_path()}")

    # Load model
    print("\nLoading model...")
    model_base = construct_model_base(args.model_path)
    model_base.model.config.pad_token_id = model_base.tokenizer.pad_token_id

    # Update system prompt if needed
    system_prompt_text = SYSTEM_PROMPTS.get(args.system_prompt)
    if 'oss' in args.model_path.lower() and hasattr(model_base, '_system_prompt'):
        model_base._system_prompt = system_prompt_text
        import functools
        from pipeline.model_utils.oss_model import tokenize_instructions_oss_chat
        model_base.tokenize_instructions_fn = functools.partial(
            tokenize_instructions_oss_chat,
            tokenizer=model_base.tokenizer,
            system=system_prompt_text,
            include_trailing_whitespace=True
        )

    print("Model loaded!")

    # Create model card
    model_card = create_model_card(model_base)
    print(f"Model card: {type(model_card).__name__}")
    print(f"Number of layers: {model_card.get_num_layers()}")

    # Load expert diffs
    expert_diffs_file = os.path.join(args.expert_diffs_path, model_card.get_expert_diffs_filename())

    if args.regenerate_diffs or not os.path.exists(expert_diffs_file):
        print(f"\nExpert diffs not found or regeneration requested...")
        print(f"Expected at: {expert_diffs_file}")
        print("Please generate expert diffs first using the appropriate script.")
        print("For OSS models: python expert_explore/compute_expert_diffs.py")
        return

    print(f"\nLoading expert diffs from: {expert_diffs_file}")
    expert_diffs = load_expert_diffs(model_card, args.expert_diffs_path)
    print(f"Loaded diffs for {len(expert_diffs)} layers")

    # Load test datasets
    print("\nLoading test datasets...")
    random.seed(42)

    harmful_datasets = {}
    for dataset_name in cfg.evaluation_datasets:
        dataset = load_dataset(dataset_name)
        if args.n_test is not None:
            dataset = random.sample(dataset, min(args.n_test, len(dataset)))
        harmful_datasets[dataset_name] = dataset

    harmless_test = load_dataset_split(harmtype='harmless', split='test')
    if args.n_test is not None:
        harmless_test = random.sample(harmless_test, min(args.n_test, len(harmless_test)))

    print(f"Loaded {len(harmful_datasets)} harmful datasets and harmless test set")

    # Build intervention configs
    configs_to_run = {}

    if args.intervention == 'all':
        if not args.skip_baseline:
            configs_to_run['baseline'] = ExpertInterventionConfig(epsilon=args.epsilon)
        configs_to_run['select_refusal'] = get_select_experts_refusal_induction_config(
            expert_diffs, args.threshold, args.epsilon
        )
        configs_to_run['select_response'] = get_select_experts_response_induction_config(
            expert_diffs, args.threshold, args.epsilon
        )
        configs_to_run['top_refusal'] = get_top_experts_refusal_config(
            expert_diffs, model_card, args.epsilon
        )
        configs_to_run['top_jailbreak'] = get_top_experts_jailbreak_config(
            expert_diffs, model_card, args.epsilon
        )
    elif args.intervention == 'baseline':
        configs_to_run['baseline'] = ExpertInterventionConfig(epsilon=args.epsilon)
    elif args.intervention == 'select_refusal':
        if not args.skip_baseline:
            configs_to_run['baseline'] = ExpertInterventionConfig(epsilon=args.epsilon)
        configs_to_run['select_refusal'] = get_select_experts_refusal_induction_config(
            expert_diffs, args.threshold, args.epsilon
        )
    elif args.intervention == 'select_response':
        if not args.skip_baseline:
            configs_to_run['baseline'] = ExpertInterventionConfig(epsilon=args.epsilon)
        configs_to_run['select_response'] = get_select_experts_response_induction_config(
            expert_diffs, args.threshold, args.epsilon
        )
    elif args.intervention == 'top_refusal':
        if not args.skip_baseline:
            configs_to_run['baseline'] = ExpertInterventionConfig(epsilon=args.epsilon)
        configs_to_run['top_refusal'] = get_top_experts_refusal_config(
            expert_diffs, model_card, args.epsilon
        )
    elif args.intervention == 'top_jailbreak':
        if not args.skip_baseline:
            configs_to_run['baseline'] = ExpertInterventionConfig(epsilon=args.epsilon)
        configs_to_run['top_jailbreak'] = get_top_experts_jailbreak_config(
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

    print("\nIntervention effects:")
    print("  - select_refusal/top_refusal: Should INCREASE refusal on harmful")
    print("  - select_response/top_jailbreak: Should DECREASE refusal on harmful")


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(args)
