"""
Paired Expert Intervention Pipeline - Refusal vs Response Induction

This script tests paired force/suppress interventions to actively shift the model
toward or away from refusal behavior.

Based on expert routing analysis:
- Layer 10, Expert 5: 30% more likely in harmful prompts (harmful-preferred)
- Layer 10, Expert 10: 13% more likely in harmless prompts (harmless-preferred)
- Layer 13, Expert 0: 25% more likely in harmful prompts (harmful-preferred)
- Layer 13, Expert 21: 15% more likely in harmless prompts (harmless-preferred)

Interventions tested:
1. Baseline (no intervention)
2. L10 Refusal Induction: Force E10, Suppress E5
3. L10 Response Induction: Force E5, Suppress E10
4. L13 Refusal Induction: Force E21, Suppress E0
5. L13 Response Induction: Force E0, Suppress E21
6. Combined Refusal Induction (both layers)
7. Combined Response Induction (both layers)
"""

import os
import sys
import json
import random
import argparse
from typing import Dict

# Add the alignment directory to path
alignment_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if alignment_dir not in sys.path:
    sys.path.insert(0, alignment_dir)

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from dataset.load_dataset import load_dataset_split, load_dataset

from expert_explore.expert_intervention_hooks_v3 import (
    ExpertInterventionConfig,
    apply_expert_interventions,
    remove_expert_interventions,
    print_intervention_summary,
    get_layer10_refusal_induction_config,
    get_layer10_response_induction_config,
    get_layer13_refusal_induction_config,
    get_layer13_response_induction_config,
    get_combined_refusal_induction_config,
    get_combined_response_induction_config
)

from pipeline.run_pipeline_subspace import (
    generate_and_save_completions_for_dataset,
    evaluate_completions_and_save_results_for_dataset
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run paired expert intervention pipeline (refusal vs response induction)"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='openai/gpt-oss-20b',
        help='Path to the model'
    )
    parser.add_argument(
        '--n_test',
        type=int,
        default=100,
        help='Number of test examples to evaluate'
    )
    parser.add_argument(
        '--skip_baseline',
        action='store_true',
        help='Skip baseline (no intervention) evaluation'
    )
    parser.add_argument(
        '--skip_combined',
        action='store_true',
        help='Skip combined (both layers) interventions'
    )
    parser.add_argument(
        '--layers',
        type=str,
        nargs='+',
        default=['all'],
        help='Which layers to test: all, l10, l13'
    )
    return parser.parse_args()


def get_paired_intervention_configs(skip_combined: bool = False) -> Dict[str, ExpertInterventionConfig]:
    """
    Get all paired intervention configurations.

    Args:
        skip_combined: If True, skip the combined (both layers) interventions

    Returns:
        Dictionary mapping intervention name to config
    """
    configs = {
        'baseline': ExpertInterventionConfig(),
        'l10_refusal_induction': get_layer10_refusal_induction_config(),
        'l10_response_induction': get_layer10_response_induction_config(),
        'l13_refusal_induction': get_layer13_refusal_induction_config(),
        'l13_response_induction': get_layer13_response_induction_config(),
    }

    if not skip_combined:
        configs['combined_refusal_induction'] = get_combined_refusal_induction_config()
        configs['combined_response_induction'] = get_combined_response_induction_config()

    return configs


def filter_configs_by_layers(
    all_configs: Dict[str, ExpertInterventionConfig],
    layers: list
) -> Dict[str, ExpertInterventionConfig]:
    """
    Filter configs based on which layers to test.

    Args:
        all_configs: All available configs
        layers: List of layer identifiers ('all', 'l10', 'l13')

    Returns:
        Filtered dictionary of configs
    """
    if 'all' in layers:
        return all_configs

    filtered = {'baseline': all_configs['baseline']}

    if 'l10' in layers:
        filtered['l10_refusal_induction'] = all_configs['l10_refusal_induction']
        filtered['l10_response_induction'] = all_configs['l10_response_induction']

    if 'l13' in layers:
        filtered['l13_refusal_induction'] = all_configs['l13_refusal_induction']
        filtered['l13_response_induction'] = all_configs['l13_response_induction']

    # Add combined if both layers are selected and it exists
    if 'l10' in layers and 'l13' in layers:
        if 'combined_refusal_induction' in all_configs:
            filtered['combined_refusal_induction'] = all_configs['combined_refusal_induction']
        if 'combined_response_induction' in all_configs:
            filtered['combined_response_induction'] = all_configs['combined_response_induction']

    return filtered


def run_paired_intervention_pipeline(
    model_path: str = 'openai/gpt-oss-20b',
    n_test: int = 100,
    skip_baseline: bool = False,
    skip_combined: bool = False,
    layers: list = ['all']
):
    """
    Run the paired intervention evaluation pipeline.

    Args:
        model_path: Path to the model
        n_test: Number of test examples
        skip_baseline: Whether to skip baseline evaluation
        skip_combined: Whether to skip combined interventions
        layers: Which layers to test
    """
    print("=" * 80)
    print("PAIRED EXPERT INTERVENTION PIPELINE")
    print("Refusal Induction vs Response Induction")
    print("=" * 80)

    # Setup configuration
    model_alias = os.path.basename(model_path) + "/paired_interventions"
    cfg = Config(model_alias=model_alias, model_path=model_path)
    cfg.n_test = n_test

    print(f"\nConfiguration:")
    print(f"  Model: {model_path}")
    print(f"  Artifact path: {cfg.artifact_path()}")
    print(f"  N_test: {cfg.n_test}")

    # Load model
    print("\nLoading model...")
    model_base = construct_model_base(model_path)
    model_base.model.config.pad_token_id = model_base.tokenizer.pad_token_id
    print("Model loaded!")

    # Get intervention configs
    all_configs = get_paired_intervention_configs(skip_combined=skip_combined)
    configs_to_run = filter_configs_by_layers(all_configs, layers)

    if skip_baseline and 'baseline' in configs_to_run:
        del configs_to_run['baseline']

    print(f"\nRunning {len(configs_to_run)} intervention configurations:")
    for config_name in configs_to_run.keys():
        print(f"  - {config_name}")

    # Load test datasets
    print("\nLoading test datasets...")
    random.seed(42)

    # Harmful test sets
    harmful_datasets = {}
    for dataset_name in cfg.evaluation_datasets:
        dataset = load_dataset(dataset_name)
        if n_test is not None:
            dataset = random.sample(dataset, min(n_test, len(dataset)))
        harmful_datasets[dataset_name] = dataset

    # Harmless test set
    harmless_test = load_dataset_split(harmtype='harmless', split='test')
    if n_test is not None:
        harmless_test = random.sample(harmless_test, min(n_test, len(harmless_test)))

    print(f"Loaded {len(harmful_datasets)} harmful datasets and harmless test set")

    # Run each intervention
    for config_name, config in configs_to_run.items():
        print("\n" + "=" * 80)
        print(f"RUNNING INTERVENTION: {config_name}")
        print("=" * 80)

        print_intervention_summary(config)

        # Apply intervention
        print("\nApplying expert intervention...")
        original_biases = apply_expert_interventions(model_base, config)

        try:
            # Generate and evaluate on harmful datasets
            print("\n--- Evaluating on harmful datasets ---")
            for dataset_name, dataset in harmful_datasets.items():
                print(f"\nProcessing dataset: {dataset_name}")

                # Generate completions
                generate_and_save_completions_for_dataset(
                    cfg=cfg,
                    model_base=model_base,
                    fwd_pre_hooks=[],
                    fwd_hooks=[],
                    intervention_label=config_name,
                    dataset_name=dataset_name,
                    topk=1,
                    coeff=1.0,
                    tau=1.0,
                    dataset=dataset
                )

                # Evaluate completions
                evaluate_completions_and_save_results_for_dataset(
                    cfg=cfg,
                    intervention_label=config_name,
                    dataset_name=dataset_name,
                    eval_methodologies=cfg.jailbreak_eval_methodologies,
                    topk=1,
                    coeff=1.0,
                    tau=1.0
                )

            # Generate and evaluate on harmless dataset
            print("\n--- Evaluating on harmless dataset ---")
            generate_and_save_completions_for_dataset(
                cfg=cfg,
                model_base=model_base,
                fwd_pre_hooks=[],
                fwd_hooks=[],
                intervention_label=config_name,
                dataset_name='harmless',
                topk=1,
                coeff=1.0,
                tau=1.0,
                dataset=harmless_test
            )

            evaluate_completions_and_save_results_for_dataset(
                cfg=cfg,
                intervention_label=config_name,
                dataset_name='harmless',
                eval_methodologies=cfg.refusal_eval_methodologies,
                topk=1,
                coeff=1.0,
                tau=1.0
            )

        finally:
            # Always restore original biases
            print("\nRestoring original model parameters...")
            remove_expert_interventions(model_base, original_biases)

        print(f"\nCompleted intervention: {config_name}")

    # Print summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {cfg.artifact_path()}/completions/")

    print("\nSummary of interventions run:")
    for config_name in configs_to_run.keys():
        print(f"  ✓ {config_name}")

    print("\n" + "=" * 80)
    print("EXPECTED OUTCOMES:")
    print("=" * 80)
    print("\nRefusal Induction (force harmless-preferred, suppress harmful-preferred):")
    print("  → Should INCREASE refusal rate on harmful prompts")
    print("  → Should maintain safety on harmless prompts")
    print("\nResponse Induction (force harmful-preferred, suppress harmless-preferred):")
    print("  → Should DECREASE refusal rate on harmful prompts (more jailbreaks)")
    print("  → May increase false refusals on harmless prompts")
    print("=" * 80)


if __name__ == "__main__":
    args = parse_arguments()

    run_paired_intervention_pipeline(
        model_path=args.model_path,
        n_test=args.n_test,
        skip_baseline=args.skip_baseline,
        skip_combined=args.skip_combined,
        layers=args.layers
    )
