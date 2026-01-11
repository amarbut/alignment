"""
Paired Expert Intervention Pipeline V4 - Using min/max +/- epsilon

This script uses the V4 intervention system which dynamically adjusts expert logits
to min/max +/- epsilon for each token, rather than using static bias modifications.

Based on expert routing analysis:
- Layer 10, Expert 5: 30% more likely in harmful prompts (harmful-preferred)
- Layer 10, Expert 10: 13% more likely in harmless prompts (harmless-preferred)
- Layer 13, Expert 0: 25% more likely in harmful prompts (harmful-preferred)
- Layer 13, Expert 21: 15% more likely in harmless prompts (harmless-preferred)

Interventions tested:
1. Baseline (no intervention)
2. L10 Refusal Induction: Force E5, Suppress E10
3. L10 Response Induction: Force E10, Suppress E5
4. L13 Refusal Induction: Force E0, Suppress E21
5. L13 Response Induction: Force E21, Suppress E0
6. Combined Refusal Induction (both layers)
7. Combined Response Induction (both layers)
8. Top Experts Refusal: Force top harmful expert, suppress top harmless expert at each layer
9. Top Experts Jailbreak: Force top harmless expert, suppress top harmful expert at each layer
10. Select Experts (threshold-based) Refusal/Response Induction
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

from expert_explore.expert_intervention_hooks_v4 import (
    ExpertInterventionConfigV4,
    apply_expert_interventions_v4,
    remove_expert_interventions_v4,
    print_intervention_summary_v4,
    get_layer10_refusal_induction_config_v4,
    get_layer10_response_induction_config_v4,
    get_layer13_refusal_induction_config_v4,
    get_layer13_response_induction_config_v4,
    get_combined_refusal_induction_config_v4,
    get_combined_response_induction_config_v4,
    get_select_experts_refusal_induction_config_v4,
    get_select_experts_response_induction_config_v4,
    get_top_experts_all_layers_refusal_induction_config_v4,
    get_top_experts_all_layers_jailbreak_induction_config_v4
)

from pipeline.run_pipeline_subspace import (
    generate_and_save_completions_for_dataset,
    evaluate_completions_and_save_results_for_dataset
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run paired expert intervention pipeline V4 (min/max +/- epsilon)"
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
        help='Which layers to test: all, all_layers, top_experts, l10, l13, select_experts, combined'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0,
        help='For select experts, use experts with diff > threshold'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.01,
        help='Epsilon value for calibrated adjustment (default: 0.01, inspired by SteerMoE)'
    )
    return parser.parse_args()


def get_paired_intervention_configs_v4(
    skip_combined: bool = False,
    threshold: float = 0,
    epsilon: float = 1e-5
) -> Dict[str, ExpertInterventionConfigV4]:
    """
    Get all paired intervention configurations for V4.

    Args:
        skip_combined: If True, skip the combined (both layers) interventions
        threshold: Threshold for select experts configs
        epsilon: Epsilon value for min/max adjustment

    Returns:
        Dictionary mapping intervention name to config
    """
    configs = {
        'baseline': ExpertInterventionConfigV4(epsilon=epsilon, use_calibration=False),
        'l10_refusal_induction': get_layer10_refusal_induction_config_v4(epsilon=epsilon),
        'l10_response_induction': get_layer10_response_induction_config_v4(epsilon=epsilon),
        'l13_refusal_induction': get_layer13_refusal_induction_config_v4(epsilon=epsilon),
        'l13_response_induction': get_layer13_response_induction_config_v4(epsilon=epsilon),
        'select_experts_refusal': get_select_experts_refusal_induction_config_v4(threshold=threshold, epsilon=epsilon),
        'select_experts_response': get_select_experts_response_induction_config_v4(threshold=threshold, epsilon=epsilon),
        'combined_refusal_induction': get_combined_refusal_induction_config_v4(epsilon=epsilon),
        'combined_response_induction': get_combined_response_induction_config_v4(epsilon=epsilon),
        'top_experts_refusal': get_top_experts_all_layers_refusal_induction_config_v4(epsilon=epsilon),
        'top_experts_jailbreak': get_top_experts_all_layers_jailbreak_induction_config_v4(epsilon=epsilon)
    }

    return configs


def filter_configs_by_layers(
    all_configs: Dict[str, ExpertInterventionConfigV4],
    layers: list
) -> Dict[str, ExpertInterventionConfigV4]:
    """
    Filter configs based on which layers to test.

    Args:
        all_configs: All available configs
        layers: List of layer identifiers ('all', 'l10', 'l13', 'all_layers', 'select_experts', 'combined', 'top_experts')

    Returns:
        Filtered dictionary of configs
    """
    if 'all' in layers:
        return all_configs

    filtered = {'baseline': all_configs['baseline']}

    if 'select_experts' in layers:
        filtered['select_experts_refusal'] = all_configs['select_experts_refusal']
        filtered['select_experts_response'] = all_configs['select_experts_response']

    if 'top_experts' in layers or 'all_layers' in layers:
        filtered['top_experts_refusal'] = all_configs['top_experts_refusal']
        filtered['top_experts_jailbreak'] = all_configs['top_experts_jailbreak']

    if 'l10' in layers:
        filtered['l10_refusal_induction'] = all_configs['l10_refusal_induction']
        filtered['l10_response_induction'] = all_configs['l10_response_induction']

    if 'l13' in layers:
        filtered['l13_refusal_induction'] = all_configs['l13_refusal_induction']
        filtered['l13_response_induction'] = all_configs['l13_response_induction']

    if 'combined' in layers:
        filtered['combined_refusal_induction'] = all_configs['combined_refusal_induction']
        filtered['combined_response_induction'] = all_configs['combined_response_induction']

    return filtered


def run_paired_intervention_pipeline_v4(
    model_path: str = 'openai/gpt-oss-20b',
    n_test: int = 100,
    skip_baseline: bool = False,
    skip_combined: bool = False,
    layers: list = ['all'],
    threshold: float = 0,
    epsilon: float = 1e-5
):
    """
    Run the paired intervention evaluation pipeline using V4 hooks.

    Args:
        model_path: Path to the model
        n_test: Number of test examples
        skip_baseline: Whether to skip baseline evaluation
        skip_combined: Whether to skip combined interventions
        layers: Which layers to test
        threshold: Threshold for select experts
        epsilon: Epsilon value for min/max adjustment
    """
    print("=" * 80)
    print("PAIRED EXPERT INTERVENTION PIPELINE V4")
    print("Min/Max +/- Epsilon Dynamic Adjustment")
    print("=" * 80)

    # Setup configuration
    model_alias = os.path.basename(model_path) + f"/paired_interventions_v4_eps{epsilon}"
    cfg = Config(model_alias=model_alias, model_path=model_path)
    cfg.n_test = n_test

    print(f"\nConfiguration:")
    print(f"  Model: {model_path}")
    print(f"  Artifact path: {cfg.artifact_path()}")
    print(f"  N_test: {cfg.n_test}")
    print(f"  Epsilon: {epsilon}")

    # Load model
    print("\nLoading model...")
    model_base = construct_model_base(model_path)
    model_base.model.config.pad_token_id = model_base.tokenizer.pad_token_id
    print("Model loaded!")

    # Get intervention configs
    all_configs = get_paired_intervention_configs_v4(
        skip_combined=skip_combined,
        threshold=threshold,
        epsilon=epsilon
    )
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

        print_intervention_summary_v4(config)

        # Apply intervention
        print("\nApplying expert interventions...")
        original_biases = apply_expert_interventions_v4(model_base, config)

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
            print("\nRestoring original biases...")
            remove_expert_interventions_v4(model_base, original_biases)

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

    run_paired_intervention_pipeline_v4(
        model_path=args.model_path,
        n_test=args.n_test,
        skip_baseline=args.skip_baseline,
        skip_combined=args.skip_combined,
        layers=args.layers,
        threshold=args.threshold,
        epsilon=args.epsilon
    )
