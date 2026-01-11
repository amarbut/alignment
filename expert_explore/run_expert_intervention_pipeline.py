"""
Complete expert intervention pipeline for OSS-20B MoE model.

This script runs the full evaluation pipeline with expert routing interventions:
1. Suppress harmful-associated experts (L10E5, L13E0)
2. Force harmful-associated experts (L10E5, L13E0)
3. Soft bias towards harmful-associated experts (L10E5, L13E0)

For each intervention type, we test:
- Individual expert interventions (L10E5 only, L13E0 only)
- Combined expert interventions (both L10E5 and L13E0)

Results are saved to the pipeline/runs directory structure.
"""

import os
import sys
import json
import random
import argparse
import torch
from typing import Dict, List

# Add the alignment directory to path so we can import from pipeline/
alignment_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if alignment_dir not in sys.path:
    sys.path.insert(0, alignment_dir)

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from dataset.load_dataset import load_dataset_split, load_dataset

# Import expert intervention functions
from expert_explore.expert_intervention_hooks_v3 import (
    ExpertInterventionConfig,
    apply_expert_interventions,
    remove_expert_interventions,
    print_intervention_summary,
    get_harmful_expert_suppression_config,
    get_harmful_expert_forcing_config,
    get_layer10_expert5_only_config,
    get_layer13_expert0_only_config
)

# Import evaluation functions from the main pipeline
from pipeline.run_pipeline_subspace import (
    generate_and_save_completions_for_dataset,
    evaluate_completions_and_save_results_for_dataset
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run expert intervention pipeline for OSS-20B MoE model"
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
        default=None,
        help='Number of test examples to evaluate (None = all)'
    )
    parser.add_argument(
        '--interventions',
        type=str,
        nargs='+',
        default=['all'],
        help='Which interventions to run: all, suppress, force, soft, baseline'
    )
    parser.add_argument(
        '--experts',
        type=str,
        nargs='+',
        default=['all'],
        help='Which expert configs to test: all, combined, l10e5, l13e0'
    )
    parser.add_argument(
        '--skip_baseline',
        action='store_true',
        help='Skip baseline (no intervention) evaluation'
    )
    return parser.parse_args()


def get_intervention_configs() -> Dict[str, ExpertInterventionConfig]:
    """
    Define all intervention configurations.

    Returns:
        Dictionary mapping intervention name to config
    """
    configs = {}

    # Baseline (no intervention)
    configs['baseline'] = ExpertInterventionConfig()

    # === SUPPRESS INTERVENTIONS ===
    # Suppress both harmful experts
    configs['suppress_both'] = get_harmful_expert_suppression_config()

    # Suppress individual experts
    configs['suppress_l10e5'] = get_layer10_expert5_only_config('suppress')
    configs['suppress_l13e0'] = get_layer13_expert0_only_config('suppress')

    # === FORCE INTERVENTIONS ===
    # Force both harmful experts
    configs['force_both'] = get_harmful_expert_forcing_config()

    # Force individual experts
    configs['force_l10e5'] = get_layer10_expert5_only_config('force')
    configs['force_l13e0'] = get_layer13_expert0_only_config('force')

    # === SOFT BIAS INTERVENTIONS ===
    # Soft bias both harmful experts
    configs['soft_both'] = ExpertInterventionConfig()
    configs['soft_both'].soft_bias_expert(layer=10, expert_id=5, strength=2.0)
    configs['soft_both'].soft_bias_expert(layer=13, expert_id=0, strength=2.0)

    # Soft bias individual experts
    configs['soft_l10e5'] = get_layer10_expert5_only_config('soft')
    configs['soft_l13e0'] = get_layer13_expert0_only_config('soft')

    return configs


def filter_configs(
    all_configs: Dict[str, ExpertInterventionConfig],
    intervention_types: List[str],
    expert_types: List[str],
    skip_baseline: bool
) -> Dict[str, ExpertInterventionConfig]:
    """
    Filter intervention configs based on command line arguments.

    Args:
        all_configs: All available configs
        intervention_types: Types of interventions to run (suppress, force, soft, baseline, all)
        expert_types: Expert configurations to test (combined, l10e5, l13e0, all)
        skip_baseline: Whether to skip baseline

    Returns:
        Filtered dictionary of configs to run
    """
    filtered = {}

    # Handle baseline
    if 'baseline' in intervention_types or 'all' in intervention_types:
        if not skip_baseline:
            filtered['baseline'] = all_configs['baseline']

    # Define intervention type prefixes
    intervention_prefixes = []
    if 'all' in intervention_types:
        intervention_prefixes = ['suppress', 'force', 'soft']
    else:
        intervention_prefixes = intervention_types

    # Define expert suffixes
    expert_suffixes = []
    if 'all' in expert_types:
        expert_suffixes = ['_both', '_l10e5', '_l13e0']
    else:
        if 'combined' in expert_types:
            expert_suffixes.append('_both')
        if 'l10e5' in expert_types:
            expert_suffixes.append('_l10e5')
        if 'l13e0' in expert_types:
            expert_suffixes.append('_l13e0')

    # Build filtered configs
    for prefix in intervention_prefixes:
        for suffix in expert_suffixes:
            config_name = f"{prefix}{suffix}"
            if config_name in all_configs:
                filtered[config_name] = all_configs[config_name]

    return filtered


def run_expert_intervention_pipeline(
    model_path: str,
    n_test: int = None,
    intervention_types: List[str] = ['all'],
    expert_types: List[str] = ['all'],
    skip_baseline: bool = False
):
    """
    Run the complete expert intervention evaluation pipeline.

    Args:
        model_path: Path to the model
        n_test: Number of test examples (None = all)
        intervention_types: Which intervention types to run
        expert_types: Which expert configurations to test
        skip_baseline: Whether to skip baseline evaluation
    """
    print("=" * 80)
    print("EXPERT INTERVENTION PIPELINE FOR OSS-20B MoE MODEL")
    print("=" * 80)

    # Setup configuration
    model_alias = os.path.basename(model_path) + "/expert_intervention"
    cfg = Config(model_alias=model_alias, model_path=model_path)

    # Override n_test if specified
    if n_test is not None:
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

    # Get all intervention configs
    all_configs = get_intervention_configs()

    # Filter configs based on arguments
    configs_to_run = filter_configs(
        all_configs,
        intervention_types,
        expert_types,
        skip_baseline
    )

    print(f"\nRunning {len(configs_to_run)} intervention configurations:")
    for config_name in configs_to_run.keys():
        print(f"  - {config_name}")

    # Load test datasets
    print("\nLoading test datasets...")
    random.seed(42)

    # Harmful test sets from evaluation datasets
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
            # Generate and evaluate completions on harmful datasets
            print("\n--- Evaluating on harmful datasets ---")
            for dataset_name, dataset in harmful_datasets.items():
                print(f"\nProcessing dataset: {dataset_name}")

                # Generate completions
                generate_and_save_completions_for_dataset(
                    cfg=cfg,
                    model_base=model_base,
                    fwd_pre_hooks=[],  # No hooks needed, intervention is in model
                    fwd_hooks=[],
                    intervention_label=config_name,
                    dataset_name=dataset_name,
                    topk=1,  # Not relevant for expert intervention
                    coeff=1.0,  # Not relevant for expert intervention
                    tau=1.0,  # Not relevant for expert intervention
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

            # Generate and evaluate completions on harmless dataset
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

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {cfg.artifact_path()}/completions/")

    # Print summary
    print("\nSummary of interventions run:")
    for config_name in configs_to_run.keys():
        print(f"  ✓ {config_name}")


if __name__ == "__main__":
    args = parse_arguments()

    run_expert_intervention_pipeline(
        model_path=args.model_path,
        n_test=args.n_test,
        intervention_types=args.interventions,
        expert_types=args.experts,
        skip_baseline=args.skip_baseline
    )
