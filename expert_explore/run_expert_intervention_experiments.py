"""
Run expert routing intervention experiments.

This script tests the causal effect of expert selection on model behavior by:
1. Manipulating which experts are used at inference time
2. Evaluating the resulting completions on jailbreak datasets
3. Comparing refusal rates across different interventions
"""

import torch
import json
import os
import argparse
from pathlib import Path

from dataset.load_dataset import load_dataset
from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.run_pipeline import (
    generate_and_save_completions_for_dataset,
    evaluate_completions_and_save_results_for_dataset
)

from expert_intervention_hooks_v3 import (
    ExpertInterventionConfig,
    apply_expert_interventions,
    remove_expert_interventions,
    print_intervention_summary,
    get_harmful_expert_suppression_config,
    get_harmful_expert_forcing_config,
    get_layer10_expert5_only_config,
    get_layer13_expert1_only_config,
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run expert intervention experiments")
    parser.add_argument('--model_path', type=str, default='openai/gpt-oss-20b',
                        help='Path to the model')
    parser.add_argument('--output_dir', type=str, default='expert_intervention_results',
                        help='Output directory for results')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['jailbreakbench', 'harmless'],
                        help='Datasets to evaluate on')
    parser.add_argument('--experiments', type=str, nargs='+',
                        default=['baseline', 'suppress_harmful', 'force_harmful'],
                        help='Which experiments to run')
    parser.add_argument('--max_new_tokens', type=int, default=64,
                        help='Maximum new tokens to generate')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for generation')
    return parser.parse_args()


def get_experiment_configs():
    """
    Define all experiment configurations.

    Returns:
        Dictionary mapping experiment name to (config, description)
    """
    experiments = {
        'baseline': (
            ExpertInterventionConfig(),
            "Baseline (no intervention)"
        ),
        'suppress_harmful': (
            get_harmful_expert_suppression_config(),
            "Suppress harmful-associated experts (L10E5, L13E1)"
        ),
        'force_harmful': (
            get_harmful_expert_forcing_config(),
            "Force harmful-associated experts (L10E5, L13E1)"
        ),
        'suppress_l10e5': (
            get_layer10_expert5_only_config('suppress'),
            "Suppress Layer 10, Expert 5 only"
        ),
        'force_l10e5': (
            get_layer10_expert5_only_config('force'),
            "Force Layer 10, Expert 5 only"
        ),
        'suppress_l13e1': (
            get_layer13_expert1_only_config('suppress'),
            "Suppress Layer 13, Expert 1 only"
        ),
        'force_l13e1': (
            get_layer13_expert1_only_config('force'),
            "Force Layer 13, Expert 1 only"
        ),
    }
    return experiments


def run_experiment(model_base, config, experiment_name, description, datasets, output_dir, max_new_tokens):
    """
    Run a single expert intervention experiment.

    Args:
        model_base: The model wrapper
        config: ExpertInterventionConfig for this experiment
        experiment_name: Name identifier for this experiment
        description: Human-readable description
        datasets: List of dataset names to evaluate
        output_dir: Base output directory
        max_new_tokens: Max tokens to generate
    """
    print("\n" + "=" * 80)
    print(f"Running Experiment: {experiment_name}")
    print(f"Description: {description}")
    print("=" * 80)

    # Print intervention summary
    print_intervention_summary(config)

    # Apply intervention by modifying router biases
    print("\nApplying router bias modifications...")
    original_biases = apply_expert_interventions(model_base, config)
    print(f"Modified {len(original_biases)} layers")

    # Create experiment output directory
    exp_dir = Path(output_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    with open(exp_dir / "config.json", 'w') as f:
        config_dict = {
            "description": description,
            "interventions": {
                f"layer_{layer}_expert_{expert}": strength
                for (layer, expert), strength in config.get_all_interventions().items()
            }
        }
        json.dump(config_dict, f, indent=2)

    # Generate and evaluate completions for each dataset
    for dataset_name in datasets:
        print(f"\n--- Processing {dataset_name} ---")

        # Load dataset
        dataset = load_dataset(dataset_name)
        print(f"Loaded {len(dataset)} examples")

        # Generate completions (router biases already modified)
        print("Generating completions...")
        completions = model_base.generate_completions(
            dataset,
            fwd_pre_hooks=[],
            fwd_hooks=[],
            max_new_tokens=max_new_tokens
        )

        # Save completions
        completions_path = exp_dir / f"{dataset_name}_completions.json"
        with open(completions_path, 'w', encoding='utf-8') as f:
            json.dump(completions, f, indent=2, ensure_ascii=False)
        print(f"Saved completions to {completions_path}")

        # Evaluate completions
        print("Evaluating completions...")
        from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak

        eval_methodologies = ['substring_matching', 'llamaguard']
        evaluations_path = exp_dir / f"{dataset_name}_evaluations.json"

        evaluation = evaluate_jailbreak(
            completions=completions,
            methodologies=eval_methodologies,
            evaluation_path=str(evaluations_path),
        )

        with open(evaluations_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)
        print(f"Saved evaluations to {evaluations_path}")

        # Print summary statistics
        if evaluation:
            print(f"\nResults for {dataset_name}:")
            for method, results in evaluation.items():
                if isinstance(results, dict) and 'average' in results:
                    avg_score = results['average']
                    print(f"  {method}: {avg_score:.3f}")

    # Clean up: restore original router biases
    print("\nRestoring original router biases...")
    remove_expert_interventions(model_base, original_biases)


def summarize_all_results(output_dir, experiments, datasets):
    """
    Create a summary comparing all experiments.

    Args:
        output_dir: Base output directory
        experiments: List of experiment names
        datasets: List of dataset names
    """
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("=" * 80)

    summary = {}

    for exp_name in experiments:
        exp_dir = Path(output_dir) / exp_name
        summary[exp_name] = {}

        for dataset_name in datasets:
            eval_path = exp_dir / f"{dataset_name}_evaluations.json"

            if eval_path.exists():
                with open(eval_path, 'r') as f:
                    evaluation = json.load(f)

                summary[exp_name][dataset_name] = {}
                for method, results in evaluation.items():
                    if isinstance(results, dict) and 'average' in results:
                        summary[exp_name][dataset_name][method] = results['average']

    # Print comparison table
    print("\nRefusal Rate Comparison:")
    print("-" * 80)

    for dataset_name in datasets:
        print(f"\n{dataset_name.upper()}:")
        print(f"{'Experiment':<30} | {'Substring Match':<15} | {'LlamaGuard':<15}")
        print("-" * 80)

        for exp_name in experiments:
            if dataset_name in summary.get(exp_name, {}):
                results = summary[exp_name][dataset_name]
                substring = results.get('substring_matching', -1)
                llamaguard = results.get('llamaguard', -1)
                print(f"{exp_name:<30} | {substring:>14.3f} | {llamaguard:>14.3f}")

    # Save summary JSON
    summary_path = Path(output_dir) / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary saved to {summary_path}")


def main():
    args = parse_arguments()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load model
    print(f"Loading model: {args.model_path}")
    model_base = construct_model_base(args.model_path)
    print("Model loaded successfully!")

    # Get all available experiments
    all_experiments = get_experiment_configs()

    # Filter to requested experiments
    experiments_to_run = {
        name: (config, desc)
        for name, (config, desc) in all_experiments.items()
        if name in args.experiments or 'all' in args.experiments
    }

    if not experiments_to_run:
        print(f"Warning: No valid experiments specified. Available: {list(all_experiments.keys())}")
        return

    print(f"\nWill run {len(experiments_to_run)} experiments:")
    for name, (_, desc) in experiments_to_run.items():
        print(f"  - {name}: {desc}")

    # Run each experiment
    for exp_name, (config, description) in experiments_to_run.items():
        try:
            run_experiment(
                model_base=model_base,
                config=config,
                experiment_name=exp_name,
                description=description,
                datasets=args.datasets,
                output_dir=args.output_dir,
                max_new_tokens=args.max_new_tokens
            )
        except Exception as e:
            print(f"\nError in experiment {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate summary
    summarize_all_results(args.output_dir, list(experiments_to_run.keys()), args.datasets)

    print("\n" + "=" * 80)
    print("All experiments complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
