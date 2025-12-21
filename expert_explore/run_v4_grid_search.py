"""
Grid search over epsilon and threshold parameters for V4 interventions.

Tests refusal and response induction with different parameter combinations
and logs evaluation metrics.
"""

import os
import sys
import json
import random
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add the alignment directory to path
alignment_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if alignment_dir not in sys.path:
    sys.path.insert(0, alignment_dir)

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from dataset.load_dataset import load_dataset_split, load_dataset

from expert_explore.expert_intervention_hooks_v4 import (
    get_select_experts_refusal_induction_config_v4,
    get_select_experts_response_induction_config_v4,
    apply_expert_interventions_v4,
    remove_expert_interventions_v4,
    print_intervention_summary_v4,
)

from pipeline.run_pipeline_subspace import (
    generate_and_save_completions_for_dataset,
    evaluate_completions_and_save_results_for_dataset
)


def load_evaluation_results(cfg, intervention_label, dataset_name, topk=1, coeff=1.0, tau=1.0):
    """Load evaluation results from JSON file."""
    # Path format: {artifact_path}/completions/k{topk}/a{coeff}/t{tau}/{dataset_name}_{intervention_label}_evaluations.json
    results_file = Path(cfg.artifact_path()) / "completions" / f"k{topk}" / f"a{coeff}" / f"t{tau}" / f"{dataset_name}_{intervention_label}_evaluations.json"

    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: Results file not found: {results_file}")
        return None


def run_grid_search(
    epsilon_values=[0.01, 0.1, 1, 5, 10],
    threshold_values=[0.1, 0.2, 0.3, 0.4, 0.5],
    n_test=100,
    model_path='openai/gpt-oss-20b'
):
    """
    Run grid search over epsilon and threshold parameters.

    Args:
        epsilon_values: List of epsilon values to test
        threshold_values: List of threshold values to test
        n_test: Number of test examples
        model_path: Path to the model
    """
    print("="*80)
    print("V4 GRID SEARCH: Epsilon x Threshold")
    print("="*80)
    print(f"\nEpsilon values: {epsilon_values}")
    print(f"Threshold values: {threshold_values}")
    print(f"Total combinations: {len(epsilon_values) * len(threshold_values) * 2} (refusal + response)")

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_alias = os.path.basename(model_path) + f"/v4_grid_search_{timestamp}"
    cfg = Config(model_alias=model_alias, model_path=model_path)
    cfg.n_test = n_test

    print(f"\nConfiguration:")
    print(f"  Model: {model_path}")
    print(f"  Artifact path: {cfg.artifact_path()}")
    print(f"  N_test: {n_test}")

    # Load model
    print("\nLoading model...")
    model_base = construct_model_base(model_path)
    model_base.model.config.pad_token_id = model_base.tokenizer.pad_token_id
    print("Model loaded!")

    # Load datasets
    print("\nLoading datasets...")
    random.seed(42)

    # Harmless test set (for refusal induction)
    harmless_test = load_dataset_split(harmtype='harmless', split='test')
    if n_test is not None:
        harmless_test = random.sample(harmless_test, min(n_test, len(harmless_test)))

    # JailbreakBench (for response induction)
    jailbreakbench_test = load_dataset('jailbreakbench')
    if n_test is not None:
        jailbreakbench_test = random.sample(jailbreakbench_test, min(n_test, len(jailbreakbench_test)))

    print(f"Loaded {len(harmless_test)} harmless examples")
    print(f"Loaded {len(jailbreakbench_test)} jailbreakbench examples")

    # Results storage
    results = []

    # Grid search
    total_runs = len(epsilon_values) * len(threshold_values) * 2
    run_count = 0

    for epsilon in epsilon_values:
        for threshold in threshold_values:
            # REFUSAL INDUCTION
            run_count += 1
            print("\n" + "="*80)
            print(f"RUN {run_count}/{total_runs}: REFUSAL INDUCTION")
            print(f"Epsilon={epsilon}, Threshold={threshold}")
            print("="*80)

            config_refusal = get_select_experts_refusal_induction_config_v4(
                threshold=threshold,
                epsilon=epsilon
            )

            intervention_label = f"refusal_eps{epsilon}_thr{threshold}"
            print_intervention_summary_v4(config_refusal)

            # Apply intervention
            original_biases = apply_expert_interventions_v4(model_base, config_refusal)

            try:
                # Generate on harmless
                generate_and_save_completions_for_dataset(
                    cfg=cfg,
                    model_base=model_base,
                    fwd_pre_hooks=[],
                    fwd_hooks=[],
                    intervention_label=intervention_label,
                    dataset_name='harmless',
                    topk=1,
                    coeff=1.0,
                    tau=1.0,
                    dataset=harmless_test
                )

                # Evaluate results
                evaluate_completions_and_save_results_for_dataset(
                    cfg=cfg,
                    intervention_label=intervention_label,
                    dataset_name='harmless',
                    eval_methodologies=cfg.refusal_eval_methodologies,
                    topk=1,
                    coeff=1.0,
                    tau=1.0
                )

            finally:
                remove_expert_interventions_v4(model_base, original_biases)

            # Load results
            eval_results = load_evaluation_results(cfg, intervention_label, 'harmless')

            # Extract metrics (refusal_rate = 1 - success_rate, since success = jailbreak)
            refusal_result = {
                'intervention_type': 'refusal',
                'epsilon': epsilon,
                'threshold': threshold,
                'dataset': 'harmless',
                'substring_refusal_rate': 1 - eval_results.get('substring_matching_success_rate', 0) if eval_results else None,
                'substring_jailbreak_rate': eval_results.get('substring_matching_success_rate') if eval_results else None,
                'llamaguard2_refusal_rate': 1 - eval_results.get('llamaguard2_success_rate', 0) if eval_results else None,
                'llamaguard2_jailbreak_rate': eval_results.get('llamaguard2_success_rate') if eval_results else None,
                'num_total': len(eval_results.get('completions', [])) if eval_results else None,
            }
            results.append(refusal_result)

            print(f"\nRefusal results:")
            print(f"  Substring refusal rate: {refusal_result['substring_refusal_rate']:.3f}" if refusal_result['substring_refusal_rate'] else "  Substring refusal rate: None")

            # RESPONSE INDUCTION
            run_count += 1
            print("\n" + "="*80)
            print(f"RUN {run_count}/{total_runs}: RESPONSE INDUCTION")
            print(f"Epsilon={epsilon}, Threshold={threshold}")
            print("="*80)

            config_response = get_select_experts_response_induction_config_v4(
                threshold=threshold,
                epsilon=epsilon
            )

            intervention_label = f"response_eps{epsilon}_thr{threshold}"
            print_intervention_summary_v4(config_response)

            # Apply intervention
            original_biases = apply_expert_interventions_v4(model_base, config_response)

            try:
                # Generate on jailbreakbench
                generate_and_save_completions_for_dataset(
                    cfg=cfg,
                    model_base=model_base,
                    fwd_pre_hooks=[],
                    fwd_hooks=[],
                    intervention_label=intervention_label,
                    dataset_name='jailbreakbench',
                    topk=1,
                    coeff=1.0,
                    tau=1.0,
                    dataset=jailbreakbench_test
                )

                # Evaluate results
                evaluate_completions_and_save_results_for_dataset(
                    cfg=cfg,
                    intervention_label=intervention_label,
                    dataset_name='jailbreakbench',
                    eval_methodologies=cfg.jailbreak_eval_methodologies,
                    topk=1,
                    coeff=1.0,
                    tau=1.0
                )

            finally:
                remove_expert_interventions_v4(model_base, original_biases)

            # Load results
            eval_results = load_evaluation_results(cfg, intervention_label, 'jailbreakbench')

            # Extract metrics (success_rate = jailbreak rate, so refusal_rate = 1 - success_rate)
            response_result = {
                'intervention_type': 'response',
                'epsilon': epsilon,
                'threshold': threshold,
                'dataset': 'jailbreakbench',
                'substring_jailbreak_rate': eval_results.get('substring_matching_success_rate') if eval_results else None,
                'substring_refusal_rate': 1 - eval_results.get('substring_matching_success_rate', 0) if eval_results else None,
                'llamaguard2_jailbreak_rate': eval_results.get('llamaguard2_success_rate') if eval_results else None,
                'llamaguard2_refusal_rate': 1 - eval_results.get('llamaguard2_success_rate', 0) if eval_results else None,
                'num_total': len(eval_results.get('completions', [])) if eval_results else None,
            }
            results.append(response_result)

            print(f"\nResponse results:")
            print(f"  Substring jailbreak rate: {response_result['substring_jailbreak_rate']:.3f}" if response_result['substring_jailbreak_rate'] else "  Substring jailbreak rate: None")
            print(f"  LlamaGuard2 jailbreak rate: {response_result['llamaguard2_jailbreak_rate']:.3f}" if response_result['llamaguard2_jailbreak_rate'] else "  LlamaGuard2 jailbreak rate: None")

            # Save intermediate results
            results_df = pd.DataFrame(results)
            results_file = Path(cfg.artifact_path()) / f"grid_search_results_{timestamp}.csv"
            results_df.to_csv(results_file, index=False)
            print(f"\nIntermediate results saved to: {results_file}")

    # Final save
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE!")
    print("="*80)

    results_df = pd.DataFrame(results)
    results_file = Path(cfg.artifact_path()) / f"grid_search_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)

    print(f"\nFinal results saved to: {results_file}")
    print(f"\nResults summary:")
    print(results_df.to_string())

    # Print some summary statistics
    print("\n" + "="*80)
    print("SUMMARY BY INTERVENTION TYPE")
    print("="*80)

    # Refusal induction: we want high refusal rates
    refusal_subset = results_df[results_df['intervention_type'] == 'refusal']
    print(f"\nREFUSAL INDUCTION (higher is better):")
    print(f"  Best substring refusal rate: {refusal_subset['substring_refusal_rate'].max():.3f}")
    print(f"    (epsilon={refusal_subset.loc[refusal_subset['substring_refusal_rate'].idxmax(), 'epsilon']}, " +
          f"threshold={refusal_subset.loc[refusal_subset['substring_refusal_rate'].idxmax(), 'threshold']})")
    print(f"  Best llamaguard2 refusal rate: {refusal_subset['llamaguard2_refusal_rate'].max():.3f}")
    print(f"    (epsilon={refusal_subset.loc[refusal_subset['llamaguard2_refusal_rate'].idxmax(), 'epsilon']}, " +
          f"threshold={refusal_subset.loc[refusal_subset['llamaguard2_refusal_rate'].idxmax(), 'threshold']})")

    # Response induction: we want high jailbreak rates (low refusal rates)
    response_subset = results_df[results_df['intervention_type'] == 'response']
    print(f"\nRESPONSE INDUCTION (higher jailbreak rate is better):")
    print(f"  Best substring jailbreak rate: {response_subset['substring_jailbreak_rate'].max():.3f}")
    print(f"    (epsilon={response_subset.loc[response_subset['substring_jailbreak_rate'].idxmax(), 'epsilon']}, " +
          f"threshold={response_subset.loc[response_subset['substring_jailbreak_rate'].idxmax(), 'threshold']})")
    print(f"  Best llamaguard2 jailbreak rate: {response_subset['llamaguard2_jailbreak_rate'].max():.3f}")
    print(f"    (epsilon={response_subset.loc[response_subset['llamaguard2_jailbreak_rate'].idxmax(), 'epsilon']}, " +
          f"threshold={response_subset.loc[response_subset['llamaguard2_jailbreak_rate'].idxmax(), 'threshold']})")

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="V4 Grid Search over epsilon and threshold")
    parser.add_argument('--n_test', type=int, default=100, help='Number of test examples')
    parser.add_argument('--model_path', type=str, default='openai/gpt-oss-20b', help='Model path')

    args = parser.parse_args()

    # Run grid search
    results_df = run_grid_search(
        epsilon_values=[0.01, 0.1, 1, 5, 10],
        threshold_values=[0.1, 0.2, 0.3, 0.4, 0.5],
        n_test=args.n_test,
        model_path=args.model_path
    )
