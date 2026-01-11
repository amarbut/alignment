"""
Quick test script for expert intervention system.

Tests the intervention on a few examples to verify it's working.
"""

import torch
from pipeline.model_utils.model_factory import construct_model_base
from expert_explore.expert_intervention_hooks_v3 import (
    ExpertInterventionConfig,
    apply_expert_interventions,
    remove_expert_interventions,
    print_intervention_summary
)
from dataset.load_dataset import load_dataset_split

def test_expert_intervention():
    """Test expert intervention with a few examples."""

    print("Loading model...")
    model_base = construct_model_base("openai/gpt-oss-20b")
    print("Model loaded!")

    # Test prompts (just take first 5 for quick testing)
    test_prompts = load_dataset_split(harmtype="harmful", split="val")[:5]

    # Test different interventions
    interventions = {
        "baseline": ExpertInterventionConfig(),
        "suppress_l10e5": ExpertInterventionConfig().suppress_expert(10, 5, -10.0),
        "force_l10e5": ExpertInterventionConfig().force_expert(10, 5, 10.0),
    }

    for intervention_name, config in interventions.items():
        print("\n" + "=" * 80)
        print(f"Testing: {intervention_name}")
        print("=" * 80)
        print_intervention_summary(config)

        # Apply interventions
        original_biases = apply_expert_interventions(model_base, config)

        # Generate completions (no hooks needed, interventions are in the model)
        completions = model_base.generate_completions(
            test_prompts,
            fwd_pre_hooks=[],
            fwd_hooks=[],
            max_new_tokens=50,
            batch_size=3
        )

        # Print results
        for i, completion in enumerate(completions):
            print(f"\nPrompt {i+1}: {completion['prompt'][:60]}...")
            print(f"Response: {completion['response'][:100]}...")

        # Remove interventions
        remove_expert_interventions(model_base, original_biases)

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_expert_intervention()
