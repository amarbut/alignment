"""
Quick test script for expert intervention system.

Tests the intervention on a few examples to verify it's working.
"""

import torch
from pipeline.model_utils.model_factory import construct_model_base
from expert_intervention_hooks import (
    ExpertInterventionConfig,
    get_expert_intervention_hooks,
    print_intervention_summary
)
from dataset.load_dataset import load_dataset_split

def test_expert_intervention():
    """Test expert intervention with a few examples."""

    print("Loading model...")
    model_base = construct_model_base("openai/gpt-oss-20b")
    print("Model loaded!")

    # Test prompts
    test_prompts = load_dataset_split(harmtype = "harmful", split = "val")

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

        # Get hooks
        fwd_pre_hooks, fwd_hooks = get_expert_intervention_hooks(model_base, config)

        # Generate completions
        completions = model_base.generate_completions(
            test_prompts,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            max_new_tokens=50,
            batch_size=3
        )

        # Print results
        for i, completion in enumerate(completions):
            print(f"\nPrompt {i+1}: {completion['instruction'][:60]}...")
            print(f"Response: {completion['completion'][:100]}...")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_expert_intervention()
