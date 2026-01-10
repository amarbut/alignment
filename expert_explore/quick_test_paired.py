"""
Quick test for paired intervention pipeline.

Tests a few examples with the new refusal/response induction interventions.
"""

import sys
sys.path.insert(0, '/media/volume/align_2_stg/alignment')

from expert_explore.run_paired_interventions import run_paired_intervention_pipeline

if __name__ == "__main__":
    print("Running quick test of paired intervention pipeline...")
    print("This will test refusal/response induction with only 5 examples.\n")

    # Run with minimal settings
    run_paired_intervention_pipeline(
        model_path='openai/gpt-oss-20b',
        n_test=5,
        skip_baseline=False,
        skip_combined=True,  # Skip combined for quicker test
        layers=['l10']  # Just test layer 10 for speed
    )

    print("\n" + "=" * 80)
    print("Quick test completed successfully!")
    print("You can now run the full paired intervention pipeline.")
    print("=" * 80)
